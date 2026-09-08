"""Per-utterance emotion / accent similarity, for per-emotion, per-accent and per-speaker breakdowns.

articulatory-tts's score_side_metric.py only writes the aggregate {mean, ci95, n}. This script embeds
the same 16 kHz {uid}_pred.wav/{uid}_gt.wav pairs with the SAME model and call path and keeps every
per-utterance value, then folds group summaries into the dataset's eval_<dataset>.json
("by_emotion"/"by_accent"/"by_speaker" -> emotion_cosine / accent_cosine) and asserts the overall
mean reproduces the merged side metric. It mirrors the two sibling implementations:
  --metric accent   /home/xoy/TTS/eval/score_accent_per_utt.py: Jzuluaga/accent-id-commonaccent_xlsr-en-english
                    via speechbrain foreign_class, encode_batch() (pooled wav2vec2-XLSR encoder output
                    before the head), cosine per pair, 0.1 s minimum duration; plus the classifier's own
                    16-way top label for pred and GT -> "accent_label_agreement" per group (a
                    classification view, kept separate from the cosine metric). Run in the eval-accent venv.
  --metric emotion  /home/xoy/EmoSpherepp/eval/emotion_cosine_per_utt.py: iic/emotion2vec_plus_large via
                    funasr (hub=hf), extract_embedding=True utterance-level feats, cosine per pair, 0.1 s
                    minimum duration; plus emotion2vec's own 9-way top label for pred and GT ->
                    "emotion_label_agreement" per group. Run in the eval-emotion venv.
Resumable: the per-utterance JSON is rewritten every 100 pairs and existing uids are skipped.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_HOME", "/data/user_data/xoy/.cache/huggingface")

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthesize_testset import load_items  # noqa: E402

MIN_PAIR_DURATION_SEC = 0.1  # same floor as score_side_metric.py

# VCTK speaker-info accent group -> CommonAccent's 16 labels (african, australia, bermuda, canada, england,
# hongkong, indian, ireland, malaysia, newzealand, philippines, scotland, singapore, southatlandtic, us, wales).
VCTK_TO_COMMONACCENT = {"American": "us", "Canadian": "canada", "English": "england", "Irish": "ireland",
                        "NorthernIrish": "ireland", "Scottish": "scotland", "Australian": "australia",
                        "Indian": "indian", "Welsh": "wales", "SouthAfrican": "african", "NewZealand": "newzealand"}
# ESD emotion -> emotion2vec+ large's 9 labels (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown)
ESD_TO_EMOTION2VEC = {"Angry": "angry", "Happy": "happy", "Neutral": "neutral", "Sad": "sad", "Surprise": "surprised"}


def summarize(values):
    values = np.array([v for v in values if v is not None], dtype=np.float64)
    if len(values) == 0:
        return {"mean": None, "ci95": None, "n": 0}
    return {"mean": float(values.mean()), "ci95": float(1.96 * values.std() / np.sqrt(len(values))), "n": int(len(values))}


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def too_short(path):
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate < MIN_PAIR_DURATION_SEC


def write_json_atomic(obj, path, **kw):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, **kw)
    os.replace(path + ".tmp", path)


def build_accent_scorer(device):
    from speechbrain.inference.interfaces import foreign_class
    classifier = foreign_class(
        source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        run_opts={"device": str(device)},
        savedir=os.path.join(os.environ["HF_HOME"], "speechbrain", "commonaccent_xlsr"),
        overrides={"wav2vec2": {"save_path": os.path.join(os.environ["HF_HOME"], "wav2vec2_checkpoints")}},
    )

    def embed(path):
        waveform = classifier.load_audio(path)
        return classifier.encode_batch(waveform.unsqueeze(0)).detach().cpu().numpy().reshape(-1)

    def label(path):
        out_prob, score, index, text_lab = classifier.classify_file(path)
        lab = text_lab[0] if isinstance(text_lab, (list, tuple)) else str(text_lab)
        if hasattr(score, "detach"):
            score = score.detach().cpu()
        return lab, float(np.asarray(score).reshape(-1)[0])
    return embed, label


def build_emotion_scorer(device):
    from funasr import AutoModel
    model = AutoModel(model="iic/emotion2vec_plus_large", hub="hf", device=str(device))

    def run(path):
        res = model.generate(path, granularity="utterance", extract_embedding=True)[0]
        emb = np.asarray(res["feats"]).reshape(-1)
        scores = np.asarray(res.get("scores", []), dtype=np.float64).reshape(-1)
        labels = res.get("labels", [])
        if len(scores) and len(labels) == len(scores):
            i = int(scores.argmax())
            lab = str(labels[i]).split("/")[-1]  # funasr labels look like "生气/angry"
            return emb, lab, float(scores[i])
        return emb, None, None
    cache = {}

    def embed(path):
        if path not in cache:
            cache[path] = run(path)
        return cache[path][0]

    def label(path):
        if path not in cache:
            cache[path] = run(path)
        return cache[path][1], cache[path][2]
    return embed, label


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", required=True, choices=["emotion", "accent"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--wav_pairs_dir", required=True)
    ap.add_argument("--results_path", required=True, help="eval_<dataset>.json to update in place (by_* groups)")
    ap.add_argument("--per_utt_out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_key = f"{args.metric}_cosine"
    label_key = f"{args.metric}_label_agreement"
    target_map = ESD_TO_EMOTION2VEC if args.metric == "emotion" else VCTK_TO_COMMONACCENT
    group_field = "emotion" if args.metric == "emotion" else "accent_label"

    items = load_items(args.dataset)
    if args.limit:
        items = items[: args.limit]
    records = json.load(open(args.per_utt_out)) if os.path.exists(args.per_utt_out) else {}
    todo = [it for it in items if it["uid"] not in records]
    print(f"{len(items)} utterances; {len(records)} already scored; scoring {len(todo)}")
    embed = label = None
    if todo:
        embed, label = (build_emotion_scorer if args.metric == "emotion" else build_accent_scorer)(device)

    n_skipped = 0
    with torch.no_grad():
        for i, it in enumerate(tqdm(todo, desc=f"{metric_key} per utt")):
            uid_safe = it["uid"].replace("/", "_")
            pred = os.path.join(args.wav_pairs_dir, f"{uid_safe}_pred.wav")
            gt = os.path.join(args.wav_pairs_dir, f"{uid_safe}_gt.wav")
            rec = {"speaker": it["speaker"], "emotion": it["emotion"], "accent_label": it["accent_label"], metric_key: None}
            if not (os.path.exists(pred) and os.path.exists(gt)):
                n_skipped += 1
                rec["skipped"] = "missing pair"
            elif too_short(pred) or too_short(gt):
                tqdm.write(f"WARNING: skipping {it['uid']} -- pred/gt wav too short (min={MIN_PAIR_DURATION_SEC}s)")
                n_skipped += 1
                rec["skipped"] = "too short"
            else:
                rec[metric_key] = cosine(embed(pred), embed(gt))
                pl, ps = label(pred)
                gl, gs = label(gt)
                rec.update({"pred_label": pl, "pred_label_score": ps, "gt_label": gl, "gt_label_score": gs})
            records[it["uid"]] = rec
            if (i + 1) % 100 == 0:
                write_json_atomic(records, args.per_utt_out, indent=1)
    write_json_atomic(records, args.per_utt_out, indent=1)
    print(f"scored {sum(1 for r in records.values() if r.get(metric_key) is not None)} pairs, skipped {n_skipped}")

    overall = summarize([r[metric_key] for r in records.values()])
    print(f"overall {metric_key}:", overall)

    def by(field):
        groups = {}
        for r in records.values():
            groups.setdefault(r[field], []).append(r)
        out = {}
        for g, recs in sorted(groups.items()):
            entry = {"n": len(recs), metric_key: summarize([r[metric_key] for r in recs])}
            # Label agreement is only meaningful for a group with ONE target label (an emotion group, an
            # accent group, or a VCTK speaker = one accent); an ESD speaker spans all five emotions.
            homogeneous = len({r[group_field] for r in recs}) == 1
            target = target_map.get(recs[0][group_field]) if homogeneous else None
            labelled = [r for r in recs if r.get("pred_label") is not None]
            if target and labelled:
                entry[label_key] = {
                    "target_label": target,
                    "pred_labelled_as_target": float(np.mean([r["pred_label"] == target for r in labelled])),
                    "gt_labelled_as_target": float(np.mean([r["gt_label"] == target for r in labelled])),
                    "pred_matches_gt_label": float(np.mean([r["pred_label"] == r["gt_label"] for r in labelled])),
                    "n": len(labelled),
                }
            out[g] = entry
        return out

    with open(args.results_path) as f:
        results = json.load(f)
    merged = results.get("metrics", {}).get(metric_key, {}).get("mean")
    if merged is not None and overall["mean"] is not None and not args.limit:
        diff = abs(merged - overall["mean"])
        print(f"consistency vs merged {metric_key} {merged:.4f}: |diff|={diff:.5f}")
        assert diff < 2e-3, f"per-utterance recomputation does not reproduce the merged {metric_key} -- not writing"
    for field, out_key in (("emotion", "by_emotion"), ("accent_label", "by_accent"), ("speaker", "by_speaker")):
        if len({r[field] for r in records.values()}) > 1:
            results.setdefault(out_key, {})
            for g, entry in by(field).items():
                results[out_key].setdefault(g, {}).update(entry)
    results[f"{metric_key}_per_utt_source"] = (f"eval/score_side_per_utt.py --metric {args.metric} "
                                              f"(same model/call path as articulatory-tts score_side_metric.py)")
    write_json_atomic(results, args.results_path, indent=2)
    for out_key in ("by_emotion", "by_accent"):
        if out_key in results and any(metric_key in e for e in results[out_key].values()):
            print(f"{out_key:14s} {'n':>5s} {metric_key:>15s} {'pred->target':>12s} {'gt->target':>10s} {'pred==gt':>8s}")
            for g, e in results[out_key].items():
                if metric_key not in e:
                    continue
                agr = e.get(label_key, {})
                m = e[metric_key]["mean"]
                print(f"{g:14s} {e['n']:5d} {m if m is None else round(m, 4)!s:>15s} "
                      f"{agr.get('pred_labelled_as_target', float('nan')):12.3f} {agr.get('gt_labelled_as_target', float('nan')):10.3f} "
                      f"{agr.get('pred_matches_gt_label', float('nan')):8.3f}")
    print(f"wrote {args.per_utt_out} and updated {args.results_path}")


if __name__ == "__main__":
    main()
