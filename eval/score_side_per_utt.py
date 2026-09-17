"""Per-utterance emotion / accent similarity, for per-emotion, per-accent and per-speaker breakdowns.

articulatory-tts's score_side_metric.py only writes the aggregate {mean, ci95, n}. This script embeds
the same 16 kHz {uid}_pred.wav/{uid}_gt.wav pairs with the SAME model and call path and keeps every
per-utterance value, then folds group summaries into the dataset's eval_<dataset>.json
("by_emotion"/"by_accent"/"by_speaker" -> emotion_cosine / accent_cosine) and asserts the overall
mean reproduces the merged side metric. It mirrors the two sibling implementations:
  --metric accent   /home/xoy/TTS/eval/score_accent_per_utt.py: GenAID (jzmzhong/GenAID, the authors'
                    GenAID_v6 checkpoint via /home/xoy/articulatory-tts/genaid_accent.py -- 64-dim preout_mlp
                    embedding after statistics pooling over the fine-tuned XLSR-53 frames), cosine per pair,
                    0.1 s minimum duration; plus GenAID's 13-way top label for pred and GT ->
                    "accent_label_agreement" per group (a classification view, kept separate from the cosine
                    metric). Run in the eval-genaid venv. 2026-09-14: replaced CommonAccent
                    (Jzuluaga/accent-id-commonaccent_xlsr-en-english, speechbrain foreign_class encode_batch()).
  --metric emotion  /home/xoy/EmoSpherepp/eval/emotion_cosine_per_utt.py: iic/emotion2vec_plus_large via
                    funasr (hub=hf), extract_embedding=True utterance-level feats, cosine per pair, 0.1 s
                    minimum duration; plus emotion2vec's own 9-way top label for pred and GT ->
                    "emotion_label_agreement" per group. Run in the eval-emotion venv.
Resumable: the per-utterance JSON is rewritten every 100 pairs and existing uids are skipped.

2026-09-16: for --metric accent, the published accent_cosine is CENTERED -- both the predicted and
ground-truth GenAID embeddings have genaid_accent.DEFAULT_CENTER_VECTOR (the mean of the six
speaker-balanced VCTK-training-speaker accent centroids, see articulatory-tts CLAUDE.md
"Accent-metric diagnostic") subtracted before the cosine, via genaid_accent.accent_cosines(); the raw
(un-centered) cosine is kept alongside per utterance and per group as accent_cosine_genaid_raw, and
each centered per-utterance record also carries its own "center_vector" path (the per-utt JSON is a
flat uid->record map with no top-level metadata slot, so this is what a re-run checks to tell an
already-centered file apart from a pre-centering raw one before deciding whether to move it aside).
--center_vector overrides the vector, --no_center reports the raw cosine only (same flags as
articulatory-tts's score_side_metric.py / score_side_per_utt.py). The by_emotion/by_accent/by_speaker
summaries and the overall consistency check against the merged eval_<dataset>.json always use the
centered accent_cosine; results["accent_center_vector"] records the vector path used. The --metric
emotion path is unaffected by any of this.
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

ART_REPO = "/home/xoy/articulatory-tts"  # genaid_accent.py lives with the reference side metric
sys.path.insert(0, ART_REPO)

MIN_PAIR_DURATION_SEC = 0.1  # same floor as score_side_metric.py

# VCTK speaker-info accent group -> GenAID's 13 labels, where one exists (label-agreement view only)
from genaid_accent import VCTK_TO_GENAID  # noqa: E402
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
    import genaid_accent
    embedder = genaid_accent.GenAIDEmbedder(device)
    print("accent model:", genaid_accent.MODEL_TAG)
    cache = {}

    def run(path):
        if path not in cache:
            cache[path] = embedder.embed_and_label(path)
        return cache[path]

    def embed(path):
        return run(path)[0]

    def label(path):
        return run(path)[1], run(path)[2]
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
    ap.add_argument("--center_vector", default=None,
                     help="--metric accent: .npy centering vector subtracted from both embeddings before the "
                          "cosine (default: genaid_accent.DEFAULT_CENTER_VECTOR, the 2026-09-16 centroid mean)")
    ap.add_argument("--no_center", action="store_true", help="--metric accent: report the raw GenAID cosine only")
    args = ap.parse_args()

    import torch
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_key = f"{args.metric}_cosine"
    raw_key = "accent_cosine_genaid_raw"  # --metric accent only, populated when centering is on
    label_key = f"{args.metric}_label_agreement"
    target_map = ESD_TO_EMOTION2VEC if args.metric == "emotion" else VCTK_TO_GENAID
    group_field = "emotion" if args.metric == "emotion" else "accent_label"

    mu, center_path, model_tag = None, None, ""
    if args.metric == "accent":
        import genaid_accent
        center_path = None if args.no_center else (args.center_vector or genaid_accent.DEFAULT_CENTER_VECTOR)
        mu = genaid_accent.load_center_vector(center_path) if center_path else None
        model_tag = genaid_accent.centered_model_tag(center_path) if center_path else genaid_accent.MODEL_TAG

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
                if args.metric == "accent":
                    centered, raw = genaid_accent.accent_cosines(embed(pred), embed(gt), mu)
                    rec[metric_key] = centered
                    if mu is not None:
                        rec[raw_key] = raw
                        # per-record marker (this file has no top-level metadata slot -- it's a flat
                        # uid->record map) so a re-run of this job can tell an already-centered
                        # eval_<dataset>_accent_per_utt.json apart from a pre-centering raw one without
                        # re-deriving it from the aggregate results JSON.
                        rec["center_vector"] = os.path.abspath(center_path)
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
            if args.metric == "accent" and mu is not None:
                entry[raw_key] = summarize([r[raw_key] for r in recs if raw_key in r])
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
                existing = results[out_key].setdefault(g, {})
                if args.metric == "accent" and mu is not None and metric_key in existing and raw_key not in existing:
                    # existing accent_cosine predates centering (a raw GenAID value from an earlier run of
                    # this script or the rescore sbatch) -- preserve it before the centered value below
                    # overwrites metric_key. Never touch any *_commonaccent key already present.
                    existing[raw_key] = existing[metric_key]
                existing.update(entry)
    results[f"{metric_key}_per_utt_source"] = (f"eval/score_side_per_utt.py --metric {args.metric} "
                                              f"(same model/call path as articulatory-tts score_side_metric.py"
                                              + (f"; {model_tag}" if model_tag else "") + ")")
    if args.metric == "accent" and mu is not None:
        results["accent_center_vector"] = os.path.abspath(center_path)
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
