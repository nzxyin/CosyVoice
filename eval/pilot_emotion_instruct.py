"""Score the ESD-val pilot that picks the --emotion_instruct_set wording for --prompt_mode cross_label.

For each condition directory written by eval/pilot_emotion_instruct.sbatch (<root>/<condition>/, with
synthesize_testset.py's wavs/ + manifest), embeds prediction and ground truth with emotion2vec+ large
through the same call path as the reference side metric (score_side_per_utt.build_emotion_scorer, 16 kHz,
0.1 s floor) and reports, per target emotion and overall: emotion cosine (pred vs GT), the fraction of
predictions emotion2vec labels as the target emotion, and the pred/GT duration ratio (a guard against an
instruction that makes the model truncate or ramble). Writes <root>/pilot_summary.json and prints a
Markdown table. Run in the eval-emotion venv.

This is a wording choice on ESD val (disjoint from test), not a result: the test-set numbers come from
the full run_synth / run_score chain.
"""
import argparse
import glob
import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_side_per_utt import ESD_TO_EMOTION2VEC, MIN_PAIR_DURATION_SEC, build_emotion_scorer, cosine, summarize  # noqa: E402

EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]


def to16k(src, dst):
    if os.path.exists(dst):
        return dst
    wav, sr = sf.read(src, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    sf.write(dst + ".tmp.wav", wav, 16000)
    os.replace(dst + ".tmp.wav", dst)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    args = ap.parse_args()
    import torch
    embed, label = build_emotion_scorer(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    summary = {}
    for cond in args.conditions:
        d = os.path.join(args.root, cond)
        pairs = os.path.join(d, "wav_pairs_16k")
        os.makedirs(pairs, exist_ok=True)
        recs = {}
        for f in sorted(glob.glob(os.path.join(d, "manifest.shard*of*.jsonl"))):
            for line in open(f):
                r = json.loads(line)
                if r.get("status") in ("ok", "existing") and os.path.exists(r["pred_wav"]):
                    recs[r["uid"]] = r
        rows = []
        for uid, r in sorted(recs.items()):
            pred = to16k(r["pred_wav"], os.path.join(pairs, f"{uid}_pred.wav"))
            gt = to16k(r["gt_wav"], os.path.join(pairs, f"{uid}_gt.wav"))
            p_sec, g_sec = sf.info(pred).duration, sf.info(gt).duration
            row = {"uid": uid, "emotion": r["emotion"], "prompt_uid": r.get("prompt_uid"),
                   "instruct_text": r.get("instruct_text"), "dur_ratio": p_sec / g_sec}
            if min(p_sec, g_sec) >= MIN_PAIR_DURATION_SEC:
                row["emotion_cosine"] = cosine(embed(pred), embed(gt))
                row["pred_label"] = label(pred)[0]
                row["gt_label"] = label(gt)[0]
            rows.append(row)
        out = {}
        for emo in EMOTIONS + ["all"]:
            rs = [x for x in rows if emo == "all" or x["emotion"] == emo]
            scored = [x for x in rs if "emotion_cosine" in x]
            out[emo] = {
                "n": len(rs),
                "emotion_cosine": summarize([x["emotion_cosine"] for x in scored]),
                "pred_labelled_as_target": float(np.mean([x["pred_label"] == ESD_TO_EMOTION2VEC[x["emotion"]] for x in scored])) if scored else None,
                "gt_labelled_as_target": float(np.mean([x["gt_label"] == ESD_TO_EMOTION2VEC[x["emotion"]] for x in scored])) if scored else None,
                "dur_ratio": summarize([x["dur_ratio"] for x in rs]),
            }
        summary[cond] = {"by_emotion": out, "per_utt": rows}
        json.dump(rows, open(os.path.join(d, "pilot_per_utt.json"), "w"), indent=1)

    with open(os.path.join(args.root, "pilot_summary.json"), "w") as f:
        json.dump({c: s["by_emotion"] for c, s in summary.items()}, f, indent=2)
    for metric, fmt in (("emotion_cosine", lambda e: f"{e['emotion_cosine']['mean']:.3f}"),
                        ("pred labelled as target", lambda e: f"{e['pred_labelled_as_target']:.2f}"),
                        ("pred/GT duration", lambda e: f"{e['dur_ratio']['mean']:.2f}")):
        print(f"\n{metric}:\n| condition | " + " | ".join(EMOTIONS + ["all"]) + " |")
        print("|---|" + "---|" * (len(EMOTIONS) + 1))
        for cond, s in summary.items():
            print(f"| {cond} | " + " | ".join(fmt(s["by_emotion"][e]) if s["by_emotion"][e]["n"] else "n/a"
                                             for e in EMOTIONS + ["all"]) + " |")
    print(f"\nwrote {os.path.join(args.root, 'pilot_summary.json')}")


if __name__ == "__main__":
    main()
