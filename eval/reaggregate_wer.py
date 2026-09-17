"""Recompute the WER variants (and per-group breakdowns) of an existing eval_<dataset>.json from its
eval_<dataset>_per_utt.json WITHOUT re-running Whisper -- the per-utterance records keep the raw and
Whisper-normalized reference/hypothesis text exactly for this purpose (same design as articulatory-tts).
Used once (2026-09-08) to bring result sets scored before the empty-normalized-reference rule was
aligned with articulatory-tts GH #32 into line; idempotent.

Usage: python eval/reaggregate_wer.py --results eval_esd.json [--per_utt eval_esd_per_utt.json]
Run under the eval venv (needs jiwer): /data/user_data/xoy/venvs/eval-articulatory-tts/bin/python
"""
import argparse
import json
import os

import jiwer


def corpus_wer(pairs):
    return {"mean": jiwer.wer([a for a, _ in pairs], [b for _, b in pairs]) if pairs else None, "ci95": None, "n": len(pairs)}


def wer_block(recs):
    out = {}
    raw = [(r["wer_reference"], r["wer_hypothesis"]) for r in recs if "wer_reference" in r]
    if raw:
        out["wer"] = corpus_wer(raw)
    norm = [(r["wer_reference_normalized"], r["wer_hypothesis_normalized"]) for r in recs
            if r.get("wer_reference_normalized", "").strip()]
    if norm:
        out["wer_whisper_normalized"] = corpus_wer(norm)
    spoken = [(r["wer_reference_spoken"], r["wer_hypothesis"]) for r in recs if r.get("wer_reference_spoken")]
    out["wer_vs_spoken_text"] = corpus_wer(spoken)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--per_utt", default=None)
    args = ap.parse_args()
    per_utt_path = args.per_utt or args.results.replace(".json", "_per_utt.json")
    d = json.load(open(args.results))
    per_utt = json.load(open(per_utt_path))
    recs = list(per_utt.values())
    before = {k: d["metrics"].get(k, {}).get("mean") for k in ("wer", "wer_whisper_normalized", "wer_vs_spoken_text")}
    d["metrics"].update(wer_block(recs))
    for key, field in (("by_accent", "accent_label"), ("by_emotion", "emotion"), ("by_speaker", "speaker")):
        if key in d:
            groups = {}
            for r in recs:
                groups.setdefault(r.get(field) or "unknown", []).append(r)
            for g, e in d[key].items():
                if g in groups:
                    e.update({k: v for k, v in wer_block(groups[g]).items() if k != "wer_vs_spoken_text"})
                    ratios = [r["pred_seconds"] / r["gt_seconds"] for r in groups[g] if r.get("gt_seconds")]
                    if ratios:
                        import numpy as np
                        e["pred_gt_dur_ratio"] = {"mean": float(np.mean(ratios)),
                                                  "ci95": float(1.96 * np.std(ratios) / np.sqrt(len(ratios))), "n": len(ratios)}
    # degenerate-output counters (added to the scorer later than the first result sets)
    d["metrics"]["n_pred_under_0p5s"] = {"mean": sum(1 for r in recs if r.get("pred_seconds", 1.0) < 0.5), "ci95": None, "n": len(recs)}
    d["metrics"]["n_pred_under_quarter_gt"] = {"mean": sum(1 for r in recs if r.get("pred_seconds", 1.0) < 0.25 * r.get("gt_seconds", 0.0)),
                                               "ci95": None, "n": len(recs)}
    d["wer_reaggregated_from_per_utt"] = True
    after = {k: d["metrics"].get(k, {}).get("mean") for k in before}
    tmp = args.results + ".tmp"
    json.dump(d, open(tmp, "w"), indent=2)
    os.replace(tmp, args.results)
    print(f"{os.path.basename(args.results)}: n={len(recs)} before={before} after={after}")


if __name__ == "__main__":
    main()
