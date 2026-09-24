"""Tabulate the eval_<dataset>.json files under
/data/user_data/xoy/cosyvoice3_eval/<ckpt>/<prompt_mode>/ as a Markdown table (one row per
dataset x prompt mode; the metrics articulatory-tts reports plus this repo's extras).

Usage (on a node with /data mounted):
  python eval/summarize_results.py [--root /data/user_data/xoy/cosyvoice3_eval/Fun-CosyVoice3-0.5B-2512] [--modes self cross cross_label]
"""
import argparse
import glob
import json
import os

COLS = [
    ("wer", "WER% raw", 100, 2), ("wer_whisper_normalized", "WER% whisper-norm", 100, 2),
    ("wer_vs_spoken_text", "WER% vs spoken", 100, 2),
    ("utmosv2", "UTMOSv2", 1, 3), ("dnsmos_ovr", "DNSMOS ovr", 1, 3), ("dnsmos_p808", "DNSMOS p808", 1, 3),
    ("dnsmos_sig", "DNSMOS sig", 1, 3), ("dnsmos_bak", "DNSMOS bak", 1, 3),
    ("speaker_cosine", "Spk cos", 1, 3), ("emotion_cosine", "Emo cos", 1, 3), ("accent_cosine", "Acc cos", 1, 3),
    ("rtf", "RTF", 1, 3), ("n_pred_under_0p5s", "#pred<0.5s", 1, 0),
]
ORDER = ["ljspeech", "libritts_test_clean", "libritts_test_other", "esd", "esd_val", "vctk", "vctk_new_accents"]


def fmt(m, scale, nd):
    if not m or m.get("mean") is None:
        return "n/a"
    s = f"{m['mean'] * scale:.{nd}f}"
    if m.get("ci95") is not None and m["ci95"] > 0:
        s += f" ±{m['ci95'] * scale:.{nd}f}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/user_data/xoy/cosyvoice3_eval/Fun-CosyVoice3-0.5B-2512")
    ap.add_argument("--modes", nargs="+", default=["self", "cross", "cross_label"])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--breakdowns_only", action="store_true")
    args = ap.parse_args()
    rows = []
    for mode in args.modes:
        for p in sorted(glob.glob(os.path.join(args.root, mode, "eval_*.json"))):
            base = os.path.basename(p)
            if base.endswith(("_per_utt.json", "_emotion.json", "_accent.json", "_accent_genaid.json", "_commonaccent.json",
                               "_genaid_centered.json", "_genaid_raw.json")):
                continue
            d = json.load(open(p))
            if "dataset" not in d:
                continue  # a side/per-utt payload that doesn't match any suffix above yet -- skip rather than KeyError
            if args.datasets and d["dataset"] not in args.datasets:
                continue
            rows.append((mode, d))
    rows.sort(key=lambda r: (args.modes.index(r[0]), ORDER.index(r[1]["dataset"]) if r[1]["dataset"] in ORDER else 99))
    header = ["prompt", "dataset", "n (scored/total)"] + [c[1] for c in COLS]
    if not args.breakdowns_only:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] * len(header)) + "|")
        for mode, d in rows:
            m = d["metrics"]
            cells = [mode, d["dataset"], f"{d['n_predicted']}/{d['n_utterances']}"]
            cells += [fmt(m.get(k), sc, nd) for k, _, sc, nd in COLS]
            print("| " + " | ".join(cells) + " |")
    for mode, d in rows:
        for key in ("by_emotion", "by_accent", "by_speaker"):
            if key not in d:
                continue
            if key == "by_speaker" and len(d[key]) > 12:
                continue  # LibriTTS has 33-39 speakers; keep the markdown readable (data stays in the JSON)
            print(f"\n{mode}/{d['dataset']} {key}:")
            print("| group | n | WER% raw | WER% whisper-norm | UTMOSv2 | DNSMOS ovr | Spk cos | Emo cos | Acc cos | pred/GT dur | pred->target label | GT->target label |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for g, e in d[key].items():
                agr = e.get("emotion_label_agreement") or e.get("accent_label_agreement") or {}
                pt = agr.get("pred_labelled_as_target"); gtl = agr.get("gt_labelled_as_target")
                print(f"| {g} | {e['n']} | {fmt(e.get('wer'), 100, 2)} | {fmt(e.get('wer_whisper_normalized'), 100, 2)} | "
                      f"{fmt(e.get('utmosv2'), 1, 3)} | {fmt(e.get('dnsmos_ovr'), 1, 3)} | {fmt(e.get('speaker_cosine'), 1, 3)} | "
                      f"{fmt(e.get('emotion_cosine'), 1, 3)} | {fmt(e.get('accent_cosine'), 1, 3)} | "
                      f"{fmt(e.get('pred_gt_dur_ratio'), 1, 2)} | "
                      f"{'n/a' if pt is None else f'{pt:.2f}'} | {'n/a' if gtl is None else f'{gtl:.2f}'} |")


if __name__ == "__main__":
    main()
