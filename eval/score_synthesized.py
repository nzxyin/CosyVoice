"""Score synthesized test-set audio (from eval/synthesize_testset.py) with the SAME metric
implementations articulatory-tts uses, so the numbers sit next to that repo's eval_*.json files:
this is a port of the score() function in /home/xoy/articulatory-tts/eval_full_testset.py (via
/home/xoy/TTS/eval/score_synthesized.py, the XTTS baseline's port of the same function) with the
SPARC-decode step replaced by "read the wav CosyVoice already wrote".

Metrics (identical model choices / preprocessing / aggregation to the reference):
  wer            corpus-level jiwer.wer over lowercased reference vs. Whisper large-v3
                 (openai/whisper-large-v3, fp32, language=en, task=transcribe) hypothesis of the
                 16 kHz-resampled prediction. Lowercasing only, punctuation kept -- exactly what
                 the reference does, so keep it that way for comparability.
  wer_whisper_normalized   same hypotheses/references passed through WhisperTokenizer.normalize
                 (transformers' port of OpenAI's EnglishTextNormalizer: punctuation, casing, number
                 words, contractions, spelling variants), pairs with an empty normalized reference
                 skipped. This is exactly what articulatory-tts's `wer` became with its GH #32 fix
                 (2026-09-08, commit b040aa1), so compare THIS key against articulatory-tts JSONs
                 produced after that fix, and the raw `wer` above against older ones / the XTTS port.
  wer_vs_spoken_text       EXTRA: corpus WER against the text CosyVoice's frontend actually spoke
                 (manifest `normalized_text`: wetext normalization + number spelling + paragraph
                 split, joined with spaces), lowercased, punctuation kept. Equals `wer` wherever the
                 frontend left the transcript unchanged; the gap isolates text-normalization
                 mismatches (digits, abbreviations) from recognition errors.
  utmosv2        UTMOSv2 (utmosv2.create_model(pretrained=True)) on the 16 kHz prediction
  dnsmos_*       torchmetrics DNSMOS (p808 / sig / bak / ovr), non-personalized, 16 kHz prediction
  speaker_cosine ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) embedding cosine, prediction vs.
                 ground-truth recording of the same utterance (>= 0.3 s each, the reference's floor)
  emotion_cosine / accent_cosine  NOT computed here -- each needs its own venv. Pass
                 --wav_pairs_dir to write {uid}_pred.wav/{uid}_gt.wav (16 kHz) and run
                 /home/xoy/articulatory-tts/score_side_metric.py --metric emotion|accent on it,
                 then merge_eval_results.py (see eval/run_score.sbatch), as the reference chain does.

Utterance universe: rebuilt from eval/synthesize_testset.py's own load_items() (the split file +
transcript/GT lookups), NOT from the shard manifests -- the manifests only contribute
per-utterance metadata (seed, prompt, timing, status). An utterance is scored iff its prediction
wav exists; every synthesizable utterance without one is reported as a failure.

Restart-safe: every scored utterance's record is appended to <results_path>.partial.jsonl as it is
produced (wav pairs written first), and a re-run skips the uids already there, so a preempted
scoring job resumes instead of redoing hours of Whisper. Delete the partial file (or pass
--no_resume) after re-synthesizing wavs with --overwrite, otherwise stale records are reused.

Summary shape per metric is the reference's {mean, ci95, n} (ci95 = 1.96 * std / sqrt(n); WER's
ci95 is None because corpus WER is a single ratio). Per-utterance records (--per_utt_out) keep the
raw reference/hypothesis text so WER can be re-aggregated over any subset without re-running
Whisper; per-speaker / per-emotion (ESD) / per-accent (VCTK) breakdowns are also written.

Run under the reference eval venv so library versions match the articulatory-tts numbers:
/data/user_data/xoy/venvs/eval-articulatory-tts/bin/python
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthesize_testset import DATASETS, PROMPT_MODES, SPLIT_STATS, assign_prompts, emotion_ref_provenance, load_items  # noqa: E402


def summarize(values):
    values = np.array(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": None, "ci95": None, "n": 0}
    return {"mean": float(values.mean()), "ci95": float(1.96 * values.std() / np.sqrt(len(values))), "n": int(len(values))}


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def write_json_atomic(obj, path, **kw):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, **kw)
    os.replace(tmp, path)


def load_manifest_metadata(patterns):
    """uid -> the most informative manifest record ('ok' beats 'existing' beats 'failed';
    among equals the later record wins). Metadata only -- see module docstring."""
    rank = {"ok": 3, "existing": 2, "failed": 1}
    records = {}
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                prev = records.get(r["uid"])
                if prev is None or rank.get(r.get("status"), 0) >= rank.get(prev.get("status"), 0):
                    records[r["uid"]] = r
    print(f"loaded metadata for {len(records)} utterances from {len(files)} manifest file(s)")
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--out_dir", required=True, help="synthesize_testset.py's --out_dir (wavs/ + manifests live here)")
    ap.add_argument("--prompt_mode", choices=PROMPT_MODES, default="self",
                    help="must match what synthesize_testset.py was run with (checked against the manifests)")
    ap.add_argument("--checkpoint", required=True, help="checkpoint identifier recorded in the results JSON")
    ap.add_argument("--results_path", required=True)
    ap.add_argument("--per_utt_out", default=None)
    ap.add_argument("--wav_pairs_dir", default=None, help="write 16 kHz {uid}_pred.wav/{uid}_gt.wav pairs for score_side_metric.py")
    ap.add_argument("--skip_audio", action="store_true", help="skip UTMOSv2 + DNSMOS")
    ap.add_argument("--skip_wer", action="store_true")
    ap.add_argument("--skip_speaker", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--partial_path", default=None, help="per-utterance JSONL checkpoint (default <results_path>.partial.jsonl)")
    ap.add_argument("--no_resume", action="store_true", help="ignore (and truncate) an existing partial checkpoint")
    args = ap.parse_args()

    import soundfile as sf
    import librosa
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = load_items(args.dataset)
    assign_prompts(items, args.prompt_mode, args.dataset)
    n_split_total = SPLIT_STATS["n_split_total"]
    wav_dir = os.path.join(args.out_dir, "wavs")
    for it in items:
        it["pred_wav"] = os.path.join(wav_dir, f"{it['uid']}.wav")
    meta = load_manifest_metadata([os.path.join(args.out_dir, "manifest.shard*of*.jsonl")])
    split_info = {}
    if os.path.exists(os.path.join(args.out_dir, "split_info.json")):
        split_info = json.load(open(os.path.join(args.out_dir, "split_info.json")))
    if args.limit:
        items = items[: args.limit]

    def has_wav(it):
        return os.path.exists(it["pred_wav"]) and os.path.getsize(it["pred_wav"]) > 1000

    ok = [it for it in items if has_wav(it)]
    missing = [it for it in items if not has_wav(it)]
    print(f"{args.dataset}: {n_split_total} split rows, {len(items)} synthesizable, "
          f"{len(ok)} with a prediction wav, {len(missing)} missing")
    recorded_modes = {meta[it["uid"]].get("prompt_mode") for it in ok if it["uid"] in meta}
    if recorded_modes and recorded_modes != {args.prompt_mode}:
        raise SystemExit(f"manifests record prompt_mode={recorded_modes} but --prompt_mode={args.prompt_mode}")

    # ---- models (same sources/settings as eval_full_testset.py) ---------------------------------
    dnsmos_fn = utmos_model = whisper_processor = whisper_model = ecapa_model = None
    if not args.skip_audio:
        # torchmetrics' DNSMOS runs ONNX models; on a node where onnxruntime cannot import (babel-l9-16,
        # 2026-09-08) it reports "not installed" instead of scoring -- fail loudly here instead.
        try:
            import onnxruntime  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise SystemExit(f"onnxruntime is not importable on {os.uname().nodename} ({e!r}); DNSMOS cannot run -- "
                             f"resubmit excluding this node (see eval/run_score.sbatch)")
        from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score as dnsmos_fn
        import utmosv2
        utmos_model = utmosv2.create_model(pretrained=True)
    if not args.skip_wer:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import jiwer
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3", torch_dtype=torch.float32
        ).to(device).eval()
        normalize = whisper_processor.tokenizer.normalize
    if not args.skip_speaker:
        from speechbrain.inference.speaker import EncoderClassifier
        ecapa_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(os.environ.get("HF_HOME", "/tmp"), "spkrec-ecapa-voxceleb"),
            run_opts={"device": str(device)},
        )
    if args.wav_pairs_dir:
        os.makedirs(args.wav_pairs_dir, exist_ok=True)

    def to16k(wav, sr):
        return wav if sr == 16000 else librosa.resample(wav, orig_sr=sr, target_sr=16000)

    def dnsmos_score(wav16):
        p808, sig, bak, ovr = dnsmos_fn(torch.from_numpy(wav16).float().to(device), 16000, personalized=False, num_threads=4)
        return {"p808": float(p808), "sig": float(sig), "bak": float(bak), "ovr": float(ovr)}

    def utmos_score(wav16):
        result = utmos_model.predict(data=wav16.astype(np.float32), sr=16000)
        return float(np.asarray(result).reshape(-1)[0])

    def whisper_transcribe(wav16):
        inputs = whisper_processor(wav16, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        ids_out = whisper_model.generate(inputs, language="en", task="transcribe")
        return whisper_processor.batch_decode(ids_out, skip_special_tokens=True)[0].strip()

    MIN_ECAPA_SEC = 0.3

    def ecapa_embed(wav16):
        emb = ecapa_model.encode_batch(torch.from_numpy(wav16).float().unsqueeze(0).to(device))
        return emb.squeeze().detach().cpu().numpy().reshape(-1)

    partial_path = args.partial_path or (args.results_path + ".partial.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(partial_path)), exist_ok=True)
    per_utt_records = {}
    if args.no_resume:
        open(partial_path, "w").close()
    elif os.path.exists(partial_path):
        n_bad = 0
        with open(partial_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    per_utt_records[r.pop("uid")] = r
                except Exception:
                    n_bad += 1  # a truncated final line from a killed job
        print(f"resuming: {len(per_utt_records)} utterances already scored in {partial_path}"
              + (f" ({n_bad} unreadable line(s) ignored)" if n_bad else ""))
    ok_uids = {it["uid"] for it in ok}
    stale = [u for u in per_utt_records if u not in ok_uids]
    for u in stale:
        del per_utt_records[u]
    if stale:
        print(f"ignoring {len(stale)} checkpointed records for utterances not in this scoring set")
    partial_f = open(partial_path, "a")
    n_resumed = len(per_utt_records)
    with torch.no_grad():
        for it in tqdm(ok, desc="scoring"):
            uid = it["uid"]
            if uid in per_utt_records:
                continue
            m = meta.get(uid, {})
            # same read path as the reference (sf.read -> float32); all five corpora are mono
            pred_wav, pred_sr = sf.read(it["pred_wav"])
            pred_wav = pred_wav.astype(np.float32)
            gt_wav, gt_sr = sf.read(it["gt_wav"])
            gt_wav = gt_wav.astype(np.float32)
            if gt_wav.ndim > 1:
                gt_wav = gt_wav.mean(axis=1)
            pred16 = to16k(pred_wav, pred_sr)
            spoken = m.get("normalized_text")
            rec = {
                "speaker": it["speaker"], "emotion": it["emotion"], "accent_label": it["accent_label"],
                "prompt_uid": m.get("prompt_uid", it.get("prompt_uid")), "seed": m.get("seed"),
                "n_segments": m.get("n_segments"), "gen_wall_seconds": m.get("gen_wall_seconds"),
                "prompt_seconds_used": m.get("prompt_seconds_used"),
                "spoken_text": " ".join(spoken) if isinstance(spoken, list) else spoken,
                "pred_seconds": round(len(pred_wav) / pred_sr, 3), "gt_seconds": round(len(gt_wav) / gt_sr, 3)}
            if m.get("instruct_text"):
                rec["instruct_text"] = m["instruct_text"]

            if not args.skip_audio:
                d = dnsmos_score(pred16)
                rec.update({f"dnsmos_{k}": v for k, v in d.items()})
                rec["utmosv2"] = utmos_score(pred16)

            if not args.skip_wer:
                hyp = whisper_transcribe(pred16)
                ref_lower, hyp_lower = it["text"].lower(), hyp.lower()
                rec.update({
                    "wer": jiwer.wer(ref_lower, hyp_lower),
                    "wer_reference": ref_lower,
                    "wer_hypothesis": hyp_lower,
                    "wer_reference_normalized": normalize(it["text"]),
                    "wer_hypothesis_normalized": normalize(hyp),
                })
                if rec["spoken_text"]:
                    rec["wer_reference_spoken"] = rec["spoken_text"].lower()

            if not args.skip_speaker:
                pred_sec, gt_sec = len(pred_wav) / pred_sr, len(gt_wav) / gt_sr
                if pred_sec < MIN_ECAPA_SEC or gt_sec < MIN_ECAPA_SEC:
                    tqdm.write(f"WARNING: skipping speaker_cosine for {uid} -- too short for ECAPA-TDNN "
                               f"(pred={pred_sec:.3f}s, gt={gt_sec:.3f}s, min={MIN_ECAPA_SEC}s)")
                else:
                    rec["speaker_cosine"] = cosine(ecapa_embed(pred16), ecapa_embed(to16k(gt_wav, gt_sr)))

            if args.wav_pairs_dir:
                uid_safe = uid.replace("/", "_")
                sf.write(os.path.join(args.wav_pairs_dir, f"{uid_safe}_pred.wav"), pred16.astype(np.float32), 16000)
                sf.write(os.path.join(args.wav_pairs_dir, f"{uid_safe}_gt.wav"), to16k(gt_wav, gt_sr).astype(np.float32), 16000)

            # checkpoint AFTER the wav pair exists, so a resumed uid never lacks its pair
            per_utt_records[uid] = rec
            partial_f.write(json.dumps({"uid": uid, **rec}) + "\n")
            partial_f.flush()
    partial_f.close()
    print(f"scored {len(per_utt_records) - n_resumed} utterances now, {n_resumed} resumed from checkpoint")

    # ---- aggregate -----------------------------------------------------------------------------
    # Everything is aggregated from the per-utterance records (fresh + resumed) so a resumed run
    # reports over the full set. Corpus WER = one jiwer.wer() call over all pairs, as in the reference.
    recs = list(per_utt_records.values())
    metrics = {}
    if not args.skip_audio:
        for k in ("p808", "sig", "bak", "ovr"):
            metrics[f"dnsmos_{k}"] = summarize([r[f"dnsmos_{k}"] for r in recs if f"dnsmos_{k}" in r])
        metrics["utmosv2"] = summarize([r["utmosv2"] for r in recs if "utmosv2" in r])
    if not args.skip_wer:
        wer_refs = [r["wer_reference"] for r in recs if "wer_reference" in r]
        wer_hyps = [r["wer_hypothesis"] for r in recs if "wer_reference" in r]
        metrics["wer"] = {"mean": jiwer.wer(wer_refs, wer_hyps) if wer_refs else None, "ci95": None, "n": len(wer_refs)}
        # Same rule as articulatory-tts's eval_full_testset.py after its GH #32 fix (2026-09-08), whose
        # primary `wer` IS this Whisper-normalized corpus WER: a pair whose reference is empty after
        # normalization is skipped (the hypothesis alone carries no reference words to score against).
        pairs = [(r["wer_reference_normalized"], r["wer_hypothesis_normalized"]) for r in recs
                 if r.get("wer_reference_normalized", "").strip()]
        metrics["wer_whisper_normalized"] = {
            "mean": jiwer.wer([a for a, _ in pairs], [b for _, b in pairs]) if pairs else None,
            "ci95": None, "n": len(pairs)}
        spoken_pairs = [(r["wer_reference_spoken"], r["wer_hypothesis"]) for r in recs if r.get("wer_reference_spoken")]
        metrics["wer_vs_spoken_text"] = {
            "mean": jiwer.wer([a for a, _ in spoken_pairs], [b for _, b in spoken_pairs]) if spoken_pairs else None,
            "ci95": None, "n": len(spoken_pairs)}
    if not args.skip_speaker:
        metrics["speaker_cosine"] = summarize([r["speaker_cosine"] for r in recs if "speaker_cosine" in r])
    # Degenerate generations (early EOS: e.g. 0.2 s of "you" for a 7.5 s target) are scored as-is like any
    # other prediction (WER takes the deletions, speaker_cosine is skipped under 0.3 s); count them so the
    # reader can see how much of the WER is this failure mode.
    metrics["n_pred_under_0p5s"] = {"mean": sum(1 for r in recs if r.get("pred_seconds", 1.0) < 0.5), "ci95": None, "n": len(recs)}
    metrics["n_pred_under_quarter_gt"] = {"mean": sum(1 for r in recs if r.get("pred_seconds", 1.0) < 0.25 * r.get("gt_seconds", 0.0)),
                                          "ci95": None, "n": len(recs)}
    metrics["attempts_mean"] = summarize([m.get("attempts", 1) for u, m in meta.items() if u in per_utt_records])
    gen = [r["gen_wall_seconds"] for r in per_utt_records.values() if r.get("gen_wall_seconds")]
    aud = [r["pred_seconds"] for r in per_utt_records.values() if r.get("gen_wall_seconds")]
    if gen:
        metrics["rtf"] = {"mean": float(sum(gen) / sum(aud)), "ci95": None, "n": len(gen)}
    metrics["pred_seconds"] = summarize([r["pred_seconds"] for r in per_utt_records.values()])
    metrics["gt_seconds"] = summarize([r["gt_seconds"] for r in per_utt_records.values()])

    def breakdown(key):
        groups = {}
        for rec in per_utt_records.values():
            groups.setdefault(rec.get(key) or "unknown", []).append(rec)
        out = {}
        for g, recs in sorted(groups.items()):
            entry = {"n": len(recs)}
            for mkey in ("utmosv2", "dnsmos_ovr", "dnsmos_p808", "speaker_cosine"):
                vals = [x[mkey] for x in recs if mkey in x]
                if vals:
                    entry[mkey] = summarize(vals)
            entry["pred_gt_dur_ratio"] = summarize([x["pred_seconds"] / x["gt_seconds"] for x in recs if x.get("gt_seconds")])
            refs = [x["wer_reference"] for x in recs if "wer_reference" in x]
            hyps = [x["wer_hypothesis"] for x in recs if "wer_reference" in x]
            if refs:
                entry["wer"] = {"mean": jiwer.wer(refs, hyps), "ci95": None, "n": len(refs)}
            nrefs = [x["wer_reference_normalized"] for x in recs if x.get("wer_reference_normalized", "").strip()]
            nhyps = [x["wer_hypothesis_normalized"] for x in recs if x.get("wer_reference_normalized", "").strip()]
            if nrefs:
                entry["wer_whisper_normalized"] = {"mean": jiwer.wer(nrefs, nhyps), "ci95": None, "n": len(nrefs)}
            out[g] = entry
        return out

    failed_uids = [it["uid"] for it in missing]
    results = {
        "checkpoint": args.checkpoint, "dataset": args.dataset,
        "split": os.path.basename(DATASETS[args.dataset]["split_path"]),
        "prompt_mode": args.prompt_mode,
        "text_frontend": split_info.get("text_frontend"), "fp16": split_info.get("fp16"),
        "instruct_prefix": split_info.get("instruct_prefix"),
        "n_split_total": n_split_total,
        "n_utterances": len(items), "n_predicted": len(ok), "n_synthesis_failed": len(missing),
        "failed_uids": failed_uids,
        "synthesis_errors": {u: meta[u].get("error") for u in failed_uids if u in meta and meta[u].get("status") == "failed"},
        "metrics": metrics,
    }
    if args.dataset.startswith("esd"):
        results.update(emotion_ref_provenance(args.prompt_mode, items, split_info))
    if len({it["accent_label"] for it in ok}) > 1:
        results["by_accent"] = breakdown("accent_label")
    if len({it["emotion"] for it in ok}) > 1:
        results["by_emotion"] = breakdown("emotion")
    n_spk = len({it["speaker"] for it in ok})
    if 1 < n_spk <= 50:
        results["by_speaker"] = breakdown("speaker")
    print(json.dumps(metrics, indent=2))

    if args.per_utt_out:
        write_json_atomic(per_utt_records, args.per_utt_out, indent=1)
        print(f"wrote per-utterance metrics for {len(per_utt_records)} utterances to {args.per_utt_out}")
    write_json_atomic(results, args.results_path, indent=2)
    print(f"wrote results to {args.results_path}")


if __name__ == "__main__":
    main()
