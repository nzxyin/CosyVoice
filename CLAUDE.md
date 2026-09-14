# CosyVoice (local clone of QwenAudio/CosyVoice, upstream FunAudioLLM/CosyVoice)

Upstream Fun-CosyVoice 3.0 code, unmodified. This clone adds an evaluation pipeline under `eval/`
that runs the released `Fun-CosyVoice3-0.5B-2512` checkpoint as a zero-shot voice-cloning baseline
on the same held-out test sets and metric stack as `/home/xoy/articulatory-tts`
(`eval_full_testset.py`), mirroring the sibling baseline evals in `/home/xoy/TTS/eval` (XTTS) and
`/home/xoy/EmoSpherepp/eval` (EmoSphere++). Protocol, file map and run commands: `eval/README.md`.

Cluster conventions (partitions, `/data`, login-node limits) live in the user-level
`~/.claude/CLAUDE.md`, not here. Chronological record of what was done: `CHANGELOG.md`.

Git: the eval code lives on branch `eval/cosyvoice3-baseline`, pushed to the user's fork
(`fork` remote = `nzxyin/CosyVoice`; `origin` = `QwenAudio/CosyVoice` is left untouched). Note
upstream's `.gitignore` has `**/*build*`, which hides `eval/build_env.sbatch` and
`eval/pip-build-constraints.txt` from `git add` -- use `git add -f` for those two.

## Layout / environment

- Conda env `cosyvoice` at `/data/user_data/xoy/miniconda3/envs/cosyvoice` (python 3.10, torch
  2.3.1+cu121, the repo's pins minus deepspeed/tensorrt). Built by `eval/build_env.sbatch`; no
  `.venv` here -- use `/data/user_data/xoy/miniconda3/envs/cosyvoice/bin/python` directly or
  `conda activate cosyvoice` (compute nodes only).
- Checkpoint: `/data/user_data/xoy/cosyvoice_models/Fun-CosyVoice3-0.5B-2512` (HF download, ~9.7 GB,
  includes `llm.rl.pt` for a possible RL-variant follow-up), symlinked as
  `pretrained_models/Fun-CosyVoice3-0.5B`.
- Eval outputs: `/data/user_data/xoy/cosyvoice3_eval/Fun-CosyVoice3-0.5B-2512/<self|cross>/`
  (`<dataset>/wavs`, manifests, `wav_pairs_16k`, `eval_<dataset>.json`).
- Scoring reuses the reference repo's venvs (`eval-articulatory-tts`, `eval-emotion`,
  `eval-genaid` under `/data/user_data/xoy/venvs/`) and its `score_side_metric.py` /
  `merge_eval_results.py` unchanged. Accent cosine = GenAID (https://github.com/jzmzhong/GenAID,
  speaker-adversarial XLSR-53 accent ID, 64-dim embedding) since 2026-09-14; CommonAccent before
  that, kept in every JSON as `accent_cosine_commonaccent` (not comparable).
- torch 2.3.1 has no Blackwell kernels: GPU jobs exclude `preempt`'s RTX PRO 6000 nodes.

## Current direction (2026-09-08)

Done: the five test sets x two prompt modes (`self` for comparability with the sibling baselines,
`cross` as the standard zero-shot protocol) are synthesized, scored and tabulated below. Possible
follow-ups, none started: the `llm.rl.pt` RL variant (same pipeline, `CKPT_NAME`/`--model_dir`),
`--no_text_frontend`, fp16/TRT speed, and re-reading the tables once articulatory-tts's GH #32
rescore publishes the normalized VCTK floor.

## Status / results

### Final results (2026-09-08; all 10 sets scored; `eval/results/summary_Fun-CosyVoice3-0.5B-2512.md` has the full tables incl. per-speaker)

WER raw = lowercased, punctuation kept (the pre-GH#32 articulatory-tts convention, also the XTTS
port's `wer`); WER norm = Whisper EnglishTextNormalizer on both sides, empty normalized references
skipped (= articulatory-tts's `wer` since its GH #32 fix, 2026-09-08). Corpus-level. n = scored /
synthesizable utterances. Spk / Emo / Acc = ECAPA / emotion2vec+ large / GenAID embedding cosine,
prediction vs. ground truth (mean; ci95 in the JSONs). **Acc cos = GenAID since 2026-09-14** (rescored on the
kept wav pairs, job 10441103); the CommonAccent values these tables showed before are kept in each JSON as
`accent_cosine_commonaccent` (self 0.853/0.893/0.862/0.853/0.877, cross 0.818/0.823/0.756/0.790/0.791 in
row order) and are not comparable. GenAID cosines sit in a compressed 0.85-0.99 band for every system
(its 64-dim post-GELU embedding shares a large common component), so read differences, not absolute values. RTF = generation wall time / audio seconds,
fp32 non-streaming on L40S/A6000/A100 (`preempt`). `<0.5s` = degenerate early-EOS outputs, scored as-is.

| prompt | dataset | n | WER% raw | WER% norm | UTMOSv2 | DNSMOS ovr | Spk cos | Emo cos | Acc cos | RTF | <0.5s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| self | ljspeech | 150/150 | 9.07 | 2.74 | 3.949 | 3.406 | 0.893 | 0.973 | 0.987 | 0.53 | 1 |
| self | libritts_test_clean | 4829/4830 | 12.48 | 3.63 | 3.483 | 3.247 | 0.842 | 0.939 | 0.985 | 0.75 | 49 |
| self | libritts_test_other | 5104/5106 | 13.93 | 4.01 | 3.320 | 3.159 | 0.813 | 0.924 | 0.973 | 0.60 | 90 |
| self | esd | 1498/1500 | 15.49 | 3.46 | 3.430 | 3.168 | 0.810 | 0.842 | 0.982 | 0.72 | 5 |
| self | vctk | 2596/2596 | 4.82 | 2.26 | 3.481 | 3.169 | 0.829 | 0.943 | 0.965 | 0.79 | 4 |
| cross | ljspeech | 150/150 | 8.61 | 1.94 | 3.946 | 3.422 | 0.787 | 0.968 | 0.987 | 0.80 | 0 |
| cross | libritts_test_clean | 4830/4830 | 11.28 | 2.40 | 3.538 | 3.271 | 0.608 | 0.918 | 0.973 | 0.54 | 17 |
| cross | libritts_test_other | 5106/5106 | 13.14 | 2.97 | 3.399 | 3.189 | 0.544 | 0.900 | 0.946 | 0.71 | 31 |
| cross | esd | 1500/1500 | 14.65 | 2.27 | 3.524 | 3.193 | 0.580 | 0.764 | 0.971 | 0.63 | 0 |
| cross | vctk | 2596/2596 | 4.49 | 1.64 | 3.603 | 3.189 | 0.632 | 0.914 | 0.921 | 0.52 | 1 |

Ground-truth WER floors (Whisper on the raw recordings, corpus-level, raw / normalized; from
articulatory-tts's `gt_transcripts_*.json` as recomputed by the EmoSphere++ session, VCTK raw from
articulatory-tts CLAUDE.md): LJSpeech 6.97 / 1.60, LibriTTS test-clean 10.60 / 2.27, test-other
13.51 / 4.11, ESD 15.05 / 2.78, VCTK 3.70 / (pending articulatory-tts's GH #32 rescore).

WER implementation parity (2026-09-08): the EmoSphere++ and XTTS eval sessions each confirmed
point-by-point that their scorers match ours (Whisper large-v3 fp32 greedy one-utterance-per-call, raw
lowercased punctuation-kept `wer`, Whisper-normalizer `wer_whisper_normalized` with
empty-normalized-reference pairs skipped, same reference texts/splits, both keys reported raw-first);
XTTS aligned the one difference it had (it used to keep pairs with an empty normalized reference; a
one-utterance change on each LibriTTS set) and reports `wer_whisper_normalized` as its headline;
articulatory-tts's `eval_full_testset.py` matches by code inspection (its `wer` is the normalized
variant since GH #32, commit b040aa1).

Missing utterances: 5 `self`-mode utterances with <=4-word texts ("please excuse me." x2, "It is also
skin.", "A lonelier place!", one test-clean) crash HiFT on every seed (the LLM, shown the tokens of the
very sentence it must say, emits nothing) and are recorded as synthesis failures (`failed_uids` in the
JSONs; scored with `ALLOW_MISSING=5`). All `cross` sets are complete.

**Per-emotion (ESD, 300 utts each; emotion cosine vs. the ESD label; "pred/GT labelled" = fraction
emotion2vec+ assigns the target emotion to the prediction / the ground truth):**

| prompt | emotion | WER% raw | WER% norm | UTMOSv2 | DNSMOS ovr | Spk cos | Emo cos | Acc cos | pred/GT dur | pred labelled | GT labelled |
|---|---|---|---|---|---|---|---|---|---|---|---|
| self | Angry | 15.43 | 3.78 | 3.440 | 3.195 | 0.811 | 0.943 | 0.982 | 1.08 | 0.92 | 0.98 |
| self | Happy | 14.29 | 3.15 | 3.477 | 3.154 | 0.793 | 0.848 | 0.984 | 1.09 | 0.71 | 0.93 |
| self | Neutral | 15.01 | 3.46 | 3.512 | 3.165 | 0.834 | 0.929 | 0.981 | 1.15 | 0.95 | 1.00 |
| self | Sad | 15.54 | 2.69 | 3.388 | 3.194 | 0.844 | 0.847 | 0.980 | 1.08 | 0.80 | 0.99 |
| self | Surprise | 17.18 | 4.20 | 3.331 | 3.132 | 0.764 | 0.644 | 0.982 | 1.10 | 0.46 | 0.95 |
| cross | Angry | 14.77 | 2.73 | 3.545 | 3.209 | 0.578 | 0.891 | 0.972 | 1.05 | 0.83 | 0.98 |
| cross | Happy | 13.89 | 1.92 | 3.551 | 3.186 | 0.553 | 0.768 | 0.974 | 1.06 | 0.60 | 0.93 |
| cross | Neutral | 15.01 | 2.28 | 3.596 | 3.204 | 0.623 | 0.912 | 0.973 | 1.09 | 0.96 | 1.00 |
| cross | Sad | 14.78 | 1.96 | 3.499 | 3.212 | 0.622 | 0.786 | 0.963 | 1.06 | 0.64 | 0.99 |
| cross | Surprise | 14.79 | 2.46 | 3.429 | 3.152 | 0.524 | 0.464 | 0.973 | 1.09 | 0.19 | 0.95 |

**Per-accent (VCTK, one held-out speaker per accent; accent cosine = GenAID embedding; "pred/GT
labelled" = fraction GenAID's 13-way classifier assigns the mapped label (American->us, Canadian->canadian,
English->english, Irish and NorthernIrish->irish, Scottish->scottish). GenAID recognises the American,
English and Scottish ground truth (95/76/44%) but not the Canadian or Northern Irish speaker (3/5%), so
the label view is informative for the first three accents only):**

| prompt | accent (speaker) | n | WER% raw | WER% norm | UTMOSv2 | DNSMOS ovr | Spk cos | Emo cos | Acc cos | pred/GT dur | pred labelled | GT labelled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| self | American (p297) | 417 | 4.96 | 2.54 | 3.627 | 3.123 | 0.801 | 0.933 | 0.978 | 0.95 | 0.94 | 0.95 |
| self | Canadian (p317) | 423 | 3.44 | 1.54 | 3.579 | 3.136 | 0.861 | 0.944 | 0.986 | 1.04 | 0.04 | 0.03 |
| self | English (p270) | 462 | 6.23 | 2.55 | 3.326 | 3.219 | 0.826 | 0.936 | 0.967 | 1.04 | 0.65 | 0.76 |
| self | Irish (p288) | 412 | 2.42 | 0.96 | 3.519 | 3.180 | 0.840 | 0.960 | 0.968 | 1.00 | 0.14 | 0.22 |
| self | NorthernIrish (p304) | 423 | 4.21 | 2.32 | 3.712 | 3.163 | 0.828 | 0.954 | 0.960 | 0.95 | 0.02 | 0.05 |
| self | Scottish (p281) | 459 | 7.32 | 3.54 | 3.166 | 3.185 | 0.821 | 0.933 | 0.931 | 1.04 | 0.24 | 0.45 |
| cross | American (p297) | 417 | 3.90 | 1.82 | 3.765 | 3.147 | 0.568 | 0.896 | 0.970 | 0.93 | 0.96 | 0.95 |
| cross | Canadian (p317) | 423 | 4.02 | 1.79 | 3.681 | 3.140 | 0.709 | 0.916 | 0.978 | 0.97 | 0.03 | 0.03 |
| cross | English (p270) | 462 | 5.74 | 1.68 | 3.431 | 3.217 | 0.646 | 0.913 | 0.922 | 1.02 | 0.52 | 0.76 |
| cross | Irish (p288) | 412 | 3.37 | 0.99 | 3.632 | 3.218 | 0.639 | 0.936 | 0.910 | 0.96 | 0.05 | 0.22 |
| cross | NorthernIrish (p304) | 423 | 3.77 | 1.55 | 3.825 | 3.182 | 0.601 | 0.931 | 0.902 | 0.94 | 0.01 | 0.05 |
| cross | Scottish (p281) | 459 | 5.87 | 1.98 | 3.325 | 3.222 | 0.627 | 0.895 | 0.853 | 1.02 | 0.07 | 0.44 |

### Findings

- **Intelligibility is at the recording floor in the standard zero-shot protocol.** `cross`
  normalized WER is 1.64-2.97% and at or below the ground-truth floor on every set with a known floor
  (LJSpeech 1.94 vs 1.60, test-clean 2.40 vs 2.27, test-other 2.97 vs 4.11, ESD 2.27 vs 2.78). Raw
  WER (8.6-14.7%) is dominated by punctuation/casing differences between Whisper's output and the
  transcripts, not recognition errors -- compare raw only against raw tables.
- **The paired `self` protocol hurts an in-context LLM instead of flattering its WER.** With the target's
  own tokens as the in-context prompt, CosyVoice3 tends to stop early: 49/4829 test-clean and 90/5104
  test-other outputs are under 0.5 s (vs 17 and 31 in `cross`), which is what lifts `self` normalized
  WER to 3.63% / 4.01% / 3.46% (ESD) against 2.40 / 2.97 / 2.27 in `cross`; five short texts fail
  outright. Speaker/emotion/accent cosines are the opposite: `self` (0.81-0.89 speaker) is inflated by
  seeing the target utterance itself; `cross` (0.54-0.79) is the honest zero-shot number. Use
  `cross` as CosyVoice3's headline row; use `self` only to sit next to the sibling baselines' paired
  numbers (XTTS `self` WER-n for reference: LJSpeech 2.44, test-clean 2.59, test-other 2.95, VCTK 1.42).
- **Speaker similarity vs. corpus:** `cross` ECAPA cosine 0.79 (LJSpeech, single speaker, prompt =
  another sentence of the same voice) > 0.63 VCTK > 0.61 test-clean > 0.58 ESD > 0.54 test-other.
  The LibriTTS/ESD values reflect short prompts (ESD ~3 s) and noisy/expressive prompts, not just the
  model; read them against the codec ceilings articulatory-tts reports (speaker 0.60-0.83 by corpus).
- **Emotion transfer is uneven (ESD, `cross`):** emotion cosine Neutral 0.912, Angry 0.891, Sad 0.786,
  Happy 0.768, Surprise 0.464; emotion2vec labels the output as the prompt's emotion 96 / 83 / 64 / 60 /
  19% of the time while it labels the ground truth correctly 93-100% of the time. Surprise is largely
  lost even with a same-speaker, same-emotion prompt; `self` (prompt = the target itself) only lifts it
  to 0.644 / 46%. Per-speaker emotion cosine is flat (0.72-0.81 `cross`).
- **Accent (VCTK, `cross`, GenAID):** accent cosine Canadian 0.978 > American 0.970 > English 0.922 >
  Irish 0.910 > NorthernIrish 0.902 > Scottish 0.853 -- the same ordering CommonAccent gave (Canadian >
  American > NorthernIrish ~ Irish ~ English > Scottish), with the North-American speakers near the
  ceiling and Scottish clearly last; Scottish and English also have the highest WER (raw 5.9 / 5.7%,
  normalized 2.0 / 1.7%) and lowest UTMOSv2 (3.33 / 3.43). `self` prompting lifts every accent (0.931-0.986),
  most for the British/Irish speakers (+0.05-0.08) and least for the North-American ones (+0.01). GenAID's
  labels: the `cross` output is labelled `us` 96% of the time for the American speaker (GT 95%), `english`
  52% for the English one (GT 76%), `scottish` only 7% for the Scottish one (GT 44%) -- i.e. under
  cross-utterance prompting the Scottish and Irish accents are partly lost even where the classifier does
  recognise the recordings; with the target as prompt (`self`) the rates rise to 65% / 24%. Speaker and
  accent are confounded (one speaker per accent).
- **Quality:** UTMOSv2 3.32-3.95 (LJSpeech best, test-other worst), DNSMOS ovr 3.16-3.42; both
  essentially identical between `self` and `cross`. RTF 0.5-0.8 in fp32 without TRT/vLLM.

- 2026-09-08: env built, checkpoint on /data, smoke test (3 utts x 5 sets x 2 modes, full scoring
  chain) passed. Full matrix in flight: 48 synthesis shards (jobs 10354132-10354179) with 10
  scoring jobs chained behind them (10354193-10354202); tracked in a GitHub issue on the
  user's fork (https://github.com/nzxyin/CosyVoice/issues/1; job-tracking issues go on `nzxyin/CosyVoice`, never on
  `QwenAudio/CosyVoice` -- explicit user direction 2026-09-08; `gh repo set-default` already points there). Numbers go here once `eval/summarize_results.py`
  has something to tabulate.
- Smoke-test sanity (n=3 each, `self` mode, not results): speaker cosine 0.88-0.89, UTMOSv2 3.2-4.1,
  DNSMOS ovr 3.1-3.4, emotion cosine 0.92 (ESD), RTF 0.9-1.2 on A100 (fp32, non-streaming).
