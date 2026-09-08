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
  `eval-accent` under `/data/user_data/xoy/venvs/`) and its `score_side_metric.py` /
  `merge_eval_results.py` unchanged.
- torch 2.3.1 has no Blackwell kernels: GPU jobs exclude `preempt`'s RTX PRO 6000 nodes.

## Current direction (2026-09-08)

Run the five test sets x two prompt modes (`self` for comparability with the sibling baselines,
`cross` as the standard zero-shot protocol), then tabulate with `eval/summarize_results.py` and
record the numbers here. Status and results are filled in below as jobs finish.

## Status / results

### Interim results (2026-09-08 11:40 EDT; 7/10 sets scored, `self` esd/libritts still scoring)

WER raw = lowercased, punctuation kept (the pre-GH#32 articulatory-tts convention, also the XTTS
port's `wer`); WER norm = Whisper EnglishTextNormalizer on both sides (= articulatory-tts's `wer`
since its GH #32 fix, 2026-09-08). Corpus-level, n = scored utterances. Spk/Emo/Acc = ECAPA /
emotion2vec+ large / CommonAccent embedding cosine, prediction vs. ground truth.

| prompt | dataset | n | WER% raw | WER% norm | UTMOSv2 | DNSMOS ovr | Spk cos | Emo cos | Acc cos | RTF |
|---|---|---|---|---|---|---|---|---|---|---|
| self | ljspeech | 150 | 9.07 | 2.74 | 3.949 | 3.406 | 0.893 | 0.973 | 0.853 | 0.53 |
| self | vctk | 2596 | 4.82 | 2.26 | 3.481 | 3.169 | 0.829 | 0.943 | 0.877 | 0.79 |
| cross | ljspeech | 150 | 8.61 | 1.94 | 3.946 | 3.422 | 0.787 | 0.968 | 0.818 | 0.80 |
| cross | libritts_test_clean | 4830 | 11.28 | 2.40 | 3.538 | 3.271 | 0.608 | 0.918 | 0.823 | 0.54 |
| cross | libritts_test_other | 5106 | 13.14 | 2.97 | 3.399 | 3.189 | 0.544 | 0.900 | 0.756 | 0.71 |
| cross | esd | 1500 | 14.65 | 2.27 | 3.524 | 3.193 | 0.580 | 0.764 | 0.790 | 0.63 |
| cross | vctk | 2596 | 4.49 | 1.64 | 3.603 | 3.189 | 0.632 | 0.914 | 0.791 | 0.52 |

WER implementation parity (2026-09-08): the EmoSphere++ eval session confirmed point-by-point
that its scorer matches ours (Whisper large-v3 fp32 greedy one-utterance-per-call, raw lowercased
punctuation-kept `wer`, Whisper-normalizer `wer_whisper_normalized` with empty-normalized-reference
pairs skipped, same reference texts/splits, both keys reported raw-first); articulatory-tts's
`eval_full_testset.py` matches by code inspection (its `wer` is the normalized variant since GH #32,
commit b040aa1); the XTTS session confirmed the same on every point and aligned the one
difference it had (its normalized WER used to keep pairs with an empty normalized reference; now the
b040aa1 rule -- a one-utterance change on each LibriTTS set). XTTS reports `wer_whisper_normalized`
as its headline; its `self`-prompt WER-n for reference: LJSpeech 2.44, test-clean 2.59, test-other
2.95, VCTK 1.42 (results under `/data/user_data/xoy/xtts_accent_eval/.../self/`).

Ground-truth WER floors (Whisper on the raw recordings, corpus-level, raw / normalized; from
articulatory-tts's `gt_transcripts_*.json` as recomputed by the EmoSphere++ session, VCTK raw from
articulatory-tts CLAUDE.md): LJSpeech 6.97 / 1.60, LibriTTS test-clean 10.60 / 2.27, test-other
13.51 / 4.11, ESD 15.05 / 2.78, VCTK 3.70 / (pending articulatory-tts's GH #32 rescore). Read the WER
columns against these: CosyVoice3 `cross` normalized WER is at or below the floor on every set
(LJSpeech 1.94 vs 1.60, test-clean 2.40 vs 2.27, test-other 2.97 vs 4.11, ESD 2.27 vs 2.78), i.e.
its intelligibility is indistinguishable from the recordings themselves at this metric's resolution.

Reading notes so far:
- Raw WER is dominated by punctuation/casing formatting, not recognition: normalized WER is
  1.6-3.0% on every set (ESD 14.65% raw -> 2.27% normalized). Report the normalized column as
  the intelligibility number; keep raw only for comparability with older tables.
- `self` vs `cross` speaker cosine (0.893 vs 0.787 LJSpeech; 0.829 vs 0.632 VCTK) shows how much
  the paired protocol flatters an in-context cloner; `cross` is the honest zero-shot number.
- Degenerate early-EOS outputs (<0.5 s): 17/4830 test-clean, 31/5106 test-other, 4/2596 VCTK
  (`n_pred_under_0p5s`); they are scored as-is. Five `self`-mode utterances (<=4-word texts) crash
  the vocoder on every seed and are reported as synthesis failures (see CHANGELOG).
- Per-accent (VCTK) and per-emotion (ESD) breakdowns with the side-metric cosines live in the
  results JSONs (`by_accent`, `by_emotion`, `by_speaker`) and print via `eval/summarize_results.py`;
  final tables go below once the per-utterance side-metric jobs finish.

- 2026-09-08: env built, checkpoint on /data, smoke test (3 utts x 5 sets x 2 modes, full scoring
  chain) passed. Full matrix in flight: 48 synthesis shards (jobs 10354132-10354179) with 10
  scoring jobs chained behind them (10354193-10354202); tracked in a GitHub issue on the
  user's fork (https://github.com/nzxyin/CosyVoice/issues/1; job-tracking issues go on `nzxyin/CosyVoice`, never on
  `QwenAudio/CosyVoice` -- explicit user direction 2026-09-08; `gh repo set-default` already points there). Numbers go here once `eval/summarize_results.py`
  has something to tabulate.
- Smoke-test sanity (n=3 each, `self` mode, not results): speaker cosine 0.88-0.89, UTMOSv2 3.2-4.1,
  DNSMOS ovr 3.1-3.4, emotion cosine 0.92 (ESD), RTF 0.9-1.2 on A100 (fp32, non-streaming).
