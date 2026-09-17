# Evaluating Fun-CosyVoice3-0.5B-2512 on the articulatory-tts test sets

Everything in this directory evaluates the released `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
checkpoint (base LLM, `llm.pt`, not the `llm.rl.pt` RL variant) as a zero-shot voice-cloning
baseline on the same held-out test splits, with the same metric implementations, as the
`/home/xoy/articulatory-tts` repo's `eval_full_testset.py`, so the numbers can be read next to
that repo's and next to the XTTS (`/home/xoy/TTS/eval`) and EmoSphere++
(`/home/xoy/EmoSpherepp/eval`) baselines built on the same protocol.

| test set | split file | utterances | notes |
|---|---|---|---|
| `ljspeech` | `LJSpeech-1.1/preprocessed/test.json` | 150 | 1 speaker |
| `libritts_test_clean` | `LibriTTS_R/test-clean.json` | 4830 | 39 speakers |
| `libritts_test_other` | `LibriTTS_R/test-other.json` | 5106 | 33 speakers |
| `esd` | `esd_english_splits/test.tsv` | 1500 | 10 speakers x 5 emotions x 30 |
| `vctk` | `vctk_globe_accent_splits/vctk_only/test.tsv` | 2596 | 6 held-out speakers, one per accent |

Metrics: corpus WER, reported twice: `wer` (Whisper large-v3 hypothesis vs. reference, lowercased,
punctuation kept -- articulatory-tts's convention before its GH #32 fix and the XTTS port's `wer`) and
`wer_whisper_normalized` (both sides through Whisper's EnglishTextNormalizer, empty normalized
references skipped -- exactly articulatory-tts's `wer` since GH #32, 2026-09-08),
UTMOSv2, DNSMOS (p808/sig/bak/ovr), ECAPA-TDNN speaker cosine (prediction vs. ground truth),
emotion2vec+ large emotion cosine and GenAID accent cosine (prediction vs. ground truth, both
scored by the reference repo's own `score_side_metric.py` and merged with its
`merge_eval_results.py`). **Since 2026-09-16 the accent cosine is CENTERED** (both embeddings minus
a fixed centering vector before the cosine -- see CLAUDE.md "Accent metric centering"); the raw
GenAID cosine is kept alongside as `accent_cosine_genaid_raw`. Extras not in the reference:
`wer_whisper_normalized` (Whisper normalizer on both sides; do not compare against
articulatory-tts JSONs), `rtf`, durations.

## Environment (conda, on /data)

`~/.condarc` routes `envs_dirs`/`pkgs_dirs` to `/data/user_data/xoy/miniconda3/`, so

```bash
sbatch eval/build_env.sbatch
```

creates `/data/user_data/xoy/miniconda3/envs/cosyvoice` (python 3.10 + `eval/requirements-inference.txt`,
which is the repo's pinned `requirements.txt` minus the training/deployment-only `deepspeed` and
`tensorrt*` wheels), downloads the checkpoint (~9.7 GB) to
`/data/user_data/xoy/cosyvoice_models/Fun-CosyVoice3-0.5B-2512`, and symlinks it to
`pretrained_models/Fun-CosyVoice3-0.5B` (the path `example.py` expects; gitignored). The job is
CPU-only (`preempt_cpu_qos`) and idempotent. `eval/pip-build-constraints.txt` pins `setuptools<80`
for pip's isolated build environments because `openai-whisper==20231117`'s sdist still imports
`pkg_resources`.

The pinned `torch==2.3.1+cu121` has no kernels for Blackwell GPUs, so every GPU job here carries an
`--exclude=` list of `preempt`'s RTX PRO 6000 nodes (regenerate with the command in the sbatch
comment). Scoring runs in the reference repo's own venvs
(`/data/user_data/xoy/venvs/eval-articulatory-tts`, `eval-emotion`, `eval-genaid`), which are not
affected.

## Running inference (standalone)

```python
import sys; sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio
m = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')   # fp32, no TRT/vLLM
for i, out in enumerate(m.inference_zero_shot(
        'Text to speak.',
        'You are a helpful assistant.<|endofprompt|>Transcript of the prompt recording.',
        'prompt.wav', stream=False)):
    torchaudio.save(f'out_{i}.wav', out['tts_speech'], m.sample_rate)   # 24 kHz
```

The prompt must be <= 30 s and >= 16 kHz. Long texts are split into <= 80-token segments, each
yielded separately. See `example.py` for cross-lingual / instruct modes.

## Files

- `synthesize_testset.py` -- test-set definitions, prompt selection, sharded + resumable synthesis;
  writes `wavs/<uid>.wav` (24 kHz) and `manifest.shard<i>of<n>.jsonl`.
- `score_synthesized.py` -- the metric port (run in the reference eval venv); writes
  `eval_<dataset>.json` (+ `_per_utt.json`, + 16 kHz wav pairs for the side metrics).
- `run_synth.sbatch <dataset> <shard> <nshards> [prompt_mode]`, `run_score.sbatch <dataset> [prompt_mode]`
  -- the SLURM wrappers (`preempt`, `--requeue`; both stages resume).
- `smoke_test.sbatch` -- 3 utterances per test set and prompt mode through the whole chain.
- `score_side_per_utt.py` / `run_side_per_utt.sbatch <dataset> [prompt_mode] [metrics]` -- per-utterance
  emotion (emotion2vec+ large, eval-emotion venv) and accent (GenAID, eval-genaid venv) cosine
  with the same models/call paths as the reference's `score_side_metric.py`, folded into the results
  JSON's `by_emotion` / `by_accent` / `by_speaker` groups together with the classifiers' own label
  agreement (mirrors `/home/xoy/TTS/eval/score_accent_per_utt.py` and
  `/home/xoy/EmoSpherepp/eval/emotion_cosine_per_utt.py` + `breakdown.py`).
- `reaggregate_wer.py` -- recompute the WER keys (raw / Whisper-normalized / vs spoken text) and
  per-group duration ratios of a results JSON from its per-utterance records, no Whisper re-run.
- `summarize_results.py` -- Markdown tables over the results JSONs (overall + per-emotion / per-accent
  / per-speaker breakdowns).

Outputs land under `/data/user_data/xoy/cosyvoice3_eval/Fun-CosyVoice3-0.5B-2512/<prompt_mode>/`.

## Typical run

```bash
sbatch eval/build_env.sbatch                                    # once
sbatch eval/smoke_test.sbatch                                   # sanity check + timing
for mode in self cross; do
  sbatch eval/run_synth.sbatch ljspeech 0 1 $mode
  for i in 0 1 2 3; do sbatch eval/run_synth.sbatch libritts_test_clean $i 4 $mode; done
  for i in 0 1 2 3; do sbatch eval/run_synth.sbatch libritts_test_other $i 4 $mode; done
  for i in 0 1;     do sbatch eval/run_synth.sbatch esd $i 2 $mode; done
  for i in 0 1;     do sbatch eval/run_synth.sbatch vctk $i 2 $mode; done
done
# after all shards report SYNTH_DONE (or chain with --dependency=afterok:...):
for mode in self cross; do for ds in ljspeech libritts_test_clean libritts_test_other esd vctk; do
  sbatch eval/run_score.sbatch $ds $mode; done; done
```

## Protocol notes

- **Prompt modes.** `self` = the target utterance's own recording and transcript are the prompt.
  This is the condition the articulatory-tts numbers correspond to (its vocoder is driven by the
  target's own speaker embedding) and what the XTTS/EmoSphere++ baselines ran. For an in-context
  LLM it is optimistic: the model sees the speech tokens of the sentence it must produce.
  `cross` = another utterance of the same speaker (same speaker *and* emotion for ESD), the
  standard zero-shot protocol. Both are run and reported.
- **Instruct prefix.** Every prompt_text starts with `You are a helpful assistant.<|endofprompt|>`,
  as in `example.py` and the libritts CosyVoice3 recipe (all training sequences carry it).
- **Text normalization.** `text_frontend=True` (default): wetext English normalization + number
  spelling on the target text before tokenization, exactly what a user of the shipped model gets
  (XTTS applies its own num2words cleaners internally too, and both are scored against the fixed
  reference transcript, the usual TTS-WER convention). The segments the frontend actually spoke are
  recorded per utterance (`normalized_text` in the manifest) and scored as the extra
  `wer_vs_spoken_text`; where it differs from `wer` the gap is text-formatting mismatch (digits,
  abbreviations), not recognition error. LJSpeech/LibriTTS-R transcripts are already normalized, so
  the two coincide there. The prompt transcript is not normalized (the `<|endofprompt|>` tag
  disables the frontend for it). `--no_text_frontend` also disables the paragraph splitter.
- **Sampling.** The checkpoint's `cosyvoice3.yaml` defaults (RAS sampling, top_p 0.8, top_k 25,
  win_size 10, tau_r 0.1), fp32, non-streaming, speed 1.0. Seed fixed per utterance
  (`seed + crc32(uid)`).
- **Prompt length.** Prompts longer than 30 s (a few LibriTTS-R utterances) are cropped to their
  first 29 s (`prompt_crops/`), recorded as `prompt_seconds_used` in the manifest.
- **Resume.** Wavs are written atomically and skipped when present; `run_score.sbatch` refuses to
  score an incomplete set unless `ALLOW_MISSING=<n>` is set. Scoring checkpoints every utterance to
  `eval_<dataset>.json.partial.jsonl` and resumes from it; delete that file (and the `eval_*.json`)
  before re-scoring wavs that were re-synthesized with `--overwrite`.
- **Failure handling.** Very short texts occasionally make the LLM emit almost no speech tokens
  and the vocoder crashes (`Kernel size can't be greater than actual input size`); the
  synthesizer re-samples with a new seed (`--max_attempts`, default 3) and records `attempts`.
  Degenerate but non-crashing outputs (early EOS, e.g. 0.2 s for a 7 s target) are scored as-is;
  `n_pred_under_0p5s` and `n_pred_under_quarter_gt` in the results count them. After the shards,
  a one-shard fill-in pass (`run_synth.sbatch <ds> 0 1 <mode>`) re-attempts any utterance still
  missing before the scorer runs.
- **Nodes.** Scoring excludes `babel-l9-16` (onnxruntime does not import there, DNSMOS silently
  reports "not installed"); synthesis excludes the Blackwell nodes (torch 2.3.1 has no kernels).

## Sharding used for the full runs (2026-09-08)

ljspeech 1, esd 3, vctk 4, libritts_test_clean 8, libritts_test_other 8 shards per prompt mode
(24 jobs per mode, ~600 utterances per shard, 1-2 h each on an L40S/A100 at ~5-8 s per utterance).
