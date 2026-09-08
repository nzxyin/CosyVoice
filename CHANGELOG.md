# Changelog

All notable changes to this clone. Upstream CosyVoice code is untouched; entries here are about the
`eval/` pipeline and the environment on Babel.

## 2026-09-08

- Initialized the `third_party/Matcha-TTS` submodule (was registered but not checked out).
- Added `eval/`: `build_env.sbatch` (conda env on /data + checkpoint download + symlink),
  `requirements-inference.txt` (repo pins minus deepspeed/tensorrt), `pip-build-constraints.txt`
  (`setuptools<80`, needed because `openai-whisper==20231117`'s sdist imports `pkg_resources`),
  `synthesize_testset.py`, `score_synthesized.py`, `run_synth.sbatch`, `run_score.sbatch`,
  `smoke_test.sbatch`, `summarize_results.py`, `README.md`.
- Added this `CHANGELOG.md` and `CLAUDE.md`.
- Jobs: env build 10353890 failed (`openai-whisper` sdist needs `pkg_resources`, gone in setuptools>=80);
  10353973 failed the same way (a `PIP_CONSTRAINT` pin does not reach pip's isolated build env);
  10353995 COMPLETED (4m17s) after building whisper with `--no-build-isolation --no-deps` against the
  env's `setuptools<80` -- env at `/data/user_data/xoy/miniconda3/envs/cosyvoice`, `pip check` clean,
  checkpoint (9.1 GB) at `/data/user_data/xoy/cosyvoice_models/Fun-CosyVoice3-0.5B-2512`, symlinked to
  `pretrained_models/Fun-CosyVoice3-0.5B`. Smoke test 10353996 chained behind it.
- Review pass (4-lens adversarial review of `eval/`, 28 agents) confirmed four defects, all fixed:
  (1) the frontend-normalized target text was not recorded, so WER could not distinguish
  text-formatting mismatch from recognition error -> manifest now carries `normalized_text`, scorer
  reports the extra `wer_vs_spoken_text`; (2) `run_score.sbatch` skipped side metrics on a non-empty
  but possibly truncated JSON -> parse-and-key check; (3) no exclusion of `babel-l9-16` (onnxruntime
  import failure makes DNSMOS silently report "not installed") -> excluded, plus a hard onnxruntime
  import check in the scorer; (4) stage-1 scoring had no intra-stage checkpoint -> per-utterance
  `.partial.jsonl` checkpoint + resume. Smoke test now also scores `cross` and re-runs one scorer
  from its checkpoint; `SKIP_SYNTH=1` re-runs only the scoring half.
- Launched the full synthesis matrix: 48 shards (jobs 10354132-10354179; ljspeech 1, esd 3, vctk 4,
  libritts_test_clean 8, libritts_test_other 8 per prompt mode, both `self` and `cross`), all on
  `preempt`. Tracked in https://github.com/nzxyin/CosyVoice/issues/1 (on the user's fork `nzxyin/CosyVoice`, created for this purpose with
  issues enabled and added as git remote `fork`; the issue mistakenly opened first on
  `QwenAudio/CosyVoice` (#1942) is closed).
- Fixed a startup race in `synthesize_testset.py`: sibling shards shared one `split_info.json.tmp`
  and shard `cross/esd/1` (job 10354158) died on a `FileNotFoundError` from a sibling's rename;
  per-shard temp names now, losing the race is tolerated. Shard resubmitted, its scorer re-chained.
- First results in: `self`/ljspeech scored (150/150; WER 9.07% raw / 2.74% Whisper-normalized,
  UTMOSv2 3.95, DNSMOS ovr 3.41, speaker cosine 0.893, emotion 0.973, accent 0.853, RTF 0.53 on L40S).
- Two model-side failure modes seen in the first shards and handled: (a) for very short texts
  ("no", "A watch.", "please excuse me.") the LLM sometimes emits ~no speech tokens and HiFT's f0
  predictor crashes on a 2-3 frame mel (`Kernel size can't be greater than actual input size`) -> no
  wav; `synthesize_testset.py` now re-samples with a new seed up to `--max_attempts 3` and records
  `attempts`; (b) degenerate early-EOS outputs (e.g. 0.2 s of "you" for a 7.5 s LJSpeech target) are
  kept and scored as-is, and the scorer now reports `n_pred_under_0p5s` / `n_pred_under_quarter_gt`.
- Re-chained the pipeline: for every (mode, dataset) except the already-scored `self`/ljspeech, the
  pending scorer was cancelled and replaced by a fill-in synthesis job (`run_synth.sbatch <ds> 0 1
  <mode>`, `afterany` on the shards; skips existing wavs, retries the crashed ones) followed by the
  scorer (`afterok` on the fill-in). Fill-in jobs 10354440-10354456 (even), scorers 10354441-10354457
  (odd).
- 11:10 EDT status: all 48 shards and 9 fill-in passes COMPLETED; 7/10 sets scored. Five `self`-mode
  utterances (ESD 0011_000747, 0016_001447 "please excuse me."; 1 in libritts_test_clean, 2 in
  test-other, all <=4-word texts) crashed on all 3 seeds -- in `self` mode the LLM is shown the
  speech tokens of the very sentence it must say and emits (near) nothing; they are recorded as
  synthesis failures (`failed_uids`/`synthesis_errors` in the results JSON) and those three sets are
  scored with `ALLOW_MISSING=5`. First attempt at that (jobs 10357901-3) died in 2 s on a zsh
  word-splitting slip in the submit loop (dataset arrived as "self esd"); resubmitted as
  10357989-10357991.
- The cross/vctk scorer 10354453 was preempted twice (15 min and 78 min in) and requeued; the
  per-utterance `.partial.jsonl` checkpoint meant the second run finished scoring and merged before
  its preemption, and the third run exited in 3 s on the "already merged" check -- the resume logic
  from the review pass paid for itself on day one.
- WER reporting aligned with articulatory-tts's GH #32 fix (commit b040aa1, 2026-09-08): the
  reference's primary `wer` is now the Whisper-normalized corpus WER (pairs with an empty normalized
  reference skipped). We report both: `wer` (raw, lowercased, punctuation kept -- comparable to the
  XTTS port and pre-#32 numbers) and `wer_whisper_normalized` (same rule as the fixed reference).
  `eval/reaggregate_wer.py` recomputes the WER keys from the saved per-utterance texts without
  re-running Whisper; applied to the finished sets. Per-group breakdowns and
  `eval/summarize_results.py` carry both columns.
- Per-accent / per-emotion evaluation (explicit user request 2026-09-08, in line with the TTS and
  EmoSphere++ evals): added `eval/score_side_per_utt.py` (per-utterance emotion2vec+/CommonAccent
  cosine + the classifiers' own label agreement, folded into `by_emotion`/`by_accent`/`by_speaker`
  of the results JSON, asserting the mean reproduces the merged side metric) and
  `eval/run_side_per_utt.sbatch`; breakdown tables in `summarize_results.py` now carry WER raw/norm,
  UTMOSv2, DNSMOS, speaker/emotion/accent cosine, pred/GT duration ratio and label agreement.
  Jobs: 10358133 (vctk self), 10358134 (vctk cross), 10358135 (esd cross), 10358136 (esd self,
  afterok the esd self rescorer 10357989). Interim overall table recorded in CLAUDE.md.
- Committed the eval pipeline on branch `eval/cosyvoice3-baseline` and pushed it to the fork
  (`nzxyin/CosyVoice`). Upstream's `.gitignore` rule `**/*build*` silently ignores
  `eval/build_env.sbatch` and `eval/pip-build-constraints.txt`; both are force-added (`git add -f`),
  so re-add them the same way after editing.
- Opened PR https://github.com/nzxyin/CosyVoice/pull/2 (fork, `eval/cosyvoice3-baseline` -> `main`);
  final result tables will be added to `CLAUDE.md` on the same branch once the remaining scoring jobs
  finish. Whisper in the scorer runs exactly as the reference does (fp32, one utterance per
  `generate()`, greedy, no `torch.compile`/batching) -- kept for comparability; that is why the
  LibriTTS test-other scorer takes ~4 h.
- `score_side_per_utt.py`: label agreement is now only computed for groups with a single target label
  (emotion/accent groups, VCTK speakers); ESD per-speaker groups span all five emotions and had been
  getting a meaningless 0.20 "GT->target" figure. cross/esd refolded. First per-emotion result
  (cross/esd): emotion cosine Neutral 0.912, Angry 0.891, Sad 0.786, Happy 0.768, Surprise 0.464;
  emotion2vec labels the output as the target emotion 96/83/64/60/19% of the time (GT: 93-100%).

