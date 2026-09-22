#!/bin/bash
#SBATCH --job-name=cosy3_smoke
#SBATCH --output=/data/user_data/xoy/slurm_logs/cosy3_smoke_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/cosy3_smoke_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# torch 2.3.1 cu121 has no Blackwell kernels -- see eval/run_synth.sbatch
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-l5-28,babel-l5-32,babel-l9-16,babel-l9-20,babel-l9-28,babel-l9-32,babel-m9-16,babel-m9-20,babel-m9-28,babel-m9-32,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28,babel-z5-32,babel-z9-16

# End-to-end smoke test on LIMIT (default 3) utterances per test set and prompt mode: synthesis
# (cosyvoice conda env) -> scoring (eval-articulatory-tts venv) -> emotion + accent side metrics
# (eval-emotion / eval-accent venvs) -> merge. Writes under .../cosyvoice3_eval/<ckpt>/smoke/ so it
# never collides with the real runs. Prints per-utterance generation time so the full runs can be
# budgeted.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/user_data/xoy/.cache/huggingface
export MODELSCOPE_CACHE=/data/user_data/xoy/.cache/modelscope
export PYTHONUNBUFFERED=1
PY=/data/user_data/xoy/miniconda3/envs/cosyvoice/bin/python
EVAL_VENV=/data/user_data/xoy/venvs/eval-articulatory-tts
EMOTION_VENV=/data/user_data/xoy/venvs/eval-emotion
ACCENT_VENV=/data/user_data/xoy/venvs/eval-genaid
ART_REPO=/home/xoy/articulatory-tts
CKPT_NAME=Fun-CosyVoice3-0.5B-2512
MODEL_DIR=/data/user_data/xoy/cosyvoice_models/$CKPT_NAME
OUT_ROOT=/data/user_data/xoy/cosyvoice3_eval/$CKPT_NAME/smoke
LIMIT=${LIMIT:-3}
DATASETS="ljspeech esd vctk libritts_test_clean libritts_test_other"
echo "job $SLURM_JOB_ID on $(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))"
# SKIP_SYNTH=1 re-runs only the scoring half against the wavs of a previous smoke run.
for mode in self cross; do
  [ "${SKIP_SYNTH:-0}" = "1" ] && break
  for ds in $DATASETS; do
    echo "=== synth $ds ($mode) ==="
    rm -rf "$OUT_ROOT/$mode/$ds"
    "$PY" -u eval/synthesize_testset.py --dataset $ds --model_dir "$MODEL_DIR" --prompt_mode $mode \
       --out_dir "$OUT_ROOT/$mode/$ds" --limit "$LIMIT" --overwrite
    cat "$OUT_ROOT/$mode/$ds"/manifest.shard0of1.jsonl | "$PY" -c "import json,sys; [print({k: r.get(k) for k in ('uid','prompt_uid','status','pred_seconds','gen_wall_seconds','n_segments','error')}) for r in map(json.loads, sys.stdin)]"
  done
done
for mode in self cross; do
  for ds in $DATASETS; do
    echo "=== score $ds ($mode) ==="
    rm -f "$OUT_ROOT/$mode/eval_${ds}.json.partial.jsonl"
    "$EVAL_VENV/bin/python" -u eval/score_synthesized.py --dataset $ds --out_dir "$OUT_ROOT/$mode/$ds" --limit "$LIMIT" \
       --checkpoint "$CKPT_NAME" --prompt_mode $mode --results_path "$OUT_ROOT/$mode/eval_${ds}.json" \
       --per_utt_out "$OUT_ROOT/$mode/eval_${ds}_per_utt.json" --wav_pairs_dir "$OUT_ROOT/$mode/$ds/wav_pairs_16k"
  done
done
echo "=== resume check: re-score ljspeech (self) from its checkpoint (should score 0 new utterances) ==="
"$EVAL_VENV/bin/python" -u eval/score_synthesized.py --dataset ljspeech --out_dir "$OUT_ROOT/self/ljspeech" --limit "$LIMIT" \
   --checkpoint "$CKPT_NAME" --prompt_mode self --results_path "$OUT_ROOT/self/eval_ljspeech.json" \
   --per_utt_out "$OUT_ROOT/self/eval_ljspeech_per_utt.json" --wav_pairs_dir "$OUT_ROOT/self/ljspeech/wav_pairs_16k"
echo "=== emotion side metric (esd) ==="
"$EMOTION_VENV/bin/python" -u "$ART_REPO/experiments/eval/score_side_metric.py" --metric emotion \
   --wav_pairs_dir "$OUT_ROOT/self/esd/wav_pairs_16k" --results_path "$OUT_ROOT/self/eval_esd_emotion.json"
echo "=== accent side metric (vctk) ==="
"$ACCENT_VENV/bin/python" -u "$ART_REPO/experiments/eval/score_side_metric.py" --metric accent \
   --wav_pairs_dir "$OUT_ROOT/self/vctk/wav_pairs_16k" --results_path "$OUT_ROOT/self/eval_vctk_accent.json"
"$EVAL_VENV/bin/python" -u "$ART_REPO/experiments/eval/merge_eval_results.py" --base "$OUT_ROOT/self/eval_esd.json" --side "$OUT_ROOT/self/eval_esd_emotion.json"
"$EVAL_VENV/bin/python" -u "$ART_REPO/experiments/eval/merge_eval_results.py" --base "$OUT_ROOT/self/eval_vctk.json" --side "$OUT_ROOT/self/eval_vctk_accent.json"
for ds in $DATASETS; do echo "--- $ds ---"; python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d['metrics'], indent=1))" "$OUT_ROOT/self/eval_${ds}.json"; done
echo "SMOKE_DONE"
