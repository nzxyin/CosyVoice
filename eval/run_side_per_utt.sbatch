#!/bin/bash
#SBATCH --job-name=cosy3_side_per_utt
#SBATCH --output=/data/user_data/xoy/slurm_logs/cosy3_side_per_utt_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/cosy3_side_per_utt_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
# The eval-genaid venv (GenAID accent metric, torch 2.3 cu121) has no Blackwell kernels: exclude preempt's
# RTX_PRO_6000 nodes (same list as run_synth.sbatch; babel-l9-16 is among them).
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-l5-28,babel-l5-32,babel-l9-16,babel-l9-20,babel-l9-28,babel-l9-32,babel-m9-16,babel-m9-20,babel-m9-28,babel-m9-32,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28,babel-z5-32,babel-z9-16

# Per-utterance emotion cosine (eval-emotion venv) and accent cosine (eval-accent venv) for one scored
# test set, folded into its eval_<dataset>.json as by_emotion / by_accent / by_speaker (+ label
# agreement), then the breakdown tables are printed. Mirrors /home/xoy/TTS/eval/run_accent_per_utt.sbatch
# and /home/xoy/EmoSpherepp/scripts/run_breakdown.sh. Requires the set's SCORE_DONE results
# (emotion_cosine/accent_cosine merged) -- the script asserts the per-utterance mean reproduces them.
#   sbatch eval/run_side_per_utt.sbatch <dataset> [prompt_mode] [metrics: "emotion accent"]
set -euo pipefail
DATASET=${1:?dataset}
PROMPT_MODE=${2:-self}
METRICS=${3:-"emotion accent"}
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/user_data/xoy/.cache/huggingface
export PYTHONUNBUFFERED=1
EMOTION_VENV=/data/user_data/xoy/venvs/eval-emotion
ACCENT_VENV=/data/user_data/xoy/venvs/eval-genaid
CKPT_NAME=${CKPT_NAME:-Fun-CosyVoice3-0.5B-2512}
OUT_ROOT=/data/user_data/xoy/cosyvoice3_eval/$CKPT_NAME/$PROMPT_MODE
RESULTS=$OUT_ROOT/eval_${DATASET}.json
echo "job $SLURM_JOB_ID on $(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) dataset=$DATASET prompt=$PROMPT_MODE metrics=$METRICS"
[ -s "$RESULTS" ] || { echo "no results JSON at $RESULTS -- run eval/run_score.sbatch first"; exit 2; }
for metric in $METRICS; do
  case $metric in emotion) PY=$EMOTION_VENV/bin/python;; accent) PY=$ACCENT_VENV/bin/python;; *) echo "unknown metric $metric"; exit 2;; esac
  "$PY" -u eval/score_side_per_utt.py --metric "$metric" --dataset "$DATASET" \
    --wav_pairs_dir "$OUT_ROOT/$DATASET/wav_pairs_16k" --results_path "$RESULTS" \
    --per_utt_out "$OUT_ROOT/eval_${DATASET}_${metric}_per_utt.json"
done
python3 eval/summarize_results.py --modes "$PROMPT_MODE" --datasets "$DATASET" --breakdowns_only || true
echo "SIDE_PER_UTT_DONE $DATASET $PROMPT_MODE"
