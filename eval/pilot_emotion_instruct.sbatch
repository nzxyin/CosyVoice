#!/bin/bash
#SBATCH --job-name=cosy3_emo_pilot
#SBATCH --output=/data/user_data/xoy/slurm_logs/cosy3_emo_pilot_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/cosy3_emo_pilot_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --requeue
# torch 2.3.1 (cosyvoice env) has no Blackwell kernels: same --exclude list as run_synth.sbatch.
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-l5-28,babel-l5-32,babel-l9-16,babel-l9-20,babel-l9-28,babel-l9-32,babel-m9-16,babel-m9-20,babel-m9-28,babel-m9-32,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28,babel-z5-32,babel-z9-16

# ESD-val pilot for --prompt_mode cross_label's instruction wording. 100 val utterances (shard 0 of 10
# of the uid-sorted split = 2 per speaker x emotion) under four conditions:
#   cross        emotion cloning (same speaker + emotion prompt), the reference point
#   label_zh     cross_label, --emotion_instruct_set zh (the repo's own emotion instruct template)
#   label_en     cross_label, --emotion_instruct_set en
#   label_none   cross_label, --emotion_instruct_set none (Neutral prompt, no emotion instruction)
# then eval/pilot_emotion_instruct.py (eval-emotion venv) prints per-emotion emotion2vec+ cosine, label
# agreement and duration ratio. Val is disjoint from the ESD test split the real runs use.
# Restart-safe (synthesis skips existing wavs, scoring is cheap).
#   sbatch eval/pilot_emotion_instruct.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/user_data/xoy/.cache/huggingface
export MODELSCOPE_CACHE=/data/user_data/xoy/.cache/modelscope
export PYTHONUNBUFFERED=1
PY=/data/user_data/xoy/miniconda3/envs/cosyvoice/bin/python
EMOTION_VENV=/data/user_data/xoy/venvs/eval-emotion
CKPT_NAME=${CKPT_NAME:-Fun-CosyVoice3-0.5B-2512}
MODEL_DIR=/data/user_data/xoy/cosyvoice_models/$CKPT_NAME
ROOT=/data/user_data/xoy/cosyvoice3_eval/$CKPT_NAME/pilot_esd_val
echo "job $SLURM_JOB_ID on $(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
for cond in cross label_zh label_en label_none; do
  case $cond in
    cross) extra=(--prompt_mode cross);;
    label_*) extra=(--prompt_mode cross_label --emotion_instruct_set "${cond#label_}");;
  esac
  "$PY" -u eval/synthesize_testset.py --dataset esd_val --model_dir "$MODEL_DIR" --out_dir "$ROOT/$cond" \
    --shard_index 0 --num_shards 10 "${extra[@]}"
done
"$EMOTION_VENV/bin/python" -u eval/pilot_emotion_instruct.py --root "$ROOT" --conditions cross label_zh label_en label_none
echo "PILOT_DONE"
