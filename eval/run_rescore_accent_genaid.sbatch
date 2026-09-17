#!/bin/bash
#SBATCH --job-name=cosy3_rescore_accent_genaid
#SBATCH --output=/data/user_data/xoy/slurm_logs/cosy3_rescore_accent_genaid_%a_%A.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/cosy3_rescore_accent_genaid_%a_%A.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-9
# The eval-genaid venv (torch 2.3 cu121) has no Blackwell kernels: exclude preempt's RTX_PRO_6000 nodes.
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-l5-28,babel-l5-32,babel-l9-16,babel-l9-20,babel-l9-28,babel-l9-32,babel-m9-16,babel-m9-20,babel-m9-28,babel-m9-32,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28,babel-z5-32,babel-z9-16

# 2026-09-14 -- accent_cosine CommonAccent -> GenAID rescore of all 10 scored sets (5 datasets x
# self/cross). The 16 kHz wav pairs from the original scoring are still on disk, so this only re-runs
# the accent side metric (articulatory-tts score_side_metric.py, GenAID) on them, merges with
# --keep_old_as commonaccent (old value -> accent_cosine_commonaccent) and, for ESD and VCTK (the
# sets that had a per-utterance accent breakdown), redoes eval/score_side_per_utt.py --metric accent
# (previous CommonAccent breakdown kept under *_commonaccent keys / files).
#   sbatch eval/run_rescore_accent_genaid.sbatch            (array over mode x dataset)
set -euo pipefail
TASKS=(self:ljspeech self:libritts_test_clean self:libritts_test_other self:esd self:vctk
       cross:ljspeech cross:libritts_test_clean cross:libritts_test_other cross:esd cross:vctk)
PROMPT_MODE=${TASKS[$SLURM_ARRAY_TASK_ID]%%:*}
DATASET=${TASKS[$SLURM_ARRAY_TASK_ID]#*:}
CKPT_NAME=${CKPT_NAME:-Fun-CosyVoice3-0.5B-2512}
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/user_data/xoy/.cache/huggingface
export PYTHONUNBUFFERED=1
EVAL_VENV=/data/user_data/xoy/venvs/eval-articulatory-tts
ACCENT_VENV=/data/user_data/xoy/venvs/eval-genaid
ART_REPO=/home/xoy/articulatory-tts
OUT_ROOT=/data/user_data/xoy/cosyvoice3_eval/$CKPT_NAME/$PROMPT_MODE
RESULTS=$OUT_ROOT/eval_${DATASET}.json
SIDE=$OUT_ROOT/eval_${DATASET}_accent_genaid.json
WAV_PAIRS=$OUT_ROOT/$DATASET/wav_pairs_16k
echo "job $SLURM_JOB_ID on $(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) dataset=$DATASET prompt=$PROMPT_MODE"
[ -s "$RESULTS" ] || { echo "no results JSON at $RESULTS"; exit 2; }
echo "pairs on disk: $(ls "$WAV_PAIRS"/*_pred.wav | wc -l)"

is_genaid_done() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d.get("metrics",{}).get("accent_cosine",{}); sys.exit(0 if isinstance(m,dict) and "GenAID" in str(m.get("model","")) else 1)' "$1" 2>/dev/null; }

if is_genaid_done "$RESULTS"; then echo "aggregate already GenAID-scored"; else
  "$ACCENT_VENV/bin/python" -u "$ART_REPO/score_side_metric.py" --metric accent \
    --wav_pairs_dir "$WAV_PAIRS" --results_path "$SIDE"
  python3 - "$RESULTS" "$(git -C "$ART_REPO" rev-parse HEAD)" "$(git rev-parse HEAD)" <<'PY'
import json, os, sys, datetime
p, art_sha, own_sha = sys.argv[1:4]
d = json.load(open(p))
d["side_metric_source_commonaccent"] = d.get("side_metric_source")
d["side_metric_source"] = (f"articulatory-tts@{art_sha} score_side_metric.py (emotion: iic/emotion2vec_plus_large, unchanged; "
                           f"accent: GenAID_v6 via genaid_accent.py, rescored {datetime.date.today().isoformat()} on the original wav pairs)")
d["eval_code_commit"] = own_sha
for grp in ("by_accent", "by_speaker", "by_emotion"):
    for g, e in d.get(grp, {}).items():
        for k in ("accent_cosine", "accent_label_agreement"):
            if k in e and f"{k}_commonaccent" not in e:
                e[f"{k}_commonaccent"] = e.pop(k)
tmp = p + ".tmp"; json.dump(d, open(tmp, "w"), indent=2); os.replace(tmp, p)
PY
  "$EVAL_VENV/bin/python" -u "$ART_REPO/merge_eval_results.py" --base "$RESULTS" --side "$SIDE" --keep_old_as commonaccent
fi

case $DATASET in esd|vctk)
  OLD=$OUT_ROOT/eval_${DATASET}_accent_per_utt.json
  if [ -s "$OLD" ] && python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); labs={v.get("pred_label") for v in r.values()}; sys.exit(0 if labs & {"us","england","canada","scotland","ireland","indian","african","wales"} - {"us"} or any(l in labs for l in ("england","canada","scotland","ireland")) else 1)' "$OLD" 2>/dev/null; then
    mv "$OLD" "${OLD%.json}_commonaccent.json"   # CommonAccent label vocabulary -> superseded file
  fi
  "$ACCENT_VENV/bin/python" -u eval/score_side_per_utt.py --metric accent --dataset "$DATASET" \
    --wav_pairs_dir "$WAV_PAIRS" --results_path "$RESULTS" --per_utt_out "$OLD"
  ;;
esac
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d["metrics"]; print("accent_cosine GenAID", m["accent_cosine"]["mean"], "n", m["accent_cosine"]["n"], "| CommonAccent", m.get("accent_cosine_commonaccent",{}).get("mean"))' "$RESULTS"
echo "RESCORE_DONE $DATASET $PROMPT_MODE"
