#!/bin/bash
#SBATCH --job-name=cosy3_rescore_accent_centered
#SBATCH --output=/data/user_data/xoy/slurm_logs/cosy3_rescore_accent_centered_%a_%A.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/cosy3_rescore_accent_centered_%a_%A.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-1
# The eval-genaid venv (torch 2.3 cu121) has no Blackwell kernels: exclude preempt's RTX_PRO_6000 nodes.
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-l5-28,babel-l5-32,babel-l9-16,babel-l9-20,babel-l9-28,babel-l9-32,babel-m9-16,babel-m9-20,babel-m9-28,babel-m9-32,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28,babel-z5-32,babel-z9-16

# 2026-09-16 -- accent_cosine raw GenAID -> centroid-centered GenAID rescore, VCTK only (self/cross;
# the other 8 scored sets have no accent breakdown and are not part of this metric). Mirrors
# run_rescore_accent_genaid.sbatch (2026-09-14, CommonAccent -> GenAID), one level up the same chain:
# genaid_accent.py in articulatory-tts (branch feature/accent-centering, merging to main before this
# runs) now exposes a DEFAULT_CENTER_VECTOR (the mean of the six speaker-balanced accent centroids of
# the VCTK training speakers) and accent_cosines() = (centered cosine, raw cosine); its own
# score_side_metric.py / score_side_per_utt.py write the centered value as accent_cosine and the raw
# value alongside as accent_cosine_genaid_raw. The 16 kHz wav pairs from the original scoring are
# still on disk, so this only re-runs the accent side metric on them, merges with --keep_old_as
# genaid_raw (old raw GenAID value -> accent_cosine_genaid_raw), and redoes
# eval/score_side_per_utt.py --metric accent for the per-accent/per-speaker breakdown (previous raw
# per-utt file kept as eval_vctk_accent_per_utt_genaid_raw.json). *_commonaccent keys/files from the
# 2026-09-14 rescore are untouched throughout. See articulatory-tts CLAUDE.md "Accent-metric
# diagnostic" -> Decision (2026-09-16) for what the centering vector is and why.
#   sbatch eval/run_rescore_accent_centered.sbatch          (array over mode x vctk)
set -euo pipefail
TASKS=(self:vctk cross:vctk)
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
SIDE=$OUT_ROOT/eval_${DATASET}_accent_genaid_centered.json
WAV_PAIRS=$OUT_ROOT/$DATASET/wav_pairs_16k
echo "job $SLURM_JOB_ID on $(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) dataset=$DATASET prompt=$PROMPT_MODE"
[ -s "$RESULTS" ] || { echo "no results JSON at $RESULTS"; exit 2; }
echo "pairs on disk: $(ls "$WAV_PAIRS"/*_pred.wav | wc -l)"

# Skip condition: the aggregate is already centered once metrics.accent_cosine.model carries GenAID's
# centered_model_tag(), which always contains "centroid-centered" (genaid_accent.CENTERING_TAG) --
# distinct from a plain (raw) GenAID tag, which does not.
is_centered_done() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d.get("metrics",{}).get("accent_cosine",{}); sys.exit(0 if isinstance(m,dict) and "centroid-centered" in str(m.get("model","")) else 1)' "$1" 2>/dev/null; }

if is_centered_done "$RESULTS"; then echo "aggregate already centered-GenAID-scored"; else
  "$ACCENT_VENV/bin/python" -u "$ART_REPO/score_side_metric.py" --metric accent \
    --wav_pairs_dir "$WAV_PAIRS" --results_path "$SIDE"
  python3 - "$RESULTS" "$(git -C "$ART_REPO" rev-parse HEAD)" "$(git rev-parse HEAD)" <<'PY'
import json, os, sys, datetime
p, art_sha, own_sha = sys.argv[1:4]
d = json.load(open(p))
d["side_metric_source_genaid_raw"] = d.get("side_metric_source")
d["side_metric_source"] = (f"articulatory-tts@{art_sha} score_side_metric.py (emotion: iic/emotion2vec_plus_large, unchanged; "
                           f"accent: GenAID_v6 via genaid_accent.py, CENTROID-CENTERED on genaid_accent.DEFAULT_CENTER_VECTOR "
                           f"since {datetime.date.today().isoformat()}; raw cosine kept as accent_cosine_genaid_raw)")
d["eval_code_commit"] = own_sha
for grp in ("by_accent", "by_speaker", "by_emotion"):
    for g, e in d.get(grp, {}).items():
        # move the pre-centering (raw GenAID) per-group value aside before the merge below overwrites
        # it with the centered one; never touch a *_commonaccent key already present from the
        # 2026-09-14 CommonAccent -> GenAID rescore.
        if "accent_cosine" in e and "accent_cosine_genaid_raw" not in e:
            e["accent_cosine_genaid_raw"] = e["accent_cosine"]
tmp = p + ".tmp"; json.dump(d, open(tmp, "w"), indent=2); os.replace(tmp, p)
PY
  "$EVAL_VENV/bin/python" -u "$ART_REPO/merge_eval_results.py" --base "$RESULTS" --side "$SIDE" --keep_old_as genaid_raw
fi

OLD=$OUT_ROOT/eval_${DATASET}_accent_per_utt.json
RAW_PER_UTT=$OUT_ROOT/eval_${DATASET}_accent_per_utt_genaid_raw.json
if [ -s "$OLD" ] && ! grep -q '"center_vector"' "$OLD"; then
  mv "$OLD" "$RAW_PER_UTT"   # pre-centering (raw GenAID) per-utt breakdown -> superseded file
fi
"$ACCENT_VENV/bin/python" -u eval/score_side_per_utt.py --metric accent --dataset "$DATASET" \
  --wav_pairs_dir "$WAV_PAIRS" --results_path "$RESULTS" --per_utt_out "$OLD"

python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d["metrics"]; print("accent_cosine centered", m["accent_cosine"]["mean"], "n", m["accent_cosine"]["n"], "| raw GenAID", m.get("accent_cosine_genaid_raw",{}).get("mean"), "| center_vector", d.get("accent_center_vector"))' "$RESULTS"
echo "RESCORE_DONE $DATASET $PROMPT_MODE"
