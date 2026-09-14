#!/bin/bash
#SBATCH --job-name=cosy3_score
#SBATCH --output=/data/user_data/xoy/slurm_logs/cosy3_score_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/cosy3_score_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
# onnxruntime fails to import on babel-l9-16 (reported by a sibling eval session 2026-09-08);
# torchmetrics' DNSMOS then reports "not installed" instead of erroring, so keep scoring off it.
# The eval-genaid venv (GenAID accent metric, torch 2.3 cu121) has no Blackwell kernels: exclude preempt's
# RTX_PRO_6000 nodes (same list as run_synth.sbatch; babel-l9-16 is among them).
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-l5-28,babel-l5-32,babel-l9-16,babel-l9-20,babel-l9-28,babel-l9-32,babel-m9-16,babel-m9-20,babel-m9-28,babel-m9-32,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28,babel-z5-32,babel-z9-16

# Score one synthesized test set: stage 1 WER/UTMOSv2/DNSMOS/speaker_cosine (eval-articulatory-tts
# venv, eval/score_synthesized.py), stage 2 emotion_cosine (eval-emotion venv) and accent_cosine
# (eval-accent venv) via articulatory-tts's own score_side_metric.py on the 16 kHz wav pairs stage 1
# writes, stage 3 merge (articulatory-tts's merge_eval_results.py). Same chain as
# articulatory-tts's run_eval_matrix_*.sh; both side metrics run on every dataset (emotion is most
# meaningful on ESD, accent on VCTK, but pred-vs-GT cosine is defined everywhere). Each stage is
# skipped if its output already exists AND parses with the expected key, so a requeue resumes; stage 1
# additionally checkpoints per utterance (<results>.partial.jsonl) so a mid-stage preemption only
# costs the utterance in flight. If wavs are re-synthesized with --overwrite, delete
# eval_<dataset>*.json and the .partial.jsonl first.
#   sbatch eval/run_score.sbatch <dataset> [prompt_mode]
# Normally chained behind the synthesis shards with --dependency=afterok:...; the completeness gate
# below is the independent check that every synthesizable utterance really has a wav (a preempted
# shard leaves a manifest but not all of its wavs). Set ALLOW_MISSING=<n> to score anyway with up
# to n missing.
set -euo pipefail
DATASET=${1:?dataset}
PROMPT_MODE=${2:-self}
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/user_data/xoy/.cache/huggingface
export PYTHONUNBUFFERED=1
EVAL_VENV=/data/user_data/xoy/venvs/eval-articulatory-tts
EMOTION_VENV=/data/user_data/xoy/venvs/eval-emotion
ACCENT_VENV=/data/user_data/xoy/venvs/eval-genaid
ART_REPO=/home/xoy/articulatory-tts
CKPT_NAME=${CKPT_NAME:-Fun-CosyVoice3-0.5B-2512}
OUT_ROOT=/data/user_data/xoy/cosyvoice3_eval/$CKPT_NAME
OUT_DIR=$OUT_ROOT/${PROMPT_MODE}/$DATASET
RESULTS=$OUT_ROOT/${PROMPT_MODE}/eval_${DATASET}.json
SIDE_EMO=$OUT_ROOT/${PROMPT_MODE}/eval_${DATASET}_emotion.json
SIDE_ACC=$OUT_ROOT/${PROMPT_MODE}/eval_${DATASET}_accent.json
WAV_PAIRS=$OUT_DIR/wav_pairs_16k
echo "job $SLURM_JOB_ID on $(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) dataset=$DATASET prompt=$PROMPT_MODE ckpt=$CKPT_NAME"

# --- completeness gate: every synthesizable utterance must have a wav ----------------------------
python3 - "$DATASET" "$OUT_DIR" "${ALLOW_MISSING:-0}" <<'PY'
import os, sys
sys.path.insert(0, "eval")
from synthesize_testset import load_items
ds, out_dir, allow = sys.argv[1], sys.argv[2], int(sys.argv[3])
items = load_items(ds)
missing = [it["uid"] for it in items
           if not (os.path.exists(os.path.join(out_dir, "wavs", it["uid"] + ".wav"))
                   and os.path.getsize(os.path.join(out_dir, "wavs", it["uid"] + ".wav")) > 1000)]
print(f"{ds}: {len(items) - len(missing)}/{len(items)} utterances have a prediction wav")
if len(missing) > allow:
    print(f"REFUSING to score: {len(missing)} missing (ALLOW_MISSING={allow}), e.g. {missing[:5]}")
    sys.exit(2)
PY

have_metric() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if sys.argv[2] in d.get("metrics",{}) else 1)' "$1" "$2" 2>/dev/null; }
# side-metric JSONs are flat ({"emotion_cosine": {...}}); an unparseable (e.g. truncated) file counts as absent
have_top_level_key() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if sys.argv[2] in d else 1)' "$1" "$2" 2>/dev/null; }

if [ -s "$RESULTS" ] && have_metric "$RESULTS" emotion_cosine && have_metric "$RESULTS" accent_cosine; then
  echo "skip $DATASET (emotion_cosine + accent_cosine already merged into $RESULTS)"
  exit 0
fi
if ! { [ -s "$RESULTS" ] && have_metric "$RESULTS" utmosv2; }; then
  "$EVAL_VENV/bin/python" -u eval/score_synthesized.py \
    --dataset "$DATASET" --out_dir "$OUT_DIR" --prompt_mode "$PROMPT_MODE" --checkpoint "$CKPT_NAME" \
    --results_path "$RESULTS" \
    --per_utt_out "$OUT_ROOT/${PROMPT_MODE}/eval_${DATASET}_per_utt.json" \
    --wav_pairs_dir "$WAV_PAIRS"
else
  echo "stage 1 results already present, skipping to side metrics"
fi
if have_top_level_key "$SIDE_EMO" emotion_cosine; then echo "emotion side metric exists"; else
  "$EMOTION_VENV/bin/python" -u "$ART_REPO/score_side_metric.py" --metric emotion \
    --wav_pairs_dir "$WAV_PAIRS" --results_path "$SIDE_EMO"
fi
if have_top_level_key "$SIDE_ACC" accent_cosine; then echo "accent side metric exists"; else
  "$ACCENT_VENV/bin/python" -u "$ART_REPO/score_side_metric.py" --metric accent \
    --wav_pairs_dir "$WAV_PAIRS" --results_path "$SIDE_ACC"
fi
# provenance first, then merge: the merge is what flips the skip condition above, so a preemption
# between the two re-runs both instead of losing the provenance line.
python3 - "$RESULTS" "$(git -C "$ART_REPO" rev-parse HEAD)" "$(git rev-parse HEAD)" <<'PY'
import json, os, sys
p, art_sha, own_sha = sys.argv[1:4]
d = json.load(open(p))
d["side_metric_source"] = f"articulatory-tts@{art_sha} score_side_metric.py (emotion: iic/emotion2vec_plus_large; accent: Jzuluaga/accent-id-commonaccent_xlsr-en-english)"
d["eval_code_commit"] = own_sha
tmp = p + ".tmp"; json.dump(d, open(tmp, "w"), indent=2); os.replace(tmp, p)
PY
"$EVAL_VENV/bin/python" -u "$ART_REPO/merge_eval_results.py" --base "$RESULTS" --side "$SIDE_EMO" "$SIDE_ACC"
echo "SCORE_DONE $DATASET $PROMPT_MODE"
