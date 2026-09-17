#!/bin/bash
#SBATCH --job-name=cosy3_env_build
#SBATCH --output=/data/user_data/xoy/slurm_logs/cosy3_env_build_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/cosy3_env_build_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# One-shot setup for the Fun-CosyVoice3 evaluation (2026-09-08):
#   1. download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 (~9.7 GB) to /data, in the background
#   2. build the `cosyvoice` conda env (python 3.10 + the repo's pinned requirements minus the
#      training/deployment-only deepspeed + tensorrt wheels, see eval/requirements-inference.txt)
#   3. import smoke test, then symlink pretrained_models/Fun-CosyVoice3-0.5B -> the /data copy
# ~/.condarc already routes envs_dirs/pkgs_dirs to /data/user_data/xoy/miniconda3/, so the env
# itself lands on /data without any extra flags. CPU-only (preempt_cpu_qos): nothing here needs
# a GPU, and /data is not mounted on the login node (which also has a 1h CPU-time rlimit).
# Safe to re-run: conda create is skipped when the env exists, pip/hf skip finished work.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PIP_CACHE_DIR=/data/user_data/xoy/.cache/pip
export HF_HOME=/data/user_data/xoy/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export MODELSCOPE_CACHE=/data/user_data/xoy/.cache/modelscope

ENV_NAME=cosyvoice
ENV_DIR=/data/user_data/xoy/miniconda3/envs/$ENV_NAME
MODEL_ROOT=/data/user_data/xoy/cosyvoice_models
MODEL_DIR=$MODEL_ROOT/Fun-CosyVoice3-0.5B-2512
mkdir -p "$MODEL_ROOT" /data/user_data/xoy/slurm_logs "$PIP_CACHE_DIR" "$MODELSCOPE_CACHE"
echo "job $SLURM_JOB_ID on $(hostname) $(date)"

# --- 1) checkpoint download, in the background ------------------------------
DL_LOG=/data/user_data/xoy/slurm_logs/cosy3_model_download_${SLURM_JOB_ID}.log
(
  set +e
  hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir "$MODEL_DIR"
  echo "hf download exit code: $?"
) > "$DL_LOG" 2>&1 &
DL_PID=$!

# --- 2) conda env -------------------------------------------------------------
source /home/xoy/miniconda3/etc/profile.d/conda.sh
if [ ! -x "$ENV_DIR/bin/python" ]; then
  echo "=== creating conda env $ENV_NAME ==="
  conda create -y -n "$ENV_NAME" python=3.10 pip
fi
conda activate "$ENV_NAME"
echo "python: $(which python) -> $(python -V)"
case "$(which python)" in "$ENV_DIR"/*) ;; *) echo "ERROR: activated python is not under $ENV_DIR"; exit 1;; esac
echo "=== pip install (inference requirements) ==="
# openai-whisper==20231117 is sdist-only and its setup.py imports pkg_resources, which setuptools>=80
# dropped; pip's isolated build env always pulls the newest setuptools (PIP_CONSTRAINT did not reach
# it, job 10353973). So: keep setuptools<80 in the env and build whisper WITHOUT isolation, --no-deps
# so it cannot drag in an unpinned torch/numpy; the pinned requirements pass right after resolves
# whisper's own deps (numba/tiktoken/more-itertools) against the pins.
export PIP_CONSTRAINT="$SLURM_SUBMIT_DIR/eval/pip-build-constraints.txt"
pip install "setuptools<80" wheel
pip install --no-build-isolation --no-deps openai-whisper==20231117
pip install -r eval/requirements-inference.txt 2>&1 | tail -30
echo "=== pip check ==="
pip check || echo "WARNING: pip check reported problems (see above)"

echo "=== import smoke test (CPU) ==="
python - <<'PY'
import sys
sys.path.append('third_party/Matcha-TTS')
import torch, torchaudio, onnxruntime, transformers, numpy, whisper
print("torch", torch.__version__, "cuda", torch.version.cuda, "torchaudio", torchaudio.__version__,
      "onnxruntime", onnxruntime.__version__, onnxruntime.get_available_providers(),
      "transformers", transformers.__version__, "numpy", numpy.__version__)
from cosyvoice.cli.cosyvoice import AutoModel  # noqa: F401
from cosyvoice.cli.frontend import CosyVoiceFrontEnd  # noqa: F401
import wetext  # noqa: F401
print("cosyvoice import OK")
PY

# --- 3) wait for the download ---------------------------------------------------
wait "$DL_PID" || true
echo "=== download log (tail) ==="; tail -20 "$DL_LOG"
grep -q "hf download exit code: 0" "$DL_LOG" || { echo "checkpoint download FAILED (see $DL_LOG)"; exit 1; }
for f in cosyvoice3.yaml llm.pt flow.pt hift.pt campplus.onnx speech_tokenizer_v3.onnx CosyVoice-BlankEN/model.safetensors CosyVoice-BlankEN/vocab.json; do
  [ -s "$MODEL_DIR/$f" ] || { echo "missing checkpoint file $f"; exit 1; }
done
echo "=== model dir ==="; ls -la "$MODEL_DIR"; du -sh "$MODEL_DIR"
# repo-local path the README/example.py expect; pretrained_models/* is gitignored
mkdir -p pretrained_models
ln -sfn "$MODEL_DIR" pretrained_models/Fun-CosyVoice3-0.5B
ls -la pretrained_models/
echo "BUILD_DONE $(date)"
