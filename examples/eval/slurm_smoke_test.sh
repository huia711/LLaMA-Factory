#!/bin/bash -l
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=lf_smoke_test
#SBATCH --export=ALL
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Auto-submit if not inside SLURM
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  SCRIPT_PATH="$(readlink -f "$0")"
  PROJECT_DIR_AUTO="$(cd "$(dirname "${SCRIPT_PATH}")/../.." && pwd)"
  SUBMIT_ACCOUNT="${ACCOUNT:-EUHPC_A06_060}"

  sbatch \
    --account="${SUBMIT_ACCOUNT}" \
    --partition="${PARTITION:-boost_usr_prod}" \
    --job-name="lf_smoke_test" \
    --time="00:30:00" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres="gpu:1" \
    --mem="64G" \
    --chdir="${PROJECT_DIR_AUTO}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR_AUTO}" \
    "${SCRIPT_PATH}"
  exit $?
fi

# Activate env
source ~/.bashrc
if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
module load gcc/12.2.0 >/dev/null 2>&1 || true
module load cuda/12.2 >/dev/null 2>&1 || true

CONDA_ENV_NAME="${CONDA_ENV_NAME:-qwen3}"
conda activate "${CONDA_ENV_NAME}" || { echo "conda env not found: ${CONDA_ENV_NAME}"; exit 127; }

# Resolve project root
if [[ -z "${PROJECT_DIR:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1

MODEL_PATH="${MODEL_PATH:-../models/Qwen3-VL-8B-Instruct}"

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CONDA_ENV=${CONDA_ENV_NAME}"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"

python3 examples/eval/smoke_test.py "${MODEL_PATH}"
