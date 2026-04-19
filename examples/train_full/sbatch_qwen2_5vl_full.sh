#!/bin/bash -l
#SBATCH --partition=accelerated-h200-8
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=qwen2_5vl_full_sft
#SBATCH --export=ALL

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module add compiler/gnu/12 mpi/openmpi devel/cuda >/dev/null 2>&1 || true
fi

CONDA_SH="${CONDA_SH:-/leonardo/home/userexternal/mli00001/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-${PROJ_DIR}/conda_env/llamafactory}"

export HF_HOME=/leonardo/home/userexternal/mli00001/.cache/huggingface
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TMPDIR=/scratch_local

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

bash "${SCRIPT_DIR}/train_qwen2_5vl_full.sh"
