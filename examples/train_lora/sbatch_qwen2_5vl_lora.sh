#!/bin/bash -l
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=sft_v5
#SBATCH --export=ALL
#SBATCH --account=EUHPC_A06_060

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CONDA_SH="${CONDA_SH:-/leonardo/home/userexternal/mli00001/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-${PROJ_DIR}/conda_env/llamafactory}"
ENABLE_TUNNEL="${ENABLE_TUNNEL:-1}"
TUNNEL_SCRIPT="${TUNNEL_SCRIPT:-${PROJ_DIR}/scripts/tunnel/osworld_tunnel_auto.sh}"

export HF_HOME=/leonardo/home/userexternal/mli00001/.cache/huggingface
export HF_DATASETS_CACHE=/leonardo_scratch/large/userexternal/mli00001/huia/cache/huggingface/datasets
export TMPDIR=/scratch_local

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

if [[ "${ENABLE_TUNNEL}" == "1" ]]; then
  if [[ ! -f "${TUNNEL_SCRIPT}" ]]; then
    echo "Tunnel script not found: ${TUNNEL_SCRIPT}" >&2
    exit 1
  fi
  source "${TUNNEL_SCRIPT}"
fi

bash "${SCRIPT_DIR}/train_qwen2_5vl_lora.sh"
