#!/bin/bash -l
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=merge_lora
#SBATCH --export=ALL
#SBATCH --account=EUHPC_A06_060

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PROJ_DIR="${PROJ_DIR:-$(cd -- "${LLAMA_FACTORY_DIR}/.." && pwd)}"

MERGE_CONFIG="${MERGE_CONFIG:-}"
CONDA_SH="${CONDA_SH:-/leonardo/home/userexternal/mli00001/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-${PROJ_DIR}/conda_env/llamafactory}"

export HF_HOME=/leonardo/home/userexternal/mli00001/.cache/huggingface
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TMPDIR=/scratch_local

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

if [[ -z "${MERGE_CONFIG}" ]]; then
  echo "ERROR: MERGE_CONFIG not set" >&2
  exit 1
fi

if [[ "${MERGE_CONFIG}" != /* ]]; then
  if [[ -f "${PROJ_DIR}/${MERGE_CONFIG}" ]]; then
    MERGE_CONFIG="${PROJ_DIR}/${MERGE_CONFIG}"
  elif [[ -f "${LLAMA_FACTORY_DIR}/${MERGE_CONFIG}" ]]; then
    MERGE_CONFIG="${LLAMA_FACTORY_DIR}/${MERGE_CONFIG}"
  elif [[ -f "${LLAMA_FACTORY_DIR}/examples/merge_lora/${MERGE_CONFIG}" ]]; then
    MERGE_CONFIG="${LLAMA_FACTORY_DIR}/examples/merge_lora/${MERGE_CONFIG}"
  fi
fi

if [[ ! -f "${MERGE_CONFIG}" ]]; then
  echo "ERROR: merge config not found: ${MERGE_CONFIG}" >&2
  exit 1
fi

python -m llamafactory.cli export "${MERGE_CONFIG}"
