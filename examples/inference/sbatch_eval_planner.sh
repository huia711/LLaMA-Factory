#!/bin/bash -l
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=eval_planner
#SBATCH --export=ALL
#SBATCH --account=EUHPC_A06_060

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CHECKPOINT="${CHECKPOINT:-}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
LORA_DIR="${LORA_DIR:-lora-sft-subtask-v4}"
EVAL_DATA="${EVAL_DATA:-agentnet_ubuntu_3k_subtask_v4_eval.jsonl}"

CONDA_SH="${CONDA_SH:-/leonardo/home/userexternal/mli00001/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-${PROJ_DIR}/conda_env/llamafactory}"
BASE_MODEL="${BASE_MODEL:-${PROJ_DIR}/models/Qwen2.5-VL-7B-Instruct}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJ_DIR}/results/train/qwen2_5vl-7b}"
DATA_ROOT="${DATA_ROOT:-${PROJ_DIR}/data/AgentNet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJ_DIR}/results/eval}"

LORA_PATH="${LORA_DIR}"
if [[ "${LORA_PATH}" != /* ]]; then
  LORA_PATH="${RESULTS_ROOT}/${LORA_PATH}"
fi

EVAL_DATA_PATH="${EVAL_DATA}"
if [[ "${EVAL_DATA_PATH}" != /* ]]; then
  EVAL_DATA_PATH="${DATA_ROOT}/${EVAL_DATA_PATH}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/$(basename -- "${LORA_PATH}")}"

export HF_HOME=/leonardo/home/userexternal/mli00001/.cache/huggingface
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TMPDIR=/scratch_local

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

CKPT_ARG=()
if [[ -n "${CHECKPOINT}" ]]; then
  CKPT_ARG=(--checkpoint "${CHECKPOINT}")
fi

python3 "${PROJ_DIR}/data/AgentNet/scripts/eval_planner.py" \
  --base-model "${BASE_MODEL}" \
  --lora-path "${LORA_PATH}" \
  "${CKPT_ARG[@]}" \
  --eval-data "${EVAL_DATA_PATH}" \
  --image-dir "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-new-tokens 2048
