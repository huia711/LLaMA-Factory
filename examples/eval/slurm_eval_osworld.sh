#!/bin/bash -l
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=250G
#SBATCH --job-name=lf_eval_osworld
#SBATCH --export=ALL
#SBATCH --output=slurm-%x-%j.out

# =============================================================================
# slurm_eval_osworld.sh - SLURM 版 OSWorld 评测（LLaMA-Factory 版）
# =============================================================================
# 对标 OpenRLHF/scripts/eval/slurm_eval_osworld.sh
# 特点：
#   - 登录节点直接 bash 执行时自动 sbatch 提交
#   - 自动建立 SSH tunnel 到 OSWorld 环境
#   - 使用 LLaMA-Factory 的 vLLM 推理（支持 Qwen3-VL）
# =============================================================================

set -euo pipefail

# 允许在登录节点直接 `bash examples/eval/slurm_eval_osworld.sh`：
# 若当前不在 Slurm 作业内，则自动二次提交为 sbatch 作业。
# 这样可统一入口，避免手动拼长 sbatch 参数。
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  SCRIPT_PATH="$(readlink -f "$0")"
  PROJECT_DIR_AUTO="$(cd "$(dirname "${SCRIPT_PATH}")/../.." && pwd)"

  SUBMIT_NNODES="${NNODES:-1}"
  SUBMIT_GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
  SUBMIT_MEM="${MEM_PER_JOB:-250G}"
  SUBMIT_TIME="${WALLTIME:-08:00:00}"
  SUBMIT_PARTITION="${PARTITION:-boost_usr_prod}"
  SUBMIT_JOB_NAME="${JOB_NAME:-lf_eval_osworld}"

  # --export=ALL,PROJECT_DIR=...：保留用户当前环境变量，并显式传入项目根目录
  # 供作业内继续定位 examples/eval/eval_osworld.sh 使用。
  SUBMIT_ACCOUNT="${ACCOUNT:-EUHPC_A06_060}"

  sbatch \
    --account="${SUBMIT_ACCOUNT}" \
    --partition="${SUBMIT_PARTITION}" \
    --job-name="${SUBMIT_JOB_NAME}" \
    --time="${SUBMIT_TIME}" \
    --nodes="${SUBMIT_NNODES}" \
    --ntasks-per-node=1 \
    --gres="gpu:${SUBMIT_GPUS_PER_NODE}" \
    --mem="${SUBMIT_MEM}" \
    --chdir="${PROJECT_DIR_AUTO}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR_AUTO}" \
    "${SCRIPT_PATH}" "$@"
  exit $?
fi

# 1) Activate runtime environment
source ~/.bashrc
if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
module load gcc/12.2.0 >/dev/null 2>&1 || true
module load cuda/12.2 >/dev/null 2>&1 || true

# 默认环境：qwen3（vllm 新版本，支持 Qwen3-VL）
if [[ -z "${CONDA_ENV_NAME:-}" ]]; then
  CONDA_ENV_NAME="qwen3"
fi
if ! conda activate "${CONDA_ENV_NAME}"; then
  echo "[slurm_eval_osworld] conda env not found: ${CONDA_ENV_NAME}" >&2
  conda info --envs || true
  exit 127
fi
echo "[slurm_eval_osworld] CONDA_ENV_NAME=${CONDA_ENV_NAME}"

# 2) Resolve and switch to project root
if [[ -z "${PROJECT_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/eval/eval_osworld.sh" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  elif [[ -f "/leonardo/home/userexternal/${USER}/GUI/LLaMA-Factory/examples/eval/eval_osworld.sh" ]]; then
    PROJECT_DIR="/leonardo/home/userexternal/${USER}/GUI/LLaMA-Factory"
  else
    echo "[slurm_eval_osworld] cannot resolve PROJECT_DIR" >&2
    exit 127
  fi
fi

cd "${PROJECT_DIR}"
echo "[slurm_eval_osworld] PROJECT_DIR=${PROJECT_DIR}"
echo "[slurm_eval_osworld] PWD=$(pwd)"
echo "[slurm_eval_osworld] NUM_HISTORY=${NUM_HISTORY:-3}, NUM_INPUT_IMAGE=${NUM_INPUT_IMAGE:-3}"
echo "[slurm_eval_osworld] AGENT_TYPE=${AGENT_TYPE:-<unset>}"
echo "[slurm_eval_osworld] CHECKPOINT=${CHECKPOINT:-<unset>}"
echo "[slurm_eval_osworld] MODEL_PATH=${MODEL_PATH:-<unset>}"

# 3) 建立/复用计算节点 -> 登录节点的中继隧道，确保可访问 OSWorld manager。
_TUNNEL_SCRIPT="${PROJECT_DIR}/../OpenRLHF/scripts/tunnel/osworld_tunnel_auto.sh"
if [[ -f "${_TUNNEL_SCRIPT}" ]]; then
  source "${_TUNNEL_SCRIPT}"
elif [[ -f "${PROJECT_DIR}/../scripts/tunnel/osworld_tunnel_auto.sh" ]]; then
  source "${PROJECT_DIR}/../scripts/tunnel/osworld_tunnel_auto.sh"
else
  echo "[slurm_eval_osworld] No tunnel script found, assuming direct connectivity."
fi

# 4) Run evaluation
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
bash "${PROJECT_DIR}/examples/eval/eval_osworld.sh"
