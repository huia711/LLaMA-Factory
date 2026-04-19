#!/bin/bash -l
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Pick a usable network interface for NCCL/GLOO.
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  IFNAME="$(ip -o -4 addr show | awk '!/ lo /{print $2; exit}')"
  if [[ -z "${IFNAME}" ]]; then
    echo "No non-loopback IPv4 interface found. Set NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME manually." >&2
    exit 1
  fi
  export NCCL_SOCKET_IFNAME="${IFNAME}"
  export GLOO_SOCKET_IFNAME="${IFNAME}"
else
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
fi

if [[ "${NCCL_SOCKET_IFNAME}" != ib* ]]; then
  export NCCL_IB_DISABLE=1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-${PROJ_DIR}/LLaMA-Factory}"
LAUNCHER_PATH="${LAUNCHER_PATH:-${LLAMA_FACTORY_DIR}/src/llamafactory/launcher.py}"
CONFIG_PATH="${CONFIG_PATH:-${LLAMA_FACTORY_DIR}/examples/train_full/qwen2_5vl_full_sft.yaml}"

NUM_GPUS="${NUM_GPUS:-4}"
NNODES="${SLURM_NNODES:-1}"
RANK="${SLURM_NODEID:-0}"

if [[ ! -f "${LAUNCHER_PATH}" ]]; then
  echo "LLaMA-Factory launcher not found: ${LAUNCHER_PATH}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Training config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ -z "${MASTER_PORT:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export MASTER_PORT="$((49152 + SLURM_JOB_ID % 16384))"
  else
    export MASTER_PORT="$(shuf -i 49152-65535 -n 1)"
  fi
fi

if [[ "${NNODES}" -gt 1 ]]; then
  MASTER_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
  export MASTER_ADDR="$(getent hosts "${MASTER_NODE}" | awk '{print $1}')"

  srun --mpi=pmix python -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes="${NNODES}" \
    --node_rank="${RANK}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    "${LAUNCHER_PATH}" \
    "${CONFIG_PATH}"
else
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

  python -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${LAUNCHER_PATH}" \
    "${CONFIG_PATH}"
fi
