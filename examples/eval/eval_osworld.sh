#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

##############################
# 参数配置
##############################

MODEL_PATH=${MODEL_PATH:-"../models/Qwen3-VL-8B-Instruct"}
CHECKPOINT=${CHECKPOINT:-"${MODEL_PATH}"}
DATA_PATH=${DATA_PATH:-"../data/osworld_test_all.jsonl"}
ENV_URL=${ENV_URL:-"http://127.0.0.1"}
ENV_MANAGER_PORT=${ENV_MANAGER_PORT:-8180}
SAVE_PATH=${SAVE_PATH:-"../results"}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-20480}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
AGENT_MAX_STEPS=${AGENT_MAX_STEPS:-25}
NUM_HISTORY=${NUM_HISTORY:-3}
NUM_INPUT_IMAGE=${NUM_INPUT_IMAGE:-5}
NUM_TEXT_HISTORY=${NUM_TEXT_HISTORY:-}
CLEAN_REMOTE_ENV=${CLEAN_REMOTE_ENV:-1}
NUM_EVAL_WORKERS=${NUM_EVAL_WORKERS:-4}
AGENT_TYPE=${AGENT_TYPE:-uitars}  #  mai
AGENT_ACTION_SPACE=${AGENT_ACTION_SPACE:-computer}
AGENT_PROMPT_LANGUAGE=${AGENT_PROMPT_LANGUAGE:-Chinese}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

##############################
# OSWorld 连通性预检
##############################

_LOCAL_BYPASS="127.0.0.1,localhost,$(hostname),10.0.0.0/8"
export no_proxy="${no_proxy:-${_LOCAL_BYPASS}}"
export NO_PROXY="${no_proxy}"

IFS=',' read -ra URL_LIST <<< "${ENV_URL}"
_primary_url="${URL_LIST[0]//[[:space:]]/}"
if [[ -z "${_primary_url}" ]]; then
  echo "[OSWorld] ENV_URL 为空" >&2
  exit 2
fi
if ! curl -fsS --max-time 5 "${_primary_url}:${ENV_MANAGER_PORT}/openapi.json" >/dev/null; then
  echo "[OSWorld] 无法访问 ${_primary_url}:${ENV_MANAGER_PORT}，请确认隧道已建立" >&2
  exit 2
fi
echo "[OSWorld] 预检通过: ${_primary_url}:${ENV_MANAGER_PORT}"

##############################
# 评估前准备
##############################

if [[ "${CLEAN_REMOTE_ENV}" == "1" ]]; then
  for raw_url in "${URL_LIST[@]}"; do
    url="${raw_url//[[:space:]]/}"
    [[ -z "${url}" ]] && continue
    if curl -fsS -m 10 -X POST "${url}:${ENV_MANAGER_PORT}/clean" >/dev/null; then
      echo "[OSWorld] clean OK: ${url}:${ENV_MANAGER_PORT}"
    else
      echo "[OSWorld] clean FAIL: ${url}:${ENV_MANAGER_PORT}"
    fi
  done
fi

mkdir -p "${SAVE_PATH}"

##############################
# 启动评估
##############################

echo "======================================"
echo "OSWorld Evaluation (LLaMA-Factory)"
echo "  Model:    ${CHECKPOINT}"
echo "  Data:     ${DATA_PATH}"
echo "  Env:      ${ENV_URL}:${ENV_MANAGER_PORT}"
echo "  Save:     ${SAVE_PATH}"
echo "  Agent:    ${AGENT_TYPE} / ${AGENT_ACTION_SPACE}"
echo "  Workers:  ${NUM_EVAL_WORKERS}"
echo "  TP:       ${TENSOR_PARALLEL}"
echo "  GPUs:     ${CUDA_VISIBLE_DEVICES}"
echo "======================================"

set +e
python3 -m osworld.eval_osworld \
    --env_type osworld \
    --env_url "${ENV_URL}" \
    --data_path "${DATA_PATH}" \
    --env_manager_port "${ENV_MANAGER_PORT}" \
    --pretrain "${CHECKPOINT}" \
    --save_path "${SAVE_PATH}" \
    --agent_type "${AGENT_TYPE}" \
    --agent_action_space "${AGENT_ACTION_SPACE}" \
    --agent_prompt_language "${AGENT_PROMPT_LANGUAGE}" \
    --action_space pyautogui \
    --observation_type screenshot \
    --agent_max_steps "${AGENT_MAX_STEPS}" \
    --num_history "${NUM_HISTORY}" \
    --num_input_image "${NUM_INPUT_IMAGE}" \
    ${NUM_TEXT_HISTORY:+--num_text_history "${NUM_TEXT_HISTORY}"} \
    --vllm_tensor_parallel_size "${TENSOR_PARALLEL}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --num_eval_workers "${NUM_EVAL_WORKERS}" \
    --save_trajectory \
    2>&1 | tee "${SAVE_PATH}/eval_osworld.log"
EVAL_EXIT_CODE=${PIPESTATUS[0]}
set -e
exit "${EVAL_EXIT_CODE}"
