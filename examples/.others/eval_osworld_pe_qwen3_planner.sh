#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Lower-bound PE eval: Qwen3-VL-8B-Instruct as Planner, GUI executor unchanged.
export AGENT_TYPE=${AGENT_TYPE:-pe}
export PLANNER_PRETRAIN=${PLANNER_PRETRAIN:-"${PROJECT_DIR}/../../models/Qwen3-VL-8B-Instruct"}
export EXECUTOR_PRETRAIN=${EXECUTOR_PRETRAIN:-"${PROJECT_DIR}/../../models/UI-TARS-1.5-7B"}
export CHECKPOINT=${CHECKPOINT:-"${EXECUTOR_PRETRAIN}"}
export DATA_PATH=${DATA_PATH:-"${PROJECT_DIR}/../data/osworld_test_all.jsonl"}
export SAVE_PATH=${SAVE_PATH:-"${PROJECT_DIR}/../results/pe_qwen3_planner"}

# Two vLLM engines share the worker GPU by default. Override these on large-memory GPUs.
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
export NUM_EVAL_WORKERS=${NUM_EVAL_WORKERS:-1}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-20480}
export NUM_HISTORY=${NUM_HISTORY:-3}
export NUM_INPUT_IMAGE=${NUM_INPUT_IMAGE:-5}
export AGENT_MAX_STEPS=${AGENT_MAX_STEPS:-25}
export PE_MAX_PLANS=${PE_MAX_PLANS:-10}
export PE_MAX_STEPS_PER_SUBTASK=${PE_MAX_STEPS_PER_SUBTASK:-15}
export PLANNER_MAX_TOKENS=${PLANNER_MAX_TOKENS:-512}
export EXECUTOR_MAX_TOKENS=${EXECUTOR_MAX_TOKENS:-1024}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}

exec "${SCRIPT_DIR}/eval_osworld.sh"
