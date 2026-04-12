#!/bin/bash -l
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=eval_v7sft_final_v7eval
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL
#SBATCH --account=EUHPC_A06_060

PROJ_DIR=/leonardo/home/userexternal/mli00001/GUI
DATA_DIR=/leonardo_scratch/large/userexternal/mli00001/huia/data/AgentNet

export HF_HOME=/leonardo/home/userexternal/mli00001/.cache/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TMPDIR=/scratch_local

source /leonardo/home/userexternal/mli00001/miniconda3/etc/profile.d/conda.sh
conda activate ${PROJ_DIR}/conda_env/llamafactory

python3 ${PROJ_DIR}/data/AgentNet/scripts/eval_planner.py \
  --base-model   ${PROJ_DIR}/models/Qwen2.5-VL-7B-Instruct \
  --lora-path    ${PROJ_DIR}/results/train/qwen2_5vl-7b/lora-sft-subtask-v7 \
  --eval-data    ${DATA_DIR}/agentnet_ubuntu_3k_subtask_v7_eval.jsonl \
  --image-dir    ${DATA_DIR} \
  --output-dir   ${PROJ_DIR}/results/eval/v7sft-final-v7eval \
  --max-samples  0 \
  --max-new-tokens 512
