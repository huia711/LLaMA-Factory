# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from copy import deepcopy

"""
命令分发与（可选的）分布式启动逻辑。

你可以把 `launcher.launch()` 理解为：
1) 从命令行取出子命令（train/chat/api/export/webchat/...）
2) 在必要时将当前进程“升级”为分布式启动器（torchrun）
3) 把控制权交给真正的业务模块

关键点（与调用链强相关）：
- 当子命令是 `train` 且满足条件时，这里会用 `torchrun` 重启自身进程，
  从而进入 PyTorch DDP / 多卡训练。
- 当不是分布式启动场景时，本模块只负责 import 对应功能模块并调用其入口函数。

与 vLLM 的关系：
- vLLM/SGLang 主要用于推理侧（chat/api/webchat），不是训练的默认路径。
"""


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   llamafactory-cli api -h: launch an OpenAI-style API server       |\n"
    + "|   llamafactory-cli chat -h: launch a chat interface in CLI         |\n"
    + "|   llamafactory-cli export -h: merge LoRA adapters and export model |\n"
    + "|   llamafactory-cli train -h: train models                          |\n"
    + "|   llamafactory-cli webchat -h: launch a chat interface in Web UI   |\n"
    + "|   llamafactory-cli webui: launch LlamaBoard                        |\n"
    + "|   llamafactory-cli env: show environment info                      |\n"
    + "|   llamafactory-cli version: show version info                      |\n"
    + "| Hint: You can use `lmf` as a shortcut for `llamafactory-cli`.      |\n"
    + "-" * 70
)


def launch():
    from .extras import logging
    from .extras.env import VERSION, print_env
    from .extras.misc import find_available_port, get_device_count, is_env_enabled, use_kt, use_ray

    logger = logging.get_logger(__name__)
    WELCOME = (
        "-" * 58
        + "\n"
        + f"| Welcome to LLaMA Factory, version {VERSION}"
        + " " * (21 - len(VERSION))
        + "|\n|"
        + " " * 56
        + "|\n"
        + "| Project page: https://github.com/hiyouga/LLaMA-Factory |\n"
        + "-" * 58
    )

    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"
    # 某些环境需要强制用 torchrun（例如 MCA 训练路径）。
    if is_env_enabled("USE_MCA"):  # force use torchrun
        os.environ["FORCE_TORCHRUN"] = "1"

    # -------------------------
    # 分布式训练启动（torchrun）
    # -------------------------
    # 条件：
    # - 子命令是 train
    # - 且满足：
    #   * FORCE_TORCHRUN=1
    #   * 或者检测到多 GPU（get_device_count() > 1）且不使用 ray、也不使用 ktransformers(kt)
    if command == "train" and (
        is_env_enabled("FORCE_TORCHRUN") or (get_device_count() > 1 and not use_ray() and not use_kt())
    ):
        # launch distributed training
        nnodes = os.getenv("NNODES", "1")
        node_rank = os.getenv("NODE_RANK", "0")
        nproc_per_node = os.getenv("NPROC_PER_NODE", str(get_device_count()))
        master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        master_port = os.getenv("MASTER_PORT", str(find_available_port()))
        logger.info_rank0(f"Initializing {nproc_per_node} distributed tasks at: {master_addr}:{master_port}")
        if int(nnodes) > 1:
            logger.info_rank0(f"Multi-node training enabled: num nodes: {nnodes}, node rank: {node_rank}")

        # elastic launch support
        max_restarts = os.getenv("MAX_RESTARTS", "0")
        rdzv_id = os.getenv("RDZV_ID")
        min_nnodes = os.getenv("MIN_NNODES")
        max_nnodes = os.getenv("MAX_NNODES")

        env = deepcopy(os.environ)
        if is_env_enabled("OPTIM_TORCH", "1"):
            # optimize DDP, see https://zhuanlan.zhihu.com/p/671834539
            # - `expandable_segments:True`：减少 CUDA 内存碎片
            # - `TORCH_NCCL_AVOID_RECORD_STREAMS`：降低某些 NCCL 场景的额外开销
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        # torchrun 支持 elastic（容错重启/动态节点数），由 rdzv_id 决定是否走 elastic 分支。
        if rdzv_id is not None:
            # launch elastic job with fault tolerant support when possible
            # see also https://docs.pytorch.org/docs/stable/elastic/train_script.html
            rdzv_nnodes = nnodes
            # elastic number of nodes if MIN_NNODES and MAX_NNODES are set
            if min_nnodes is not None and max_nnodes is not None:
                rdzv_nnodes = f"{min_nnodes}:{max_nnodes}"

            process = subprocess.run(
                (
                    "torchrun --nnodes {rdzv_nnodes} --nproc-per-node {nproc_per_node} "
                    "--rdzv-id {rdzv_id} --rdzv-backend c10d --rdzv-endpoint {master_addr}:{master_port} "
                    "--max-restarts {max_restarts} {file_name} {args}"
                )
                .format(
                    rdzv_nnodes=rdzv_nnodes,
                    nproc_per_node=nproc_per_node,
                    rdzv_id=rdzv_id,
                    master_addr=master_addr,
                    master_port=master_port,
                    max_restarts=max_restarts,
                    file_name=__file__,
                    args=" ".join(sys.argv[1:]),
                )
                .split(),
                env=env,
                check=True,
            )
        else:
            # NOTE: DO NOT USE shell=True to avoid security risk
            process = subprocess.run(
                (
                    "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                    "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
                )
                .format(
                    nnodes=nnodes,
                    node_rank=node_rank,
                    nproc_per_node=nproc_per_node,
                    master_addr=master_addr,
                    master_port=master_port,
                    file_name=__file__,
                    args=" ".join(sys.argv[1:]),
                )
                .split(),
                env=env,
                check=True,
            )

        sys.exit(process.returncode)

    # -------------------------
    # 非 torchrun 启动：按子命令分发
    # -------------------------
    elif command == "api":
        # OpenAI-style API 服务（推理侧；可选择 HF/vLLM/SGLang 等后端）
        from .api.app import run_api

        run_api()

    elif command == "chat":
        # CLI 交互式聊天（推理侧）
        from .chat.chat_model import run_chat

        run_chat()

    elif command == "eval":
        raise NotImplementedError("Evaluation will be deprecated in the future.")

    elif command == "export":
        # 导出模型：合并 LoRA / 导出权重 / 保存 tokenizer & processor 等
        from .train.tuner import export_model

        export_model()

    elif command == "train":
        # 训练入口：解析参数 -> 选择 stage -> 运行 sft/pt/ppo/...（见 train/tuner.py）
        from .train.tuner import run_exp

        run_exp()

    elif command == "webchat":
        # Web chat demo（推理侧）
        from .webui.interface import run_web_demo

        run_web_demo()

    elif command == "webui":
        # LlamaBoard / WebUI（通常包含推理/训练/配置等可视化入口）
        from .webui.interface import run_web_ui

        run_web_ui()

    elif command == "env":
        print_env()

    elif command == "version":
        print(WELCOME)

    elif command == "help":
        print(USAGE)

    else:
        print(f"Unknown command: {command}.\n{USAGE}")


if __name__ == "__main__":
    from llamafactory.train.tuner import run_exp  # use absolute import

    run_exp()
