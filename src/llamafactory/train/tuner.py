# Copyright 2025 the KVCache.AI team, Approaching AI, and the LlamaFactory team.
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
import shutil
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from transformers import EarlyStoppingCallback, PreTrainedModel

from ..data import get_template_and_fix_tokenizer
from ..extras import logging
from ..extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_mcore_adapter_available, is_ray_available
from ..hparams import get_infer_args, get_ray_args, get_train_args, read_args
from ..model import load_model, load_tokenizer
from .callbacks import LogCallback, PissaConvertCallback, ReporterCallback
from .dpo import run_dpo
from .kto import run_kto
from .ppo import run_ppo
from .pt import run_pt
from .rm import run_rm
from .sft import run_sft
from .trainer_utils import get_ray_trainer, get_swanlab_callback

"""
训练/导出总入口（tuner）。

从命令行执行 `llamafactory-cli train ...` 后，调用链大致为：
    setup.py(console_scripts)
      -> llamafactory.cli:main
      -> llamafactory.launcher.launch (子命令分发/torchrun)
      -> llamafactory.train.tuner.run_exp
      -> _training_function (解析参数/分发 stage)
      -> run_sft / run_pt / run_ppo / ...

本文件核心做两件事：
1) `run_exp()`：训练入口，支持单机/torchrun 分布式，也支持 Ray 分布式（ray_args.use_ray）
2) `export_model()`：导出入口，用于合并 LoRA、保存 tokenizer/processor、生成 Ollama Modelfile 等

与 Prompt/Template 的关系：
- 训练时 template 主要在各 stage workflow 内使用（例如 SFT workflow 会调用 `get_template_and_fix_tokenizer`）。
- export 时也会读取 template 生成 Ollama Modelfile。

与 vLLM 的关系：
- 训练阶段默认 **不使用 vLLM**（`hparams/parser.py` 会强制训练 `infer_backend == HF`）。
- vLLM 主要用于推理端（chat/api/webchat）或离线 batch generation（scripts/vllm_infer.py）。
"""


if is_ray_available():
    import ray
    from ray.train.huggingface.transformers import RayTrainReportCallback


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = logging.get_logger(__name__)


def _training_function(config: dict[str, Any]) -> None:
    """
    单次训练进程的执行体（在 Ray 下也会作为 train loop 被远程调用）。

    参数:
      - config["args"]：来自 CLI/yaml/json 的参数（dict 或 list[str]），会被 `get_train_args` 解析成 dataclasses
      - config["callbacks"]：训练回调列表（logging/swanlab/early-stopping 等）
    """
    args = config.get("args")
    callbacks: list[Any] = config.get("callbacks")
    # 解析训练参数（Model/Data/Training/Finetuning/Generating）
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    # ---- 标准回调：日志 / PiSSA / swanlab / early stopping / reporter ----
    callbacks.append(LogCallback())
    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())

    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))

    if finetuning_args.early_stopping_steps is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=finetuning_args.early_stopping_steps))

    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))  # add to last

    # ------------------------------------------------------------
    # MCA（mcore_adapter）训练路径：
    # 仅对 ["pt","sft","dpo"] 等部分 stage 提供替代实现。
    # 由 CLI 传入 `--use_mca True` 或环境变量触发（见 launcher.py / hparams/parser.py）
    # ------------------------------------------------------------
    if finetuning_args.stage in ["pt", "sft", "dpo"] and finetuning_args.use_mca:
        if not is_mcore_adapter_available():
            raise ImportError("mcore_adapter is not installed. Please install it with `pip install mcore-adapter`.")
        if finetuning_args.stage == "pt":
            from .mca import run_pt as run_pt_mca

            run_pt_mca(model_args, data_args, training_args, finetuning_args, callbacks)
        elif finetuning_args.stage == "sft":
            from .mca import run_sft as run_sft_mca

            run_sft_mca(model_args, data_args, training_args, finetuning_args, callbacks)
        elif finetuning_args.stage == "dpo":
            from .mca import run_dpo as run_dpo_mca

            run_dpo_mca(model_args, data_args, training_args, finetuning_args, callbacks)

    elif finetuning_args.stage == "pt":
        # 预训练（PT）路径
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        # 监督微调（SFT）路径
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "rm":
        # Reward Model（RM）训练路径
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "ppo":
        # PPO（RLHF）训练路径
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo":
        # DPO 训练路径
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "kto":
        # KTO 训练路径
        run_kto(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError(f"Unknown task: {finetuning_args.stage}.")

    if is_ray_available() and ray.is_initialized():
        # Ray 模式下，Ray runtime 会接管进程生命周期；直接 return 避免破坏 Ray 管理的进程组。
        return  # if ray is intialized it will destroy the process group on return

    try:
        if dist.is_initialized():
            # 非 Ray 下，尽量显式销毁进程组，避免下次启动报错。
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Failed to destroy process group: {e}.")


def run_exp(args: Optional[dict[str, Any]] = None, callbacks: Optional[list["TrainerCallback"]] = None) -> None:
    """
    训练总入口。

    - 支持直接传 dict（编程式调用），也支持从 CLI/yaml/json 解析（read_args）
    - 支持两种分布式：
      - torchrun/DDP：由 launcher.py 判断并重启进程（本函数只负责单进程逻辑）
      - Ray：当 ray_args.use_ray=True 时，使用 Ray Trainer 分发 `_training_function`
    """
    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)

    ray_args = get_ray_args(args)
    callbacks = callbacks or []
    if ray_args.use_ray:
        # Ray 模式：使用 RayTrainReportCallback 把 metrics 回传给 Ray Driver
        callbacks.append(RayTrainReportCallback())
        trainer = get_ray_trainer(
            training_function=_training_function,
            train_loop_config={"args": args, "callbacks": callbacks},
            ray_args=ray_args,
        )
        trainer.fit()
    else:
        # 普通模式：直接在当前进程执行训练逻辑
        _training_function(config={"args": args, "callbacks": callbacks})


def export_model(args: Optional[dict[str, Any]] = None) -> None:
    """
    导出/合并模型入口。

    典型用途：
    - 合并 LoRA adapter 到 base model
    - 保存 tokenizer/processor
    - 为 Ollama 生成 `Modelfile`
    - RM 阶段可能额外复制 value head 权重
    """
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir` to save model.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    # 导出阶段也会读取 template（例如用于生成 Ollama Modelfile 或确保 special tokens 一致）
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    # 必须在 fix tokenizer 之后 load_model，以便必要时 resize vocab
    model = load_model(tokenizer, model_args, finetuning_args)  # must after fixing tokenizer to resize vocab

    if getattr(model, "quantization_method", None) is not None and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None) is not None:  # quantized model adopts float16 type
        setattr(model.config, "torch_dtype", torch.float16)
    else:
        if model_args.infer_dtype == "auto":
            output_dtype = getattr(model.config, "torch_dtype", torch.float32)
            if output_dtype == torch.float32:  # if infer_dtype is auto, try using half precision first
                output_dtype = infer_optim_dtype(torch.bfloat16)
        else:
            output_dtype = getattr(torch, model_args.infer_dtype)

        setattr(model.config, "torch_dtype", output_dtype)
        model = model.to(output_dtype)
        logger.info_rank0(f"Convert model dtype to: {output_dtype}.")

    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size=f"{model_args.export_size}GB",
        safe_serialization=(not model_args.export_legacy_format),
    )
    if model_args.export_hub_model_id is not None:
        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            max_shard_size=f"{model_args.export_size}GB",
            safe_serialization=(not model_args.export_legacy_format),
        )

    if finetuning_args.stage == "rm":
        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        if os.path.exists(os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_SAFE_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")
        elif os.path.exists(os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

        if processor is not None:
            processor.save_pretrained(model_args.export_dir)
            if model_args.export_hub_model_id is not None:
                processor.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

    except Exception as e:
        logger.warning_rank0(f"Cannot save tokenizer, please copy the files manually: {e}.")

    ollama_modelfile = os.path.join(model_args.export_dir, "Modelfile")
    with open(ollama_modelfile, "w", encoding="utf-8") as f:
        f.write(template.get_ollama_modelfile(tokenizer))
        logger.info_rank0(f"Ollama modelfile saved in {ollama_modelfile}")
