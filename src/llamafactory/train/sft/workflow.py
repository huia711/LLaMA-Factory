# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from typing import TYPE_CHECKING, Optional

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.packages import is_transformers_version_greater_than
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer

"""
SFT（Supervised Fine-Tuning）训练工作流。

从整体调用链看，本文件位于：
    `llamafactory-cli train ...`
      -> `train/tuner.py::_training_function()`
      -> `run_sft(...)`  （本文件）

本文件核心流程（高度概括）：
1) 加载 tokenizer/processor（多模态模型会有 processor）
2) 根据 `--template` 选择 prompt/template，并修正 tokenizer
3) 加载/处理数据集：把原始样本 → prompt token ids / labels / multimodal payload
4) 加载模型（full/LoRA/freeze/量化等策略由 finetuning_args/model_args 控制）
5) 构造 data collator（关键：label pad、4D attention mask、多模态列对齐等）
6) 构造 Trainer 并执行 train/eval/predict

与 Prompt/Template 的关系（最关键）：
- `template = get_template_and_fix_tokenizer(tokenizer, data_args)`
  这一步决定了：
  - prompt 的拼接格式（例如 qwen2_vl / qwen3_nothink）
  - stop words / eos 替换等策略
  - 多模态 token（image/video/audio placeholder）的处理策略（通过 template.mm_plugin）

与 vLLM 的关系：
- 训练本身不走 vLLM；但在 `do_predict` 时会提示批量生成更推荐用 `scripts/vllm_infer.py`。
"""


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    """
    执行 SFT。

    参数说明：
    - model_args: 模型相关配置（路径、dtype、量化、device_map、多模态像素上限等）
    - data_args: 数据与模板配置（dataset、template、cutoff_len、packing 等）
    - training_args: HF Trainer 参数（batch/epoch/lr/ddp/deepspeed 等）
    - finetuning_args: 微调策略（full/lora/freeze、stage、额外回调等）
    - generating_args: 生成相关参数（主要供评估/预测阶段使用）
    """
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    # 1) 模板选择与 tokenizer 修正（决定 prompt 格式）
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    # 2) 数据集加载：把样本加工成训练所需字段（input_ids/labels/可能还有 images/videos/audios）
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    # 3) 加载模型：会根据 finetuning_args (LoRA/freeze/full) 走不同的加载/封装逻辑
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    # 4) Data collator：负责把 dataset sample 组装成 batch，并处理 padding/mask/label 对齐等细节
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Metric utils
    metric_module = {}
    if model_args.use_kt:
        # KTransformers 的 SFT 目前对 predict_with_generate / compute_accuracy 有限制
        if training_args.predict_with_generate:
            raise NotImplementedError("`predict_with_generate` is not supported in KTransformers SFT yet.")
        elif finetuning_args.compute_accuracy:
            raise NotImplementedError("`compute_accuracy` is not supported in KTransformers SFT yet.")

    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)

    # Compatible with Transformers v4 and Transformers v5
    if is_transformers_version_greater_than("4.58.0"):
        # transformers>=4.58 里 additional_special_tokens_ids 取法可能不同，这里做兼容处理
        extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
        if not isinstance(extra_ids, list):
            extra_special_tokens = getattr(tokenizer, "_extra_special_tokens", [])
            string_tokens = [str(t) for t in extra_special_tokens]
            extra_ids = tokenizer.convert_tokens_to_ids(string_tokens)
        all_eos_ids = [tokenizer.eos_token_id] + [i for i in extra_ids if i != -1]
        unique_eos_ids = list(dict.fromkeys(all_eos_ids))
        gen_kwargs["eos_token_id"] = unique_eos_ids
    else:
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Initialize our Trainer
    if model_args.use_kt:
        # 5) KTransformers 特殊 Trainer（与 HF Trainer 不同）
        from ktransformers.sft.lora import KTrainer  # type: ignore
        from ktransformers.util.globals import GLOBAL_CONFIG  # type: ignore

        GLOBAL_CONFIG._config["mod"] = "sft"

        trainer = KTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer_module,
            data_collator=data_collator,
            callbacks=callbacks,
            **dataset_module,
            **metric_module,
        )
        trainer.model_accepts_loss_kwargs = False
        model.config.use_cache = False

    else:
        # 6) 默认使用 CustomSeq2SeqTrainer（HF Trainer 扩展）
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            callbacks=callbacks,
            gen_kwargs=gen_kwargs,
            **dataset_module,
            **tokenizer_module,
            **metric_module,
        )

    # Training
    if training_args.do_train:
        # 7) 训练主循环（支持 resume_from_checkpoint）
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    if training_args.predict_with_generate:
        # 生成阶段建议左 padding（很多 decoder-only 模型对 left-pad 更友好）
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        # 训练内置的 predict（HF generate）可能很慢；仓库提供 `scripts/vllm_infer.py` 做 vLLM 批量生成
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
