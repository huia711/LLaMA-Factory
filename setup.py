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
import re

from setuptools import find_packages, setup

"""
本文件用于 Python 包发布/安装（setuptools）。

你从命令行执行的：
    `llamafactory-cli ...`
就是在这里通过 `entry_points["console_scripts"]` 注册出来的。

本文件**不会参与训练/推理的业务逻辑**，但它决定：
- CLI 入口函数指向哪里（`llamafactory.cli:main`）
- 可选依赖(extra)有哪些（例如 vLLM / SGLang / DeepSpeed 等）

与本仓库常见工作流的关系：
- 训练：`llamafactory-cli train ...`
- 推理：`llamafactory-cli chat/api/webchat ...`（可选 vLLM/SGLang 后端）
- 导出：`llamafactory-cli export ...`
"""


def get_version() -> str:
    """从 `src/llamafactory/extras/env.py` 读取版本号字符串。"""
    with open(os.path.join("src", "llamafactory", "extras", "env.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("VERSION")
        (version,) = re.findall(pattern, file_content)
        return version


def get_requires() -> list[str]:
    """从仓库根目录 `requirements.txt` 读取基础依赖（install_requires）。"""
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


def get_console_scripts() -> list[str]:
    """
    定义 console_scripts 入口（pip 安装后会生成同名可执行文件）。

    - `llamafactory-cli`：主入口
    - `lmf`：可选短别名（由 `ENABLE_SHORT_CONSOLE` 环境变量控制）
    """
    console_scripts = ["llamafactory-cli = llamafactory.cli:main"]
    if os.getenv("ENABLE_SHORT_CONSOLE", "1").lower() in ["true", "y", "1"]:
        console_scripts.append("lmf = llamafactory.cli:main")

    return console_scripts


extra_require = {
    # 这里的 key 对应 pip 安装时的 extra 选择：
    #   pip install "llamafactory[vllm]"
    #   pip install "llamafactory[deepspeed]"
    # 等等。
    #
    # 注意：训练脚本（`llamafactory-cli train`）通常不需要 vLLM；
    # vLLM 主要用于推理后端（chat/api/webchat）或离线 batch generation。
    "torch": ["torch>=2.0.0", "torchvision>=0.15.0"],
    "torch-npu": ["torch==2.7.1", "torch-npu==2.7.1", "torchvision==0.22.1", "decorator"],
    "metrics": ["nltk", "jieba", "rouge-chinese"],
    "deepspeed": ["deepspeed>=0.10.0,<=0.16.9"],
    "liger-kernel": ["liger-kernel>=0.5.5"],
    "bitsandbytes": ["bitsandbytes>=0.39.0"],
    "hqq": ["hqq"],
    "eetq": ["eetq"],
    "gptq": ["optimum>=1.24.0", "gptqmodel>=2.0.0"],
    "aqlm": ["aqlm[gpu]>=1.1.0"],
    # vLLM 推理引擎（OpenAI-style API / Chat / WebUI 等推理场景使用）
    "vllm": ["vllm>=0.4.3,<=0.11.0"],
    # SGLang 推理引擎（同样主要用于推理侧）
    "sglang": ["sglang[srt]>=0.4.5", "transformers==4.51.1"],
    "galore": ["galore-torch"],
    "apollo": ["apollo-torch"],
    "badam": ["badam>=1.2.1"],
    "adam-mini": ["adam-mini"],
    "minicpm_v": [
        "soundfile",
        "torchvision",
        "torchaudio",
        "vector_quantize_pytorch",
        "vocos",
        "msgpack",
        "referencing",
        "jsonschema_specifications",
    ],
    "openmind": ["openmind"],
    "swanlab": ["swanlab"],
    "fp8": ["torchao>=0.8.0", "accelerate>=1.10.0"],
    "fp8-te": ["transformer_engine[pytorch]>=2.0.0", "accelerate>=1.10.0"],
    "fp8-all": ["torchao>=0.8.0", "transformer_engine[pytorch]>=2.0.0", "accelerate>=1.10.0"],
    "dev": ["pre-commit", "ruff", "pytest", "build"],
}


def main():
    """setuptools setup()：打包/安装入口。"""
    setup(
        name="llamafactory",
        version=get_version(),
        author="hiyouga",
        author_email="hiyouga@buaa.edu.cn",
        description="Unified Efficient Fine-Tuning of 100+ LLMs",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords=["AI", "LLM", "GPT", "ChatGPT", "Llama", "Transformer", "DeepSeek", "Pytorch"],
        license="Apache 2.0 License",
        url="https://github.com/hiyouga/LLaMA-Factory",
        package_dir={"": "src"},
        packages=find_packages("src"),
        python_requires=">=3.9.0",
        install_requires=get_requires(),
        extras_require=extra_require,
        entry_points={"console_scripts": get_console_scripts()},
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    main()
