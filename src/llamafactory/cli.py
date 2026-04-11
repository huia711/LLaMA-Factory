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


"""
CLI 入口模块（console_scripts 指向 `llamafactory.cli:main`）。

你在命令行执行的：
    - `llamafactory-cli train ...`
    - `llamafactory-cli chat ...`
    - `llamafactory-cli api ...`
最终都会先进入本文件的 `main()`，再由 launcher 做子命令分发。

本模块的核心职责只有一个：
    根据环境变量选择 “v1 launcher” 或 “默认 launcher”，然后调用 `launcher.launch()`。

常用开关：
- `USE_V1=1`：走 `llamafactory.v1.launcher`（若你有一套 v1 兼容逻辑/参数体系）
- 否则：走 `llamafactory.launcher`（默认主线）
"""


def main():
    from .extras.misc import is_env_enabled

    # 根据环境变量选择 launcher 版本。
    # 这里不做任何业务逻辑（不解析参数、不创建模型），仅选择分发入口。
    if is_env_enabled("USE_V1"):
        from .v1 import launcher
    else:
        from . import launcher

    # 进入分发器：根据子命令（train/chat/api/export/...）调用对应模块。
    launcher.launch()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
