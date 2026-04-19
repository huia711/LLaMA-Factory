"""
Agent 架构：Subtask Planner + Executor（Option 2：稀疏干预）

Planner 输出格式（对齐 Offline GRPO）：
    Thought: <对当前状态的分析>
    Subtask: <子任务指令 | DONE | FAIL>
    Crop: [ymin, xmin, ymax, xmax]

Executor 输出格式（与 UITARSAgent 一致）：
    Thought: ...
    Action: ...
"""

import os
import re
import math
import json
import ast
from io import BytesIO
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor

from .uitars import (
    SCREEN_LOGIC_SIZE,
    FINISH_WORD,
    WAIT_WORD,
    ENV_FAIL_WORD,
    CALL_USER,
    parse_action,
    escape_single_quotes,
    fix_click_output,
    fix_drag_output,
    parse_action_qwen2vl,
    smart_resize,
    parsing_response_to_pyautogui_code,
    parsing_response_to_android_action_code,
    add_box_token,
)


# Max pixels for Planner input images (limits visual token count per image).
# With Qwen2.5-VL using 28×28 merged patches: 1280×720 ≈ 1175 tokens/image.
# 6 images (5 history + 1 current) × 1175 = ~7050 tokens, well within max_model_len=16384.
_PLANNER_MAX_PIXELS = 1280 * 720  # 921600


def _resize_to_max_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    """Resize PIL image so total pixel count does not exceed max_pixels."""
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    new_w = max(28, int(w * scale))
    new_h = max(28, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _flatten_mm_messages_for_text_only_chat_template(messages: List[Dict]) -> List[Dict]:
    """
    将 Qwen2VL 风格的多模态 message.content(list[dict]) 降级为纯文本 content(str)。
    适用于旧版 chat_template 不支持 list[dict] content 的情况。
    """
    flattened: List[Dict] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            parts: List[str] = []
            for c in content:
                if not isinstance(c, dict):
                    continue
                c_type = c.get("type")
                if c_type == "text":
                    parts.append(c.get("text", ""))
                elif c_type == "image":
                    parts.append("<|vision_start|><|image_pad|><|vision_end|>")
                elif c_type == "video":
                    parts.append("<|vision_start|><|video_pad|><|vision_end|>")
            flattened.append({**m, "content": "".join(parts)})
        else:
            flattened.append(m)
    return flattened


# ══════════════════════════════════════════════════════════════════════
# Planner Prompts — 对齐 Offline GRPO（agentnet_subtask_v4_rl.jsonl）
# ══════════════════════════════════════════════════════════════════════

PLANNER_SYSTEM_PROMPT = r"""You are a **VCoT Subtask Planner** for a GUI agent system.

The Executor is a capable GUI agent that autonomously handles multi-step UI operations. Your role is to **observe the current state, verify the Executor's progress, and assign the next high-level subtask**. You do not micromanage individual clicks or keystrokes — the Executor will figure out the atomic steps needed to complete whatever goal you give it.

## Inputs

1. **Instruction**: The user's global GUI task goal.
2. **Completed Subtasks**: Previous subtasks you assigned and their outcomes.
3. **Subtask Executor Actions** (optional): The Executor's recent steps within the last subtask, each paired with a screenshot showing the UI state at that step.
4. **Current Screenshot**: The latest screen state after the Executor's last action.

## Output Format (strict)

```
Thought: <one paragraph of natural-language reasoning>
Subtask: <high-level goal instruction for the Executor> / DONE / FAIL
Crop: [ymin, xmin, ymax, xmax]
```

### Field Details

**Thought**:
- What you observe on the current screen (UI elements, application state, visible values).
- Whether the Executor's last actions achieved the expected result — compare the expected UI change against what is actually visible.
- Based on the global instruction and current visual state, what the next step should be.

**Subtask**:
- A precise, context-aware goal instruction grounded in currently visible UI elements (names, positions, current state). The Executor will complete it autonomously through multiple atomic steps.
- Write goal-level instructions, not procedure-level: e.g., `Set the departure date to March 16` ✓ — NOT `Click the date field, then click 16` ✗.
- `DONE`: The global task has been fully and correctly completed — verifiable from the current screenshot.
- `FAIL`: The global task is infeasible or the Executor is in an unrecoverable state (e.g., wrong application, broken environment).

**Crop**:
- The UI region most relevant to the current subtask.
- Coordinates normalized to `[0, 1000]`, where `[0, 0]` is the top-left and `[1000, 1000]` is the bottom-right.

## Guidelines

- Do NOT repeat the global instruction verbatim — rewrite it with specific UI context (element names, positions, current state).
- If the Executor failed or is stuck (e.g., repeated identical screenshots), consider providing a more atomic or alternative subtask.
- **Verify before proceeding**: Confirm that the previous subtask's effects are actually visible in the current screenshot before assigning the next subtask.
- Ground all reasoning in visual evidence from the screenshots and the Executor's actions."""


# ══════════════════════════════════════════════════════════════════════
# Executor Prompts
# ══════════════════════════════════════════════════════════════════════

MOBILE_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
long_press(start_box='<|box_start|>(x1,y1)<|box_end|>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
open_app(app_name=\'\')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
press_home()
press_back()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.
"""

COMPUTER_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.
"""

EXECUTOR_SYSTEM_PROMPT = "You are a helpful assistant."

EXECUTOR_USER_PROMPT = """You are a GUI agent. You are given a task and your action history with screenshots. You need to perform the next action to complete the task.

## Task
{task_instruction}

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
{action_space}

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
"""


# ══════════════════════════════════════════════════════════════════════
# PlannerAgent 
# ══════════════════════════════════════════════════════════════════════

class PlannerAgent:
    """
    Subtask Agent

    仅在 subtask 边界被调用，每次输入：
      - 全局指令
      - 已完成 subtask 轻量历史（旧 subtask 纯文本）
      - 最近一个 subtask 的 Executor 执行历史（含截图，最多 history_n 步）
      - 当前截图

    输出格式：Thought + Subtask + Crop
    """

    def __init__(
        self,
        tokenizer_path: Optional[str],
        history_n: int = 4,
        screen_size: Tuple[int, int] = SCREEN_LOGIC_SIZE,
        language: str = 'Chinese',
    ):
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True, use_fast=False
            )
            self.processor = AutoProcessor.from_pretrained(
                tokenizer_path, trust_remote_code=True, use_fast=False
            )
        else:
            self.tokenizer = None
            self.processor = None
        self.history_n = history_n
        self.screen_size = screen_size
        self.language = language

        self.reset()

    def reset(self):
        """重置 Planner 状态（开始新任务）。"""
        self.completed_subtasks: List[Dict] = []
        self.current_subtask: Optional[str] = None
        self.current_crop: Optional[List[float]] = None

    def record_subtask_outcome(
        self,
        subtask: str,
        outcome: str,
        num_steps: int,
        executor_steps: Optional[List[Dict]] = None,
        post_screenshot: Optional[bytes] = None,
    ) -> None:
        """
        记录一个已完成 subtask 的结果（在外层循环每轮结束时调用）。

        Args:
            subtask: Planner 给出的子任务指令文本
            outcome: "Done" 或 "Failed"
            num_steps: Executor 实际执行的步数
            executor_steps: Executor 在该 subtask 中的执行记录列表，
                每条为 {action_nl: str, action_code: str, screenshot: bytes|None}
            post_screenshot: subtask 结束后的屏幕截图（= 下一个 subtask 的初始状态），
                显示在 Completed Subtasks 里供 Planner 确认执行结果
        """
        self.completed_subtasks.append({
            "subtask": subtask,
            "outcome": outcome,
            "num_steps": num_steps,
            "executor_steps": executor_steps or [],
            "post_screenshot": post_screenshot,
        })

    def _build_user_content_and_images(
        self, instruction: str, obs: Dict
    ) -> Tuple[List[Dict], List[Image.Image]]:
        """
        构建 Planner 输入（对齐 VCOT Subtask Planner 训练数据格式）。

        无历史时：## Instruction + ## Current Screenshot
        有历史时：## Instruction
                  ## Completed Subtasks（旧 subtasks 纯文字摘要）
                  ## Subtask Executor Actions（最近一个 subtask 的步骤，带截图，最多 history_n 步）
                  ## Current Screenshot
        """
        user_content: List[Dict] = []
        image_input_list: List[Image.Image] = []
        completed = self.completed_subtasks

        if not completed:
            # 第一次调用：只有 instruction + 当前截图
            user_content.append({
                "type": "text",
                "text": f"## Instruction\n{instruction}\n\n## Current Screenshot\n",
            })
        else:
            text_buffer = f"## Instruction\n{instruction}\n\n"

            # 所有已完成 subtasks：旧的纯文字，最近一个额外附带 post_screenshot
            text_buffer += "## Completed Subtasks\n"
            older = completed[:-1]
            latest_completed = completed[-1]
            for i, record in enumerate(older):
                text_buffer += (
                    f'{i + 1}. "{record["subtask"]}" '
                    f'→ {record["outcome"]} ({record["num_steps"]} steps)\n'
                )
            # 最近一个 subtask：文字 + post_screenshot（= 当前 subtask 的初始状态）
            idx = len(completed)
            text_buffer += (
                f'{idx}. "{latest_completed["subtask"]}" '
                f'→ {latest_completed["outcome"]} ({latest_completed["num_steps"]} steps)\n'
            )
            post_shot = latest_completed.get("post_screenshot")
            if post_shot is not None:
                user_content.append({"type": "text", "text": text_buffer})
                text_buffer = ""
                try:
                    img = Image.open(BytesIO(post_shot))
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img = _resize_to_max_pixels(img, _PLANNER_MAX_PIXELS)
                    image_input_list.append(img)
                    user_content.append({"type": "image", "image": ""})
                except Exception:
                    pass
            text_buffer += "\n"

            # 最近一个 subtask 的 executor 步骤（带截图，最多 history_n 步取最新的）
            latest_completed = completed[-1]
            steps = list(latest_completed.get("executor_steps") or [])
            if self.history_n and len(steps) > self.history_n:
                steps = steps[-self.history_n:]

            if steps:
                text_buffer += "## Subtask Executor Actions\n"
                for i, step in enumerate(steps):
                    has_screenshot = step.get("screenshot") is not None
                    step_num = i + 1
                    desc = step.get("action_nl", "")
                    code = step.get("action_code", "")
                    if desc and code:
                        step_desc = f"[{desc}] {code}"
                    elif code:
                        step_desc = code
                    elif desc:
                        step_desc = f"[{desc}]"
                    else:
                        step_desc = ""
                    if has_screenshot:
                        # 先 flush 文字 buffer，再插入图片，再写 step 描述
                        text_buffer += f"{step_num}. "
                        user_content.append({"type": "text", "text": text_buffer})
                        text_buffer = ""
                        try:
                            img = Image.open(BytesIO(step["screenshot"]))
                            if img.mode != "RGB":
                                img = img.convert("RGB")
                            img = _resize_to_max_pixels(img, _PLANNER_MAX_PIXELS)
                            image_input_list.append(img)
                            user_content.append({"type": "image", "image": ""})
                        except Exception:
                            pass
                        text_buffer = f" {step_desc}\n"
                    else:
                        text_buffer += f"{step_num}. {step_desc}\n"
            else:
                text_buffer += (
                    f'## Subtask Executor Actions\n'
                    f'(No recorded steps for "{latest_completed["subtask"]}")\n'
                )

            text_buffer += "\n## Current Screenshot\n"
            user_content.append({"type": "text", "text": text_buffer})

        # Current screenshot image
        try:
            cur_img = Image.open(BytesIO(obs["screenshot"]))
            if cur_img.mode != "RGB":
                cur_img = cur_img.convert("RGB")
            cur_img = _resize_to_max_pixels(cur_img, _PLANNER_MAX_PIXELS)
            image_input_list.append(cur_img)
            user_content.append({"type": "image", "image": ""})
        except Exception as e:
            raise RuntimeError(f"Error opening current screenshot: {e}")

        return user_content, image_input_list

    def get_api_messages(
        self, instruction: str, obs: Dict
    ) -> Tuple[List[Dict], List[Image.Image]]:
        """
        构建 API 调用用的 OpenAI 格式 messages（不需要 tokenizer）。
        图片占位符 {"type": "image", "image": ""} 由调用方替换为 base64 data URL。

        Returns:
            (messages, image_input_list)
        """
        user_content, image_input_list = self._build_user_content_and_images(instruction, obs)
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return messages, image_input_list

    def get_model_inputs(self, instruction: str, obs: Dict) -> Dict:
        """
        构建 Planner 的 vLLM 输入（需要 tokenizer）。
        """
        if self.tokenizer is None:
            raise RuntimeError("PlannerAgent tokenizer not loaded; use get_api_messages() for API mode")

        user_content, image_input_list = self._build_user_content_and_images(instruction, obs)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": PLANNER_SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]

        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(
                _flatten_mm_messages_for_text_only_chat_template(messages),
                tokenize=False,
                add_generation_prompt=True,
            )

        return {"prompt": prompt_text, "multi_modal_data": {"image": image_input_list}}

    def parse_response(self, response: str) -> Dict:
        """
        解析 Planner 的输出，提取 Thought / Subtask / Crop。

        Returns:
            {
                "thought": str | None,
                "subtask": str | None,
                "subtask_type": "new_subtask" | "DONE" | "FAIL" | None,
                "crop": [ymin, xmin, ymax, xmax] | None,   (0-1000 scale)
                "format_ok": bool,
            }
        """
        raw = (response or "").strip()

        thought_m = re.search(
            r"(?m)^Thought\s*:\s*(.+?)(?=\nSubtask\s*:|$)", raw, re.DOTALL
        )
        subtask_m = re.search(
            r"(?m)^Subtask\s*:\s*(.+?)(?=\nCrop\s*:|$)", raw, re.DOTALL
        )
        crop_m = re.search(r"(?m)^Crop\s*:\s*(\[[\d\s,.\-]+\])", raw)

        thought = thought_m.group(1).strip() if thought_m else None
        subtask = subtask_m.group(1).strip() if subtask_m else None

        crop = None
        if crop_m:
            try:
                vals = json.loads(crop_m.group(1))
                if len(vals) == 4:
                    crop = [float(v) for v in vals]
            except Exception:
                pass

        subtask_type = None
        if subtask is not None:
            s_upper = subtask.strip().upper()
            if s_upper == "DONE":
                subtask_type = "DONE"
            elif s_upper == "FAIL":
                subtask_type = "FAIL"
            elif s_upper == "CONTINUE":
                subtask_type = "CONTINUE"
            else:
                subtask_type = "new_subtask"

        format_ok = (thought is not None) and (subtask is not None) and (crop is not None)

        self.current_subtask = subtask
        self.current_crop = crop

        return {
            "thought": thought,
            "subtask": subtask,
            "subtask_type": subtask_type,
            "crop": crop,
            "format_ok": format_ok,
        }


# ══════════════════════════════════════════════════════════════════════
# ExecutorAgent
# ══════════════════════════════════════════════════════════════════════

class ExecutorAgent:
    """
    Executor Agent
    """

    @staticmethod
    def _detect_pixel_coord_mode(tokenizer_path: str) -> bool:
        model_name = tokenizer_path.lower()
        if "ui-tars-1.5" in model_name or "uitars-1.5" in model_name:
            return True
        if "zerogui" in model_name:
            return False
        try:
            config_path = os.path.join(tokenizer_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    model_type = config.get("model_type", "")
                    if "qwen2_5_vl" in model_type:
                        return True
                    elif "qwen2_vl" in model_type:
                        return False
        except Exception:
            pass
        return False

    def __init__(
        self,
        tokenizer_path: str,
        max_trajectory_length: int = 15,
        history_n: int = 3,
        screen_size: Tuple[int, int] = SCREEN_LOGIC_SIZE,
        action_space: str = 'computer',
        language: str = 'Chinese',
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, use_fast=False
        )
        self.processor = AutoProcessor.from_pretrained(
            tokenizer_path, trust_remote_code=True, use_fast=False
        )

        self.max_trajectory_length = max_trajectory_length
        self.history_n = history_n
        self.screen_size = screen_size
        self.action_space = action_space
        self.language = language

        self.use_pixel_coords = self._detect_pixel_coord_mode(tokenizer_path)
        print(f"[ExecutorAgent] coord mode: {'pixel (smart_resize)' if self.use_pixel_coords else '0-1000 normalized'}")

        if action_space == 'mobile':
            self.prompt_action_space = MOBILE_ACTION_SPACE
            self.action_code_mapper = parsing_response_to_android_action_code
        else:
            self.prompt_action_space = COMPUTER_ACTION_SPACE
            self.action_code_mapper = parsing_response_to_pyautogui_code

        self.action_parse_res_factor = 1000
        self.customize_action_parser = parse_action_qwen2vl

        self.reset()

    def reset(self):
        """重置 Agent 状态。"""
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
        self.last_model_image_size = None
        self.last_original_image_size = None
        self.pending_crop_image = None
        self.pending_crop_box = None

    # ── Planner crop 注入所需的坐标工具方法 ──

    @staticmethod
    def _parse_box_value(box_value):
        if box_value is None:
            return None
        if isinstance(box_value, (list, tuple)):
            values = list(box_value)
        else:
            try:
                values = ast.literal_eval(str(box_value))
            except Exception:
                return None
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            return None
        try:
            return [float(v) for v in values]
        except Exception:
            return None

    @staticmethod
    def _clamp_box(box):
        x1, y1, x2, y2 = box
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return [x1, y1, x2, y2]

    @staticmethod
    def _map_box_to_full(box, crop_box):
        cx1, cy1, cx2, cy2 = crop_box
        crop_w = max(cx2 - cx1, 0.0)
        crop_h = max(cy2 - cy1, 0.0)
        if crop_w == 0.0 or crop_h == 0.0:
            return box
        x1, y1, x2, y2 = box
        mapped = [
            cx1 + x1 * crop_w,
            cy1 + y1 * crop_h,
            cx1 + x2 * crop_w,
            cy1 + y2 * crop_h,
        ]
        return ExecutorAgent._clamp_box(mapped)

    # ── 核心方法 ──

    def get_model_inputs(self, task_instruction: str, obs: Dict) -> Dict:
        """
        构建 Executor 的模型输入。

        Args:
            task_instruction: 当前任务指令（turn 0 为全局 instruction，后续 turn 为 planner 给出的 subtask）
            obs: 包含 screenshot 的观察
        """
        current_image = obs["screenshot"]

        if self.pending_crop_image is not None:
            current_image = self.pending_crop_image
            self.pending_crop_image = None
        self.history_images.append(current_image)

        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n:]

        self.observations.append(
            {"screenshot": obs["screenshot"], "accessibility_tree": None}
        )

        user_prompt = EXECUTOR_USER_PROMPT.format(
            task_instruction=task_instruction,
            action_space=self.prompt_action_space,
            language=self.language,
        )

        images = []
        for image_bytes in self.history_images:
            try:
                image = Image.open(BytesIO(image_bytes))
            except Exception as e:
                raise RuntimeError(f"Error opening image: {e}")

            ori_w, ori_h = image.size
            self.last_original_image_size = (ori_w, ori_h)

            new_h, new_w = smart_resize(ori_h, ori_w)
            if self.use_pixel_coords:
                self.last_model_image_size = (new_w, new_h)
            else:
                self.last_model_image_size = None

            if image.mode != "RGB":
                image = image.convert("RGB")
            images.append(image)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": EXECUTOR_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ]

        image_num = 0
        image_input_list = []
        if len(self.history_responses) > 0:
            for history_idx, history_response in enumerate(self.history_responses):
                if history_idx + self.history_n > len(self.history_responses):
                    cur_image = images[image_num]
                    image_input_list.append(cur_image)
                    messages.append({
                        "role": "user",
                        "content": [{"type": "image", "image": ""}],
                    })
                    image_num += 1

                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": add_box_token(history_response)}],
                })

            cur_image = images[image_num]
            image_input_list.append(cur_image)
            messages.append({
                "role": "user",
                "content": [{"type": "image", "image": ""}],
            })
            image_num += 1
        else:
            cur_image = images[image_num]
            image_input_list.append(cur_image)
            messages.append({
                "role": "user",
                "content": [{"type": "image", "image": ""}],
            })
            image_num += 1

        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(
                _flatten_mm_messages_for_text_only_chat_template(messages),
                tokenize=False,
                add_generation_prompt=True,
            )

        return {"prompt": prompt_text, "multi_modal_data": {"image": image_input_list}}

    def parse_action(self, response: str) -> List[str]:
        """
        解析 Executor 的响应，提取动作代码。
        """
        self.history_responses.append(response)
        self.thoughts.append(response)

        try:
            parsed_responses = self.customize_action_parser(
                response,
                self.action_parse_res_factor,
                self.screen_size[1],
                self.screen_size[0],
                model_image_size=self.last_model_image_size,
            )
        except Exception as e:
            print(f"Parsing action error: {response}, with error:\n{e}")
            return ["DONE"]

        actions = []

        for parsed_response in parsed_responses:
            if "action_type" in parsed_response:
                action_type = parsed_response["action_type"]

                if self.action_space != 'mobile' and action_type == FINISH_WORD:
                    self.pending_crop_box = None
                    self.actions.append(actions)
                    return ["DONE"]
                elif action_type == WAIT_WORD:
                    self.pending_crop_box = None
                    self.actions.append(actions)
                    return ["WAIT"]
                elif action_type == ENV_FAIL_WORD:
                    self.pending_crop_box = None
                    self.actions.append(actions)
                    return ["FAIL"]
                elif action_type == CALL_USER:
                    self.pending_crop_box = None
                    self.actions.append(actions)
                    return ["FAIL"]

            if self.pending_crop_box is not None:
                action_inputs = parsed_response.get("action_inputs", {})
                for key, value in list(action_inputs.items()):
                    if "start_box" in key or "end_box" in key:
                        box = self._parse_box_value(value)
                        if not box:
                            continue
                        mapped = self._map_box_to_full(box, self.pending_crop_box)
                        action_inputs[key] = str(mapped)

            try:
                pyautogui_code = self.action_code_mapper(
                    parsed_response,
                    self.screen_size[1],
                    self.screen_size[0],
                    input_swap=False,
                )
                actions.append(pyautogui_code)
            except Exception as e:
                print(f"Parsing pyautogui code error: {parsed_response}, with error:\n{e}")

        if actions and self.pending_crop_box is not None:
            self.pending_crop_box = None
        self.actions.append(actions)

        if len(self.actions) >= self.max_trajectory_length:
            actions = ["FAIL"]

        return actions


# ══════════════════════════════════════════════════════════════════════
# PEAgent — Planner-Executor 协调器
# ══════════════════════════════════════════════════════════════════════

class PEAgent:
    """
    PE Agent（Planner-Executor 协调器）

    管理 PlannerAgent（Subtask Planner）和 ExecutorAgent 的交互。
    """

    def __init__(
        self,
        planner_tokenizer_path: str,
        executor_tokenizer_path: str,
        max_trajectory_length: int,
        history_n: int,
        screen_size: Tuple[int, int] = SCREEN_LOGIC_SIZE,
        action_space: str = 'computer',
        language: str = 'Chinese',
    ):
        self.planner = PlannerAgent(
            tokenizer_path=planner_tokenizer_path,
            history_n=history_n,
            screen_size=screen_size,
            language=language,
        )

        self.executor = ExecutorAgent(
            tokenizer_path=executor_tokenizer_path,
            max_trajectory_length=max_trajectory_length,
            history_n=history_n,
            screen_size=screen_size,
            action_space=action_space,
            language=language,
        )

        self.max_trajectory_length = max_trajectory_length
        self.history_n = history_n
        self.screen_size = screen_size
        self.action_space = action_space
        self.language = language

        self.current_subtask: Optional[str] = None
        self.current_crop: Optional[List[float]] = None
        self.step_count: int = 0

    def reset(self):
        """重置两个 Agent。"""
        self.planner.reset()
        self.executor.reset()
        self.current_subtask = None
        self.current_crop = None
        self.step_count = 0

    def get_planner_inputs(self, instruction: str, obs: Dict) -> Dict:
        """获取 Planner 的模型输入。"""
        return self.planner.get_model_inputs(instruction, obs)

    def get_executor_inputs(self, task_instruction: str, obs: Dict) -> Dict:
        """获取 Executor 的模型输入。"""
        return self.executor.get_model_inputs(task_instruction, obs)

    def parse_planner_response(self, response: str) -> Dict:
        """
        解析 Planner 的响应。

        Returns:
            {thought, subtask, subtask_type, crop, format_ok}
        """
        result = self.planner.parse_response(response)
        self.current_subtask = result.get("subtask")
        self.current_crop = result.get("crop")
        return result

    def parse_executor_response(self, response: str) -> List[str]:
        """解析 Executor 的响应，返回动作代码。"""
        self.step_count += 1
        return self.executor.parse_action(response)
