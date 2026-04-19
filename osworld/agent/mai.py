"""
MAI-UI Agent 模块

目标：
1. 对接 MAI-UI 的 function-calling 输出（<tool_call> JSON）
2. 将动作映射为 OSWorld 可执行的 pyautogui 代码
3. 保持与现有 Agent 接口一致（get_model_inputs / parse_action / reset）
"""

from __future__ import annotations

import ast
import json
import math
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200
SCALE_FACTOR = 999


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """复用 Qwen 系列常见的 smart_resize 规则。"""
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)

    return h_bar, w_bar


MAI_TOOL_DESCS = [
    {
        "type": "function",
        "function": {
            "name": "computer",
            "description": "Execute one desktop GUI action for the current task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "click",
                            "double_click",
                            "right_click",
                            "hover",
                            "drag",
                            "scroll",
                            "type",
                            "press",
                            "hotkey",
                            "wait",
                            "terminate",
                            "done",
                            "fail",
                            "answer",
                        ],
                    },
                    "coordinate": {"type": "array", "items": {"type": "number"}},
                    "start_coordinate": {"type": "array", "items": {"type": "number"}},
                    "end_coordinate": {"type": "array", "items": {"type": "number"}},
                    "pixels": {"type": "number"},
                    "direction": {"type": "string"},
                    "text": {"type": "string"},
                    "content": {"type": "string"},
                    "key": {"type": "string"},
                    "keys": {"type": "array", "items": {"type": "string"}},
                    "status": {"type": "string"},
                    "time": {"type": "number"},
                },
                "required": ["action"],
            },
        },
    }
]


MAI_SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
For each function call, return the thinking process in <thinking> </thinking> tags, and a json object with function name and arguments within <tool_call></tool_call> XML tags:
```
<thinking>
...
</thinking>
<tool_call>
{"name": "computer", "arguments": <args-json-object>}
</tool_call>
```

## Action Space

{"action": "click", "coordinate": [x, y]}
{"action": "double_click", "coordinate": [x, y]}
{"action": "right_click", "coordinate": [x, y]}
{"action": "hover", "coordinate": [x, y]}
{"action": "type", "text": ""} # Use escape characters \\', \\", and \\n in text part to ensure we can parse the text in normal python string format.
{"action": "scroll", "direction": "up or down or left or right", "coordinate": [x, y]} # "coordinate" is optional. Use the "coordinate" if you want to scroll a specific UI element.
{"action": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]}
{"action": "press", "key": "key_name"}
{"action": "hotkey", "keys": ["ctrl", "c"]}
{"action": "wait"}
{"action": "terminate", "status": "success or fail"}

## Note
- Write a small plan and finally summarize your next action (with its target element) in one sentence in <thinking></thinking> part.
- Coordinates should be in [0, 999], based on the current screenshot.
- You must follow the Action Space strictly, and return the correct json object within <thinking> </thinking> and <tool_call></tool_call> XML tags.
""".strip()


def _safe_load_json_or_literal(text: str):
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def parse_tagged_text(text: str) -> Dict[str, Any]:
    """
    Parse MAI-style tagged output and extract thinking + tool_call.

    Official MAI output format:
      <thinking>...</thinking>
      <tool_call>{"name":"...","arguments":{...}}</tool_call>
    """
    if "</think>" in text and "</thinking>" not in text:
        text = text.replace("</think>", "</thinking>")
        text = "<thinking>" + text

    pattern = r"<thinking>(.*?)</thinking>.*?<tool_call>(.*?)</tool_call>"
    result: Dict[str, Any] = {"thinking": None, "tool_call": None}

    match = re.search(pattern, text, re.DOTALL)
    if match:
        result = {
            "thinking": match.group(1).strip().strip('"'),
            "tool_call": match.group(2).strip().strip('"'),
        }

    if result["tool_call"]:
        parsed = _safe_load_json_or_literal(result["tool_call"])
        if not isinstance(parsed, dict):
            raise ValueError("Invalid JSON in <tool_call>")
        result["tool_call"] = parsed

    return result


def _normalize_coord_999(coord: Any, field_name: str) -> List[float]:
    if isinstance(coord, str):
        parsed = _safe_load_json_or_literal(coord)
        if parsed is not None:
            coord = parsed

    point_x: Optional[float] = None
    point_y: Optional[float] = None

    if isinstance(coord, (list, tuple)):
        values = list(coord)
        if len(values) == 2:
            point_x, point_y = float(values[0]), float(values[1])
        elif len(values) == 4:
            x1, y1, x2, y2 = [float(v) for v in values]
            point_x = (x1 + x2) / 2
            point_y = (y1 + y2) / 2
        else:
            raise ValueError(
                f"Invalid {field_name} format: expected 2 or 4 values, got {len(values)}"
            )
    elif isinstance(coord, dict):
        if {"x", "y"}.issubset(set(coord.keys())):
            point_x = float(coord["x"])
            point_y = float(coord["y"])
        elif {"x1", "y1", "x2", "y2"}.issubset(set(coord.keys())):
            point_x = (float(coord["x1"]) + float(coord["x2"])) / 2
            point_y = (float(coord["y1"]) + float(coord["y2"])) / 2
        else:
            raise ValueError(f"Invalid {field_name} dict format: missing x/y or x1/y1/x2/y2")
    else:
        raise ValueError(f"Invalid {field_name} type: {type(coord)}")

    if point_x is None or point_y is None:
        raise ValueError(f"Failed to parse {field_name}")

    if abs(point_x) <= 1.0 and abs(point_y) <= 1.0:
        return [point_x, point_y]
    return [point_x / SCALE_FACTOR, point_y / SCALE_FACTOR]


def parse_action_to_structure_output(text: str) -> Dict[str, Any]:
    """
    Parse model output to structured action.

    This follows the official MAI parser logic:
    1) parse <thinking>/<tool_call>
    2) read tool_call["arguments"]
    3) normalize coordinate-like fields by SCALE_FACTOR=999.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response")

    results = parse_tagged_text(text)
    tool_call = results["tool_call"]
    if not isinstance(tool_call, dict):
        raise ValueError("No valid <tool_call> found")

    action = tool_call.get("arguments")
    if not isinstance(action, dict):
        raise ValueError("Missing or invalid tool_call.arguments")

    action = dict(action)
    for key in ("coordinate", "start_coordinate", "end_coordinate"):
        if key in action and action[key] is not None:
            action[key] = _normalize_coord_999(action[key], key)

    return {
        "thinking": results["thinking"],
        "action_json": action,
    }


def extract_tool_calls(text: str) -> List[dict]:
    """
    从 MAI-UI 输出中提取 <tool_call> 块。
    兼容：
    - 标准 XML 包裹
    - 仅输出单个 {"name": ..., "arguments": ...} JSON 的场景
    """
    tool_calls: List[dict] = []
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    for match in re.findall(pattern, text, flags=re.DOTALL):
        obj = _safe_load_json_or_literal(match)
        if isinstance(obj, dict):
            tool_calls.append(obj)

    if tool_calls:
        return tool_calls

    for block in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL):
        obj = _safe_load_json_or_literal(block)
        if isinstance(obj, dict) and ("arguments" in obj or "name" in obj):
            return [obj]

    json_like = re.search(r"(\{[\s\S]*\"name\"[\s\S]*\"arguments\"[\s\S]*\})", text)
    if json_like:
        obj = _safe_load_json_or_literal(json_like.group(1))
        if isinstance(obj, dict) and ("arguments" in obj or "name" in obj):
            return [obj]

    obj = _safe_load_json_or_literal(text)
    if isinstance(obj, dict) and ("arguments" in obj or "name" in obj):
        return [obj]

    return []


def _ensure_dict(value) -> Dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        obj = _safe_load_json_or_literal(value)
        if isinstance(obj, dict):
            return obj
    return {}


def _to_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "0", "no", "n", "off"}:
            return False
    return default


def _parse_coord(candidate) -> Optional[Tuple[float, float]]:
    if candidate is None:
        return None

    if isinstance(candidate, dict):
        if "x" in candidate and "y" in candidate:
            try:
                return float(candidate["x"]), float(candidate["y"])
            except Exception:
                return None
        if {"x1", "y1", "x2", "y2"}.issubset(set(candidate.keys())):
            try:
                x = (float(candidate["x1"]) + float(candidate["x2"])) / 2.0
                y = (float(candidate["y1"]) + float(candidate["y2"])) / 2.0
                return x, y
            except Exception:
                return None

    values = None
    if isinstance(candidate, (list, tuple)):
        values = list(candidate)
    elif isinstance(candidate, str):
        parsed = _safe_load_json_or_literal(candidate)
        if isinstance(parsed, (list, tuple)):
            values = list(parsed)
        elif isinstance(parsed, dict):
            return _parse_coord(parsed)
        else:
            nums = re.findall(r"-?\d+(?:\.\d+)?", candidate)
            if len(nums) >= 2:
                values = [float(nums[0]), float(nums[1])]

    if not values:
        return None
    try:
        numbers = [float(v) for v in values]
    except Exception:
        return None

    if len(numbers) >= 4:
        x = (numbers[0] + numbers[2]) / 2.0
        y = (numbers[1] + numbers[3]) / 2.0
        return x, y
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    return None


def _map_to_screen(
    coord: Tuple[float, float],
    screen_width: int,
    screen_height: int,
    model_image_size: Optional[Tuple[int, int]] = None,
    original_image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    x, y = coord
    sx, sy = x, y

    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        sx = x * screen_width
        sy = y * screen_height
    elif 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0 and (x > 1.0 or y > 1.0):
        sx = x / 1000.0 * screen_width
        sy = y / 1000.0 * screen_height
    elif model_image_size and original_image_size:
        mw, mh = model_image_size
        ow, oh = original_image_size
        if mw > 0 and mh > 0 and 0.0 <= x <= mw * 1.1 and 0.0 <= y <= mh * 1.1:
            sx = x * (ow / mw)
            sy = y * (oh / mh)

    sx = int(round(max(0.0, min(float(screen_width - 1), sx))))
    sy = int(round(max(0.0, min(float(screen_height - 1), sy))))
    return sx, sy


def _extract_first_coord(args: Dict, keys: Sequence[str]) -> Optional[Tuple[float, float]]:
    for key in keys:
        if key in args:
            coord = _parse_coord(args.get(key))
            if coord is not None:
                return coord
    return None


def _normalize_hotkey(value) -> List[str]:
    if isinstance(value, (list, tuple)):
        keys = [str(v).strip() for v in value if str(v).strip()]
        return keys
    if isinstance(value, str):
        text = value.strip()
        if "+" in text:
            return [k.strip() for k in text.split("+") if k.strip()]
        if text:
            return [text]
    return []


def _escape_single_quotes(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'")


def mai_to_pyautogui(
    tool_calls: List[dict],
    image_height: int,
    image_width: int,
    model_image_size: Optional[Tuple[int, int]] = None,
    original_image_size: Optional[Tuple[int, int]] = None,
) -> str:
    """
    将 MAI tool call 转换为 pyautogui 代码或控制信号（DONE/FAIL/WAIT）。
    """
    pyautogui_code = "import pyautogui\nimport time\n"
    has_action_code = False

    for idx, call in enumerate(tool_calls):
        name = str(call.get("name", "")).strip().lower()
        args = _ensure_dict(call.get("arguments", {}))
        action = str(args.get("action", "")).strip().lower() or name

        if idx > 0 and has_action_code:
            pyautogui_code += "\ntime.sleep(0.3)\n"

        if action in {"done", "finish", "finished", "completed", "success", "answer"}:
            return "DONE"
        if action in {"fail", "failure", "env_fail", "error"}:
            return "FAIL"
        if action == "terminate":
            status = str(args.get("status", "success")).strip().lower()
            return "DONE" if status in {"success", "done", "completed"} else "FAIL"
        if action == "wait" and len(tool_calls) == 1:
            return "WAIT"
        if action in {"ask_user", "mcp_call"}:
            return "FAIL"

        if action in {"click", "left_click", "left_single", "tap"}:
            coord = _extract_first_coord(args, ["coordinate", "coords", "position", "point", "start_box"])
            if coord is not None:
                x, y = _map_to_screen(coord, image_width, image_height, model_image_size, original_image_size)
                pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                has_action_code = True

        elif action in {"double_click", "doubleclick", "left_double"}:
            coord = _extract_first_coord(args, ["coordinate", "coords", "position", "point", "start_box"])
            if coord is not None:
                x, y = _map_to_screen(coord, image_width, image_height, model_image_size, original_image_size)
                pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                has_action_code = True

        elif action in {"right_click", "right_single"}:
            coord = _extract_first_coord(args, ["coordinate", "coords", "position", "point", "start_box"])
            if coord is not None:
                x, y = _map_to_screen(coord, image_width, image_height, model_image_size, original_image_size)
                pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                has_action_code = True

        elif action in {"hover", "mouse_move", "move"}:
            coord = _extract_first_coord(args, ["coordinate", "coords", "position", "point"])
            if coord is not None:
                x, y = _map_to_screen(coord, image_width, image_height, model_image_size, original_image_size)
                pyautogui_code += f"\npyautogui.moveTo({x}, {y})"
                has_action_code = True

        elif action in {"drag", "select", "swipe"}:
            start = _extract_first_coord(
                args,
                ["start_coordinate", "start_coords", "start", "start_point", "start_box", "coordinate"],
            )
            end = _extract_first_coord(
                args,
                ["end_coordinate", "end_coords", "end", "end_point", "end_box"],
            )
            if start is not None and end is not None:
                sx, sy = _map_to_screen(start, image_width, image_height, model_image_size, original_image_size)
                ex, ey = _map_to_screen(end, image_width, image_height, model_image_size, original_image_size)
                pyautogui_code += (
                    f"\npyautogui.moveTo({sx}, {sy})"
                    f"\npyautogui.dragTo({ex}, {ey}, duration=0.8)"
                )
                has_action_code = True

        elif action == "scroll":
            amount = args.get("pixels", args.get("amount", 0))
            direction = str(args.get("direction", "")).strip().lower()
            try:
                amount = int(round(float(amount)))
            except Exception:
                amount = 0
            if amount == 0:
                if direction in {"up", "scroll_up"}:
                    amount = 600
                elif direction in {"down", "scroll_down"}:
                    amount = -600
            if amount != 0:
                coord = _extract_first_coord(args, ["coordinate", "coords", "position", "point"])
                if coord is not None:
                    x, y = _map_to_screen(coord, image_width, image_height, model_image_size, original_image_size)
                    pyautogui_code += f"\npyautogui.scroll({amount}, x={x}, y={y})"
                else:
                    pyautogui_code += f"\npyautogui.scroll({amount})"
                has_action_code = True

        elif action in {"type", "input_text", "write"}:
            text = str(args.get("text", args.get("content", args.get("value", ""))))
            if text:
                coord = _extract_first_coord(args, ["coordinate", "coords", "position", "point"])
                if coord is not None:
                    x, y = _map_to_screen(coord, image_width, image_height, model_image_size, original_image_size)
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                    pyautogui_code += "\ntime.sleep(0.2)"
                escaped = _escape_single_quotes(text)
                pyautogui_code += "\nimport pyperclip"
                pyautogui_code += f"\npyperclip.copy('{escaped}')"
                pyautogui_code += "\npyautogui.hotkey('ctrl', 'v')"
                if _to_bool(args.get("press_enter"), default=False):
                    pyautogui_code += "\npyautogui.press('enter')"
                has_action_code = True

        elif action in {"hotkey", "key", "shortcut"}:
            keys = _normalize_hotkey(args.get("keys", args.get("key", args.get("hotkey", ""))))
            if keys:
                if len(keys) == 1:
                    pyautogui_code += f"\npyautogui.press({repr(keys[0])})"
                else:
                    pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in keys])})"
                has_action_code = True

        elif action in {"press", "keypress"}:
            key = str(args.get("key", "")).strip()
            if key:
                pyautogui_code += f"\npyautogui.press({repr(key)})"
                has_action_code = True

        elif action == "visit_url":
            url = str(args.get("url", "")).strip()
            if url:
                escaped = _escape_single_quotes(url)
                pyautogui_code += "\npyautogui.hotkey('ctrl', 'l')"
                pyautogui_code += "\ntime.sleep(0.2)"
                pyautogui_code += "\nimport pyperclip"
                pyautogui_code += f"\npyperclip.copy('{escaped}')"
                pyautogui_code += "\npyautogui.hotkey('ctrl', 'v')"
                pyautogui_code += "\npyautogui.press('enter')"
                has_action_code = True

        elif action == "web_search":
            query = str(args.get("query", "")).strip()
            if query:
                escaped = _escape_single_quotes(query)
                pyautogui_code += "\npyautogui.hotkey('ctrl', 'l')"
                pyautogui_code += "\ntime.sleep(0.2)"
                pyautogui_code += "\nimport pyperclip"
                pyautogui_code += f"\npyperclip.copy('{escaped}')"
                pyautogui_code += "\npyautogui.hotkey('ctrl', 'v')"
                pyautogui_code += "\npyautogui.press('enter')"
                has_action_code = True

        elif action == "history_back":
            pyautogui_code += "\npyautogui.hotkey('alt', 'left')"
            has_action_code = True

        elif action == "wait":
            seconds = args.get("time", args.get("seconds", 1.0))
            try:
                seconds = max(0.0, float(seconds))
            except Exception:
                seconds = 1.0
            pyautogui_code += f"\ntime.sleep({seconds:.2f})"
            has_action_code = True

    if not has_action_code:
        return "FAIL"
    return pyautogui_code


class MAIAgent:
    """
    MAI-UI Agent：面向 OSWorld 的 function-calling 适配层。
    """

    def __init__(
        self,
        tokenizer_path,
        max_trajectory_length: int = 15,
        history_n: int = 5,
        screen_size: Tuple[int, int] = (1920, 1080),
        action_space: str = "computer",
        language: str = "English",
        use_fast: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, use_fast=use_fast
        )
        self.processor = AutoProcessor.from_pretrained(
            tokenizer_path, trust_remote_code=True, use_fast=use_fast
        )

        self.max_trajectory_length = max_trajectory_length
        if isinstance(history_n, int) and history_n > 0:
            self.history_n = history_n
        else:
            self.history_n = max(1, int(max_trajectory_length))
        self.screen_size = screen_size
        self.action_space = action_space
        self.language = language

        self.last_model_image_size: Optional[Tuple[int, int]] = None
        self.last_original_image_size: Optional[Tuple[int, int]] = None

        print(f"[MAIAgent] Initialized with screen_size={screen_size}, action_space={action_space}")
        self.reset()

    def get_model_inputs(self, instruction: str, obs: Dict):
        self.history_images.append(obs["screenshot"])
        self.observations.append({"screenshot": obs["screenshot"]})

        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n:]

        if isinstance(self.history_images, bytes):
            self.history_images = [self.history_images]
        elif isinstance(self.history_images, np.ndarray):
            self.history_images = list(self.history_images)
        elif not isinstance(self.history_images, list):
            raise TypeError(f"Unidentified images type: {type(self.history_images)}")

        images = []
        for image_data in self.history_images:
            image = Image.open(BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")

            ori_w, ori_h = image.size
            self.last_original_image_size = (ori_w, ori_h)
            new_h, new_w = smart_resize(ori_h, ori_w)
            self.last_model_image_size = (new_w, new_h)
            images.append(image)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": MAI_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": f"Task: {instruction}"}]},
        ]

        image_input_list = []
        if len(self.history_responses) > 0:
            for history_idx, history_response in enumerate(self.history_responses):
                if history_idx + self.history_n > len(self.history_responses):
                    cur_image = images[len(image_input_list)]
                    image_input_list.append(cur_image)
                    messages.append({"role": "user", "content": [{"type": "image", "image": ""}]})

                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": str(history_response)}],
                    }
                )

            cur_image = images[len(image_input_list)]
            image_input_list.append(cur_image)
            messages.append({"role": "user", "content": [{"type": "image", "image": ""}]})
        else:
            cur_image = images[0]
            image_input_list.append(cur_image)
            messages.append({"role": "user", "content": [{"type": "image", "image": ""}]})

        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tools=MAI_TOOL_DESCS,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            simple_messages = []
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            text_parts.append("<|vision_start|><|image_pad|><|vision_end|>")
                        elif isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    simple_messages.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": "".join(text_parts),
                        }
                    )
                else:
                    simple_messages.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": str(content),
                        }
                    )
            prompt_text = self.tokenizer.apply_chat_template(
                simple_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return {"prompt": prompt_text, "multi_modal_data": {"image": image_input_list}}

    def parse_action(self, response: str):
        self.history_responses.append(response)
        try:
            parsed = parse_action_to_structure_output(response)
            action_json = parsed["action_json"]
        except Exception as e:
            print(f"[MAIAgent] Official parser failed: {e}")
            self.actions.append([])
            return ["FAIL"]

        tool_calls: List[dict] = [{"name": "computer", "arguments": action_json}]

        try:
            pyautogui_code = mai_to_pyautogui(
                tool_calls,
                self.screen_size[1],
                self.screen_size[0],
                model_image_size=self.last_model_image_size,
                original_image_size=self.last_original_image_size,
            )
        except Exception as e:
            print(f"[MAIAgent] Error parsing action: {e}")
            self.actions.append([])
            return ["FAIL"]

        if pyautogui_code in {"DONE", "FAIL", "WAIT"}:
            self.actions.append([])
            return [pyautogui_code]

        self.actions.append([pyautogui_code])
        if len(self.history_responses) >= self.max_trajectory_length:
            return ["FAIL"]
        return [pyautogui_code]

    def reset(self):
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
        self.last_model_image_size = None
        self.last_original_image_size = None
