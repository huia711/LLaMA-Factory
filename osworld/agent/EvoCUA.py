"""
EvoCUA Agent adapter for OpenRLHF evaluation loop.

This implementation follows the official EvoCUA S2 (tool-calling) workflow:
- prompt: S2_SYSTEM_PROMPT + S2_DESCRIPTION_PROMPT_TEMPLATE + build_s2_tools_def
- parsing: "Action:" + <tool_call>{...}</tool_call> -> pyautogui / DONE / FAIL / WAIT
"""

from __future__ import annotations

import json
import math
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


S2_ACTION_DESCRIPTION = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `key_down`: Press and HOLD the specified key(s) down in order (no release). Use this for stateful holds like holding Shift while clicking.
* `key_up`: Release the specified key(s) in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
""".strip()


S2_DESCRIPTION_PROMPT_TEMPLATE = """Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
{resolution_info}
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked."""


S2_SYSTEM_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_xml}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {{"name": <function-name>, "arguments": <args-json-object>}}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call."""


def build_s2_tools_def(description_prompt: str) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name_for_human": "computer_use",
            "name": "computer_use",
            "description": description_prompt,
            "parameters": {
                "properties": {
                    "action": {
                        "description": S2_ACTION_DESCRIPTION,
                        "enum": [
                            "key",
                            "type",
                            "mouse_move",
                            "left_click",
                            "left_click_drag",
                            "right_click",
                            "middle_click",
                            "double_click",
                            "triple_click",
                            "scroll",
                            "wait",
                            "terminate",
                            "key_down",
                            "key_up",
                        ],
                        "type": "string",
                    },
                    "keys": {"description": "Required only by `action=key`.", "type": "array"},
                    "text": {"description": "Required only by `action=type`.", "type": "string"},
                    "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                    "pixels": {"description": "The amount of scrolling.", "type": "number"},
                    "time": {"description": "The seconds to wait.", "type": "number"},
                    "status": {
                        "description": "The status of the task.",
                        "type": "string",
                        "enum": ["success", "failure"],
                    },
                },
                "required": ["action"],
                "type": "object",
            },
            "args_format": "Format the arguments as a JSON object.",
        },
    }


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 56 * 56,
    max_pixels: int = 16 * 16 * 4 * 12800,
    max_long_side: int = 8192,
) -> Tuple[int, int]:
    if height < 2 or width < 2:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {height} / {width}")

    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return h_bar, w_bar


class EvoCUAAgent:
    """
    OpenRLHF adapter of official EvoCUA S2 desktop agent.
    """

    def __init__(
        self,
        tokenizer_path,
        max_trajectory_length: int = 50,
        history_n: int = 4,
        screen_size: Tuple[int, int] = (1920, 1080),
        action_space: str = "computer",
        language: str = "English",
        coordinate_type: str = "relative",
        resize_factor: int = 32,
        use_fast: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, use_fast=use_fast
        )
        self.processor = AutoProcessor.from_pretrained(
            tokenizer_path, trust_remote_code=True, use_fast=use_fast
        )

        self.max_trajectory_length = max_trajectory_length
        self.history_n = max(int(history_n), 0)
        self.screen_size = screen_size
        self.action_space = action_space
        self.language = language
        self.coordinate_type = coordinate_type
        self.resize_factor = resize_factor
        # Qwen3-VL/EvoCUA 当前处理器单轮最多支持 3 张图（含当前帧）。
        self.max_images_per_prompt = 3
        self._warned_image_cap = False

        self.last_model_image_size: Optional[Tuple[int, int]] = None
        self.last_original_image_size: Optional[Tuple[int, int]] = None

        print(
            f"[EvoCUAAgent] Initialized with screen_size={screen_size}, "
            f"coordinate_type={coordinate_type}, history_n={self.history_n}"
        )
        self.reset()

    def reset(self):
        self.actions: List[str] = []
        self.observations: List[Dict[str, Any]] = []
        self.history_images: List[bytes] = []
        self.history_responses: List[str] = []
        self.responses: List[str] = []
        self.last_model_image_size = None
        self.last_original_image_size = None
        self._warned_image_cap = False

    @staticmethod
    def _image_from_bytes(image_data: bytes) -> Image.Image:
        image = Image.open(BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _build_instruction_prompt(self, instruction: str, previous_actions_str: str) -> str:
        return f"""
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {instruction}

Previous actions:
{previous_actions_str}""".strip()

    def _build_system_prompt(self, model_width: int, model_height: int) -> str:
        if self.coordinate_type == "absolute":
            resolution_info = f"* The screen's resolution is {model_width}x{model_height}."
        else:
            resolution_info = "* The screen's resolution is 1000x1000."
        description_prompt = S2_DESCRIPTION_PROMPT_TEMPLATE.format(
            resolution_info=resolution_info
        )
        tools_def = build_s2_tools_def(description_prompt)
        return S2_SYSTEM_PROMPT.format(
            tools_xml=json.dumps(tools_def, ensure_ascii=False)
        )

    def get_model_inputs(self, instruction: str, obs: Dict):
        screenshot_bytes = obs["screenshot"]
        self.history_images.append(screenshot_bytes)
        self.observations.append({"screenshot": screenshot_bytes})

        current_image = self._image_from_bytes(screenshot_bytes)
        original_width, original_height = current_image.size
        self.last_original_image_size = (original_width, original_height)

        resized_height, resized_width = smart_resize(
            height=original_height,
            width=original_width,
            factor=self.resize_factor,
            max_pixels=16 * 16 * 4 * 12800,
        )
        self.last_model_image_size = (resized_width, resized_height)

        system_prompt = self._build_system_prompt(resized_width, resized_height)
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        ]
        image_input_list: List[Image.Image] = []

        current_step = len(self.actions)
        history_start_idx = max(0, current_step - self.history_n)
        previous_actions: List[str] = []
        for i in range(history_start_idx):
            if i < len(self.actions):
                previous_actions.append(f"Step {i + 1}: {self.actions[i]}")
        previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

        requested_history_len = min(self.history_n, len(self.history_responses))
        max_history_images = max(self.max_images_per_prompt - 1, 0)
        history_len = min(requested_history_len, max_history_images)
        if requested_history_len > history_len and not self._warned_image_cap:
            print(
                f"[EvoCUAAgent] history truncated from {requested_history_len} to {history_len} "
                f"to satisfy max_images_per_prompt={self.max_images_per_prompt}."
            )
            self._warned_image_cap = True
        if history_len > 0:
            hist_responses = self.history_responses[-history_len:]
            hist_imgs = self.history_images[-history_len - 1 : -1]

            for i in range(history_len):
                if i < len(hist_imgs):
                    hist_image = self._image_from_bytes(hist_imgs[i])
                    image_input_list.append(hist_image)
                    if i == 0:
                        instruction_prompt = self._build_instruction_prompt(
                            instruction, previous_actions_str
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": ""},
                                    {"type": "text", "text": instruction_prompt},
                                ],
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": [{"type": "image", "image": ""}],
                            }
                        )

                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": str(hist_responses[i])}],
                    }
                )

        if history_len == 0:
            instruction_prompt = self._build_instruction_prompt(
                instruction, previous_actions_str
            )
            image_input_list.append(current_image)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ""},
                        {"type": "text", "text": instruction_prompt},
                    ],
                }
            )
        else:
            image_input_list.append(current_image)
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image", "image": ""}],
                }
            )

        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
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
                        {"role": msg.get("role", "user"), "content": str(content)}
                    )
            prompt_text = self.tokenizer.apply_chat_template(
                simple_messages, tokenize=False, add_generation_prompt=True
            )

        return {"prompt": prompt_text, "multi_modal_data": {"image": image_input_list}}

    def _adjust_coordinates(
        self,
        x: float,
        y: float,
        processed_width: Optional[int],
        processed_height: Optional[int],
        original_width: int,
        original_height: int,
    ) -> Tuple[int, int]:
        if self.coordinate_type == "absolute":
            if processed_width and processed_height:
                x_scale = original_width / processed_width
                y_scale = original_height / processed_height
                return int(x * x_scale), int(y * y_scale)
            return int(x), int(y)

        x_scale = original_width / 999.0
        y_scale = original_height / 999.0
        return int(x * x_scale), int(y * y_scale)

    def _process_tool_call(
        self,
        json_str: str,
        pyautogui_code: List[str],
        processed_width: Optional[int],
        processed_height: Optional[int],
        original_width: int,
        original_height: int,
    ) -> None:
        try:
            tool_call = json.loads(json_str)
            if tool_call.get("name") != "computer_use":
                return

            args = tool_call["arguments"]
            action = args["action"]
        except Exception:
            return

        def _clean_keys(raw_keys):
            keys = raw_keys if isinstance(raw_keys, list) else [raw_keys]
            cleaned_keys = []
            for key in keys:
                if isinstance(key, str):
                    if key.startswith("keys=["):
                        key = key[6:]
                    if key.endswith("]"):
                        key = key[:-1]
                    if key.startswith("['") or key.startswith('["'):
                        key = key[2:] if len(key) > 2 else key
                    if key.endswith("']") or key.endswith('"]'):
                        key = key[:-2] if len(key) > 2 else key
                    cleaned_keys.append(key.strip())
                else:
                    cleaned_keys.append(key)
            return cleaned_keys

        if action in {"left_click", "click"}:
            if "coordinate" in args:
                x, y = args["coordinate"]
                ax, ay = self._adjust_coordinates(
                    float(x),
                    float(y),
                    processed_width,
                    processed_height,
                    original_width,
                    original_height,
                )
                pyautogui_code.append(f"pyautogui.click({ax}, {ay})")
            else:
                pyautogui_code.append("pyautogui.click()")

        elif action == "right_click":
            if "coordinate" in args:
                x, y = args["coordinate"]
                ax, ay = self._adjust_coordinates(
                    float(x),
                    float(y),
                    processed_width,
                    processed_height,
                    original_width,
                    original_height,
                )
                pyautogui_code.append(f"pyautogui.rightClick({ax}, {ay})")
            else:
                pyautogui_code.append("pyautogui.rightClick()")

        elif action == "middle_click":
            if "coordinate" in args:
                x, y = args["coordinate"]
                ax, ay = self._adjust_coordinates(
                    float(x),
                    float(y),
                    processed_width,
                    processed_height,
                    original_width,
                    original_height,
                )
                pyautogui_code.append(f"pyautogui.middleClick({ax}, {ay})")
            else:
                pyautogui_code.append("pyautogui.middleClick()")

        elif action == "double_click":
            if "coordinate" in args:
                x, y = args["coordinate"]
                ax, ay = self._adjust_coordinates(
                    float(x),
                    float(y),
                    processed_width,
                    processed_height,
                    original_width,
                    original_height,
                )
                pyautogui_code.append(f"pyautogui.doubleClick({ax}, {ay})")
            else:
                pyautogui_code.append("pyautogui.doubleClick()")

        elif action == "triple_click":
            if "coordinate" in args:
                x, y = args["coordinate"]
                ax, ay = self._adjust_coordinates(
                    float(x),
                    float(y),
                    processed_width,
                    processed_height,
                    original_width,
                    original_height,
                )
                pyautogui_code.append(f"pyautogui.tripleClick({ax}, {ay})")
            else:
                pyautogui_code.append("pyautogui.tripleClick()")

        elif action == "type":
            text = args.get("text", "")
            try:
                text = text.encode("latin-1", "backslashreplace").decode("unicode_escape")
            except Exception:
                pass
            result = ""
            for char in text:
                if char == "\n":
                    result += "pyautogui.press('enter')\n"
                elif char == "'":
                    result += 'pyautogui.press("\\\'")\n'
                elif char == "\\":
                    result += "pyautogui.press('\\\\')\n"
                elif char == '"':
                    result += "pyautogui.press('\"')\n"
                else:
                    result += f"pyautogui.press('{char}')\n"
            pyautogui_code.append(result)

        elif action == "key":
            keys = _clean_keys(args.get("keys", []))
            keys_str = ", ".join([f"'{k}'" for k in keys])
            if len(keys) > 1:
                pyautogui_code.append(f"pyautogui.hotkey({keys_str})")
            else:
                pyautogui_code.append(f"pyautogui.press({keys_str})")

        elif action == "key_down":
            keys = _clean_keys(args.get("keys", []))
            for k in keys:
                pyautogui_code.append(f"pyautogui.keyDown('{k}')")

        elif action == "key_up":
            keys = _clean_keys(args.get("keys", []))
            for k in reversed(keys):
                pyautogui_code.append(f"pyautogui.keyUp('{k}')")

        elif action == "scroll":
            pixels = args.get("pixels", 0)
            pyautogui_code.append(f"pyautogui.scroll({pixels})")

        elif action == "wait":
            pyautogui_code.append("WAIT")

        elif action == "terminate":
            status = str(args.get("status", "success")).lower()
            if status == "failure":
                pyautogui_code.append("FAIL")
            else:
                pyautogui_code.append("DONE")

        elif action == "mouse_move":
            if "coordinate" in args:
                x, y = args["coordinate"]
                ax, ay = self._adjust_coordinates(
                    float(x),
                    float(y),
                    processed_width,
                    processed_height,
                    original_width,
                    original_height,
                )
                pyautogui_code.append(f"pyautogui.moveTo({ax}, {ay})")
            else:
                pyautogui_code.append("pyautogui.moveTo(0, 0)")

        elif action == "left_click_drag":
            if "coordinate" in args:
                x, y = args["coordinate"]
                ax, ay = self._adjust_coordinates(
                    float(x),
                    float(y),
                    processed_width,
                    processed_height,
                    original_width,
                    original_height,
                )
                duration = args.get("duration", 0.5)
                pyautogui_code.append(f"pyautogui.dragTo({ax}, {ay}, duration={duration})")
            else:
                pyautogui_code.append("pyautogui.dragTo(0, 0)")

    def _parse_response_s2(
        self,
        response: str,
        processed_width: Optional[int],
        processed_height: Optional[int],
        original_width: int,
        original_height: int,
    ) -> Tuple[str, List[str]]:
        low_level_instruction = ""
        pyautogui_code: List[str] = []

        if response is None or not response.strip():
            return low_level_instruction, pyautogui_code

        lines = response.split("\n")
        inside_tool_call = False
        current_tool_call: List[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.lower().startswith("action:"):
                if not low_level_instruction:
                    low_level_instruction = line.split(":", 1)[-1].strip()
                continue

            if line.startswith("<tool_call>"):
                inside_tool_call = True
                continue
            if line.startswith("</tool_call>"):
                if current_tool_call:
                    self._process_tool_call(
                        "\n".join(current_tool_call),
                        pyautogui_code,
                        processed_width,
                        processed_height,
                        original_width,
                        original_height,
                    )
                    current_tool_call = []
                inside_tool_call = False
                continue

            if inside_tool_call:
                current_tool_call.append(line)
                continue

            if line.startswith("{") and line.endswith("}"):
                try:
                    json_obj = json.loads(line)
                    if "name" in json_obj and "arguments" in json_obj:
                        self._process_tool_call(
                            line,
                            pyautogui_code,
                            processed_width,
                            processed_height,
                            original_width,
                            original_height,
                        )
                except Exception:
                    pass

        if current_tool_call:
            self._process_tool_call(
                "\n".join(current_tool_call),
                pyautogui_code,
                processed_width,
                processed_height,
                original_width,
                original_height,
            )

        if not low_level_instruction and pyautogui_code:
            first_action = pyautogui_code[0]
            if "." in first_action:
                action_type = first_action.split(".", 1)[1].split("(", 1)[0]
            else:
                action_type = first_action.lower()
            low_level_instruction = f"Performing {action_type} action"

        return low_level_instruction, pyautogui_code

    def parse_action(self, response: str):
        self.history_responses.append(response)
        self.responses.append(response)

        if self.last_model_image_size is not None:
            processed_width, processed_height = self.last_model_image_size
        else:
            processed_width, processed_height = None, None

        if self.last_original_image_size is not None:
            original_width, original_height = self.last_original_image_size
        else:
            original_width, original_height = self.screen_size

        try:
            low_level_instruction, pyautogui_code = self._parse_response_s2(
                response,
                processed_width,
                processed_height,
                original_width,
                original_height,
            )
        except Exception as e:
            print(f"[EvoCUAAgent] Official parser failed: {e}")
            self.actions.append("Parse failed")
            return ["FAIL"]

        if not pyautogui_code:
            self.actions.append(low_level_instruction or "No action parsed")
            return ["FAIL"]

        current_step = len(self.actions) + 1
        first_action = pyautogui_code[0] if pyautogui_code else ""
        if current_step >= self.max_trajectory_length and str(first_action).upper() not in (
            "DONE",
            "FAIL",
        ):
            low_level_instruction = "Fail the task because reaching the maximum step limit."
            pyautogui_code = ["FAIL"]

        self.actions.append(low_level_instruction or "Action parsed")
        return pyautogui_code
