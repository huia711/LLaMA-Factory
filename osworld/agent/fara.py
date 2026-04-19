"""
Fara-7B Agent 模块
基于 Microsoft Fara-7B 模型的 GUI Agent 实现

核心功能：
1. 使用 Fara-7B 的 function calling 格式
2. 支持 <tool_call> XML 标签输出解析
3. 将 Fara 动作映射到 pyautogui 代码

输出格式：
<tool_call>
{"name": "<function-name>", "arguments": <args-json-object>}
</tool_call>

动作空间：
- mouse_move: 移动鼠标到指定坐标
- left_click: 左键单击
- type: 输入文本
- scroll: 滚动
- key: 按键
- terminate: 结束任务
"""

import json
import re
from io import BytesIO
from typing import Dict, List
import math

import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor


# ========== Fara MLM 处理器配置（参考官方实现）==========
# 来源: fara/src/fara/fara_agent.py
MLM_PROCESSOR_IM_CFG = {
    "min_pixels": 3136,
    "max_pixels": 12845056,
    "patch_size": 14,
    "merge_size": 2,
}

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, 
    factor: int = IMAGE_FACTOR, 
    min_pixels: int = MIN_PIXELS, 
    max_pixels: int = MAX_PIXELS
) -> tuple:
    """官方 smart_resize 函数"""
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    
    return h_bar, w_bar

def convert_resized_coords_to_original(
    coords: List[float], rsz_w: int, rsz_h: int, og_w: int, og_h: int
) -> List[float]:
    """
    将模型输出的坐标（基于缩放后图像）转换为原始屏幕坐标
    参考: fara/src/fara/fara_agent.py::convert_resized_coords_to_original
    """
    scale_x = og_w / rsz_w
    scale_y = og_h / rsz_h
    return [coords[0] * scale_x, coords[1] * scale_y]


# ========== Fara-7B System Prompt ==========
FARA_SYSTEM_PROMPT = """You are a web automation agent that performs actions on websites to fulfill user requests by calling various tools.

You should stop execution at **Critical Points**. A Critical Point occurs in tasks like:
* Checkout
* Book
* Purchase
* Call
* Email
* Order

A Critical Point requires the user's permission or personal/sensitive information (name, email, credit card, address, payment information, resume, etc.) to complete a transaction (purchase, reservation, sign-up, etc.), or to communicate as a human would (call, email, apply to a job, etc.).

**Guideline:** Solve the task as far as possible **up until a Critical Point**.

**Examples:**
* If the task is to "call a restaurant to make a reservation," do **not** actually make the call. Instead, navigate to the restaurant's page and find the phone number.
* If the task is to "order new size 12 running shoes," do **not** place the order. Instead, search for the right shoes that meet the criteria and add them to the cart.

Some tasks, like answering questions, may not encounter a Critical Point at all.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_descs}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


# ========== Fara 动作空间定义 ==========
FARA_TOOL_DESCS = [
    {
        "type": "function",
        "function": {
            "name": "do",
            "description": "Performs a comprehensive action. Note:\n* For search bars, you may need to press_enter=False and instead separately call left_click() on the search button to submit the search query. This is especially true of search bars that have auto-suggest popups for e.g. locations\n* For calendar widgets, you usually need to left_click() on arrows to move between months and left_click() on dates to select them; type() is not typically used to input dates there.",
            "parameters": {
                "properties": {
                    "action": {
                        "description": "The action to perform. The available actions are:\n* key: Performs key down presses on the arguments passed in order, then performs key releases in reverse order. Includes 'Enter', 'Alt', 'Shift', 'Tab', 'Control', 'Backspace', 'Delete', 'Escape', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'PageDown', 'PageUp', 'Shift', etc.\n* type: Type a string of text on the keyboard.\n* mouse_move: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* left_click: Click the left mouse button.\n* scroll: Performs a scroll of the mouse scroll wheel.\n* visit_url: Visit a specified URL.\n* web_search: Perform a web search with a specified query.\n* history_back: Go back to the previous page in the browser history.\n* pause_and_memorize_fact: Pause and memorize a fact for future reference.\n* wait: Wait specified seconds for the change to happen.\n* terminate: Terminate the current task and report its completion status.",
                        "enum": ["key", "type", "mouse_move", "left_click", "scroll", "visit_url", "web_search", "history_back", "pause_and_memorize_fact", "wait", "terminate"],
                        "type": "string"
                    },
                    "keys": {"description": "Required only by action=key.", "type": "array"},
                    "text": {"description": "Required only by action=type.", "type": "string"},
                    "coordinate": {"description": "(x, y) coordinates for mouse actions. Required only by action=left_click, action=mouse_move, and action=type.", "type": "array"},
                    "pixels": {"description": "Amount of scrolling. Positive = up, Negative = down. Required only by action=scroll.", "type": "number"},
                    "url": {"description": "The URL to visit. Required only by action=visit_url.", "type": "string"},
                    "query": {"description": "The query to search for. Required only by action=web_search.", "type": "string"},
                    "fact": {"description": "The fact to remember for the future. Required only by action=pause_and_memorize_fact.", "type": "string"},
                    "time": {"description": "Seconds to wait. Required only by action=wait.", "type": "number"},
                    "status": {"description": "Status of the task. Required only by action=terminate.", "type": "string", "enum": ["success", "failure"]}
                },
                "required": ["action"],
                "type": "object"
            }
        }
    }
]


def extract_tool_calls(text: str) -> List[dict]:
    """
    从 Fara-7B 输出中提取 tool_call
    
    输入格式:
        <tool_call>
        {"name": "do", "arguments": {"action": "left_click", "coordinate": [960, 540]}}
        </tool_call>
    
    返回: List[dict], 每个字典包含 name 和 arguments
    """
    tool_calls = []
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            tool_call = json.loads(match)
            tool_calls.append(tool_call)
        except json.JSONDecodeError as e:
            print(f"Failed to parse tool_call: {match}, error: {e}")
            continue
    
    return tool_calls


def proc_coords(
    coords: List[float] | None,
    mlm_w: int,
    mlm_h: int,
    viewport_w: int,
    viewport_h: int,
) -> List[float] | None:
    """
    处理坐标：从模型输入尺寸映射到viewport尺寸
    参考: fara/src/fara/fara_agent.py::proc_coords
    """
    if not coords or len(coords) < 2:
        return coords
    
    tgt_x, tgt_y = coords[0], coords[1]
    return convert_resized_coords_to_original(
        [tgt_x, tgt_y], mlm_w, mlm_h, viewport_w, viewport_h
    )


def fara_to_pyautogui(tool_calls: List[dict], image_height: int, image_width: int, 
                       model_image_size=None, original_image_size=None) -> str:
    """
    将 Fara tool_call 转换为 pyautogui 代码
    参考: fara/src/fara/fara_agent.py::execute_action
    
    参数:
        tool_calls: Fara 动作列表
        image_height: 屏幕高度（viewport height）
        image_width: 屏幕宽度（viewport width）
        model_image_size: 模型输入图像尺寸 (width, height)，经过 smart_resize 后的
        original_image_size: 原始图像尺寸 (width, height)
    
    返回:
        pyautogui 代码字符串
    """
    pyautogui_code = "import pyautogui\nimport time\n"
    
    for idx, call in enumerate(tool_calls):
        name = call.get("name", "")
        args = call.get("arguments", {})
        
        # 容错：忽略 name 字段，直接检查 arguments 中的 action
        # Fara 模型可能输出错误的函数名（如 "funciton"）
        action = args.get("action", "")
        
        # 坐标映射：将模型输出坐标映射到真实屏幕坐标
        if "coordinate" in args and model_image_size and original_image_size:
            mlm_w, mlm_h = model_image_size
            viewport_w, viewport_h = original_image_size
            mapped_coords = proc_coords(
                args["coordinate"], mlm_w, mlm_h, viewport_w, viewport_h
            )
            if mapped_coords:
                args["coordinate"] = mapped_coords
        
        if idx > 0:
            pyautogui_code += "\ntime.sleep(0.5)\n"
        
        if action == "left_click":
            coordinate = args.get("coordinate", [])
            if len(coordinate) == 2:
                x, y = int(coordinate[0]), int(coordinate[1])
                pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
        
        elif action == "mouse_move":
            coordinate = args.get("coordinate", [])
            if len(coordinate) == 2:
                x, y = int(coordinate[0]), int(coordinate[1])
                pyautogui_code += f"\npyautogui.moveTo({x}, {y})"
        
        elif action == "type" or action == "input_text":
            # 参考: fara_agent.py 第 549-567 行
            text = args.get("text", args.get("text_value", ""))
            press_enter = args.get("press_enter", True)
            
            # 如果有坐标，先点击
            coordinate = args.get("coordinate")
            if coordinate and len(coordinate) == 2:
                x, y = int(coordinate[0]), int(coordinate[1])
                pyautogui_code += f"\npyautogui.click({x}, {y})"
                pyautogui_code += f"\ntime.sleep(0.2)"
            
            # 输入文本（使用剪贴板避免特殊字符问题）
            escaped_text = text.replace("'", "\\'")
            pyautogui_code += f"\nimport pyperclip"
            pyautogui_code += f"\npyperclip.copy('{escaped_text}')"
            pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
            
            if press_enter or text.endswith("\n"):
                pyautogui_code += f"\npyautogui.press('enter')"
        
        elif action == "scroll":
            # Fara 的 scroll 是 Page Up/Down，不是具体像素值
            # 参考: fara_agent.py 第 513-520 行
            pixels = args.get("pixels", 0)
            if pixels > 0:
                # Page Up
                pyautogui_code += f"\npyautogui.press('pageup')"
            elif pixels < 0:
                # Page Down
                pyautogui_code += f"\npyautogui.press('pagedown')"
        
        elif action == "key":
            keys = args.get("keys", [])
            if keys:
                keys_repr = ', '.join([repr(k) for k in keys])
                pyautogui_code += f"\npyautogui.hotkey({keys_repr})"
        
        elif action == "terminate":
            status = args.get("status", "success")
            if status == "success":
                return "DONE"
            else:
                return "FAIL"
        
        elif action == "wait":
            time_sec = args.get("time", 5)
            pyautogui_code += f"\ntime.sleep({time_sec})"
        
        elif action == "visit_url":
            # 参考: fara_agent.py 第 466-491 行
            url = args.get("url", "")
            if url:
                escaped_url = url.replace("'", "\\'")
                pyautogui_code += f"\n# Visit URL: {escaped_url}"
                pyautogui_code += f"\npyautogui.hotkey('ctrl', 'l')"
                pyautogui_code += f"\ntime.sleep(0.5)"
                pyautogui_code += f"\nimport pyperclip"
                pyautogui_code += f"\npyperclip.copy('{escaped_url}')"
                pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                pyautogui_code += f"\ntime.sleep(0.2)"
                pyautogui_code += f"\npyautogui.press('enter')"
                pyautogui_code += f"\ntime.sleep(3)  # Wait for page load"
        
        elif action == "web_search":
            # 参考: fara_agent.py 第 499-512 行
            query = args.get("query", "")
            if query:
                escaped_query = query.replace("'", "\\'")
                pyautogui_code += f"\n# Web search: {escaped_query}"
                pyautogui_code += f"\npyautogui.hotkey('ctrl', 'l')"
                pyautogui_code += f"\ntime.sleep(0.5)"
                pyautogui_code += f"\nimport pyperclip"
                pyautogui_code += f"\npyperclip.copy('{escaped_query}')"
                pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                pyautogui_code += f"\ntime.sleep(0.2)"
                pyautogui_code += f"\npyautogui.press('enter')"
                pyautogui_code += f"\ntime.sleep(3)  # Wait for search results"
        
        elif action == "history_back":
            # 使用 Alt+Left 或浏览器的后退快捷键
            pyautogui_code += f"\npyautogui.hotkey('alt', 'Left')"
            pyautogui_code += f"\ntime.sleep(1)"
        
        elif action == "pause_and_memorize_fact":
            # 这个动作不需要实际执行，只是记录
            fact = args.get("fact", "")
            pyautogui_code += f"\n# Memorized: {fact}"
    
    return pyautogui_code


class FaraAgent:
    """
    Fara-7B Agent 实现
    基于官方 Fara Agent: fara/src/fara/fara_agent.py
    
    主要特性:
    - 使用 Qwen2.5-VL 作为视觉语言模型
    - 坐标基于模型输入尺寸（smart_resize 后），需要映射回真实屏幕坐标
    - 支持高级动作: visit_url, web_search, left_click, type, scroll, key 等
    """
    
    def __init__(self,
                 tokenizer_path,
                 max_trajectory_length=15,
                 history_n=5,
                 screen_size=(1920, 1080),
                 action_space='computer',
                 language='English',
                 use_fast: bool = True):
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, use_fast=use_fast
        )
        self.processor = AutoProcessor.from_pretrained(
            tokenizer_path, trust_remote_code=True, use_fast=use_fast
        )
        
        self.max_trajectory_length = max_trajectory_length
        self.history_n = history_n
        self.screen_size = screen_size
        self.action_space = action_space
        self.language = language
        
        # MLM（Multimodal Language Model）图像尺寸
        # 这是模型实际看到的图像尺寸（经过 smart_resize）
        self.last_model_image_size = None  # (width, height)
        
        # 原始 viewport 尺寸（真实屏幕尺寸）
        self.last_original_image_size = None  # (width, height)
        
        print(f"[FaraAgent] Initialized with screen_size={screen_size}")
        self.reset()
    
    def get_model_inputs(self, instruction: str, obs: Dict):
        """构建 Fara-7B 的模型输入"""
        
        self.history_images.append(obs["screenshot"])
        base64_image = obs["screenshot"]
        self.observations.append({"screenshot": base64_image})
        
        # 构建 tool descriptions JSON
        tool_descs_json = json.dumps(FARA_TOOL_DESCS, ensure_ascii=False, indent=2)
        system_prompt = FARA_SYSTEM_PROMPT.format(tool_descs=tool_descs_json)
        
        # 限制历史图像数量
        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n:]
        
        # 处理图像
        messages, images = [], []
        if isinstance(self.history_images, bytes):
            self.history_images = [self.history_images]
        elif isinstance(self.history_images, np.ndarray):
            self.history_images = list(self.history_images)
        
        for turn, image_data in enumerate(self.history_images):
            try:
                image = Image.open(BytesIO(image_data))
            except Exception as e:
                raise RuntimeError(f"Error opening image: {e}")
            
            ori_w, ori_h = image.size
            self.last_original_image_size = (ori_w, ori_h)
            
            # 使用 smart_resize 计算模型输入尺寸
            new_h, new_w = smart_resize(ori_h, ori_w)
            self.last_model_image_size = (new_w, new_h)
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            images.append(image)
        
        # 构建消息列表
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Task: {instruction}"}]
            }
        ]
        
        # 添加历史对话和图像
        image_input_list = []
        if len(self.history_responses) > 0:
            for history_idx, history_response in enumerate(self.history_responses):
                if history_idx + self.history_n > len(self.history_responses):
                    cur_image = images[len(image_input_list)]
                    image_input_list.append(cur_image)
                    messages.append({
                        "role": "user",
                        "content": [{"type": "image", "image": ""}]
                    })
                
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": history_response}]
                })
            
            # 添加当前图像
            cur_image = images[len(image_input_list)]
            image_input_list.append(cur_image)
            messages.append({
                "role": "user",
                "content": [{"type": "image", "image": ""}]
            })
        else:
            # 第一次，直接添加图像
            cur_image = images[0]
            image_input_list.append(cur_image)
            messages.append({
                "role": "user",
                "content": [{"type": "image", "image": ""}]
            })
        
        # 应用 chat template
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = {"prompt": prompt_text, "multi_modal_data": {'image': image_input_list}}
        return inputs
    
    def parse_action(self, response: str):
        """解析 Fara-7B 的输出"""
        
        self.history_responses.append(response)
        
        # 提取 tool_calls
        tool_calls = extract_tool_calls(response)
        
        if not tool_calls:
            print(f"[FaraAgent] No tool_call found in response: {response}")
            return ["FAIL"]
        
        # 转换为 pyautogui 代码
        try:
            pyautogui_code = fara_to_pyautogui(
                tool_calls,
                self.screen_size[1],
                self.screen_size[0],
                model_image_size=self.last_model_image_size,
                original_image_size=self.last_original_image_size
            )
            
            # 检查是否完成
            if pyautogui_code in ["DONE", "FAIL"]:
                return [pyautogui_code]
            
            self.actions.append([pyautogui_code])
            
            if len(self.history_responses) >= self.max_trajectory_length:
                return ["FAIL"]
            
            return [pyautogui_code]
        
        except Exception as e:
            print(f"[FaraAgent] Error parsing action: {e}")
            return ["FAIL"]
    
    def reset(self):
        """重置 agent 状态"""
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
        self.last_model_image_size = None
        self.last_original_image_size = None
