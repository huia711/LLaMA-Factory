# ============================================================================
# 相对于原版 ZeroGUI/openrlhf/agent/uitars.py 的改动：
# 1. 新增 smart_resize 相关函数（IMAGE_FACTOR/MIN_PIXELS/MAX_PIXELS 等），
#    支持 UI-TARS-1.5 (Qwen2.5-VL) 的像素坐标模式
# 2. Prompt 模板增加 {current_todo_list} 占位符和坐标说明 Note
# 3. parse_action_qwen2vl: 新增 model_image_size 参数，支持像素坐标归一化，
#    并增加安全裁剪 [0,1]
# 4. UITARSAgent.__init__: 新增 gt_todo_path/enable_crop/use_fast 参数，
#    增加像素坐标模式检测(_detect_pixel_coord_mode)，
#    processor 增加 trust_remote_code=True
# 5. 新增 crop 功能：_handle_crop_action/_crop_current_image/_map_box_to_full 等
# 6. 新增 GT TODO list 功能：_load_gt_todo_map/_format_gt_todo_items
# 7. get_model_inputs: 使用官方 smart_resize（不手动缩放图像），
#    支持 crop 图像注入，增加 apply_chat_template TypeError 降级处理
# 8. parse_action: 解析失败返回 ["FAIL"] 而非 ["DONE"]（防止 reward hacking）
# ============================================================================
"""
UI-TARS Agent 模块

动作空间：
- Computer: click, drag, type, scroll, hotkey, wait, finished 等
- Mobile: click, long_press, type, scroll, open_app, press_home, press_back 等
"""

import ast
import json
import os
import re
from io import BytesIO
from typing import Dict, List, Optional
import math
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor


# ========== 常量定义 ==========
# 屏幕逻辑尺寸（用于坐标归一化）
SCREEN_LOGIC_SIZE = (1920, 1080)

# 特殊动作标识
FINISH_WORD = "finished"      # 任务完成
WAIT_WORD = "wait"            # 等待（暂停 5 秒后重新截图）
ENV_FAIL_WORD = "error_env"   # 环境错误
CALL_USER = "call_user"       # 需要用户帮助

# ========== 图像预处理常量（严格按照官方 README_coordinates.md）==========
# 参考: https://github.com/bytedance/UI-TARS/blob/main/README_coordinates.md
IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28      # 78400
MAX_PIXELS = 16384 * 28 * 28    # 12845056
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer >= 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer <= 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, 
    factor: int = IMAGE_FACTOR, 
    min_pixels: int = MIN_PIXELS, 
    max_pixels: int = MAX_PIXELS
) -> tuple:
    """
    官方 smart_resize 函数，严格按照 README_coordinates.md 实现。
    计算模型实际看到的图像尺寸，模型输出的坐标是基于这个尺寸的。
    返回 (new_height, new_width)。
    """
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
    
    return h_bar, w_bar  # (new_height, new_width)

# ========== 动作空间定义 ==========

# 桌面（Computer）动作空间
# 坐标格式：<|box_start|>(x,y)<|box_end|>，其中 x,y 是像素坐标（0-1920, 0-1080）
UITARS_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
"""

# 移动端（Mobile）动作空间
UITARS_MOBILE_ACTION_SPACE = """
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

# 带用户调用功能的动作空间（用于需要用户帮助的场景）
UITARS_CALL_USR_ACTION_SPACE = """
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

# 可选 crop 动作（由 enable_crop 控制是否注入到 action space）
CROP_ACTION_LINE = (
    "crop(start_box='<|box_start|>(x1,y1)<|box_end|>', "
    "end_box='<|box_start|>(x3,y3)<|box_end|>') "
    "# If the target element is small, ambiguous, or not clearly visible in the current view, "
    "you MUST use the `crop` action first to zoom in."
)

# ========== Prompt 模板定义 ==========

# Prompt 模板 1: 无思考模式（直接输出动作）
UITARS_USR_PROMPT_NOTHOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Action: ...
```
## Action Space
{UITARS_CALL_USR_ACTION_SPACE}
## TO-DO List
{current_todo_list}
## User Instruction
{instruction}
"""

# Prompt 模板 2: 思考模式（Thought + Action）
UITARS_USR_PROMPT_THOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

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

## TO-DO List
{current_todo_list}

## User Instruction
{instruction}
"""


# ========== 动作解析函数 ==========

def parse_action(action_str):
    """
    定义一个函数来解析每个 action
    使用 Python AST（抽象语法树）解析动作字符串，提取函数名和参数。
    """
    try:
        # 使用 AST 解析字符串为语法树
        # mode='eval' 表示这是一个表达式（而非语句）
        node = ast.parse(action_str, mode='eval')

        # 确保节点是一个表达式
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        # 获取表达式的主体（应该是函数调用）
        call = node.body

        # 确保主体是一个函数调用
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # 提取函数名
        if isinstance(call.func, ast.Name):
            # 直接函数名：click(...)
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            # 属性调用：pyautogui.click(...)
            func_name = call.func.attr
        else:
            func_name = None

        # 提取关键字参数
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            # 处理不同类型的值，这里假设都是常量
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # 兼容 Python < 3.8
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {
            'function': func_name,
            'args': kwargs
        }

    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None
    
def escape_single_quotes(text):
    """
    转义字符串中的单引号

    示例:
        >>> escape_single_quotes("It's a test")
        "It\\'s a test"
    """
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)

def _strip_tool_call_from_box(action_str: str) -> str:
    """Remove <tool_call> prefix inside start_box/end_box values.

    Qwen3-VL outputs coordinates as start_box='<tool_call>(x, y)' instead of
    start_box='(x,y)'. This strips the <tool_call> tag so downstream parsing
    works unchanged.
    """
    return re.sub(r"</?tool_call>", "", action_str)


def fix_click_output(output: str) -> str:
    """
    修复格式错误的 click 动作输出

    示例:
        >>> fix_click_output("click(start_box='=x(409,173)')")
        "click(start_box='(409,173)')"
        >>> fix_click_output("click(start_box='<tool_call>(392, 405)')")
        "click(start_box='(392,405)')"
    """
    matches = re.findall(r'(\d+)\s*,\s*(\d+)', output)

    if matches:
        x, y = matches[-1]
        return f"click(start_box='({x},{y})')"
    else:
        return None


def fix_drag_output(output: str) -> str:
    """
    修复格式错误的 drag 动作输出

    示例:
        >>> fix_drag_output("drag(start_box='=(624,470)', end_box='=(288,505)')")
        "drag(start_box='(624,470)', end_box='(288,505)')"
    """
    matches = re.findall(r'(\d+)\s*,\s*(\d+)', output)

    if matches and len(matches) >= 2:
        x1, y1 = matches[-2]
        x2, y2 = matches[-1]
        return f"drag(start_box='({x1},{y1})', end_box='({x2},{y2})')"
    else:
        return None


def fix_scroll_output(output: str) -> str:
    """Fix scroll actions with <tool_call> or malformed coordinates.

    Example:
        >>> fix_scroll_output("scroll(start_box='<tool_call>(515, 567)', direction='down')")
        "scroll(start_box='(515,567)', direction='down')"
    """
    coords = re.findall(r'(\d+)\s*,\s*(\d+)', output)
    direction_m = re.search(r"direction\s*=\s*['\"](\w+)['\"]", output)
    if coords and direction_m:
        x, y = coords[-1]
        direction = direction_m.group(1)
        return f"scroll(start_box='({x},{y})', direction='{direction}')"
    return None


def fix_box_action_output(action_name: str, output: str) -> str:
    """Fix single-box actions (left_double, right_single, etc.) with <tool_call>.

    Example:
        >>> fix_box_action_output("left_double", "left_double(start_box='<tool_call>(957, 814)')")
        "left_double(start_box='(957,814)')"
    """
    coords = re.findall(r'(\d+)\s*,\s*(\d+)', output)
    if coords:
        x, y = coords[-1]
        return f"{action_name}(start_box='({x},{y})')"
    return None

def parse_action_qwen2vl(
    text,
    factor,
    image_height,
    image_width,
    model_image_size=None,
):
    """
    解析 Qwen2-VL 模型的响应文本，提取动作列表
    
    这是 UITARSAgent 的核心解析函数，负责：
    1. 提取 Thought/Reflection 部分（思考过程）
    2. 提取 Action 部分（动作代码）
    3. 解析每个动作为结构化字典
    4. 处理坐标归一化（从像素坐标转为相对坐标）
    """
    text = text.strip()
    
    # ========== 步骤 1: 提取思考部分 ==========
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Action_Summary: "
    else:
        # 默认使用 Thought 格式
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            # 单一思考部分
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            # Reflection + Action_Summary 两部分
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    
    # ========== 步骤 2: 提取 Action 部分 ==========
    assert "Action:" in text, "Response must contain 'Action:' field"
    action_str = text.split("Action:")[-1]  # 取 Action: 之后的所有内容

    # ========== 步骤 3: 分割多个动作 ==========
    # 模型可能输出多个动作（用 \n\n 分隔）
    tmp_all_action = action_str.split("\n\n")
    all_action = []
    
    for action_str in tmp_all_action:
        # ========== 步骤 3.0: 全局清理 <tool_call> 标签 ==========
        if "<tool_call>" in action_str:
            action_str_clean = _strip_tool_call_from_box(action_str)
            if action_str_clean != action_str:
                action_str = action_str_clean

        # ========== 步骤 3.1: 修复 type 动作 ==========
        if "type(content" in action_str:
            def escape_quotes(match):
                content = match.group(1)  
                return content

            pattern = r"type\(content='(.*?)'\)"
            content = re.sub(pattern, escape_quotes, action_str)

            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        
        # ========== 步骤 3.2: 修复 click 动作 ==========
        elif "click(start_box" in action_str:
            action_str_fixed = fix_click_output(action_str)
            if (action_str_fixed is not None) and (action_str_fixed != action_str):
                print('[CLICK ACTION FIXED]', action_str, '->', action_str_fixed)
                action_str = action_str_fixed
        
        # ========== 步骤 3.3: 修复 drag 动作 ==========
        elif "drag(start_box" in action_str:
            action_str_fixed = fix_drag_output(action_str)
            if (action_str_fixed is not None) and (action_str_fixed != action_str):
                print('[DRAG ACTION FIXED]', action_str, '->', action_str_fixed)
                action_str = action_str_fixed

        # ========== 步骤 3.4: 修复 scroll 动作 ==========
        elif "scroll(" in action_str:
            action_str_fixed = fix_scroll_output(action_str)
            if (action_str_fixed is not None) and (action_str_fixed != action_str):
                print('[SCROLL ACTION FIXED]', action_str, '->', action_str_fixed)
                action_str = action_str_fixed

        # ========== 步骤 3.5: 修复 left_double / right_single 等单坐标动作 ==========
        else:
            for _act_name in ("left_double", "right_single"):
                if f"{_act_name}(start_box" in action_str:
                    action_str_fixed = fix_box_action_output(_act_name, action_str)
                    if (action_str_fixed is not None) and (action_str_fixed != action_str):
                        print(f'[{_act_name.upper()} ACTION FIXED]', action_str, '->', action_str_fixed)
                        action_str = action_str_fixed
                    break
        
        all_action.append(action_str)

    # ========== 步骤 4: 解析每个动作字符串 ==========
    # 使用 parse_action 函数将字符串解析为结构化字典
    parsed_actions = [parse_action(action.replace("\n","\\n").lstrip()) for action in all_action]
    
    # ========== 步骤 5: 构建最终动作列表 ==========
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        # 跳过解析失败的动作
        if action_instance is None:
            print(f"Action can't parse: {raw_str}")
            continue

        action_type = action_instance["function"]
        params = action_instance["args"]

        # 处理动作参数
        action_inputs = {}
        for param_name, param in params.items():
            if param == "":
                continue
            param = param.lstrip()  # 去掉多余的空格
            action_inputs[param_name.strip()] = param
            
            # ========== 步骤 5.1: 处理坐标参数（start_box / end_box） ==========
            # 输入格式：(x,y) 或 [x,y] 或 x,y
            # 输出格式：[x1, y1, x2, y2]（归一化到 0-1 范围）
            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                # 移除括号和方括号，按逗号分割
                numbers = (
                    ori_box.replace("(", "")
                    .replace(")", "")
                    .replace("[", "")
                    .replace("]", "")
                    .split(",")
                )

                # ========== 坐标归一化（严格按照官方 README_coordinates.md）==========
                # 官方公式: new_coordinate = (model_output / new_size * original_size)
                # 即: 归一化坐标 = model_output / smart_resize_size
                # 模型输出的坐标是基于 smart_resize 后分辨率的像素坐标
                float_numbers = []
                model_w = model_h = None
                if model_image_size and len(model_image_size) == 2:
                    model_w, model_h = model_image_size

                for idx, num in enumerate(numbers):
                    num = num.strip()
                    if not num:
                        continue
                    try:
                        raw_val = float(num)
                    except ValueError:
                        continue

                    if model_w and model_h:
                        # model_output / smart_resize_size
                        dim = model_w if idx % 2 == 0 else model_h
                        v = raw_val / float(dim)
                    else:
                        # 回退到 factor（通常是 1000）
                        v = raw_val / float(factor)

                    # 安全裁剪到 [0, 1]
                    v = max(0.0, min(1.0, v))
                    float_numbers.append(v)

                # 如果只有两个数字（x, y），扩展为四个数字（x1, y1, x2, y2）
                if len(float_numbers) == 2:
                    float_numbers = [
                        float_numbers[0],
                        float_numbers[1],
                        float_numbers[0],
                        float_numbers[1],
                    ]

                action_inputs[param_name.strip()] = str(float_numbers)

        # 构建最终的动作字典
        actions.append(
            {
                "reflection": reflection,  # 反思文本（可选）
                "thought": thought,  # 思考文本（可选）
                "action_type": action_type,  # 动作类型
                "action_inputs": action_inputs,  # 动作参数
                "text": text,  # 原始响应文本
            }
        )

    return actions

def action_space_mapping(input_text: str) -> str:
    # 定义替换规则：正则表达式模式和对应的替换模板
    rules = [
        # 1. click(start_box='<|box_start|>(x1,y1)<|box_end|>')
        (
            r"click\(start_box='(?:<\|box_start\|>)?\(([0-9]+),([0-9]+)\)(?:<\|box_end\|>)?'\)",
            lambda m: f'do(action="Tap", element=[{int(m.group(1))/1000:.3f}, {int(m.group(2))/1000:.3f}])'
        ),
        # 2. long_press(start_box='<|box_start|>(x1,y1)<|box_end|>', time='')
        (
            r"long_press\(start_box='(?:<\|box_start\|>)?\(([0-9]+),([0-9]+)\)(?:<\|box_end\|>)?', time=''?\)",
            lambda m: f'do(action="Long Press", element=[{int(m.group(1))/1000:.3f}, {int(m.group(2))/1000:.3f}])'
        ),
        # 2. long_press(start_box='<|box_start|>(x1,y1)<|box_end|>')
        (
            r"long_press\(start_box='(?:<\|box_start\|>)?\(([0-9]+),([0-9]+)\)(?:<\|box_end\|>)?'\)",
            lambda m: f'do(action="Long Press", element=[{int(m.group(1))/1000:.3f}, {int(m.group(2))/1000:.3f}])'
        ),
        # 3. type(content='')
        (
            r"type\(content='((?:\'|[^'])*?)'\)",
            r'do(action="Type", text="\1")'
        ),
        # 4. scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
        (
            r"scroll\(start_box='(?:<\|box_start\|>)?\(([0-9]+),([0-9]+)\)(?:<\|box_end\|>)?', end_box='(?:<\|box_start\|>)?\(([0-9]+),([0-9]+)\)(?:<\|box_end\|>)?'\)",
            lambda m: f'do(action="Swipe Precise", start=[{int(m.group(1))/1000:.3f}, {int(m.group(2))/1000:.3f}], end=[{int(m.group(3))/1000:.3f}, {int(m.group(4))/1000:.3f}])'
        ),
        # 5. scroll(direction='up')
        (
            r"scroll\(direction='((?:up|down|left|right))'\)",
            r'do(action="Swipe", direction="\1")'
        ),
        # 6. press_home()
        (
            r"press_home\(\)",
            r'do(action="Home")'
        ),
        # 7. press_back()
        (
            r"press_back\(\)",
            r'do(action="Back")'
        ),
        # 8. finished(content='')
        (
            r"finished\(content='((?:\'|[^'])*?)'\)",
            r'finish(message="\1")'
        ),
        # 9. finished()
        (
            r"finished\(\)",
            r'finish(message="")'
        ),
        # 10. drag(start_box='(624,470)', end_box='(288,505)')
        (
            r"drag\(start_box='\(([0-9]+),([0-9]+)\)', end_box='\(([0-9]+),([0-9]+)\)'\)",
            lambda m: f'do(action="Swipe Precise", start=[{int(m.group(1))/1000:.3f}, {int(m.group(2))/1000:.3f}], end=[{int(m.group(3))/1000:.3f}, {int(m.group(4))/1000:.3f}])'
        ),
        # 11. scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
        (
            r"scroll\(start_box='(?:<\|box_start\|>)?\(([0-9]+),([0-9]+)\)(?:<\|box_end\|>)?', direction='(down|up|left|right)'\)",
            lambda m: f'do(action="Swipe", element=[{int(m.group(1))/1000:.3f}, {int(m.group(2))/1000:.3f}], direction="{m.group(3)}")'
        ),
        # 12. open_app(app_name=\'\')
        (
            r"open_app\(app_name='([^']+)'\)",
            lambda m: f'do(action="Launch", app="{m.group(1)}")'
        )
    ]

    # 匹配整体输出格式：Thought: ...\nAction: ...\n
    output_pattern = r'(Thought:.*Action:.*)'
    
    def replace_action(match):
        line = match.group(1)
        # 提取 Action 部分
        action_match = re.search(r'Action: (.*?)(?=\n|$)', line)
        if not action_match:
            return line

        action = action_match.group(1)
        # 尝试每条替换规则
        for pattern, replacement in rules:
            if re.match(pattern, action):
                if callable(replacement):
                    # 使用lambda函数处理替换
                    action = re.sub(pattern, replacement, action)
                else:
                    # 普通替换
                    action = re.sub(pattern, replacement, action)
                break
        return action
        
    # 处理整个输入文本
    result = re.sub(output_pattern, replace_action, input_text, flags=re.DOTALL)
    return result

def parsing_response_to_android_action_code(responses, image_height: int, image_width:int, input_swap:bool=True) -> str:
    if isinstance(responses, dict):
        responses = [responses]
    action_code = ""
    for response_id, response in enumerate(responses):
        input_text = response["text"]
        action_code += action_space_mapping(input_text)

    return action_code

def parsing_response_to_pyautogui_code(responses, image_height: int, image_width:int, input_swap:bool=True) -> str:
    """
    将模型输出解析为 pyautogui 代码字符串

    参数:
        responses: 动作字典或字典列表
            单个动作字典结构：
        {
                "action_type": "click",  # 动作类型
                "action_inputs": {       # 动作参数
                    "start_box": "[0.5, 0.3, 0.5, 0.3]",  # 归一化坐标 [x1, y1, x2, y2]
                    "end_box": None,     # 终点坐标（drag 动作）
                    "content": "text",   # 输入内容（type 动作）
                    "direction": "down", # 滚动方向（scroll 动作）
                    "key": "ctrl v",     # 快捷键（hotkey 动作）
                },
                "thought": "...",        # 思考文本（可选）
                "observation": "..."     # 观测文本（可选）
            }
        image_height: 图片高度（像素），用于坐标转换
        image_width: 图片宽度（像素），用于坐标转换
        input_swap: 是否使用剪贴板方式输入（True 使用 ctrl+v，False 使用 pyautogui.write）
            - True: 更可靠，支持特殊字符，但需要 pyperclip
            - False: 更简单，但可能不支持某些特殊字符
    
    返回:
        str: 生成的 pyautogui 代码字符串
            例如：
            ```
            import pyautogui
            import time
            '''
            Observation: ...
            Thought: ...
            '''
            pyautogui.click(960, 324)
            ```
        
        特殊返回值：
        - "DONE": 任务完成
        - "WAIT": 等待
        - "FAIL": 失败
    
    坐标转换说明：
        - 输入：归一化坐标 [x1, y1, x2, y2]（0-1 范围）
        - 输出：绝对像素坐标（基于 image_width 和 image_height）
        - 计算：x_abs = (x1 + x2) / 2 * image_width
    """
    # 初始化代码头部
    pyautogui_code = f"import pyautogui\nimport time\n"
    
    # 统一处理：确保 responses 是列表
    if isinstance(responses, dict):
        responses = [responses]
    for response_id, response in enumerate(responses):
        if "observation" in response:
            observation = response["observation"]
        else:
            observation = ""

        if "thought" in response:
            thought = response["thought"]
        else:
            thought = ""

        if response_id == 0:
            pyautogui_code += f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
        else:
            pyautogui_code += f"\ntime.sleep(3)\n"

        action_dict = response
        action_type = action_dict.get("action_type")
        action_inputs = action_dict.get("action_inputs", {})

        if action_type == "hotkey":
            # Parsing hotkey action
            if "key" in action_inputs:
                hotkey = action_inputs.get("key", "")
            else:
                hotkey = action_inputs.get("hotkey", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"
            
            elif hotkey == "arrowup":
                hotkey = "up"
            
            elif hotkey == "arrowdown":
                hotkey = "down"

            if hotkey:
                # Handle other hotkeys
                keys = hotkey.split()  # Split the keys by space
                convert_keys = []
                for key in keys:
                    if key == "space":
                        key = ' '
                    convert_keys.append(key)
                pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})"

        elif action_type == "press":
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")

            if key_to_press == "arrowleft":
                key_to_press = "left"
            elif key_to_press == "arrowright":
                key_to_press = "right"
            elif key_to_press == "arrowup":
                key_to_press = "up"
            elif key_to_press == "arrowdown":
                key_to_press = "down"
            elif key_to_press == "space":
                key_to_press = " "

            if key_to_press:
                pyautogui_code += f"\npyautogui.press({repr(key_to_press)})"
            
        elif action_type == "keyup":
            key_to_up = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyUp({repr(key_to_up)})"
        
        elif action_type == "keydown":
            key_to_down = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyDown({repr(key_to_down)})"

        elif action_type == "type":
            # Parsing typing action using clipboard
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            stripped_content = content
            if content.endswith("\n") or content.endswith("\\n"):
                stripped_content = stripped_content.rstrip("\\n").rstrip("\n")
            if content:
                if input_swap:
                    pyautogui_code += f"\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy('{stripped_content}')"
                    pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"
                else:
                    pyautogui_code += f"\npyautogui.write('{stripped_content}', interval=0.1)"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"


        elif action_type in ["drag", "select"]:
            # Parsing drag or select action based on start and end_boxes
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                sx = round(float((x1 + x2) / 2) * image_width, 3)
                sy = round(float((y1 + y2) / 2) * image_height, 3)
                x1, y1, x2, y2 = eval(end_box)  # Assuming box is in [x1, y1, x2, y2]
                ex = round(float((x1 + x2) / 2) * image_width, 3)
                ey = round(float((y1 + y2) / 2) * image_height, 3)
                pyautogui_code += (
                    f"\npyautogui.moveTo({sx}, {sy})\n"
                    f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n"
                )

        elif action_type == "scroll":
            # Parsing scroll action
            start_box = action_inputs.get("start_box")
            if start_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)

                # # 先点对应区域，再滚动
                # pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
            else:
                x = None
                y = None
            direction = action_inputs.get("direction", "")

            if x == None:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5)"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5)"
            else:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

        elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            # Parsing mouse click actions
            start_box = action_inputs.get("start_box")
            start_box = str(start_box)
            if start_box:
                start_box = eval(start_box)
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2 = x1
                    y2 = y1
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                if action_type == "left_single" or action_type == "click":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"

        elif action_type in ["finished"]:
            pyautogui_code = f"DONE"

        else:
            pyautogui_code += f"\n# Unrecognized action type: {action_type}"

    return pyautogui_code

def add_box_token(input_string):
    """
    将模型输出中的坐标参数，自动补全为带有 <|box_start|> / <|box_end|> 标记的格式。
    
    作用场景：
    - 一些 UI-TARS / Qwen2-VL 模型在生成 `Action:` 时，只输出形如
      `click(start_box='(x,y)')` 的裸坐标；
    - 而我们在上游 prompt / 解析链路里，约定使用
      `click(start_box='<|box_start|>(x,y)<|box_end|>')` 这种格式，方便统一用正则解析。
    
    本函数会：
    1）按 `Action: ` 切分多个动作段落；
    2）用正则匹配每个动作里的 `start_box='(x,y)'` / `end_box='(x,y)'`；
    3）把它们替换成带 `<|box_start|>` / `<|box_end|>` 的版本；
    4）再把所有动作重新拼接成完整字符串返回。
    """
    # Step 1: Split the string into individual actions
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            # Step 2: Extract coordinates (start_box or end_box) using regex
            coordinates = re.findall(r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)
            
            updated_action = action  # Start with the original action
            for coord_type, x, y in coordinates:
                # Convert x and y to integers
                updated_action = updated_action.replace(f"{coord_type}='({x},{y})'", f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'")
            processed_actions.append(updated_action)
        
        # Step 5: Reconstruct the final string
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string


class UITARSAgent:
    @staticmethod
    def _format_gt_todo_items(items: List[str]) -> str:
        lines: List[str] = []
        first = True
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            if text.startswith("["):
                lines.append(text)
                if first and "[o]" in text:
                    first = False
            else:
                marker = "[o]" if first else "[ ]"
                lines.append(f"{marker} {text}")
                first = False
        return "\n".join(lines) if lines else "None"

    @staticmethod
    def _load_gt_todo_map(gt_todo_path: Optional[str]) -> Dict[str, List[str]]:
        if not gt_todo_path:
            return {}
        path = os.path.expanduser(gt_todo_path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if not os.path.exists(path):
            print(f"[WARN] GT TODO file not found: {path}")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load GT TODO file: {e}")
            return {}
        if not isinstance(data, dict):
            print("[WARN] GT TODO file must be a JSON object (example_id -> list of items).")
            return {}
        cleaned: Dict[str, List[str]] = {}
        for eid, items in data.items():
            if isinstance(items, list):
                cleaned_items = [str(item).strip() for item in items if str(item).strip()]
                if cleaned_items:
                    cleaned[str(eid)] = cleaned_items
            elif isinstance(items, str):
                text = items.strip()
                if text:
                    cleaned[str(eid)] = [text]
        return cleaned

    @staticmethod
    def _detect_pixel_coord_mode(tokenizer_path: str) -> bool:
        """
        检测模型的坐标输出模式
        
        Returns:
            True: 模型输出像素坐标（UI-TARS-1.5-7B）
            False: 模型输出 0-1000 归一化坐标（ZeroGUI-OSWorld-7B）
        """
        # 方法 1: 检查模型名称
        model_name = tokenizer_path.lower()
        if "ui-tars-1.5" in model_name or "uitars-1.5" in model_name:
            return True
        if "zerogui" in model_name:
            return False
        
        # 方法 2: 检查模型配置文件（model_type）
        try:
            import os
            import json
            config_path = os.path.join(tokenizer_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    model_type = config.get("model_type", "")
                    # Qwen2.5-VL 使用像素坐标，Qwen2-VL 使用 0-1000 归一化
                    if "qwen2_5_vl" in model_type:
                        return True
                    elif "qwen2_vl" in model_type:
                        return False
        except Exception:
            pass
        
        # 默认：使用 0-1000 归一化（更安全的回退方案）
        return False

    @staticmethod
    def _append_crop_action_space(action_space: str) -> str:
        """
        根据 enable_crop 将 crop 动作注入到 action space（避免重复注入）。
        """
        if "crop(" in action_space:
            return action_space
        return action_space.rstrip() + "\n" + CROP_ACTION_LINE + "\n"
    
    def __init__(self,
                 tokenizer_path,
                 max_trajectory_length=15,
                 history_n=5,
                 text_history_n=None,
                 screen_size=SCREEN_LOGIC_SIZE,
                 action_space='computer',
                 infer_mode='qwen2vl_user',
                 prompt_style='qwen2vl_user',
                 input_swap=False,
                 language='Chinese',
                 gt_todo_path: Optional[str] = None,
                 enable_crop: bool = False,
                 use_fast: bool = True,
                 ):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
        self.processor = AutoProcessor.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=use_fast,
        )
        
        # 【已通过直接修改 tokenizer_config.json 解决】
        # UI-TARS-1.5-7B 的 chat_template 已经在配置文件中修改为多模态格式，
        # 不再需要运行时动态替换。如需恢复原始配置，参见备份文件：
        #   checkpoints/UI-TARS-1.5-7B/tokenizer_config.json.backup
        #
        # chat_tmpl = getattr(self.tokenizer, "chat_template", "") or ""
        # processor_name = self.processor.__class__.__name__
        # if "<|vision_start|>" not in chat_tmpl and "Qwen2" in processor_name and "VL" in processor_name:
        #     print(f"[UITARSAgent] 检测到 {processor_name}，但 chat_template 缺少视觉占位符")
        #     print(f"[UITARSAgent] 自动替换为标准多模态 chat_template")
        #     self.tokenizer.chat_template = UITARS_QWEN2VL_MM_CHAT_TEMPLATE
        
        self.max_trajectory_length = max_trajectory_length
        self.history_n = history_n
        self.text_history_n = text_history_n  # None = keep all text history
        self.screen_size = screen_size
        self.action_space = action_space
        self.infer_mode = infer_mode 
        self.prompt_style = prompt_style
        self.input_swap = input_swap
        self.language = language
        self.gt_todo_path = gt_todo_path
        self.gt_todo_map = self._load_gt_todo_map(gt_todo_path)
        self.last_model_image_size = None  # 模型输入图像尺寸 (w, h)
        self.last_original_image_size = None  # 原始截图尺寸 (w, h)
        self.enable_crop = enable_crop
        self.pending_crop_image = None  # 等待注入的 crop 图像（字节）
        self.pending_crop_box = None  # crop 区域（归一化到完整屏幕）
        
        # 检测模型坐标输出模式
        # UI-TARS-1.5 (Qwen2.5-VL): 输出像素坐标（基于 smart_resize 后的分辨率）
        # ZeroGUI-1.0 (Qwen2-VL): 输出 0-1000 范围的归一化坐标
        self.use_pixel_coords = self._detect_pixel_coord_mode(tokenizer_path)
        print(f"[UITARSAgent] 坐标模式: {'像素坐标 (smart_resize)' if self.use_pixel_coords else '0-1000 归一化'}")
        
        self.prompt_action_space = UITARS_ACTION_SPACE
        self.customize_action_parser = parse_action_qwen2vl
        self.action_parse_res_factor = 1000
        if self.infer_mode == "qwen2vl_user":
            self.prompt_action_space = UITARS_CALL_USR_ACTION_SPACE
        if action_space == 'mobile':
            self.prompt_action_space = UITARS_MOBILE_ACTION_SPACE
            self.action_code_mapper = parsing_response_to_android_action_code
        else:
            self.action_code_mapper = parsing_response_to_pyautogui_code

        self.prompt_template = UITARS_USR_PROMPT_THOUGHT
        
        if self.prompt_style == "qwen2vl_user":
            self.prompt_template = UITARS_USR_PROMPT_THOUGHT
        elif self.prompt_style == "qwen2vl_no_thought":
            self.prompt_template = UITARS_USR_PROMPT_NOTHOUGHT

        self.reset()

    @staticmethod
    def _parse_box_value(box_value):
        """
        将 box 值统一解析为 float 列表。

        支持输入：
        - 已经是 list/tuple
        - 字符串形式，比如 "[x1, y1, x2, y2]" 或 "(x, y)"

        返回：
        - 至少 2 个数字的 list[float]
        - 解析失败返回 None
        """
        if box_value is None:
            return None
        if isinstance(box_value, (list, tuple)):
            values = list(box_value)
        else:
            try:
                # 安全解析字符串形式的 list/tuple（不执行任意代码）。
                values = ast.literal_eval(str(box_value))
            except Exception:
                return None
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            return None
        try:
            # 逐个转成 float，任意失败则整体失败。
            return [float(v) for v in values]
        except Exception:
            return None

    @staticmethod
    def _box_center(box_values):
        """
        计算 box 的中心点。

        - [x1, y1, x2, y2] -> 矩形中心点
        - [x, y] -> 视为单点
        """
        if len(box_values) >= 4:
            x1, y1, x2, y2 = box_values[:4]
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if len(box_values) == 2:
            return box_values[0], box_values[1]
        return None

    @staticmethod
    def _clamp_box(box):
        """
        将 box 裁剪到 [0, 1]，并保证顺序合法（x1 <= x2, y1 <= y2）。
        """
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
        """
        将裁剪图内的局部坐标映射回全图坐标。

        - box: 裁剪图内的归一化坐标（0..1）
        - crop_box: 该裁剪区域在全图中的归一化坐标（0..1）
        """
        cx1, cy1, cx2, cy2 = crop_box
        crop_w = max(cx2 - cx1, 0.0)
        crop_h = max(cy2 - cy1, 0.0)
        if crop_w == 0.0 or crop_h == 0.0:
            return box
        x1, y1, x2, y2 = box
        # 先在裁剪区域内缩放，再平移回全图坐标系。
        mapped = [
            cx1 + x1 * crop_w,
            cy1 + y1 * crop_h,
            cx1 + x2 * crop_w,
            cy1 + y2 * crop_h,
        ]
        return UITARSAgent._clamp_box(mapped)

    def _crop_current_image(self, crop_box):
        """
        使用归一化坐标生成白色蒙版图：保留原始分辨率，crop 区域外填白色。
        始终从原始截图（observations）裁剪，避免连续 crop 时从蒙版图裁剪导致全白。
        """
        if not self.observations:
            return None
        try:
            raw_screenshot = self.observations[-1]["screenshot"]
            image = Image.open(BytesIO(raw_screenshot))
        except Exception:
            return None
        width, height = image.size
        x1, y1, x2, y2 = self._clamp_box(crop_box)
        # 将归一化坐标换算为像素边界。
        left = max(0, min(width, int(round(x1 * width))))
        right = max(0, min(width, int(round(x2 * width))))
        top = max(0, min(height, int(round(y1 * height))))
        bottom = max(0, min(height, int(round(y2 * height))))
        # 避免出现零面积裁剪。
        if right <= left:
            right = min(width, left + 1)
        if bottom <= top:
            bottom = min(height, top + 1)
        try:
            masked = Image.new("RGB", (width, height), (255, 255, 255))
            crop_region = image.crop((left, top, right, bottom))
            masked.paste(crop_region, (left, top))
            buf = BytesIO()
            masked.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None

    def _handle_crop_action(self, parsed_response):
        """
        处理解析后的 crop 动作：
        - 用起点/终点计算裁剪框
        - 裁剪当前截图
        - 缓存裁剪图供下一轮模型输入
        - 缓存裁剪区域在全图中的位置用于坐标映射
        """
        action_inputs = parsed_response.get("action_inputs", {})
        start_box = self._parse_box_value(action_inputs.get("start_box"))
        end_box = self._parse_box_value(action_inputs.get("end_box"))
        # 必须同时有起点和终点，才能确定裁剪框。
        if not start_box or not end_box:
            return False
        start_center = self._box_center(start_box)
        end_center = self._box_center(end_box)
        # 任一端点无法归约为坐标点则失败。
        if not start_center or not end_center:
            return False
        sx, sy = start_center
        ex, ey = end_center
        # 裁剪框由两个端点的中心坐标确定。
        local_box = self._clamp_box([min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey)])
        full_box = local_box
        # 如果已经在上一层裁剪中，需要映射回全图坐标。
        if self.pending_crop_box is not None:
            full_box = self._map_box_to_full(local_box, self.pending_crop_box)
        cropped_bytes = self._crop_current_image(local_box)
        if not cropped_bytes:
            return False
        # 缓存下一轮输入图像及其在全图中的范围。
        self.pending_crop_image = cropped_bytes
        self.pending_crop_box = full_box
        return True

    def get_model_inputs(self, instruction: str, obs: Dict):
        
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(self.thoughts), \
            "The number of observations and actions should be the same."

        current_image = obs["screenshot"]
        if self.enable_crop and self.pending_crop_image is not None:
            current_image = self.pending_crop_image
            self.pending_crop_image = None
        self.history_images.append(current_image)

        base64_image = obs["screenshot"]
        self.observations.append(
            {"screenshot": base64_image, "accessibility_tree": None}
        )

        example_id = obs.get("example_id")
        if not example_id:
            task_meta = obs.get("task_meta")
            if isinstance(task_meta, dict):
                example_id = task_meta.get("example_id")

        current_todo_list = "None"
        if self._override_todo_list:
            current_todo_list = self._override_todo_list
        elif example_id and self.gt_todo_map:
            gt_items = self.gt_todo_map.get(str(example_id))
            if gt_items:
                current_todo_list = self._format_gt_todo_items(gt_items)
        self.current_todo_list = current_todo_list
        
        if self.infer_mode == "qwen2vl_user":
            user_prompt = self.prompt_template.format(
                instruction=instruction,
                action_space=self.prompt_action_space,
                language=self.language,
                current_todo_list=current_todo_list
            )
        elif self.infer_mode == "qwen2vl_no_thought":
            user_prompt = self.prompt_template.format(
                instruction=instruction,
                current_todo_list=current_todo_list
            )

        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n:]

        messages, images = [], []
        if isinstance(self.history_images, bytes):
            self.history_images = [self.history_images]
        elif isinstance(self.history_images, np.ndarray):
            self.history_images = list(self.history_images)
        elif isinstance(self.history_images, list):
            pass
        else:
            raise TypeError(f"Unidentified images type: {type(self.history_images)}")

        # 使用 smart_resize 对图像做尺度适配，保持与模型输入的尺寸一致
        for turn, image in enumerate(self.history_images):
            try:
                image = Image.open(BytesIO(image))
            except Exception as e:
                raise RuntimeError(f"Error opening image: {e}")

            ori_w, ori_h = image.size
            self.last_original_image_size = (ori_w, ori_h)

            # 使用官方 smart_resize 计算模型输入尺寸（与 README_coordinates.md 一致）
            # 模型输出的坐标是基于 smart_resize 后的分辨率
            new_h, new_w = smart_resize(ori_h, ori_w)
            
            # 只有当模型使用像素坐标模式（UI-TARS-1.5）时，才记录 smart_resize 后的尺寸
            # 对于使用 0-1000 归一化的模型（ZeroGUI），让 parse_action_qwen2vl 回退到 factor=1000
            if self.use_pixel_coords:
                self.last_model_image_size = (new_w, new_h)
            else:
                self.last_model_image_size = None
            
            # 注意：这里不对图像做缩放，让 vLLM/Processor 内部处理
            # 但我们记录 smart_resize 的结果用于坐标反算

            if image.mode != "RGB":
                image = image.convert("RGB")

            images.append(image)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]
        
        image_num = 0
        image_input_list = []
        # 截断文字历史（text_history_n=None 时保留全部，与原始行为一致）
        text_hist = self.history_responses
        if self.text_history_n is not None and len(text_hist) > self.text_history_n:
            text_hist = text_hist[-self.text_history_n:]
        if len(text_hist) > 0:
            for history_idx, history_response in enumerate(text_hist):
                # send at most history_n images to the model
                if history_idx + self.history_n > len(text_hist):
                    cur_image = images[image_num]
                    image_input_list.append(cur_image)
                    messages.append({
                        "role": "user",
                        # 多模态消息的“图片占位符”：
                        # - 这里放的是结构化 content（type=image），并不包含真实图片像素。
                        # - tokenizer.apply_chat_template 会把它渲染成类似：
                        #   <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>
                        # - 真正的图片对象在 image_input_list 中，最终在 processor 内部转成
                        #   pixel_values / image_grid_thw 送入视觉编码器。
                        "content": [{"type": "image", "image": ""}]
                    })
                    image_num += 1
                    
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": add_box_token(history_response)}]
                })

            cur_image = images[image_num]
            image_input_list.append(cur_image)
            messages.append({
                "role": "user",
                # 同上：这里只是占位；图片本体在 image_input_list（multi_modal_data）里。
                "content": [{"type": "image", "image": ""}]
            })
            image_num += 1
        else:
            cur_image = images[image_num]
            image_input_list.append(cur_image)
            messages.append({
                "role": "user",
                # 同上：这里只是占位；图片本体在 image_input_list（multi_modal_data）里。
                "content": [{"type": "image", "image": ""}]
            })
            image_num += 1

        # 【改动标记 4】【新增逻辑】异常处理和降级方案（1062-1093行）
        # 原始版本：prompt_text = self.tokenizer.apply_chat_template(messages, ...)
        # 新版本：增加 try-except，处理不支持多模态格式的 chat_template
        #
        # 某些模型（例如 UI-TARS-1.5-7B）使用的是旧版 Qwen 文本 chat_template，
        # 期望 message.content 为字符串，而不是多模态的 list[dict]。
        # 这里先按「多模态消息」格式尝试一次；如果触发 TypeError（list 无法拼接到 str），
        # 则回退到一个简化版本：把 list 中的所有 text 拼成字符串，image 用占位符标记。
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except TypeError:
            simple_messages = []
            for m in messages:
                content = m.get("content")
                if isinstance(content, list):
                    text_parts = []
                    has_image = False
                    for c in content:
                        if isinstance(c, dict):
                            if c.get("type") == "text":
                                text_parts.append(c.get("text", ""))
                            elif c.get("type") == "image":
                                has_image = True
                    text = "".join(text_parts)
                    if has_image:
                        text = "<|vision_start|><|image_pad|><|vision_end|>" + text
                    simple_messages.append(
                        {
                            "role": m.get("role", "user"),
                            "content": text,
                        }
                    )
                else:
                    simple_messages.append(m)

            prompt_text = self.tokenizer.apply_chat_template(
                simple_messages, tokenize=False, add_generation_prompt=True
            )

        inputs = {"prompt": prompt_text, "multi_modal_data": {'image': image_input_list}}
        # prompt_text: 文本 prompt（会包含 <|vision_start|><|image_pad|><|vision_end|> 这样的视觉占位 token）
        # multi_modal_data["image"]: 真实图片（PIL.Image）列表，底层会转成 pixel_values 等张量喂给模型
        return inputs

    def parse_action(self, response: str):
        
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
            # 解析失败通常意味着模型没有按约定输出 `Thought:` / `Action:`（例如只输出 '!!!'）。
            # 旧逻辑直接返回 DONE，会让策略“用乱码快速结束任务”成为捷径，极易导致训练坍塌。
            # 这里改为 FAIL：显式标记为失败终止，避免被当作成功完成的终止动作。
            self.pending_crop_box = None
            self.actions.append([])
            return ["FAIL"]

        actions = []
        if self.enable_crop:
            crop_response = None
            for parsed_response in parsed_responses:
                if parsed_response.get("action_type") == "crop":
                    crop_response = parsed_response
                    break
            if crop_response is not None:
                if not self._handle_crop_action(crop_response):
                    print(f"[WARN] Failed to handle crop action: {response}")
                self.actions.append(actions)
                if len(self.history_responses) >= self.max_trajectory_length:
                    return ["FAIL"]
                return []

        for parsed_response in parsed_responses:
            if "action_type" in parsed_response:

                if self.action_space != 'mobile' and parsed_response["action_type"] == FINISH_WORD:
                    self.pending_crop_box = None
                    self.actions.append(actions)

                    return ["DONE"]
                
                elif parsed_response["action_type"] == WAIT_WORD:
                    self.pending_crop_box = None
                    self.actions.append(actions)
                    return ["WAIT"]
                
                elif parsed_response["action_type"] == ENV_FAIL_WORD:
                    self.pending_crop_box = None
                    self.actions.append(actions)
                    return ["FAIL"]

                elif parsed_response["action_type"] == CALL_USER:
                    self.pending_crop_box = None
                    self.actions.append(actions)
                    return ["FAIL"]

            if self.enable_crop and self.pending_crop_box is not None:
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
                    self.input_swap
                )
                actions.append(pyautogui_code)
            except Exception as e:
                print(f"Parsing pyautogui code error: {parsed_response}, with error:\n{e}")

        if actions and self.pending_crop_box is not None:
            self.pending_crop_box = None
        self.actions.append(actions)

        if len(self.history_responses) >= self.max_trajectory_length:
            # Default to FAIL if exceed max steps
            actions = ["FAIL"]

        return actions

    def reset(self):
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
        self.current_todo_list = ""
        self._override_todo_list = None
        self.last_model_image_size = None
        self.last_original_image_size = None
        self.pending_crop_image = None
        self.pending_crop_box = None
