"""
Agent 模块初始化文件

"""

from .aguvis import AguvisAgent
from .uitars import UITARSAgent
from .fara import FaraAgent
from .mai import MAIAgent
from .EvoCUA import EvoCUAAgent
# ========== PE 框架新增导入 ==========
from .pe import PlannerAgent, ExecutorAgent, PEAgent


def create_agent(args):
    """
    Agent 创建工厂函数
    
    根据 args.agent_type 创建对应的 Agent 实例。这是评估和训练的统一入口。
    返回 Agent 实例（AguvisAgent / UITARSAgent / PEAgent / PlannerAgent / ExecutorAgent）
    """
    use_fast = not getattr(args, "disable_fast_tokenizer", False)
    if args.agent_type == 'aguvis':
        # AguVis Agent: 使用 Planning + Execution 格式
        # 输出格式：Action: ... + pyautogui 代码
        agent = AguvisAgent(
            tokenizer_path=args.pretrain,
            history_n=args.num_history,
            screen_size=(args.screen_width, args.screen_height),
        )
    elif args.agent_type == 'uitars':
        # UI-TARS Agent: 使用 Thought + Action 格式
        # 输出格式：Thought: ... Action: ...
        # 支持 computer 和 mobile 两种动作空间
        agent = UITARSAgent(
            tokenizer_path=args.pretrain,
            max_trajectory_length=args.agent_max_steps,
            history_n=args.num_history,
            text_history_n=getattr(args, 'num_text_history', None),
            screen_size=(args.screen_width, args.screen_height),
            action_space=args.agent_action_space,
            language=args.agent_prompt_language,
            gt_todo_path=getattr(args, 'gt_todo_path', None),
            enable_crop=getattr(args, 'enable_crop', False),
            use_fast=use_fast,
        )
    # ========== Fara-7B Agent ==========
    elif args.agent_type == 'fara':
        # Fara-7B: Microsoft 的 Computer Use Agent
        # 使用 function calling 格式（<tool_call> XML 标签）
        agent = FaraAgent(
            tokenizer_path=args.pretrain,
            max_trajectory_length=args.agent_max_steps,
            history_n=args.num_history,
            screen_size=(args.screen_width, args.screen_height),
            action_space=args.agent_action_space,
            language=getattr(args, 'agent_prompt_language', 'English'),
            use_fast=use_fast,
        )
    elif args.agent_type == 'mai':
        # MAI-UI: Qwen3-VL 系列 GUI Agent
        # 使用 function calling 格式（<tool_call> XML 标签）
        agent = MAIAgent(
            tokenizer_path=args.pretrain,
            max_trajectory_length=args.agent_max_steps,
            history_n=args.num_history,
            screen_size=(args.screen_width, args.screen_height),
            action_space=args.agent_action_space,
            language=getattr(args, 'agent_prompt_language', 'English'),
            use_fast=use_fast,
        )
    elif args.agent_type in {'evocua', 'EvoCUA'}:
        # EvoCUA: 官方 S2 工具调用风格（Action + <tool_call>）
        agent = EvoCUAAgent(
            tokenizer_path=args.pretrain,
            max_trajectory_length=args.agent_max_steps,
            history_n=args.num_history,
            screen_size=(args.screen_width, args.screen_height),
            action_space=args.agent_action_space,
            language=getattr(args, 'agent_prompt_language', 'English'),
            coordinate_type=getattr(args, 'coordinate_type', 'relative'),
            resize_factor=getattr(args, 'resize_factor', 32),
            use_fast=use_fast,
        )
    # ========== PE 框架：Subtask Planner / Executor ==========
    elif args.agent_type == 'planner':
        planner_path = getattr(args, 'planner_pretrain', args.pretrain)
        agent = PlannerAgent(
            tokenizer_path=planner_path,
            history_n=args.num_history,
            screen_size=(args.screen_width, args.screen_height),
            language=getattr(args, 'agent_prompt_language', 'Chinese'),
        )
    elif args.agent_type == 'executor':
        executor_path = getattr(args, 'executor_pretrain', args.pretrain)
        agent = ExecutorAgent(
            tokenizer_path=executor_path,
            max_trajectory_length=args.agent_max_steps,
            history_n=args.num_history,
            screen_size=(args.screen_width, args.screen_height),
            action_space=args.agent_action_space,
            language=getattr(args, 'agent_prompt_language', 'Chinese'),
        )
    else:
        raise NotImplementedError(f"Agent type '{args.agent_type}' not implemented")
    return agent
