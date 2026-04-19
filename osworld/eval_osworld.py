# =============================================================================
# eval_osworld.py - GUI Agent OSWorld 评测入口（LLaMA-Factory 版）
# =============================================================================
# 支持多 worker 并行评测：每个 worker 独占 1 GPU + 1 env + 1 agent。
# 无 Ray 依赖，使用 Python multiprocessing。
# =============================================================================

import os
import sys
import json
import copy
import time
import signal
import argparse
import datetime
import multiprocessing as mp
from tqdm import tqdm


def _str2bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "y", "on"):
        return True
    if text in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def run_single_task(llm, agent, env, env_config, task_meta, sampling_params, save_dir):
    """执行单个 OSWorld 任务: reset → (infer → act)循环 → evaluate"""
    from vllm import SamplingParams as _SP

    _task_start = time.perf_counter()
    _env_reset_ms = _env_step_ms = _llm_generate_ms = _eval_ms = 0.0

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    example_id = task_meta.get("example_id", "")
    task_save_dir = None
    if save_dir:
        task_save_dir = os.path.join(save_dir, f"{timestamp}_{example_id}")
        os.makedirs(task_save_dir, exist_ok=True)

    agent.reset()

    try:
        task_config = env.get_task_config(**task_meta)
        instruction = task_config["instruction"]
    except Exception as e:
        print(f"[ERROR] get_task_config failed for {example_id}: {e!r}")
        return {"reward": -1, "num_steps": 0, "is_valid": False, "error": str(e)}

    _t0 = time.perf_counter()
    obs = env.reset(task_config=task_config)
    _env_reset_ms = (time.perf_counter() - _t0) * 1000

    if obs is None:
        print(f"[ERROR] env.reset failed for {example_id}")
        return {"reward": -1, "num_steps": 0, "is_valid": False, "error": "env reset failed"}

    if task_save_dir:
        with open(os.path.join(task_save_dir, "step_0.png"), "wb") as f:
            f.write(obs["screenshot"])
        with open(os.path.join(task_save_dir, "config.json"), "w") as f:
            json.dump(task_config, f, indent=4)

    done = False
    is_step_error = False
    step_idx = 0
    traj_for_eval = {"screenshots": [obs["screenshot"]], "actions": []}

    while not done and step_idx < env_config["max_steps"]:
        if obs is not None:
            obs["example_id"] = example_id

        inputs = agent.get_model_inputs(instruction, obs)

        _t0 = time.perf_counter()
        outputs = llm.generate(
            [{"prompt": inputs["prompt"], "multi_modal_data": inputs["multi_modal_data"]}],
            sampling_params=sampling_params,
        )
        _step_llm_ms = (time.perf_counter() - _t0) * 1000
        _llm_generate_ms += _step_llm_ms
        response = outputs[0].outputs[0].text

        actions = agent.parse_action(response)

        if task_save_dir:
            with open(os.path.join(task_save_dir, "model_output.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "parsed_action": actions,
                    "output_text": response,
                }, ensure_ascii=False) + "\n")

        for action in actions:
            pause = float(env_config.get("step_pause", 2))
            _t0 = time.perf_counter()
            obs, reward, done, info = env.step(action, pause=pause)
            _action_env_ms = (time.perf_counter() - _t0) * 1000
            _env_step_ms += _action_env_ms

            if obs is None and done:
                is_step_error = True
                break

            traj_for_eval["screenshots"].append(obs["screenshot"])
            traj_for_eval["actions"].append(action)

            if task_save_dir:
                with open(os.path.join(task_save_dir, f"step_{step_idx + 1}.png"), "wb") as f:
                    f.write(obs["screenshot"])

            if done:
                break

        step_idx += 1

    if is_step_error:
        result = -1
        eval_outputs = {"reward": -1, "error_type": "osworld_action_failed"}
    else:
        _t0 = time.perf_counter()
        eval_outputs = env.evaluate(task_config, traj_for_eval)
        _eval_ms = (time.perf_counter() - _t0) * 1000
        result = eval_outputs["reward"]

    _total_ms = (time.perf_counter() - _task_start) * 1000
    _profile = {
        "env_reset_ms": _env_reset_ms,
        "env_step_ms": _env_step_ms,
        "llm_generate_ms": _llm_generate_ms,
        "eval_ms": _eval_ms,
        "total_ms": _total_ms,
    }

    if task_save_dir:
        with open(os.path.join(task_save_dir, "eval.json"), "w") as f:
            json.dump(eval_outputs, f, indent=4)
        with open(os.path.join(task_save_dir, "profile.json"), "w") as f:
            json.dump(_profile, f, indent=4)

    return {
        "reward": float(result),
        "num_steps": step_idx,
        "is_valid": result >= 0,
        "profile": _profile,
    }


# =====================================================================
# Worker process: each worker owns 1 GPU + 1 vLLM engine + 1 env
# =====================================================================

def _worker_loop(worker_id, gpu_id, args_dict, task_queue, result_queue):
    """在独立进程中运行：加载模型、创建 env，循环消费任务队列。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import LLM, SamplingParams
    from osworld.env import create_env
    from osworld.agent import create_agent

    args = argparse.Namespace(**args_dict)

    env, env_config = create_env(args, env_idx=worker_id)
    agent = create_agent(args)

    engine_kwargs = {
        "model": args.pretrain,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "limit_mm_per_prompt": {"image": args.num_input_image},
    }
    if args.max_model_len:
        engine_kwargs["max_model_len"] = args.max_model_len

    print(f"[worker-{worker_id}] Loading model on GPU {gpu_id}...", flush=True)
    llm = LLM(**engine_kwargs)
    print(f"[worker-{worker_id}] Ready", flush=True)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        frequency_penalty=args.frequency_penalty,
        max_tokens=1024,
    )

    task_timeout = int(os.environ.get("OSWORLD_TASK_TIMEOUT_SEC", "600"))

    while True:
        item = task_queue.get()
        if item is None:
            break
        task_meta, save_dir, max_retries = item
        example_id = task_meta.get("example_id", "")

        best_reward = -1
        best_outputs = None
        for attempt in range(max_retries):
            _t0 = time.perf_counter()
            try:
                outputs = run_single_task(llm, agent, env, env_config, task_meta, sampling_params, save_dir=save_dir)
            except Exception as e:
                elapsed = time.perf_counter() - _t0
                print(f"[worker-{worker_id}] Task {example_id} attempt {attempt} crashed after {elapsed:.0f}s: {e!r}", flush=True)
                outputs = {"reward": -1, "num_steps": 0, "is_valid": False, "error": str(e)}
            elapsed = time.perf_counter() - _t0
            if elapsed > task_timeout:
                print(f"[worker-{worker_id}] Task {example_id} attempt {attempt} took {elapsed:.0f}s (>{task_timeout}s), skipping retries", flush=True)
                if outputs["reward"] > best_reward:
                    best_reward = outputs["reward"]
                    best_outputs = outputs
                break
            if outputs["reward"] > best_reward:
                best_reward = outputs["reward"]
                best_outputs = outputs
            if outputs["reward"] == 1:
                break

        if best_outputs is None:
            best_outputs = {"reward": -1, "num_steps": 0, "is_valid": False, "error": "all attempts failed"}
        print(f"[worker-{worker_id}] Finished {example_id}: reward={best_outputs['reward']}", flush=True)
        result_queue.put((task_meta, best_outputs))

    try:
        env.close()
    except Exception:
        pass


# =====================================================================
# Main
# =====================================================================

def _load_finished_ids(save_dir, domain_dirs):
    """从已有 task_info.jsonl 读取已完成的 example_id。"""
    finished_ids = set()
    paths = [os.path.join(save_dir, "task_info.jsonl")]
    for dd in domain_dirs.values():
        paths.append(os.path.join(dd, "task_info.jsonl"))
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        eid = json.loads(line).get("example_id")
                        if eid:
                            finished_ids.add(eid)
                    except Exception:
                        continue
        except Exception:
            pass
    return finished_ids


def main(args):
    # Determine number of workers
    cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = [x.strip() for x in cuda_devs.split(",") if x.strip()] if cuda_devs else ["0"]

    if args.vllm_tensor_parallel_size > 1:
        num_workers = 1
    else:
        num_workers = min(len(gpu_ids), getattr(args, "num_eval_workers", len(gpu_ids)))
    num_workers = max(1, num_workers)

    print(f"[eval] num_workers={num_workers}, GPUs={gpu_ids[:num_workers]}", flush=True)

    # Load task list
    with open(args.data_path, "r") as f:
        task_metas = [json.loads(line) for line in f if line.strip()]

    # Domain dirs
    domain_dirs = {}
    for meta in task_metas:
        domain = meta.get("domain")
        if domain and domain not in domain_dirs:
            domain_dir = os.path.join(args.save_dir, domain)
            os.makedirs(domain_dir, exist_ok=True)
            domain_dirs[domain] = domain_dir

    # Resume
    finished_ids = _load_finished_ids(args.save_dir, domain_dirs)
    filtered = [m for m in task_metas if m.get("example_id") not in finished_ids]

    if finished_ids:
        print(f"[INFO] Found {len(finished_ids)} finished tasks, will skip them.")
    if not filtered:
        print("[INFO] All tasks already evaluated. Nothing to do.")
        return

    print(f"[INFO] Will evaluate {len(filtered)} tasks (total {len(task_metas)}).", flush=True)

    result_file = os.path.join(args.save_dir, "task_info.jsonl")

    def save_task_result(dump_info, domain):
        with open(result_file, "a") as f:
            f.write(json.dumps(dump_info) + "\n")
        if domain and domain in domain_dirs:
            domain_result_file = os.path.join(domain_dirs[domain], "task_info.jsonl")
            with open(domain_result_file, "a") as f:
                f.write(json.dumps(dump_info) + "\n")

    # ========== Single worker: simple sequential loop ==========
    if num_workers == 1:
        from vllm import LLM, SamplingParams
        from osworld.env import create_env
        from osworld.agent import create_agent

        env, env_config = create_env(args)
        agent = create_agent(args)

        print(f"[eval] Loading model: {args.pretrain}", flush=True)
        engine_kwargs = {
            "model": args.pretrain,
            "trust_remote_code": True,
            "tensor_parallel_size": args.vllm_tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": True,
            "limit_mm_per_prompt": {"image": args.num_input_image},
        }
        if args.max_model_len:
            engine_kwargs["max_model_len"] = args.max_model_len

        llm = LLM(**engine_kwargs)
        print("[eval] vLLM engine ready", flush=True)

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            frequency_penalty=args.frequency_penalty,
            max_tokens=1024,
        )

        task_timeout = int(os.environ.get("OSWORLD_TASK_TIMEOUT_SEC", "600"))
        task_acc = []
        for task_meta in tqdm(filtered, desc="Evaluating"):
            domain = task_meta.get("domain", "")
            example_id = task_meta.get("example_id", "")
            domain_save_dir = domain_dirs.get(domain, args.save_dir) if domain else args.save_dir

            best_reward = -1
            best_outputs = None
            for attempt in range(args.max_retries):
                _t0 = time.perf_counter()
                try:
                    outputs = run_single_task(llm, agent, env, env_config, task_meta, sampling_params, save_dir=domain_save_dir)
                except Exception as e:
                    elapsed = time.perf_counter() - _t0
                    print(f"[eval] Task {example_id} attempt {attempt} crashed after {elapsed:.0f}s: {e!r}", flush=True)
                    outputs = {"reward": -1, "num_steps": 0, "is_valid": False, "error": str(e)}
                elapsed = time.perf_counter() - _t0
                if elapsed > task_timeout:
                    print(f"[eval] Task {example_id} took {elapsed:.0f}s (>{task_timeout}s), skipping retries", flush=True)
                    if outputs["reward"] > best_reward:
                        best_reward = outputs["reward"]
                        best_outputs = outputs
                    break
                if outputs["reward"] > best_reward:
                    best_reward = outputs["reward"]
                    best_outputs = outputs
                if outputs["reward"] == 1:
                    break

            if best_outputs is None:
                best_outputs = {"reward": -1, "num_steps": 0, "is_valid": False, "error": "all attempts failed"}
            task_acc.append(best_outputs["reward"])
            dump_info = copy.deepcopy(task_meta)
            dump_info.update({"timestamp": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), "acc": best_outputs["reward"]})
            save_task_result(dump_info, domain)

        try:
            env.close()
        except Exception:
            pass

    # ========== Multi worker: parallel processes ==========
    else:
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()

        args_dict = vars(args)

        workers = []
        for i in range(num_workers):
            p = ctx.Process(
                target=_worker_loop,
                args=(i, gpu_ids[i], args_dict, task_queue, result_queue),
            )
            p.start()
            workers.append(p)

        def _kill_workers(signum=None, frame=None):
            for p in workers:
                if p.is_alive():
                    p.terminate()
            if signum is not None:
                sys.exit(1)

        signal.signal(signal.SIGTERM, _kill_workers)
        signal.signal(signal.SIGINT, _kill_workers)

        try:
            for task_meta in filtered:
                domain = task_meta.get("domain", "")
                domain_save_dir = domain_dirs.get(domain, args.save_dir) if domain else args.save_dir
                task_queue.put((task_meta, domain_save_dir, args.max_retries))

            for _ in workers:
                task_queue.put(None)

            task_acc = []
            pbar = tqdm(total=len(filtered), desc=f"Evaluating ({num_workers} workers)")
            collected = 0
            while collected < len(filtered):
                task_meta, outputs = result_queue.get()
                task_acc.append(outputs["reward"])
                domain = task_meta.get("domain", "")
                dump_info = copy.deepcopy(task_meta)
                dump_info.update({"timestamp": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), "acc": outputs["reward"]})
                save_task_result(dump_info, domain)
                collected += 1
                pbar.update(1)
            pbar.close()
        finally:
            for p in workers:
                p.join(timeout=60)
                if p.is_alive():
                    p.terminate()

    # Summary
    valid_acc = [r for r in task_acc if r >= 0]
    n_invalid = len(task_acc) - len(valid_acc)
    avg = sum(valid_acc) / len(valid_acc) if valid_acc else 0.0
    print(f"\nAverage success rate: {avg:.4f} ({len(valid_acc)} valid, {n_invalid} invalid)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSWorld Evaluation (vLLM, multi-worker)")

    parser.add_argument("--data_path", type=str, default="./data/osworld_test_all.jsonl")
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--num_eval_workers", type=int, default=None,
                        help="并行 worker 数（默认=GPU 数量，TP>1 时强制为 1）")

    parser.add_argument("--env_type", type=str, default="osworld")
    parser.add_argument("--env_url", type=str, default=None)
    parser.add_argument("--env_port", type=int, default=None)
    parser.add_argument("--env_manager_port", type=int, default=None)
    parser.add_argument("--action_space", type=str, default="pyautogui")
    parser.add_argument("--observation_type", choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"], default="screenshot")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--agent_max_steps", "--max_steps", dest="agent_max_steps", type=int, default=50)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--save_trajectory", action="store_true", default=False)
    parser.add_argument("--use_llm_evaluator", action="store_true", default=False)
    parser.add_argument("--test_task_llm_eval", action="store_true", default=False)

    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--agent_type", type=str, default="uitars")
    parser.add_argument("--agent_action_space", type=str, default="computer")
    parser.add_argument("--agent_prompt_language", type=str, default="Chinese")
    parser.add_argument("--num_history", type=int, default=5)
    parser.add_argument("--num_text_history", type=int, default=None)
    parser.add_argument("--gt_todo_path", type=str, default=None)
    parser.add_argument("--enable_crop", nargs="?", const=True, default=False, type=_str2bool)
    parser.add_argument("--num_input_image", type=int, default=5)

    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=1.0)

    args = parser.parse_args()

    model_name = os.path.basename(args.pretrain)
    args.save_dir = os.path.join(args.save_path, "eval", model_name)
    os.makedirs(args.save_dir, exist_ok=True)

    args_json_path = os.path.join(args.save_dir, "args.json")
    with open(args_json_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"[INFO] Args saved to: {args_json_path}")

    main(args)
