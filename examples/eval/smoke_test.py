"""
Smoke test: verify imports, agent creation, and vLLM model loading.
Does NOT require a running OSWorld environment.
"""
import sys
import time
import argparse

def main():
    print("=" * 60)
    print("SMOKE TEST: LLaMA-Factory OSWorld Eval Pipeline")
    print("=" * 60)

    # 1) Import test
    print("\n[1/4] Testing imports...", flush=True)
    t0 = time.time()
    try:
        from osworld.env import create_env, RemoteDesktopEnv
        from osworld.agent import create_agent, UITARSAgent, MAIAgent
        from osworld.eval_osworld import run_single_task
        from vllm import LLM, SamplingParams
        print(f"  OK  imports passed ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  FAIL  import error: {e}")
        sys.exit(1)

    # 2) Agent creation test
    print("\n[2/4] Testing agent creation (MAI / Qwen3-VL)...", flush=True)
    t0 = time.time()
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not model_path:
        print("  SKIP  no model path provided (pass as first argument)")
    else:
        try:
            args = argparse.Namespace(
                pretrain=model_path,
                agent_type="mai",
                agent_max_steps=5,
                num_history=3,
                screen_width=1920,
                screen_height=1080,
                agent_action_space="computer",
                agent_prompt_language="Chinese",
                disable_fast_tokenizer=False,
            )
            agent = create_agent(args)
            print(f"  OK  MAIAgent created ({time.time()-t0:.1f}s)")
            print(f"       type={type(agent).__name__}")
        except Exception as e:
            print(f"  FAIL  agent creation error: {e}")
            import traceback; traceback.print_exc()
            sys.exit(2)

    # 3) vLLM engine loading test
    print("\n[3/4] Testing vLLM engine loading...", flush=True)
    if not model_path:
        print("  SKIP  no model path")
    else:
        t0 = time.time()
        try:
            llm = LLM(
                model=model_path,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,
                enforce_eager=True,
                max_model_len=4096,
                limit_mm_per_prompt={"image": 3},
            )
            print(f"  OK  vLLM engine loaded ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  FAIL  vLLM error: {e}")
            import traceback; traceback.print_exc()
            sys.exit(3)

    # 4) Dummy inference test (text-only, no image)
    print("\n[4/4] Testing dummy inference...", flush=True)
    if not model_path:
        print("  SKIP  no model path")
    else:
        t0 = time.time()
        try:
            sp = SamplingParams(temperature=0.0, max_tokens=32)
            out = llm.generate(["Hello, I am a"], sp)
            text = out[0].outputs[0].text[:80]
            print(f"  OK  generation works ({time.time()-t0:.1f}s)")
            print(f"       output: {text!r}")
        except Exception as e:
            print(f"  FAIL  generation error: {e}")
            import traceback; traceback.print_exc()
            sys.exit(4)

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
