# -*- coding: utf-8 -*-
"""
prefix_ttft_lab.py — 生产栈 prompt/prefix cache TTFT 对比实验
llama-cpp-python 0.3.35 (CUDA) / RTX 3060 6GB / gemma-4-E4B-it-Q4_K_M.gguf

协议（单次模型加载内完成）：
  warmup:  reset+clear -> eval(P+Q1) -> sample 1        （冷启动首轮，兼作参考）
  每轮 trial i:
    cache-off: reset()+kv_cache_clear() -> 计时[ eval(P+Q2) -> sample 1 ]   # 真·从头重算
    cache-on : n_tokens=len(P) + kv_cache_seq_rm 截断 -> 计时[ eval(Q2) -> sample 1 ]
               # 第二轮只算新增 token，KV 中保留同一长前缀（= llama.cpp server / vLLM APC 的原地前缀复用语义）
TTFT = 从提交第二轮输入到采出第一个 token 的墙钟时间。
"""
import os, sys, time, ctypes, argparse, json, subprocess, statistics, traceback

# --- CUDA DLL 预载（照抄 gpu-test/test_gpu_load.py 头部） ---
BASE = r"C:\temp\g4env-gpu\Lib\site-packages"
CU = os.path.join(BASE, "nvidia", "cublas", "bin")
RT = os.path.join(BASE, "nvidia", "cuda_runtime", "bin")
os.add_dll_directory(CU)
os.add_dll_directory(RT)
for f in ["cudart64_12.dll"]:
    ctypes.CDLL(os.path.join(RT, f))
for f in ["cublas64_12.dll", "cublasLt64_12.dll"]:
    ctypes.CDLL(os.path.join(CU, f))

import llama_cpp
from llama_cpp import Llama

MODEL = r"C:\temp\g4models\gemma-4-E4B-it-Q4_K_M.gguf"
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")

PARA = ("西安高新区的这家智能制造企业成立于2012年，主要从事工业机器人核心部件的研发与生产。"
        "公司现有员工八百余人，其中研发人员占比超过四成。过去五年，公司营业收入年均增长百分之二十三，"
        "主要产品销往东南亚与欧洲市场。2024年，公司启动新一轮数字化转型，引入视觉检测与预测性维护系统，"
        "并与三所高校共建联合实验室，重点攻关高精度减速器的国产替代。管理层认为，未来三年的关键变量包括"
        "原材料价格、海外订单稳定性以及高端人才引进。\n")

Q1 = "<start_of_turn>user\n第一轮：用一句话概括上文的核心观点。<end_of_turn>\n<start_of_turn>model\n"
Q2 = "<start_of_turn>user\n第二轮：请基于上文给出一条可执行的建议。<end_of_turn>\n<start_of_turn>model\n"


def vram_used_mb():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception as e:
        return f"nvidia-smi failed: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ctx", type=int, default=6144)
    ap.add_argument("--gpu-layers", type=int, default=-1)
    ap.add_argument("--prefix-tokens", type=int, default=3200)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    res = {
        "config": {
            "model": MODEL, "n_ctx": args.n_ctx, "n_gpu_layers": args.gpu_layers,
            "prefix_target_tokens": args.prefix_tokens, "trials": args.trials,
            "llama_cpp_version": llama_cpp.__version__,
            "gpu_offload_supported": bool(llama_cpp.llama_supports_gpu_offload()),
        },
        "vram_before_load": vram_used_mb(),
        "warmup": None, "rounds": [],
    }

    def flush():
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    t0 = time.time()
    llm = Llama(model_path=MODEL, n_ctx=args.n_ctx, n_gpu_layers=args.gpu_layers,
                use_mmap=True, verbose=True, seed=42)
    res["config"]["load_time_s"] = round(time.time() - t0, 2)
    res["vram_after_load"] = vram_used_mb()
    print(f"\n=== LOAD {res['config']['load_time_s']}s | VRAM before={res['vram_before_load']} after={res['vram_after_load']} ===", flush=True)

    P = llm.tokenize(PARA.encode("utf-8"), add_bos=True)
    while len(P) < args.prefix_tokens:
        P += llm.tokenize(PARA.encode("utf-8"), add_bos=False)
    P = P[: args.prefix_tokens]
    q1 = llm.tokenize(Q1.encode("utf-8"), add_bos=False)
    q2 = llm.tokenize(Q2.encode("utf-8"), add_bos=False)
    np_, n1, n2 = len(P), len(q1), len(q2)
    assert np_ + max(n1, n2) + 8 < args.n_ctx, "ctx too small"
    res["config"].update({"prefix_tokens": np_, "q1_tokens": n1, "q2_tokens": n2})
    print(f"=== TOKENS prefix={np_} q1={n1} q2={n2} | off 重算量={np_ + n2} on 计算量={n2} ===", flush=True)
    flush()

    def sample_one():
        return llm.sample()

    llm.reset()
    llm._ctx.kv_cache_clear()
    t0 = time.perf_counter()
    llm.eval(P + q1)
    sample_one()
    warm = time.perf_counter() - t0
    res["warmup"] = {"round1_cold_s": round(warm, 4),
                     "tokens": np_ + n1,
                     "prefill_tps": round((np_ + n1) / warm, 1)}
    print(f"=== WARMUP round1 cold: {warm:.3f}s ({(np_ + n1) / warm:.1f} tok/s) ===", flush=True)
    flush()

    for i in range(1, args.trials + 1):
        llm.reset()
        llm._ctx.kv_cache_clear()
        t0 = time.perf_counter()
        llm.eval(P + q2)
        sample_one()
        off = time.perf_counter() - t0

        llm.n_tokens = np_
        llm._ctx.kv_cache_seq_rm(-1, np_, -1)
        t0 = time.perf_counter()
        llm.eval(q2)
        sample_one()
        on = time.perf_counter() - t0

        row = {"trial": i, "ttft_off_s": round(off, 4), "ttft_on_s": round(on, 4),
               "speedup": round(off / on, 2) if on > 0 else None,
               "off_prefill_tps": round((np_ + n2) / off, 1)}
        res["rounds"].append(row)
        print(f"=== TRIAL {i}: off={off:.3f}s on={on:.4f}s speedup={row['speedup']}x "
              f"(off prefill {row['off_prefill_tps']} tok/s) ===", flush=True)
        flush()

    offs = [r["ttft_off_s"] for r in res["rounds"]]
    ons = [r["ttft_on_s"] for r in res["rounds"]]
    res["summary"] = {
        "ttft_off_median_s": round(statistics.median(offs), 4),
        "ttft_on_median_s": round(statistics.median(ons), 4),
        "speedup_median_x": round(statistics.median(offs) / statistics.median(ons), 2),
        "token_ratio_off_over_on": round((np_ + n2) / n2, 1),
    }
    flush()
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(res["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
