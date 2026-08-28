# -*- coding: utf-8 -*-
"""
kv_cache_lab.py — KV cache (past_key_values) vs 无 cache 全量重前向 对比实验
路线 A：distilgpt2 / fp32 / CPU-only / batch=1

用法（venv 在 C:\\temp\\hfenv）：
  python kv_cache_lab.py                 # 主实验：100+ token 中文 prompt，生成 32 token
  python kv_cache_lab.py --sweep         # 加分项：prompt 长度 16/64/256 扫描
  python kv_cache_lab.py --gen-tokens 32 --runs 3 --prompt-len 128
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 模型下载走镜像
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# huggingface_hub 直连镜像 HEAD 校验失败，改为 curl 预下载 + 本地路径加载；
# 若本地目录不存在则回退到 HF_ENDPOINT 镜像在线下载。
_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "distilgpt2")
MODEL_ID = _LOCAL if os.path.exists(os.path.join(_LOCAL, "model.safetensors")) else "distilgpt2"

# 固定中文 prompt（distilgpt2 的 GPT-2 BPE 对中文按字节切分，这段约 130+ token）
PROMPT = (
    "Transformer 模型的核心是自注意力机制。在自回归生成时，模型每产生一个新词，"
    "都要重新计算整段上下文的注意力。如果每次都把已经算过的键和值丢弃，"
    "那么计算量会随着序列长度迅速增长，造成大量重复劳动。键值缓存的做法是"
    "把每一层历史词的键和值保留下来，下一步只为新词计算注意力，"
    "这样每步的计算量近似恒定， decoding 速度因此大幅提升。"
)


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    model.to("cpu")
    return tok, model


def sync():
    pass  # CPU 无需同步，占位以便与 GPU 版脚本对齐


@torch.inference_mode()
def prefill(model, input_ids):
    """一次性全量前向，返回 (耗时秒, next_token, past_key_values)"""
    sync()
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, use_cache=True)
    sync()
    dt = time.perf_counter() - t0
    next_tok = int(out.logits[0, -1].argmax())
    return dt, next_tok, out.past_key_values


@torch.inference_mode()
def decode_cached(model, first_tok, past, n_steps):
    """带 cache 逐 token 解码 n_steps 步，返回 (每步耗时列表, 生成的token列表, 最终past)"""
    step_times, tokens = [], []
    cur = torch.tensor([[first_tok]], dtype=torch.long)
    for _ in range(n_steps):
        sync()
        t0 = time.perf_counter()
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        sync()
        step_times.append(time.perf_counter() - t0)
        past = out.past_key_values
        cur = torch.tensor([[int(out.logits[0, -1].argmax())]], dtype=torch.long)
        tokens.append(int(cur[0, 0]))
    return step_times, tokens, past


@torch.inference_mode()
def decode_nocache(model, prompt_ids, first_tok, n_steps):
    """无 cache：每步把全序列重新前向一遍，返回 (每步耗时列表, 生成的token列表)"""
    seq = list(prompt_ids) + [first_tok]
    step_times, tokens = [], []
    for _ in range(n_steps):
        ids = torch.tensor([seq], dtype=torch.long)
        sync()
        t0 = time.perf_counter()
        out = model(input_ids=ids, use_cache=False)
        sync()
        step_times.append(time.perf_counter() - t0)
        nxt = int(out.logits[0, -1].argmax())
        seq.append(nxt)
        tokens.append(nxt)
    return step_times, tokens


def run_once(model, prompt_ids, gen_tokens):
    """完整跑一轮：prefill + 两侧各生成 gen_tokens 个 token"""
    pf_t, tok1, past = prefill(model, prompt_ids)
    c_steps, c_toks, _ = decode_cached(model, tok1, past, gen_tokens - 1)
    n_steps, n_toks = decode_nocache(model, list(prompt_ids[0]), tok1, gen_tokens - 1)
    return {
        "prefill_s": pf_t,
        "cached_steps": c_steps,
        "nocache_steps": n_steps,
        "match": ([tok1] + c_toks) == ([tok1] + n_toks),
        "gen_text": tok1,
        "cached_toks": [tok1] + c_toks,
        "nocache_toks": [tok1] + n_toks,
    }


def experiment(model, tok, prompt_ids, gen_tokens, runs, label=""):
    L = len(prompt_ids[0])
    run_once(model, prompt_ids, min(4, gen_tokens))
    results = [run_once(model, prompt_ids, gen_tokens) for _ in range(runs)]

    prefill_ms = statistics.median(r["prefill_s"] for r in results) * 1e3
    cached_all = [t for r in results for t in r["cached_steps"]]
    nocache_all = [t for r in results for t in r["nocache_steps"]]
    cached_ms = statistics.median(cached_all) * 1e3
    nocache_ms = statistics.median(nocache_all) * 1e3
    match = all(r["match"] for r in results)

    tot_cached = prefill_ms + cached_ms * (gen_tokens - 1)
    tot_nocache = prefill_ms + nocache_ms * (gen_tokens - 1)

    print(f"\n=== {label or 'main'} | prompt={L} tok, gen={gen_tokens} tok, runs={runs} ===")
    print(f"prefill (全量首前向)      : {prefill_ms:8.1f} ms")
    print(f"增量解码 (cache, ms/tok)  : {cached_ms:8.2f} ms")
    print(f"无 cache 重前向 (ms/tok)  : {nocache_ms:8.2f} ms")
    print(f"单步加速比 (nocache/cache): {nocache_ms / cached_ms:8.1f} x")
    print(f"总耗时 cache / nocache    : {tot_cached:8.0f} ms / {tot_nocache:.0f} ms"
          f"  (总比 {tot_nocache / tot_cached:.1f} x)")
    print(f"两侧贪心生成序列一致      : {match}")
    return {
        "prompt_len": L, "gen": gen_tokens, "runs": runs,
        "prefill_ms": prefill_ms, "cached_ms": cached_ms,
        "nocache_ms": nocache_ms, "speedup_step": nocache_ms / cached_ms,
        "total_cached_ms": tot_cached, "total_nocache_ms": tot_nocache,
        "match": match,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--prompt-len", type=int, default=0, help=">0 时把 prompt 截/循环到指定 token 数")
    ap.add_argument("--sweep", action="store_true", help="prompt 长度扫描 16/64/256")
    args = ap.parse_args()

    print(f"torch={torch.__version__} threads={torch.get_num_threads()} cuda={torch.cuda.is_available()}")
    import transformers
    print(f"transformers={transformers.__version__}")

    tok, model = load_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={MODEL_ID} params={n_params / 1e6:.1f}M dtype=fp32 device=cpu")

    base_ids = tok(PROMPT, return_tensors="pt").input_ids
    print(f"固定中文 prompt 长度: {base_ids.shape[1]} tokens")

    summaries = []

    if args.sweep:
        for L in (16, 64, 256):
            ids = base_ids
            while ids.shape[1] < L:
                ids = torch.cat([ids, base_ids], dim=1)
            ids = ids[:, :L]
            summaries.append(experiment(model, tok, ids, args.gen_tokens, args.runs,
                                        label=f"sweep L={L}"))
    else:
        ids = base_ids
        if args.prompt_len > 0:
            while ids.shape[1] < args.prompt_len:
                ids = torch.cat([ids, base_ids], dim=1)
            ids = ids[:, :args.prompt_len]
        summaries.append(experiment(model, tok, ids, args.gen_tokens, args.runs))

    print("\n--- SUMMARY ---")
    print("prompt_len,gen,prefill_ms,cached_ms_per_tok,nocache_ms_per_tok,step_speedup,total_speedup,match")
    for s in summaries:
        ts = s["total_nocache_ms"] / s["total_cached_ms"]
        print(f"{s['prompt_len']},{s['gen']},{s['prefill_ms']:.1f},{s['cached_ms']:.2f},"
              f"{s['nocache_ms']:.2f},{s['speedup_step']:.1f},{ts:.1f},{s['match']}")


if __name__ == "__main__":
    main()
