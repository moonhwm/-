# -*- coding: utf-8 -*-
"""prefix_reuse_lab.py — 幻16 真机验证「KV 缓存前缀复用」报告结论
场景：同一长前缀多轮对话。A 路径：每轮全量重算（前缀+新增）；B 路径：前缀 prefill 一次存 cache，每轮只算新增。
注意：distilgpt2 n_positions=1024，原报告 N=2048 超窗，本实验用 N=512, M=64。
venv: C:\\temp\\hfenv（torch 2.13.0+cpu + transformers 5.16.1，模型在 ../kv-cache-lab/models/distilgpt2）"""
import os
import statistics
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "distilgpt2")
N_PREFIX = 512
M_NEW = 64
ROUNDS = 10
RUNS = 3

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval().to("cpu")

base = tok("西安是中国历史上建都朝代最多的城市之一，" * 60, return_tensors="pt").input_ids
while base.shape[1] < N_PREFIX:
    base = torch.cat([base, base], dim=1)
prefix_ids = base[:, :N_PREFIX]
new_ids = tok("请接着上文继续介绍它的历史地位。" * 8, return_tensors="pt").input_ids
while new_ids.shape[1] < M_NEW:
    new_ids = torch.cat([new_ids, new_ids], dim=1)
new_ids = new_ids[:, :M_NEW]
print(f"prefix={prefix_ids.shape[1]} tok, new/round={new_ids.shape[1]} tok, rounds={ROUNDS}")


@torch.inference_mode()
def full_forward(ids):
    t0 = time.perf_counter()
    model(input_ids=ids, use_cache=False)
    return time.perf_counter() - t0


@torch.inference_mode()
def prefill(ids):
    t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True)
    return time.perf_counter() - t0, out.past_key_values


@torch.inference_mode()
def incremental(ids, past):
    t0 = time.perf_counter()
    model(input_ids=ids, past_key_values=past, use_cache=True)
    return time.perf_counter() - t0


full_forward(prefix_ids[:, :64])

a_times = []
for _ in range(RUNS):
    per_round = statistics.median(
        full_forward(torch.cat([prefix_ids, new_ids], dim=1)) for _ in range(3)
    )
    a_times.append(per_round)
a_per_round = statistics.median(a_times)

b_prefill, past0 = prefill(prefix_ids)
b_inc = statistics.median(incremental(new_ids, past0) for _ in range(RUNS))

print(f"\nA 无缓存每轮全量重算: {a_per_round*1e3:.1f} ms/轮")
print(f"B 前缀 prefill 一次性: {b_prefill*1e3:.1f} ms")
print(f"B 增量（复用缓存）   : {b_inc*1e3:.1f} ms/轮")
print(f"每轮节省: {(1-b_inc/a_per_round)*100:.1f}%  |  单轮加速: {a_per_round/b_inc:.1f}x")
print("\n轮数, 无缓存累计ms, 有缓存累计ms(含一次性prefill), 倍数")
for R in (1, 5, 10, 50):
    a_tot = a_per_round * R * 1e3
    b_tot = (b_prefill + b_inc * R) * 1e3
    print(f"{R}, {a_tot:.0f}, {b_tot:.0f}, {a_tot/b_tot:.1f}x")
