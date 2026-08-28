# -*- coding: utf-8 -*-
"""snapshot_seed_lab.py — 磁盘 KV 快照「种子」可行性实测（纯 CPU，不占显存）
问题：save_state/load_state 的存取成本是多少？跨重置恢复后续写是否与连续推理一致？
环境: C:\\temp\\g4env（CPU 版 llama-cpp-python 0.3.35）；模型 Q4_K_M 4.97GB（mmap）
"""
import os, sys, time, pickle
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from llama_cpp import Llama

HERE = Path(__file__).parent
MODEL = r"C:\temp\g4models\gemma-4-E4B-it-Q4_K_M.gguf"
SNAP = HERE / "seed_snapshot.bin"

PREFIX = ("西安这家智能制造企业成立于2012年，主营工业机器人核心部件研发。"
          "公司研发人员占比四成，过去五年营收年均增长百分之二十三，"
          "产品销往东南亚与欧洲，2024年启动数字化转型。") * 12
QUESTION = "管理层认为未来三年的关键变量是什么？请简要回答。"

llm = Llama(model_path=MODEL, n_ctx=2048, n_threads=4, n_gpu_layers=0,
            use_mmap=True, verbose=False, seed=42)

P = llm.tokenize(PREFIX.encode(), add_bos=True)
Q = llm.tokenize(QUESTION.encode(), add_bos=False)
print(f"prefix={len(P)} tok, question={len(Q)} tok", flush=True)

t0 = time.perf_counter(); llm.eval(P); t_prefill = time.perf_counter() - t0
t0 = time.perf_counter()
_state = llm.save_state()
with open(SNAP, 'wb') as f: pickle.dump(_state, f)
t_save = time.perf_counter() - t0
snap_mb = SNAP.stat().st_size / 2**20
t0 = time.perf_counter(); llm.eval(Q); next_tok_cont = llm.sample(); t_ans = time.perf_counter() - t0
print(f"prefill {t_prefill:.2f}s | save_state {t_save:.2f}s ({snap_mb:.1f} MB) | 答问 {t_ans:.2f}s", flush=True)

llm.reset()
t0 = time.perf_counter()
with open(SNAP, 'rb') as f: _state2 = pickle.load(f)
llm.load_state(_state2)
t_load = time.perf_counter() - t0
t0 = time.perf_counter(); llm.eval(Q); next_tok_rest = llm.sample(); t_ans2 = time.perf_counter() - t0
print(f"load_state {t_load:.2f}s | 恢复后答问 {t_ans2:.2f}s", flush=True)

llm.reset()
t0 = time.perf_counter(); llm.eval(P); llm.eval(Q); next_tok_re = llm.sample(); t_full = time.perf_counter() - t0
print(f"无快照重算 {t_full:.2f}s", flush=True)

same = next_tok_cont == next_tok_rest == next_tok_re
print(f"\n=== 种子 viability ===")
print(f"恢复快照总成本: {t_load + t_ans2:.2f}s  vs  重算: {t_full:.2f}s  →  {(t_full)/(t_load+t_ans2):.1f}x")
print(f"三条路径 next_token 一致: {same}")
print(f"快照体积: {snap_mb:.1f} MB（前缀 {len(P)} tok → {snap_mb*1024/len(P):.1f} KB/tok）")
