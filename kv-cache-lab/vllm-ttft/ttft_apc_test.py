# -*- coding: utf-8 -*-
"""
vLLM prefix caching 开/关 TTFT 对比实验
用法: python ttft_apc_test.py on|off
口径: TTFT 近似 = 对同一约 6K token 前缀的第二轮 generate(max_tokens=8) 的总耗时
      (prefill 主导, decode 8 token 占比小; 冷轮=第一轮, 暖轮=第二轮)
"""
import sys, time, json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = r"C:\temp\vllm-models\Qwen2.5-3B-Instruct-AWQ"
flag = sys.argv[1] if len(sys.argv) > 1 else "on"
apc = flag == "on"

tok = AutoTokenizer.from_pretrained(MODEL)

# 构造约 6000 token 的长前缀(科普散文循环, 每段带序号避免纯重复被完全去重)
para = (
    "在大规模语言模型的推理过程中,注意力机制需要对每个新生成的词元访问此前所有词元的键值缓存。"
    "键值缓存的容量随着序列长度线性增长,因此长上下文推理既受显存容量限制,也受显存带宽限制。"
    "前缀缓存技术的核心思想是:如果两次请求共享相同的前缀,那么这段前缀计算得到的键值张量可以被复用,"
    "从而避免重复的预填充计算。在分批服务和多轮对话场景中,系统提示词往往占据绝大部分输入长度,"
    "因此前缀缓存可以显著降低首词元延迟。此外,量化技术通过降低权重精度来压缩显存占用,"
    "使得消费级显卡也能运行七十亿参数级别的模型。"
)
pieces, n = [], 0
i = 0
while n < 6000:
    i += 1
    s = f"第{i}章。{para}"
    pieces.append(s)
    n += len(tok(s)["input_ids"])
prefix = "".join(pieces)
prefix_ids = tok(prefix)["input_ids"][:6000]
prefix = tok.decode(prefix_ids)
print(f"PREFIX_TOKENS={len(prefix_ids)}")

q1 = prefix + "\n\n问题:用一句话总结上文的核心观点。"
q2 = prefix + "\n\n问题:上文提到了哪几类关键技术手段?请列举。"

llm = LLM(
    model=MODEL,
    max_model_len=8192,
    gpu_memory_utilization=0.80,
    enforce_eager=True,
    enable_prefix_caching=apc,
    disable_log_stats=True,
)
sp = SamplingParams(temperature=0, max_tokens=8)

t0 = time.perf_counter()
llm.generate([q1], sp)
t1 = time.perf_counter()
llm.generate([q2], sp)
t2 = time.perf_counter()
q3 = prefix + "\n\n问题:作者对消费级显卡运行大模型持什么态度?"
llm.generate([q3], sp)
t3 = time.perf_counter()

result = {
    "apc": apc,
    "prefix_tokens": len(prefix_ids),
    "cold_round_s": round(t1 - t0, 3),
    "warm_round2_s": round(t2 - t1, 3),
    "warm_round3_s": round(t3 - t2, 3),
}
print("RESULT_JSON=" + json.dumps(result, ensure_ascii=False))
