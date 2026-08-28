# KV Cache 对比实验结果（路线 A：distilgpt2 / CPU-only）

实验日期：2026-08-29
脚本：[kv_cache_lab.py](kv_cache_lab.py)（可重复运行，参数 `--gen-tokens / --runs / --prompt-len / --sweep` 可调）

## 1. 机器与环境

| 项 | 值 |
|---|---|
| 机器 | 幻16 2022 笔记本，i7-12700H（6P+8E），16GB RAM（实验时系统高负载 80–95%） |
| GPU | RTX 3060 6GB —— **全程未使用**（被本机 LLM 服务占用；torch.cuda.is_available() = False） |
| Python | 3.12.14，venv 位于 `C:\temp\hfenv` |
| torch | 2.13.0+cpu（官方 CPU 源 `download.pytorch.org/whl/cpu`，无 CUDA 负载），threads = 14 |
| transformers | 5.16.1 |
| 模型 | distilgpt2（81.9M 参数，fp32，CPU，batch=1；权重 352,824,413 B safetensors 经 hf-mirror.com 预下载到 `models/distilgpt2/`） |
| 计时方法 | `time.perf_counter()`，每轮 warmup 1 次，正式 3 轮取中位数；`torch.inference_mode()`；贪心 argmax 解码（确定性） |

## 2. 主实验：固定中文 prompt（331 token），生成 32 token

| 指标 | 实测值 |
|---|---|
| **prefill**（一次性全量前向 331 tok） | **285.5 ms** |
| **增量解码**（带 past_key_values） | **17.33 ms / tok** |
| **无 cache 重前向**（use_cache=False，每步重算全序列） | **284.83 ms / tok** |
| **单步加速比**（无cache / cache） | **16.4 ×** |
| 生成 32 token 总耗时 | cache 路径 **823 ms** vs 无 cache 路径 **9115 ms**，总加速 **11.1 ×** |
| 正确性校验 | 两侧贪心生成序列**完全一致**（True）——cache 只是省计算，不改变输出 |

## 3. 加分项：prompt 长度扫描（生成 32 token）

| prompt 长度 | prefill ms | cache ms/tok | 无cache ms/tok | 单步加速比 | 总加速比 |
|---|---|---|---|---|---|
| 16 | 37.1 | 16.53 | 47.24 | 2.9 × | 2.7 × |
| 64 | 47.1 | 14.12 | 53.88 | 3.8 × | 3.5 × |
| 256 | 128.8 | 15.77 | 130.38 | 8.3 × | 6.8 × |
| 331（主实验） | 285.5 | 17.33 | 284.83 | 16.4 × | 11.1 × |

曲线形态：
- **cache 路径**每 token 耗时基本恒定（14–17 ms，不随 prompt 长度增长）→ 近似 **O(1)** 每步（注意力部分理论上随历史长度线性增长，但在 ≤331 token 规模下被每步固定开销淹没）。
- **无 cache 路径**每 token 耗时随序列长度快速增长（47 → 285 ms，L 从 16 到 331 增长约 20 倍、单步耗时增长约 6 倍）。总成本 = 每步成本 × 步数，因此对生成长度 N 呈超线性（含注意力 O(L²) 项）增长。
- 加速比随 prompt 变长而扩大：序列越长，重算浪费越大，KV cache 收益越高。

## 4. 与理论预期的对照

| 理论预期 | 实测是否符合 |
|---|---|
| prefill 与无 cache 单步同量级（都是一次全序列前向） | ✅ 285.5 ms vs 284.83 ms，几乎相等（无 cache 时序列略长 1–31 token，差异被噪声淹没） |
| cache 单步 ≈ 单 token 前向，远小于全量前向 | ✅ 17.33 ms，约为全量的 1/16 |
| cache 不改变模型输出（数学等价） | ✅ 贪心解码序列逐 token 一致 |
| 无 cache 成本随长度近似平方增长 | ⚠️ 方向正确（超线性），但本规模下线性项（FFN/投影按 L 增长）占主导，纯二次项需更长序列才显著 |
| cache 单步近似恒定 | ✅ 14–17 ms 平台期 |

## 5. 局限声明

1. **微缩模型无代表性**：distilgpt2 仅 82M 参数、6 层。生产级模型（7B+）的绝对耗时、cache 内存占用、加速比量级完全不同；本实验只验证机制与趋势。
2. **微缩 attention 提醒**：本实验用的是完整 transformers 模型（含 FFN/归一化），但模型本身小，**绝对耗时不能作为生产基准**；单步 ~17 ms 中有相当比例是 Python/算子调度等固定开销，而非纯计算。
3. **CPU + 高负载环境**：测量时系统内存占用 80–95%，torch 用 14 线程，结果受 E 核调度、内存带宽竞争影响，绝对数字抖动大；相对比例（加速比）更稳健。
4. **未测 cache 内存开销**：KV cache 用显存/内存换时间，本实验未量化其内存增长（distilgpt2 fp32、331 token 下约 6 层 × 2(K,V) × 331 × 768 × 4B ≈ 12 MB，微不足道；大模型长上下文下才是瓶颈）。
5. **batch=1、贪心解码**：未覆盖 batching、采样、beam search 等场景。
6. **transformers 5.16.1**：新版 API 中 `torch_dtype` 已弃用（应写 `dtype`），脚本保留旧写法仅触发警告，不影响结果。

## 6. 踩坑记录

- `huggingface.co` 本机直连超时；`hf-mirror.com` 用 curl 可通（200，0.44 s），但 `huggingface_hub` 的 HEAD 元数据校验对镜像返回 `FileMetadataError` → 解决方案：curl 预下载 6 个文件到 `models/distilgpt2/`，脚本检测本地目录存在则走本地路径，否则回退 `HF_ENDPOINT` 在线下载。
- PyPI 官方源的 Windows torch 默认带 CUDA（约 2.5 GB），已按要求用 `download.pytorch.org/whl/cpu` 避开。
