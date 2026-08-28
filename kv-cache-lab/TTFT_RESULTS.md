# Prompt/Prefix Cache TTFT 对比实验结果

**结论（先给数字）**：同一 3200-token 中文长前缀下的第二轮提问（新增 39 token），
**cache-off 中位 TTFT ≈ 1.61–1.74 s，cache-on 中位 TTFT ≈ 0.047–0.049 s，提速 ≈ 34–35×**。
高于 vLLM 社区常见经验区间（5–20×），原因见「对照与解读」。

- 实验日期：2026-08-29（本机墙钟）
- 脚本：[`prefix_ttft_lab.py`](prefix_ttft_lab.py)；API 探针：[`api_probe.py`](api_probe.py)；原始数据：`results.json`（第 2 次运行）、`run2.log`（完整 llama.cpp 日志）

## 1. 环境与最终配置

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU（VRAM 6143 MiB，cc 8.6） |
| Python 环境 | `C:\temp\g4env-gpu\Scripts\python.exe`（需照 `gpu-test/test_gpu_load.py` 预载 cudart64_12 / cublas64_12 / cublasLt64_12 DLL） |
| llama-cpp-python | 0.3.35（CUDA 版，`llama_supports_gpu_offload()=True`） |
| 模型 | `C:\temp\g4models\gemma-4-E4B-it-Q4_K_M.gguf`（Gemma-4-E4B-It，42 层，7.5B，Q4_K_M，4.97 GB） |
| **n_ctx** | **6144**（未触发 OOM，未降级） |
| **n_gpu_layers** | **-1（日志确认 offloaded 43/43 layers to GPU）** |
| 显存占用（llama.cpp 日志） | 模型 buffer CUDA0 2883.5 MiB + KV buffer 360+96 MiB + compute buffer 575 MiB ≈ **3.9 GB**；另有 2208 MiB CPU_Mapped（gemma4 `per_layer_token_embd` 等张量按架构走 mmap，不占显存） |
| `nvidia-smi` | 本机 NVML 初始化失败（`Failed to initialize NVML: Unknown Error`），显存数字以 llama.cpp buffer 日志为准 |
| 采样 | `llm.sample()` 默认参数，每轮只采 1 个 token（TTFT 终点） |

## 2. API 核查结论（先查证再动手）

打印 `dir()` + `inspect.signature` 确认（探针输出见会话日志）：

- `llama_cpp.llama_cache` 提供 `LlamaRAMCache(capacity_bytes)` / `LlamaDiskCache(cache_dir, capacity_bytes)`，经 `Llama.set_cache(cache)` 挂载——但它是 **`LlamaState` 全量快照缓存**（`save_state()`/`load_state()` 序列化整段 KV），按 token 序列做最长前缀键匹配。恢复快照本身有反序列化/拷贝开销，**不是生产栈（llama.cpp server slot / vLLM APC）的“原地 KV 前缀复用”语义**。
- `Llama.generate()` 内部（llama.py L917–950）已实现自动最长前缀匹配：`kv_cache_seq_rm(-1, reuse_prefix, -1)` 后只 decode 后缀。
- `Llama.eval(tokens)` 在 `self.n_tokens` 处续写 decode，入口处先清 `n_tokens` 之后的 KV；`Llama.reset()` 置 `n_tokens=0`。
- 因此实验采用**低层路线**：cache-on = 同实例截断到前缀长度后只 eval 新增 token；cache-off = `reset()` + `kv_cache_clear()` 后从头重算。这与 llama.cpp server 连续同前缀请求的原地复用等价。

## 3. 实验协议

- 前缀 P：中文段落循环拼接，**3200 token**（含 BOS）；Q1 38 token，Q2 39 token（`<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n` 格式）。
- warmup 1 次（冷启动首轮 P+Q1，兼作首轮参考）。
- 每轮 trial：先 off（reset+clear → 计时 eval(P+Q2)+采样 1 token），后 on（截断 `n_tokens=3200` + `kv_cache_seq_rm` → 计时 eval(Q2)+采样 1 token）。off/on 交替，共用同一模型实例、同一后端配置，对比公平。
- 各 3 轮取中位数；独立复跑 1 次验证。

## 4. 数字表

| 轮次 | cache-off TTFT (s) | cache-on TTFT (s) | 提速倍数 | off 侧 prefill (tok/s) |
|---|---|---|---|---|
| Run1-T1 | 1.540 | 0.0643 | 23.95× | 2103 |
| Run1-T2 | 1.613 | 0.0471 | 34.24× | 2009 |
| Run1-T3 | 1.675 | 0.0465 | 36.00× | 1934 |
| **Run1 中位** | **1.613** | **0.0471** | **34.24×** | — |
| Run2-T1 | 1.583 | 0.0522 | 30.30× | 2047 |
| Run2-T2 | 1.779 | 0.0475 | 37.48× | 1820 |
| Run2-T3 | 1.742 | 0.0492 | 35.41× | 1859 |
| **Run2 中位** | **1.742** | **0.0492** | **35.40×** | — |

参考：冷启动首轮（P+Q1，3238 token）TTFT ≈ 2.08–2.11 s（~1540 tok/s，含 CUDA graph 首次构建）。
off/on 的 token 计算量比 = 3239 / 39 ≈ **83.1**（提速的理论上界）。

## 5. 与 vLLM 社区预期（5–20×）对照

实测 34–35×，**高于**该区间。解读：

1. 本实验测的是**纯计算墙钟**（eval+单 token 采样），不含 vLLM 服务端到端 TTFT 里的调度、tokenize/detokenize、HTTP、采样器开销；那些固定开销会摊薄倍数。
2. token 比 83:1 是上界；实测倍数低于上界，因为 off 侧长序列 attention 更贵（每 token 成本随上下文增长），而 on 侧 39 token 里固定开销（batch 构建、采样）占比不小（39 token / 0.049 s ≈ 800 tok/s）。
3. 前缀越长，off 侧 prefill 越贵，倍数越大；3.2K token 属于中等长度，vLLM 社区数字多基于更长前缀+更重模型+并发场景。

## 6. 坑与局限

- **CUDA DLL 预载是必须的**：不预载 cudart/cublas/cublasLt 直接 `import llama_cpp` 会挂（模板头部原样照抄有效）。
- **NVML 不可用**：`nvidia-smi` 与 pynvml 均报 `Failed to initialize NVML: Unknown Error`，显存只能读 llama.cpp buffer 日志。
- gemma4 有 ~2.2 GB 张量（per-layer embedding 类）走 CPU_Mapped——mmap 映射不占显存，且日志确认 43/43 层计算图在 GPU；off/on 两侧条件相同，不影响对比有效性。
- cache-on 语义为**同实例原地 KV 复用**；未测 `LlamaDiskCache` 快照恢复路径（跨进程场景，恢复开销另算）。
- 单实例、单并发、无 LoRA/投机解码；采样参数对 TTFT 影响未分离（每轮只采 1 token，开销 < 1 ms 量级）。
- 时间盒内未完成项：无——两轮完整运行 + 复跑验证均完成，实验期间未触碰 8000 端口与 gemma4-deploy 目录。
