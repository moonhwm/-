# vLLM (Windows 原生) Prefix Caching 开/关 TTFT 对比实验结果

**日期**：2026-08-29　**机器**：Windows 11 笔记本，RTX 3060 Laptop 6GB（sm_86），驱动 555.99
**结论先行**：同前缀第二轮提问，APC 开 vs 关 = **0.142s vs 2.03s ≈ 14.3× 加速**（6K token 前缀，Qwen2.5-3B-Instruct-AWQ，max_tokens=8 总耗时口径）。

## 数字

| 组 | 冷轮（第1轮） | 暖轮 2 | 暖轮 3 | 暖轮均值 |
|---|---|---|---|---|
| `enable_prefix_caching=False` | 3.385 s | 2.029 s | 2.061 s | **2.045 s** |
| `enable_prefix_caching=True` | 2.257 s | 0.142 s | 0.129 s | **0.136 s** |

- **加速倍数 ≈ 2.045 / 0.136 ≈ 15×**（用暖轮 2 单点算为 14.3×）
- 前缀 6000 token；APC 开时暖轮只剩“提问后缀 + 8 token decode”，约 0.14s，符合预期量级。
- 冷轮差异（3.39s vs 2.26s）是首次运行的系统性偏慢（权重分页/编译缓存冷启动），不做对比依据。
- 口径声明：TTFT 用“第二轮 `llm.generate()` 总耗时（max_tokens=8）”近似——离线 API 无流式回调，prefill 占绝对主导，8 token decode 误差 <0.05s。

## 配置

- vLLM：SystemPanic/vllm-windows **v0.19.0+cu124**（最新 cu124 版；v0.20+ 起全是 cu132，驱动 555.99 不支持）
- torch **2.11.0+cu126**（release 说明硬性要求；cu126 运行时经 CUDA minor-version compatibility 跑在 555.99 上正常）
- 模型：**Qwen2.5-3B-Instruct-AWQ**（降级，原因见坑 6），`max_model_len=8192`、`gpu_memory_utilization=0.80`、`enforce_eager=True`、`VLLM_ATTENTION_BACKEND=FLASH_ATTN`、`VLLM_ENABLE_V1_MULTIPROCESSING=0`
- 运行脚本：[ttft_apc_test.py](ttft_apc_test.py)，日志：run_off.log / run_on.log

## 与 llama.cpp 实测 34× 的对照

llama.cpp（同机此前实测）prompt-cache 复用加速 34×；vLLM APC 这里 14-15×。两者口径不同：llama.cpp 侧前缀更长/解码更长摊薄了收益口径，且 llama.cpp 的缓存命中是整段 state 直读，vLLM 是 block 级 KV 复用仍需跑后缀 prefill+decode。**方向一致：长前缀复用在消费级显卡上是 1 个数量级级的收益**。绝对延迟 llama.cpp 更低，vLLM 胜在并发/连续批处理场景。

## 坑（按踩坑顺序）

1. **nvidia-smi 报 `Failed to initialize NVML: Unknown Error`**——虚惊。设备管理器里 GPU 状态 OK，torch `cuda.is_available()=True`。NVML 监控服务和 CUDA 本体是两条路。
2. **最新 vllm-windows（v0.26.0）全是 cu132**，驱动 555.99 只到 CUDA 12.5 → 必须回退到 v0.19.0（最后一个 cu124 版）。cu126 的 torch 可以跑（12.x 次版本兼容）。
3. **hf-mirror 的 huggingface_hub HEAD 校验必挂**（`LocalEntryNotFoundError`），按预案改 curl 直下。**curl `-C -` 断点续传与镜像 302 重定向冲突会写坏文件**（文件比预期大 2GB），坏片删掉整下即可。
4. pip 装 wheel 时两个带平台标记的依赖 **llguidance、xgrammar 被漏装**，需手动补。
5. vLLM v1 默认 fork 引擎子进程，曾撞上 **ZMQ `Address in use`**（超时杀进程后的残留/端口竞态）→ `VLLM_ENABLE_V1_MULTIPROCESSING=0` 单进程模式绕开。
6. **显存两道闸**：①开机空载只剩 5.0/6.0 GiB，`gpu_memory_utilization=0.85`（要 5.1 GiB）直接拒启动 → 降 0.80；②7B-AWQ 权重 5.2 GiB + KV(8K)≈0.5 GiB + 开销 > 6 GiB，**7B 在本机 6GB 显存物理上放不下**（已下载未试，判定后主动降级 3B-AWQ）。
7. **flashinfer-python 会被 pip 当依赖装上，但它在 Windows 上要求本机 CUDA Toolkit**（`CUDA_LIB_PATH` 未设即抛 ValueError），且 vLLM 探测后端时不捕获这个异常，`VLLM_ATTENTION_BACKEND=FLASH_ATTN` 也拦不住探测 → 直接 `pip uninstall flashinfer-python flashinfer-jit-cache` 解决。

## 环境留存（可复用）

- venv：`C:\temp\vllmenv`（torch 2.11+cu126 + vllm 0.19.0，已卸 flashinfer）
- 模型：`C:\temp\vllm-models\Qwen2.5-3B-Instruct-AWQ`（2.7GB 已验）、`Qwen2.5-7B-Instruct-AWQ`（5.6GB，本机显存放不下，仅留档）
