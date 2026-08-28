# 建档库（-）

本仓库是 Kimi 在本机项目的**规范建档路径**（canonical archive，用户 2026-08-29 指定）。

## 当前建档：Claw 本地推理栈建设全记录

幻16 2022（GU603ZM：i7-12700H + RTX 3060 Laptop 6GB + 16GB DDR5）上把 Gemma-4 E4B BF16 全量包（15 GB）实装为局域网推理服务的完整发现流程。

- [DISCOVERY_LOG.md](DISCOVERY_LOG.md) —— 全量时间线 + 证据链 + 可证伪断言登记
- [gemma4-deploy/OPTIMIZATION.md](gemma4-deploy/OPTIMIZATION.md) —— 硬件档案、性能基线、CUDA 三坑、回退规程
- [gemma4-deploy/serve.py](gemma4-deploy/serve.py) —— 主服务（OpenAI 兼容 + 网页聊天 + APK 托管 + /healthz + 配置化）
- [gemma4-deploy/watchdog.py](gemma4-deploy/watchdog.py) —— 看门狗（120s 探活，25s 超时 × 3 次阈值）
- [gemma4-deploy/server.config.json](gemma4-deploy/server.config.json) —— 调优接口

## 核心结论

- 15 GB BF16 模型在 16 GB 内存机器上：mmap 换页主导，CPU 4 线程最优（加线程反而慢 3 倍）
- RTX 3060 6GB 部分卸载 14 层为生产甜点：热态 ≈18 tok/s（25×）；显存超订（>6GB）立刻崩塌
- Windows CUDA wheel 三坑：缺 cublas/cudart 运行时、须 ctypes 预载 DLL、ggml-cpu.dll 非法指令需换 CPU 版
- 看门狗须防长推理误杀：GIL 阻塞 healthz，需 25s 超时 + 3 次失败阈值

## 归档纪律

- 所有记录带 data_cutoff 与 conf 分级（empirical / estimated / assumed）
- 归档后核证回写进 Supabase `discovery_log` 表（项目 moonhwm's Project）
- 本仓库内容与本地工作区 `Documents/kimi/tasks/2026-08-27/22-20-45-c3ffff44/` 同步

data_cutoff: 2026-08-29