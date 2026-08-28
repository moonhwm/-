# 幻16 2022（GU603ZM）本地推理栈 · 优化与运维交接

> 版本 v1.1 · 2026-08-29 · 本文件随 `gemma4-deploy/` 目录走
> 规范建档：github.com/moonhwm/- ｜ 结构化副本：Supabase discovery_log 表

## 1. 硬件档案（WMI 实证）

| 部件 | 实测 | 备注 |
|---|---|---|
| 机型 | ROG Zephyrus M16 GU603ZM（幻16 2022 款） | Win32_ComputerSystem |
| CPU | i7-12700H，6P+8E 核 / 20 线程 | 纯 CPU 推理 4 线程最优（见 §3） |
| dGPU | RTX 3060 Laptop，6 GB GDDR6，算力 8.6 | 驱动 555.99（CUDA ≤12.5） |
| iGPU | Intel Iris Xe | 桌面渲染可走核显 |
| RAM | 16 GB DDR5（可用 15.7 GB） | 全局最大瓶颈 |
| 已知异常 | nvidia-smi 报 NVML 初始化失败 | CUDA 本体正常，仅监控接口不可用，疑 dGPU 低功耗状态；未修 |

## 2. 性能基线（32 tokens，实测）

| 配置 | 冷启动 | 热缓存 | 相对 CPU 基线 |
|---|---|---|---|
| CPU 4 线程（原配置） | 44.3 s | — | 1×（0.72 tok/s） |
| CPU 3 / 6 线程 | 145 s / 161 s | — | 更慢：换页主导，勿调 |
| GPU 20 层（显存超订 6.5 GB > 6 GB） | 81 s | 63 s | 回退！显存溢出比纯 CPU 还慢 |
| **GPU 14 层（生产现用）** | **8.7 s** | **1.7–1.8 s（≈18 tok/s）** | **约 25×** |

教训：显存一旦超订（模型+KV+桌面 > 6 GB）就走共享内存，性能崩塌。留 1 GB 给桌面是硬约束。独测环境里 22 层能跑 4 tok/s，但那是没桌面占显存的理想条件。

## 3. 关键结论

1. **换页主导论**：15 GB 模型 vs 16 GB 内存，mmap 分页是主瓶颈时，增加线程反而加剧并发缺页。CPU 路径保持 4 线程。
2. **GPU 甜点层数 = 14**（生产）/ 22（独测）。经验式：`n_gpu_layers ≈ (显存可用GB − 1.2GB 桌面预留) / 每层约 0.23 GB`。
3. **Windows CUDA 轮三坑**（重装 wheel 必重做，见 `gpu-test/README.md`）：
   - 预编译 wheel 不带 cublas/cudart → 需 pip 装 `nvidia-cublas-cu12==12.5.3.2` + `nvidia-cuda-runtime-cu12==12.5.82`
   - 必须 ctypes 预载 3 个 DLL（已固化在 `serve.py` 的 `_maybe_preload_cuda()`）
   - cu124 wheel 的 `ggml-cpu.dll` 在 i7-12700H 上非法指令崩溃 → 已用 g4env CPU 版同版本 DLL 覆盖（备份 `*.cu124.bak`）

## 4. 架构与文件

```
gemma4-deploy/
├── serve.py            # 主服务（OpenAI 兼容 + 网页聊天 + APK 下载 + /healthz）
├── start_bf16.py       # 启动器：g4env-gpu 存在则用 GPU 环境，否则回退 g4env
├── watchdog.py         # 看门狗：120s 探活，25s 超时 × 3 次失败阈值
├── server.config.json  # ★ 唯一调优接口（环境变量可覆盖任意键）
├── webchat.html        # 浏览器/手机聊天页
└── server_bf16.log / watchdog.log
```

- 计划任务：`Gemma4-BF16-Local`（登录起服务）、`Gemma4-BF16-Watchdog`（登录起看门狗）
- 防火墙：规则「Gemma4 BF16 8000」放行 TCP 8000
- 电源：插电休眠已关（`powercfg /change standby-timeout-ac 0`）
- Claw Desktop fallback 链：云端主模型 → kimi-coding/k2p6 → local-bf16/gemma4

## 5. 面向未来的接口（预留的优化余地）

| 接口 | 位置 | 用途 |
|---|---|---|
| `server.config.json` | 全部运行参数 | 改 JSON 即调优，无需碰代码；环境变量可临时覆盖 |
| `/healthz` | HTTP | 配置回显 + 版本，供监控/未来调优脚本读取 |
| `n_gpu_layers` | 配置键 | 未来换更大显存显卡/关桌面渲染时上调（独测 22 层可用） |
| `flash_attn` | 配置键 | 已在代码接线，模型支持时开启可省 KV 内存 |
| `gpu-test/test_gpu_load.py` | 脚本 | 任何层数/参数的 A/B 基准：`python test_gpu_load.py <层数>` |
| `start_bf16.py` 双环境 | 启动器 | 未来升级 llama-cpp-python 时新建 `g4env-gpu2` 灰度切换，改一行目录名即切 |

## 6. 已知风险与回退

- **回退纯 CPU**：把 `server.config.json` 的 `n_gpu_layers` 改 0，或把 `C:\temp\g4env-gpu` 重命名，重启即回 g4env 纯 CPU 环境。
- **打游戏/跑显存重活前**：服务会与之争显存导致双方变慢，建议先 `n_gpu_layers=0` 或停服务。
- **DLL 混装组合**（cu124 wheel + CPU 版 ggml-cpu.dll）在 llama-cpp-python 升级后必须重新验证。
- 局域网无鉴权：仅家庭 Wi-Fi 使用，勿做公网端口映射。
- **速度是双模态的**：热页面 + 显存空闲 → 18 tok/s；冷页面/内存压力/显存竞争 → 可跌回 <1 tok/s（15 GB 模型 × 16 GB 内存的物理现实）。单日实测横跨 1.7 s–161 s / 32 tok，单次数字不可外推。

## 7. 看门狗实战验证记录（2026-08-28 23:28）

- **kill 实测**：23:28:09 杀主服务 → 23:29:58 首检失败 → 23:32:00 二次失败触发拉起 → 23:32:41 恢复健康。中断 4 分 32 秒，拉起后稳定存活 4 分钟+。conf: empirical。
- **修复的缺陷**（实测暴露）：长推理时 llama.cpp 占 GIL，`/healthz` 会超时，原 8s 超时 × 2 次失败阈值会在一次正常长生成中误判服务死亡 → 已改为 25s 超时 + 连续 3 次失败才拉起。23:43 起生效。
- **已知残留**：推理请求服务端串行排队，客户端放弃（超时/断连）不会终止服务端生成；连续塞请求会累积排队。
- 勿对 `Gemma4-BF16-Watchdog` 重复 `schtasks /run`：会叠出多个看门狗进程（23:43 已清理重复实例，当前仅 1 个）。
