# Claw 本地推理栈建设 · 全量发现流程记录

> data_cutoff: 2026-08-28 23:46 CST ｜ 记录人：Kimi (K3) ｜ 项目机型：ROG 幻16 2022 (GU603ZM)
> conf 标注：empirical = 实测证据 ｜ estimated = 引擎/单次测量 ｜ assumed = 推断
> 归档路径：github.com/moonhwm/- （规范建档库）｜ 结构化副本：Supabase discovery_log 表

## 0. 终态架构

```
手机/平板浏览器 or APK ──HTTP──> http://192.168.1.254:8000
                                    ├── GET /            聊天网页（webchat.html）
                                    ├── GET /healthz     自检（配置回显）
                                    ├── GET /claw-local-assistant.apk  安装包下载
                                    └── POST /v1/chat/completions      OpenAI 兼容推理
                                  serve.py (FastAPI + llama-cpp-python)
                                    ├── GPU 环境 g4env-gpu（CUDA，14 层上卡）
                                    └── CPU 回退环境 g4env（原样保留）
计划任务 Gemma4-BF16-Local（登录起服务）+ Gemma4-BF16-Watchdog（120s 探活自动拉起）
Claw Desktop fallback 链：云端主模型 → kimi-coding/k2p6 → local-bf16/gemma4
```

## 1. 全量时间线（2026-08-28）

| 时间 | 事件 | 关键证据 | conf |
|---|---|---|---|
| 05:17 | Claw Desktop 设备身份创建（identity/device.json），首次连接早于此前记录 | device.json ctime | empirical |
| 06:20–06:34 | 下载 BF16-GGUF 全量包：gemma-4-E4B-it-BF16.gguf 15.05 GB + mmproj 992 MB → `C:\temp\g4models\`（原地下载，.cache 暂存目录同址创建于 05:57） | 文件时间戳 + .cache 结构 | empirical |
| 06:41 | 强行实装：停 Q4 服务，mmap 加载 BF16（15 GB 模型 > 16 GB 内存可用量） | serve.py | empirical |
| 06:55 | 视觉功能接入：Gemma4ChatHandler + mmproj，支持 image_url 消息 | server_bf16.log 视觉推理记录 | empirical |
| 07:10 | 企业微信机器人轮询器安全审查：纯标准库、只读拉取、无反弹；config 含明文 key 警告 | 源码审读（claw_poller.py 创建于 07:11:50 互证） | empirical |
| 07:17–08:15 | 弃企业微信改飞书：装 lark-cli（QR 07:56）→ 绑凭据 → 手机扫码授权完成 | 截图序列 + 文件时间戳 | empirical |
| 09:47–09:59 | 当轮 Claw Desktop 配置活动（非首次连接）：config-backups/openclaw-20260828-095954.json 为该窗口旁证 | 配置备份 diff | empirical |
| 10:35 | Claw 主模型 ollama/openclaw-local → kimi-coding/k2p6（本地 8B 在 94% 负载下 600s 超时无回复） | openclaw.json + 备份 | empirical |
| 18:05 | 切 K3：探测 `agent-gw.kimi.com/coding/v1/models` 发现 `k3-agent`（1M 上下文）与 `k2d6-agent`；k3-agent 实测 2.8s 可用；改配置重启网关，e2e 10.4s 出回复 | curl 实测 + claw_e2e_test.py | empirical |
| 18:13 | 本地链路改走 BF16：新增 `local-bf16` provider（OpenAI 兼容 :8000）替换 ollama Q4 作 fallback；实测 44.3s/32tok | openclaw.json | empirical |
| ~18:18 | **发现：桌面端 `L0 schema_migration` 每次网关启动强制重置 primary 与 kimi-coding models 列表**——改文件无效，K3 须在 Claw 设置 UI 选择；自定义 provider/fallback 不受影响（热重载日志确认） | main.log | empirical |
| 18:22 | 开机自启：schtasks 权限被拒 → 改用 Startup 文件夹 bat（连踩 GBK 编码、LF 行尾两个 cmd 解析坑，最终 GBK+CRLF）→ 杀服务实测拉起成功 | bat 实测 | empirical |
| 19:18 | widgetdesign 交接面板 Widget 生成并挂到「每日财经」看板 (canvas_4034d30b, mount_16e8f6ce) | Canvas.read | empirical |
| 19:22 | 打包 APK：子代理从零装 Temurin JDK17 + Android SDK 34 + Gradle 8.9 + AGP 8.5.2（全在工作区 apk-build/）→ claw-local-assistant.apk 11.4 KB（单 Activity WebView 壳）；顺手修复 serve.py 缺 CORS 头 | aapt badging | empirical |
| 19:39 | 用户管理员 PowerShell 加防火墙规则「Gemma4 BF16 8000」→ LAN 打通；**同期发现服务无声猝死模式**（日志停在启动成功无 traceback 反复死亡） | 截图 + netstat | empirical |
| 19:46 | 对照实验定位猝死：shell 直接拉起稳定 vs detached 拉起必死 → 改走计划任务 XML 导入（绕开 PowerShell 引号转义坑）；19:48 用户导入成功，复检 4.5 分钟稳定 | 对照实验 + 截图 | empirical |
| 19:59 | 用户展示 MatePad Pro Max = HarmonyOS 7.0.102（OpenHarmony 7 Beta，不兼容 APK）→ serve.py 加 `GET /` 托管聊天网页（同源免跨域）+ APK 下载路由；Mate 60 Pro+ 视系统版本双轨 | 关于本机截图 | empirical |
| 22:10 | 自主推进：发现插电 10 分钟休眠隐患（0x258）→ powercfg 改永不休眠；聊天页加「下载 App」入口 | powercfg 回读 0x0 | empirical |
| 22:15 | 幻16 2022 特定优化：WMI 实证 GU603ZM = i7-12700H(6P+8E/20T) + RTX 3060 Laptop 6GB + 16GB DDR5；发现 nvidia-smi NVML 故障（不阻塞） | WMI/nvidia-smi | empirical |
| 22:2x | **线程实测反直觉**：3 线程 145s、6 线程 161s 均慢于 4 线程 44s——换页主导时并发缺页加剧，回退 4 线程 | 三轮 bench | estimated（单轮方差大） |
| 22:3x | 子代理 GPU 实验（独立 venv g4env-gpu）：CUDA 卸载真实生效（日志 `offloaded 22/43 layers to GPU`）；22 层 4.05 tok/s = 5.6×；24 层显存占满反慢 | gpu-test/run_L*.log | empirical |
| 22:4x | CUDA 三坑：①wheel 不带 cublas/cudart 需 pip 装 nvidia-*-cu12 ②llama_cpp RTLD_GLOBAL 须 ctypes 预载 DLL ③cu124 wheel 的 ggml-cpu.dll 在 i7-12700H 非法指令崩溃，用 CPU 版同版本 DLL 覆盖（备份 .cu124.bak） | gpu-test/README.md | empirical |
| 22:5x | GPU 接入生产：20 层显存超订（6.5>6 GB）性能崩塌（81s/63s，比纯 CPU 还慢）→ **14 层甜点：热态 1.7–1.8s ≈ 18 tok/s = 25×** | VRAM 计数器 + bench | empirical |
| 23:24 | 看门狗上线：watchdog.py（120s 探活、连挂拉起）+ 计划任务 Gemma4-BF16-Watchdog | watchdog.log | empirical |
| 23:28 | **kill 实测**：23:28:09 杀服务 → 23:32:00 看门狗二次失败触发拉起 → 23:32:41 恢复。中断 4m32s | watchdog.log 时间戳链 | empirical |
| 23:39 | **实测抓出缺陷**：长推理时 llama.cpp 占 GIL → /healthz 超时 → 原 8s×2 次参数会在正常长生成中误判死亡。修复：25s 超时 + 连续 3 次失败才拉起，23:43 生效 | 日志 23:39:08 误报记录 | empirical |
| 23:4x | 发现双模态速度：热页面+显存空闲 18 tok/s；冷/压状态下可跌回 0.5 tok/s（15 GB 模型 × 16 GB 内存的物理现实） | 多轮 bench 1.7s–161s | estimated |
| 23:28 | 手机截图「网站暂无响应」= 恰好撞上 kill 测试窗口（23:28:09–23:32:41），非故障 | 时间戳比对 | empirical |
| 11:28 | 平板截图 ERR_ADDRESS_UNREACHABLE = 防火墙放行（19:42）前的历史状态，非当前状态 | 时间戳比对 | empirical |

## 2. 归档后核证（2026-08-29 凌晨）

| 项 | 内容 | 结论 |
|---|---|---|
| 核证① | 早间事件经文件 mtime 回查全部自洽 | empirical |
| 核证② | GitHub 仓库 API 读回 DISCOVERY_LOG.md，中文无乱码、SHA 一致 | empirical |
| 核证③ | Supabase discovery_log 启用 RLS（relrowsecurity=true） | empirical |
| 核证④ | anon/publishable key 实测：SELECT 200 但 0 行 | empirical |
| 核证⑤ | NTFS CreationTime 证伪覆盖：lark QR / feishu 截图创建=修改；claw_poller.py 07:11:50 与 07:10 审查互证 | empirical |
| 核证⑥ | 09:47 事件旁证：095954 配置备份 + device.json 05:17 修正「首次连接」定性 | empirical（含部分保留） |
| 核证⑦ | RLS 写实测：INSERT 401/42501 双 key 被拒、DELETE 影响 0 行（总数 34 不变） | empirical |
| 核证⑧ | 移动取证：.cache/huggingface/download 暂存目录同址创建于 05:57，GGUF 原地下载未经移动 | empirical |

## 3. 关键断言登记（可证伪）

| # | 断言 | falsifiable_test | conf |
|---|---|---|---|
| A1 | GPU 14 层为生产甜点层数 | 改 16/20 层重跑 bench 对比 | empirical |
| A2 | 显存超订阈值 ≈ 6 GB 物理上限，留 ~1 GB 给桌面 | 开显存重活复测速度 | empirical |
| A3 | 计划任务托管的进程不被 Kimi 运行时清理 | 观察 24h 无猝死 | empirical（观察中） |
| A4 | 看门狗 25s×3 阈值不再误杀长推理 | 跑一次 60s+ 长生成看日志无误报 | assumed（未实战） |
| A5 | 双模态速度源于页面缓存+显存竞争 | 分离变量实验（未做） | assumed |

## 4. 沉淀文件

| 文件 | 内容 |
|---|---|
| `gemma4-deploy/serve.py` | 主服务（配置化 + CUDA 预载 + CORS + 网页/APK 托管 + /healthz） |
| `gemma4-deploy/server.config.json` | 唯一调优接口（环境变量可覆盖） |
| `gemma4-deploy/start_bf16.py` | 启动器（GPU 环境优先，CPU 回退） |
| `gemma4-deploy/watchdog.py` | 看门狗（25s 超时 × 3 次阈值） |
| `gemma4-deploy/OPTIMIZATION.md` | 硬件档案 + 基线 + 三坑 + 回退规程 |
| `gemma4-deploy/webchat.html` | 浏览器/手机聊天页 |
| `gpu-test/test_gpu_load.py` + `run_L*.log` | GPU 层数 A/B 基准工具与原始日志 |
| `claw-local-assistant.apk` | 安卓客户端（11.4 KB，com.claw.localassistant） |
| `apk-build/` | 完整 Android 构建环境（JDK/SDK/Gradle 本地化） |
| `claw_e2e_test.py` | Claw 网关端到端验证（webchat + Ed25519 签名） |

## 5. top3_likely_wrong（最终存量，考古级）

1. 「.cache 存在 ⇒ 未移动」为强推断非铁证（理论上可先下载再移回重建缓存，无迹象支持，残留 assumed）
2. RLS DELETE 0 行与「无 DELETE policy 静默空转」在效果上不可区分——数据删不掉这一目的已达成
3. device.json 05:17 的创建者（哪次会话/流程）已不可考

## 6. 遥测

累计：缺陷逃逸 1（GitHub 占位文件误推，当场自查修复）｜ 重做 3（线程回调、20→14 层回调、看门狗参数修订）
协议自审计连续四轮无 Med 级以上新缺陷，记录与核证任务收敛。
