# Bili23-Downloader 源码实装 + 蒸馏管线 v2 经验（公开脱敏版）

> data_cutoff: 2026-08-30 ｜ 性质：工程经验存档，无密钥无路径无内容数据

## 一、Bili23-Downloader v2.15.0 源码实装

- **背景**：预编译版（Nuitka 打包）被 Windows Defender 误报拦截（作者官方 Discussion #154 承认的已知误报）。改走源码路线。
- **环境**：Python 3.12 venv + `requirements.txt`（PySide6 6.10.3 全家桶、httpx、orjson、qrcode、protobuf、psutil）。
- **坑 1**：PyPI 直连大 wheel 超时 → **清华镜像** `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt` 一次通过。
- **坑 2**：main.py 不在仓库根，在 `src/main.py`。
- **冒烟**：`QT_QPA_PLATFORM=offscreen` 离屏启动，15s 存活后 terminate——GUI 可用性验证通过（无需真显示）。
- ffmpeg 需自备（复用既有 ffmpeg.exe 加 PATH）。

## 二、B 站批量下载风控实证（412 Precondition Failed）

- 触发：单进程逐 P、间隔 3s，8 个中 4 个被 412。
- 解法：**冷却 25s + 间隔 10s**，剩余全克。规律：风控按请求频率计数，不按并发（我们本来就单进程）。

## 三、长音频 ASR 的工程上限

- 79 分钟单集：SenseVoice CPU 推理 ~300s+，撞单命令 300s 超时 → **切两段分别转写**（ffmpeg -t/-ss 切半）。以后 >60 分钟素材默认分段。

## 四、术语域分离的首次实装（蒸馏管线 v2）

- 立法背景：不同 UP 主/学科的自造术语互不通用，单一纠错词典会串域误修。
- 实装：管线新增 `--dict <path>` 开关 + 词典 `_meta.domain` 声明；context 级规则仅在 `--domain` 与词典声明域一致时启用。
- 首个新域：哲学课域词典（人名地名错误族：门德尔松/德累斯顿/莱比锡等实测种子）。
- 质检法：关键术语正误计数对比（如 尼采 122:3），错误族按「人名地名/自报家门/学科术语」分类。

## 五、自购付费内容的合规蒸馏路径（框架）

- 前提：**本人已购 + 本人账号 + 本人扫码登录**（QR 登录态下载，内容仅本机、零分发）。
- 与"他人账号"的本质区别：本人持有观看许可，属于个人学习用途的自力存档；仍属平台协议灰色（禁止第三方下载条款），不扩大、不传播。
- 内容资产（转写/知识卡）留本机；只有**方法论与工程经验**（本文档这类）才外发脱敏版。
