# ASR 试点实测报告（2026-08-29）

> 试点：BV1BKuHzcELe 开头 10 分钟切片（610.7s，16kHz mono）｜ 环境：funasr 1.4.6 + torch 2.13.0+cpu（纯 CPU，i7-12700H）
> 目的：为「B站长教学视频蒸馏」验证云 4核8G CPU 生产管线可行性

## 速率实测

| 模型 | 610.7s 音频耗时 | ×实时倍率 |
|---|---|---|
| SenseVoice-Small + FSMN-VAD | 27.8s | **≈22–24×** |
| Paraformer-Large + FSMN-VAD + CT-PUNC | 40.2s | ≈15–16× |

模型加载仅 5–6s；内存峰值 <3GB。

## 云 4核8G 推算

4 vCPU 保守折算 4–8×RT → 55h 全量 ≈ 7–14 小时，**夜间批跑一夜跑完，可行**（¥388/年 Flexus L，无需 GPU 服务器）。推荐云端 SenseVoice-Small + FSMN-VAD（更快、断句更整、更省内存），逐 P 队列。

## 文本质量初判

- 通用口语：两模型均流畅，标点可用
- 物理术语错 6+ 处/10min：「薛定谔方程」几乎全错（薛定额/序定/确定方程）、「束缚态」参半、「μ 粒子→谬的例子」、「势场→市场」、「陈鄂生→陈赫成/陈克生」
- 结论：通用层没问题，**专业术语必须走后处理**（热词词典 + LLM 润色）——印证精度方案 L2 层的必要性
- Paraformer 30s 后断句碎片化（喂 LLM 前需重组）；SenseVoice 句子完整自然但不输出时间戳

## 坑位记录

1. B站批量下载 412 风控 → 逐 P 独立进程 + sleep 6s
2. sentencepiece 不吃中文路径 → MODELSCOPE_CACHE 迁 ASCII 路径
3. funasr 1.4.6 需手动补 kaldi-native-fbank
4. merge_vad=True 丢时间戳；Paraformer 需 sentence_timestamp=True
5. ffmpeg 用 BtbN 静态包

## 下一步

1. 建物理术语热词库（量子力学册 300-500 词），Paraformer 热词注入 + gemma-4 后纠
2. 云采购决策：Flexus L 4核8G ¥388/年（已具备下单条件）
3. 达标验收按 ASR_PRECISION_PLAN.md 的 G1/G2 双级门执行
