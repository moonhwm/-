# 磁盘 KV 快照「种子」可行性实测（幻16 / 纯 CPU / Q4 7.5B）

> data_cutoff: 2026-08-29 ｜ 触发：生产栈 API 核查时发现 `save_state/load_state/LlamaDiskCache` 存在但未测恢复路径
> 脚本：`snapshot_seed_lab.py`（环境 `C:\temp\g4env`，CPU 版 llama-cpp-python 0.3.35，不占显存）

## 数字（661 token 前缀 + 15 token 提问）

| 环节 | 耗时 |
|---|---|
| 前缀 prefill | 9.22 s |
| save_state（pickle 落盘） | 1.44 s |
| **快照体积** | **548.2 MB（849 KB/token）** |
| load_state（读盘恢复） | 0.66 s |
| 恢复后答问 | 0.32 s |
| **恢复路径总计** | **0.98 s** |
| 无快照从头重算 | 9.43 s |
| **恢复 vs 重算** | **9.6×** |
| 三路径 next_token 一致性（连续 / 恢复快照 / 重算） | ✅ 逐 token 相同 |

## 结论

「磁盘 KV 快照」种子**可行且已验证**：跨重置（等价跨进程）恢复后生成与连续推理逐 token 一致，恢复成本仅为重算的 1/10（且前缀越长优势越大——重算 O(N) 起，恢复是固定 I/O）。

## 体积约束（长程 Agent 的硬约束）

- 快照 ≈ **0.85 MB/token**（gemma4 的 SWA cache 按全长存储 + V  padding 到 1024，fp16 KV + logits/state）。
- 外推：100K token 会话 ≈ 85 GB——**快照适合做“会话检查点”，不适合存无限历史**。长程 Agent 的正确姿势是“摘要压缩历史 + 关键节点快照”，而不是全量快照链。
- 若要压缩体积：KV 量化（Q8）可砍半；去掉 scores/logits 只留 llama_state bytes 可再省（pickle 里含 numpy scores）。

## 与 Mooncake/AgentENV 的关系

本机这条 save_state/load_state 路径 = 单机版 KV 检查点；kvcache-ai/Mooncake 是分布式版。语义上已验证同边界成对恢复可行（同模型、同量化、同 llama.cpp 版本——换任一项快照即失效）。
