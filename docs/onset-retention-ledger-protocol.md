# 起音保留账本协议

本工具只汇总已冻结的《樱花草》JSON，不运行选音器、神经推理或音频处理。
目标是把候选行的既有选择理由与最终起音保留情况对应起来，并把同一声部、
同一原始起音时间的多个模型类别折叠成一个声部起音，同时保留每个候选 ID。

## 冻结输入

- `test-data/sakura-tsumugi-expression-onset-v3/analysis.json`：
  `26a0250568b2ce149022c4aa1549af2d9dd49b045372959a6b26b5342ba5313a`
- `test-data/chord-evidence-ties-v1/10k/candidates.json`：
  `c20742509baea4d94c63493d6f5454fa3ee5c4b72636917f4ea2e89fab071895`
- `test-data/expression-precision-v1/default/10k/selection.json`：
  `24f23cc1d59e7452e5ede424f10054d285b0d2ecdd26b33095e651716f0b0b44`
- `test-data/expression-rhythm-v1/default/full/10k/scopes.json`：
  `185634495f21d9ad98cd33362ac4474c2a146966e81e67fe371b3343338ad46d`

脚本逐项核验这些 SHA-256；候选与选音决策 ID 必须一一对应，定轨候选必须是分析
候选的子集，所有选中ID也必须在定轨候选中；既有 `scopes.json` 必须恰好覆盖定轨
候选一次。任何输入变化都会停止，不会自动覆盖或更新冻结值。

## 计数语义

- 总体决策仍按每个候选 ID 记录。声部起音按 `section + voice + analysis.json 的 start`
  分组；跨声部同一时刻另作并列索引。对齐后的 `chart_start` 单独保存，不用量化后的
  时间合并原始起音。
- 同一原始时刻的不同 `voice` 分别记为声部起音；同一 `voice` 下的多个
  `instrument_id` 只保留为模型类别标签，不计作多个声源。每一候选 ID、选音原因、
  `represented_by` 和选择组索引均留存在对应行。
- `selection_group_index` 直接复制选音决策中的 `group` 字段。它是该次选择记录中的
  组索引，不是稳定的物理乐器身份，也不能跨运行当作相同求解变量。
- 省略理由严格照录选择器输出。`weak_detection`、`voice_not_selected`、
  `group_capacity_or_density`、`represented_by_chord` 和 `duplicate_pitch` 不是听感、
  声源真值或音乐价值结论。`strength` 只保存原值，不解释为置信度或价值分。
- 不计算准确率、可闻性、Timing 正确率或音乐分数；不读取音频、Reference 谱面或
  MAESTRO 保留测试，不修改候选或应用默认行为。

## 运行

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_onset_retention.py
.\.venv\Scripts\python.exe scripts/summarize_onset_retention.py --output test-data/onset-retention-ledger-final-20261113
```

输出目录必须是新的 ignored `test-data/` 子目录。输入哈希、来源代码哈希、逐时刻账本、
按声部及段落汇总和明确的验收限制写入 `report.json`。
