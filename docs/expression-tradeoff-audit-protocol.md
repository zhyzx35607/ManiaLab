# 表达取舍静态审计协议

本工具只核对已冻结的《樱花草》10K默认选择结果，为候选起音与LN取舍提供
可复核的定位信息。它不运行选音器、模型、音频处理或参考谱评价，也不改变
应用默认行为。

## 冻结输入

- `test-data/sakura-tsumugi-expression-onset-v3/analysis.json`：
  `26a0250568b2ce149022c4aa1549af2d9dd49b045372959a6b26b5342ba5313a`
- `test-data/expression-precision-v1/default/10k/selection.json`：
  `24f23cc1d9e7452e5ede424f10054d285b0d2ecdd26b33095e651716f0b0b44`
- `test-data/expression-precision-v1/execution-plan.json`：
  `e6d47956ec469a98c81997f9e09d36011b8c2da6a8b75041aca4c4ad42b1b31e`

所有分析候选与选择决策 ID 必须一一对应；原始起音、声部、段落、选中/省略
总数及省略理由必须一致。输入哈希不同则停止，不会覆盖旧输出。

## LN 定位标记

“省略的持续音资格候选”按现有表达阶段的必要条件从冻结字段重建：来源为
`pitch`，声部模式允许持续音，候选具有至少 `min_ln` 的 `expression_end`，
释放不是 `UNOBSERVED_RELEASES` 中的未观测边界，且不是 articulation tap。
本快照 `min_ln=240ms`。这表示候选在进入整体编排前符合持续音条件，不表示
其音乐上正确、可闻、应被选中或可无冲突加回。

结果按 `section + voice + 原始 analysis start` 折叠多个候选 ID。导出字段
`chart_start/chart_end` 仅用于识别实际导出的 LN；区间重叠使用同一原始分析时钟
上的 `start/expression_end`，不用于Timing验收。

`individually_uncovered_ms` 表示某候选起音组的最长持续区间与当前同声部选中
LN覆盖区间比较后，单独考虑时未被覆盖的部分。不同组会互相重叠，数值不能相加
或解释为可增加LN总时长；求解容量和密度约束也没有重算。结构复核队列按该值
排序，仅供以后挑选人工检查对象，不是音乐价值或质量排名。

## 运行

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_expression_tradeoffs.py
.\.venv\Scripts\python.exe scripts/summarize_expression_tradeoffs.py --output test-data\expression-tradeoff-audit-v1-20261113
```

输出只能写到新的 ignored `test-data/` 子目录。报告保留输入与来源 SHA-256，并
明确记录未运行神经推理、未读音频或Reference、未计算音乐价值、未验收Timing或
音乐质量。之后仍须真实音频听审与osu!试打。
