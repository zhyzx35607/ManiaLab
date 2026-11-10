# 预先固定的 Tsumugi 起音回收方案

固定于首次读取本次对照匹配标签之前。本方案仅用于离线消融，不启用产品功能、不改冻结解码器。

1. 输入必须通过 V1 construction/window/finalize/raw-notes 逐字段审计。
2. 候选只来自审计 `filtered_early_unclosed_onset_source_indices`：原构造 `has_onset=True` 且 `has_offset=False`，所在组被官方 finalize 过滤。保留是否有 boundary head、其原始 flags、window、原始模型区间和 source index。
3. 输出候选 onset=start/sample_rate，offset=onset，release_reason=`unresolved_or_postprocessed`，application_ln_eligible=false。零时长只表达起音事实，不能当作完整物理区间；原始区间只作为证据保留。
4. 类别保留该构造的模型类别及候选，program=-1；不按真实乐器标签改类别，不过滤错误类，包括 drums。
5. 去重固定为同 pitch、同 slot，起音样本距离满足 `abs(delta_samples)*1000 <= 50*sample_rate`。先对所有已有最终起音去重，再按 `(start_sample,pitch,slot,source_index)` 升序，贪心对已接受回收候选去重。不同类别同 pitch/slot 仍按同样规则；不同 slot 不去重。不做传递闭包，也不按置信度选候选。拒绝原因和对应来源全部记录。
6. 落在已存长音内部但距离其起音超过 50ms 的候选仍保留，以观测漏重弹；不裁剪、延长或删除原始音。
7. 全分母消融必须等待冻结 100 例全部完成并验证 manifest、计划、完成标记、comparison、报告与证据 hash 一致。60 GuitarSet/strings、40 vocadito/A1、40 vocadito/A2 独立汇总，禁止挑更好的人声标注。
8. 在原 scoring window 按 onset 选样，所有类进入预测分母。复用原匹配器的最大一对一匹配、精确半音和固定 50ms/40ms 起音容差；drums 保留在分母但不能匹配有音高标签。只输出起音 TP/FP/FN、P/R/F1 和起音误差，不输出任何音尾误差或完整区间分数。
9. 在完整100例到齐前，只对四组 smoke 做结构控制。不读标签、不挑例调整阈值。输入与旧结果均不覆盖。
