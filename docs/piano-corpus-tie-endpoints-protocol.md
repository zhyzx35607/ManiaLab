# 审计集合之外的延音端点类型诊断

2026-11-09，在 `beethoven_piano_sonatas_05-3` 的只读初查之后、扩展检查全部 22 个审计失败文件之前冻结。初查已确认该文件的 6 个端点均为已注册的 `GraceNote`；`part.iter_all(Note)` 默认不含子类。因此本轮不预设“集合之外”等于导入器悬空或源数据错误。

范围仅为冻结 `corpus-ties-v2/report.json` 中 22 个含 error 的原 train/validation 文件。报告 SHA-256 为 `1fbdeb843e237f97396ae7258ec349ca07f9015274f5a296b729fa11f408bb8d`。逐文件核验 manifest 身份后，只读导入一次，枚举普通 `Note` 指向其默认遍历集合之外的 tie_prev/tie_next 端点。记录实际 Python 类型、是否属于 `Note` 子类集合、`GenericNote` 集合和已注册时间线对象、起止是否存在，以及归一化关系数量；不调用上游拆 tie、不改缓存、不重跑全量审计或模型。

同时直接读取原 XML，统计带 sound tie 的 grace/cue/rest 与普通音符，并在本地保留这些特殊音符的结构位置、声明及必要属性。原 XML 没有显式 ID 时，不凭 Partitura 自动 ID 猜测逐节点对应，也不将同一文件内两类数量相近当作一一映射证据。完整 tie 配对仍由独立源读取层负责。

产物写入新的 `test-data/piano-svsep/corpus-tie-endpoints-v1`，拒绝覆盖；记录运行器、协议、Partitura 导入/score 代码及来源哈希，每份原文件执行前后均不变。公开汇总只含文件身份、类型/声明计数和异常，不含逐音符内容。旧 151 份覆盖统计和 22 份异常身份保持冻结；本轮仅解释覆盖缺口，不补写缺失训练指标，也不视为支持 tied grace/cue/rest 的实现。
