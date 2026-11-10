# 真实语料延音审计的等价加速复跑

2026-11-09，在 `corpus-ties-v2` 首次运行前冻结。基础定义、173 份 train/validation 范围、全部过滤与验收条件沿用 [piano-corpus-ties-protocol.md](piano-corpus-ties-protocol.md)，不读取 test 内容、不改缓存或上游。

初次 `corpus-ties-v1` 的运行器 SHA-256 为 `7579cd1e5a2a46ed6d88890ca79cbb85b3adb7a52f74a748c78662bf538d317b`。最先两个文件已完整完成，其中首文件约 59 秒；第三文件在导入后的指针检查遇到 `Imported tie_prev endpoint is outside written notes`。旧目录、源码/协议快照及进度全部保留，初次运行已主动停止，没有完整报告，也不计为完整语料审计。

已确认的性能原因是审计器 `score_snapshot` 在每个音符上重新访问 Partitura 的 `measure_number_map` 属性，该属性会重建同一 part 的完整映射。此次仅将该映射移到每 part 的音符循环之外，一份快照中的每个 part 只创建一次映射。上游预处理函数、导入器、窗口定义、过滤、关系与事件计数保持一致；不借机处理首轮发现的导入异常。

输出改为新的 `test-data/piano-svsep/corpus-ties-v2`，拒绝覆盖。初次完成的两份文件必须与此次相应文件逐项比较，排除 elapsed_seconds 后，所有摘要、书写几何、关系和事件差异完全相同；原/处理后数组也必须完全相同。初次异常记录同样比较，不能因为换实现而静默丢掉失败例。记录新旧运行器和协议身份、此次实际源码快照；旧目录仍保持不变。
