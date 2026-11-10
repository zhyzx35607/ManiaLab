# MusicXML 源持续事件适配与时钟核验

2026-11-09，首次运行本协议下的真实语料核验前冻结。现有训练缓存把已拆分书写片段当成音符；本轮从源 XML 恢复两层表示，避免把同一个持续音的记谱延续用作新起音监督。

只处理 `dataset-v2/manifest.json` 中 ready 的157份 train 和16份 validation，保留原分组，完整源文件口径；不读 test 数组或运行模型。透明记录：编写适配器前，为查看格式曾打开 `mozart_sonatas/K279-1.musicxml` 前180行，随后确认它属于 test。这次结构查看没有用于预测、评分或训练；不能声称整个开发过程中从未查看 test 源文件。后续批量运行器严格排除 test。

依据 W3C MusicXML 4.0 官方元素说明（2026-11-09已取回核对）：`tie` 表示声音连接、`tied` 表示记谱；显式 `tied number` 可消歧；和弦附加音不推进时钟且可以比首音短；`backup` 不越过小节和 divisions 变化；`duration/divisions` 给出四分音符单位时间。来源：

- https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/tie/
- https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/tied/
- https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/chord/
- https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/backup/
- https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/divisions/

适配器保存结构路径 ID、原 XML ID、part、书写 measure label/序号、staff、voice、声学/记谱 tie 声明。用有理数 quarter 时间读 partwise 中带明确 duration 的音、休止、和弦、backup/forward、连音及弱起；只支持小节边界 divisions 变化，统一转换到无损整数单位。grace/cue 显式排除并计数；带 tie 的排除音、无法解释的时钟、移调声部等显式报错，不猜时间。反复段不展开，谱面时间不等于播放时间、物理松键时间或 LN 长度。

只有两端都声明 sound tie、同 part/pitch、时间精确相接且互为唯一候选时才连接；双方明确 notation number 不同则不配对。有多种并行同音匹配时保留 unknown，即使 voice/staff 看似可以猜出连接，也不当真值。声明缺失、冲突、time-only 条件、未支持的 notation、性能起止偏移均禁止相关监督，歧义沿候选连接传播。已知不存在 tie 的独立音仍可标记 score attack。新层不会倒推修复旧缓存。

自动测试覆盖源声明和时钟；另外对旧七份自造 XML 直接解析，不读取作者连接作为推理输入。六个无歧义例应恢复原事件数；并行同音且无 number 的例应有4个 unknown 书写片段和0条已确认边，与之前作者明确给边时得到两个事件的口径不同。

真实语料每份独立读取，然后对内存 XML 副本的所有 note 注入唯一 ID，仅修改 ID 属性，由固定 Partitura 1.8.0 导入，逐 note 比较 part、pitch、quarter 起止、staff、voice。副本的 ID 显式映射回 XML 结构路径，不用相似音高或时间猜身份，不使用 Partitura tie 指针做真值。时间容差1e-8 quarter；缺失或额外音符、时间/字段差异逐项保留，不能删除失败文件后宣称全部通过。Partitura 本身可能有错误，因此不一致是待诊断证据，不自动否定源表示或自动改源时间。

输出新目录 `test-data/piano-svsep/musicxml-events-v1`，禁止覆盖。保留逐文件源表示、未知原因、导入差异、异常、输入和代码哈希及快照。原语料、旧缓存和权重不变。公开仅汇总；本轮没有训练、应用启用或音乐验收。

## 第二版运行前修订

首轮在完成51个文件记录后主动中断，快照与产物保留，不视作完整核验。子代理用额外自造例指出三处监督边界问题：不完整tie的未标记潜在对端仍会成为attack、note级time-only漏检、多part控制小节时钟未核对。修复后，缺失/冲突声明沿同pitch精确相接的潜在对端传播unknown；保存并屏蔽note/tie的time-only以及性能偏移条件；所有part的控制小节编号顺序和起止必须一致，non-controlling显式不支持。拒绝不完整时钟比默许错误同步更适合本轮标签准备。

第二版仍按原冻结173份完整文件运行，输出 `musicxml-events-v2`；不覆盖或混用v1。新增回归例与修复版测试必须先通过。v1原测试、七例和前51份原始结果只证明该版本的已查范围，不能作为修复后结果。
