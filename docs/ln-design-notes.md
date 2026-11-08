# LN 设计迭代的依据与验收

本轮从 `916cc67` 开始，重点是 6–10K、尤其 10K 的按下／保持／释放关系。对音错误与键型偏好分别验证；改定轨时不能改变声音起止来掩盖问题。

## 参考资料

2026-11-08 实际查阅以下一手资料。社区指南是作者经验，评级规范是提交规范，都不能直接变成普遍的“好谱分数”。

1. [Torosu](https://github.com/Manolo-dl/Torosu)，本地参考提交 `b283791f46622b0dd1f47f53afa10bd62d700e66`。重点读了 `analysis/layout.rs`、`analysis/patterns.rs`、`tasks/key_converter.rs` 和 `tasks/fln_converter.rs`。借鉴的是手指布局、保持状态、短释放间隙和连续搜索；FLN 转换器按下一音填长的功能不能证明音乐持续到了那里。
2. [osu!mania Hold note](https://osu.ppy.sh/wiki/en/Gameplay/Hit_object/Hold_note)：区分 anchor、inverse、shield、reverse shield、stair和release。尤其不能把长条后面的短键直接计成 LN→LN 的反键。
3. [osu!mania Pattern](https://osu.ppy.sh/wiki/en/Beatmap/Pattern/osu!mania)：键型是有意组织的事件序列，混合键型不能仅用短键／长条比例描述。
4. [osu!mania Ranking Criteria](https://osu.ppy.sh/wiki/en/Ranking_criteria/osu!mania)：持续音的表达、尾端与声音或节拍的关系、LN间隔、同手配合和密度都与难度有关。这里的难度分级不是本程序自动生成质量的验收结论。
5. [Beatmap tags](https://osu.ppy.sh/wiki/en/Beatmap/Beatmap_tags)：hybrid、long note、release 等是描述不同体验的类别，不能当成同一根“LN更多”的滑块。
6. [osu!mania Strain 实现](https://github.com/ppy/osu/blob/master/osu.Game.Rulesets.Mania/Difficulty/Skills/Strain.cs)：参考保持与释放关系进入难度计算的方式；不复制其数值作为人体工学真值，也不把本工具的统计称为官方星级。
7. [The Guide for 10K Mapping](https://osu.ppy.sh/community/forums/topics/1336072)：作者介绍10K的双拇指、五指手型、持续层与短键层配合、同拍或异拍放手等，并链接10K资源表。按作者观点理解，不推广为所有玩家偏好。
8. [10K社区资源表](https://docs.google.com/spreadsheets/d/1alQQ7evfoZMF6Z2l4Il6oeTt1uygWWYFSxD2JyGBUh0/htmlview) 与 [10K难度表](https://docs.google.com/spreadsheets/d/1UGetQtAWjoD9HKlCgrNAqJ9TVasBDP-gWCmP3gvlXpw/htmlview)：用于寻找LN/Inverse参考，具体表项还需核对谱面、键数、作者和来源。下载素材仅留本地。
9. [Rhythm Game Automatic Chart Generation and Evaluation Using Music Data](https://openaccess-api.cms-conferences.org/articles/download/978-1-964867-13-7_19)：研究模型与传统算法组合生成谱面，包含有限用户评价；游戏、训练条件及问卷规模与本任务不同，不能据论文结果声称本项目已达到相同水平。

## 本轮可检验的改变

- 定轨评价包含没有新起音的释放时刻，报告部分放手、同手按下与放手、长按中短键、同轨LN接续、Shield和Reverse shield。
- LN数量比例与占用轨道时间分别计算。相同的LN数量可以产生完全不同的占轨和放手密度。
- 连续搜索增加不同释放次序的和弦分配；不再始终强制低音到高音的轨道排列。已有音高连贯偏好仍参与评价。
- 同手协调、放手分离、反键倾向只影响定轨。反键只奖励已有LN之间的正释放间隙，不能把静音填成长条。
- 新增单物件修正：音频模式可改起止、轨道、增删，起止按候选的拍线量化；严格重定轨只允许换轨。同轨冲突拒绝保存，修改作为新候选，原始采音和先前候选保留。

本工具使用 `min(0.3拍,150ms)` 统计近邻接续，参考Torosu的启发式窗口；这个数值不是社区对所有Inverse或Shield的唯一标准。7K/9K共享中键的手部负担只是各半近似，实际手指选择尚未逐事件求解。

## 固定验收

先在原有14份完整谱面上比较基线与新默认，覆盖6–10K。它们是已经看过的开发样本，不是新歌盲测。另补充高LN比例10K，不能用大量短键谱替代LN测试。

硬要求是所有起止、物件身份、类型和原始音效字段保持一致、零同轨结构冲突。默认短放手再按和快速同轨的总数不得比基线增加超过 `max(5,5%)`；单曲退步也要保留。Hybrid和反键参数需在有对应素材的样例上改变目标统计，同时报告负担和耗时的变化。没有素材支撑时，参数无效必须如实显示。

这些验收验证保真、可控和实现行为；音乐性、玩家舒适度、自动采音准确度需要另行评价，不用这份定轨对照代替。

## Timing 的独立修复

旧版深度模式选用鼓声后会省略鼓点不足的12秒窗口。已定位到一首前35秒与后段不同的参考曲：后段鼓点稳定，使旧版把全曲标成稳定，前奏却没有被检查。

新逻辑保持候选BPM、Offset和原门槛，用混音起音补查这些遗漏窗口。它只能撤销错误的自动通过，不能把原本未通过的结果放行。完整67曲开发缓存对照单独写入 `timing-coverage-validation.json`；这只是重新检查既有起音，不能冒充重新运行模型或独立音频真值。
