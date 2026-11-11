# 续接进度

更新：2026-11-11，Asia/Shanghai。本文件用于中断或额度耗尽后恢复工作；
先核对实际进程、Git 和产物，不能把这里记录的运行中状态当作已经完成。

## 已保存的应用

- 可下载版本：`v0.1.0-alpha.6-dev.4`，含声轨导入、逐段表达、可选钢琴、
  Tsumugi 贝斯/其他伴奏及 FL 吉他采音。使用方式见
  [乐器采音选择](instrument-choices.md)。
- 开发基线提交 `63fcc08baff631d642e1bc8ae69620a83cc7a1cb` 已推送，
  Windows validation `42839955554` 成功。此前本地完整检查为
  1866 passed、85 skipped；不是本轮重新运行的结果。
- 之前81个开发提交已通过 PR #1 归入主线，合并提交
  `9b11b10d913acb19070f228397bb7cbf8bc5233c`；PR检查 `42846636872` 成功。
- 模型恢复、比较入口和续接说明已通过 PR #2 合入主线，合并提交
  `0a43911477f7c11eea58a31e57bfdfa88c753e3c`；PR检查 `42847322903` 成功。
- 鼓组剩余容量补充已通过 PR #3 合入主线，合并提交
  `40e1a061c35caf89ad7b004a858b9619892e1fb5`；PR检查 `42852204830` 成功。
- Mega53长窗和采音结果已通过 PR #4 合入主线，合并提交
  `a7b2478c2df09e0f6dc2a3d0dc409ea356dabdac`；PR检查 `42854329970` 成功。
- Harmonica显式FP32对照及完整结果已通过 PR #5 合入主线，合并提交
  `28c9a51613d33e991f1b7bb888e6c1940a627cf9`；PR检查 `42856082126` 成功。
- 当前Nova other原声与落线对照已通过 PR #6 合入主线，合并提交
  `3a1aa6d139d24b41508143e82773973d3789f5f6`；PR检查 `42859344954` 成功。
- MuScriptor完整结果及独审已通过 PR #7 合入主线，合并提交
  `2510e103ffe83a38ae3fb4963d35a9fe1b498ecf`；PR检查 `42860878982` 成功。
- 等价候选去重修复已通过 PR #8 合入主线，合并提交
  `3b11f6d1b0c513b9e83fa2d3c4bc4c9004018933`；PR检查 `42863989466` 成功，
  GitHub检查1901 passed、85 skipped。
- 已知乐器音色验证入口已通过 PR #9 合入主线，合并提交
  `279bf6e1e9cdca9d906301de7b12781481a60d38`；PR检查 `42865447918` 成功，
  GitHub检查1905 passed、85 skipped。
- 音色分类与声部分组完整结果已通过 PR #10 合入主线，合并提交
  `cc3e82817a9940008c0e2703e34cda777c776430`；PR检查 `42868880993` 成功，
  GitHub检查1908 passed、85 skipped。
- 区间空隙保护已通过 PR #11 合入主线，合并提交
  `3dfb85da843fbaddd46a8d9c3b53949a3498e7b6`；PR检查 `42872567983` 成功，
  GitHub检查1928 passed、85 skipped。
- 钢琴模型低活动上下文已通过 PR #12 合入主线，合并提交
  `ff4d7fed161bbe30ab9d43b0cda42ab60f57b4a9`；PR检查 `42875415387` 成功，
  GitHub检查1950 passed、85 skipped；研究规则未默认启用。
- 钢琴低活动原声核查已通过 PR #13 合入主线，合并提交
  `46cb32647c1fc62b3ee99c33e774013eac8f117c`；PR检查 `42877636337` 成功，
  GitHub检查1961 passed、85 skipped。
- 钢琴跨窗持续修复已通过 PR #14 合入主线，合并提交
  `ad9c8c05ed69848d5bfc0538c834c7729ee4b2f8`；PR检查 `42882871715` 成功，
  GitHub检查1980 passed、85 skipped；没有新Release。
- 同权重和弦起音排序已通过 PR #15 合入主线，合并提交
  `12bb01c01f5daa6cec0a454c7e504071f94b1f80`；PR检查 `42885147076` 成功，
  GitHub检查1992 passed、85 skipped；没有新Release。
- 相邻节奏组连贯取舍研究已通过 PR #16 合入主线，合并提交
  `abf8e31547cb71d7cf23330eebb8b8ac278967ad`；PR检查 `42887691860` 成功，
  GitHub检查2005 passed、85 skipped；研究参数仍默认0，没有新Release。
- 原区间乐器评分恢复入口已通过 PR #17 合入主线，合并提交
  `2179391207ecaadcd6bb966dd0bdf3ef04533d2d`；PR检查 `42890087106` 成功，
  GitHub检查2021 passed、85 skipped。后续完整结果另开PR；没有新Release。
- 乐器评分完整结果已通过 PR #18 合入主线，合并提交
  `97f9fc1`；GitHub检查 `42892770435`、`42892755994` 均为2021 passed、
  85 skipped；没有新Release。
- 有限持续覆盖研究已通过 PR #19 合入主线，合并提交
  `1f90d1c06530ed59124d7882ccdbda46b0ec2fd9`；检查 `42969262042`、
  `42969130741`成功。研究权重仍默认0，没有新Release。
- 自动 Timing、真实混音的漏采/串音、声部连续性和 LN 音乐性尚未最终验收。
  不宣称已超越 Mapperatorinator。质量标准见[目标](music-quality-target.md)。

## 本阶段新增

- 表达求解精度对照完成，当前分支`research/expression-tradeoffs`。
  旧持续覆盖9K方案在10K完整约束中可行且目标更高；原0.005相对间隙允许
  这部分损失。固定10组对照全部gap=0，主选择2.60–5.20秒，原15秒/5秒
  时间上限不增加。默认6–10K及持续覆盖6–8K逐物件不变；持续覆盖9K和
  10K均1491物件/425LN。只修正数值取舍，持续覆盖仍不默认启用。
  生产两处gap改0，`attack_optimal`要求两个阶段status0且gap0；不表示
  音乐验收或联合最优。新增9项及相关检查共97 passed。
  生产修复后完整套件2053 passed、85 skipped；72份case产物、14份源码
  与85份输入独审通过，默认selected/placed全部字段与顺序一致。
  入口`scripts/check_expression_precision.py`，产物
  `test-data/expression-precision-v1`，session `49169`已exit 0，勿重跑。
  完整[结果](expression-precision-results.md)与[协议](expression-precision-protocol.md)。
  无推理/训练。后续研究段内节奏骨架必须用确认过的全曲拍格，不能从每个
  短组首重置相位，也不能把全部support声部误认为重复伴奏。
- 固定节奏骨架研究已完成，当前分支`research/rhythm-policy`。新增显式恒速
  拍格策略接口，默认不启用；每个段落/声部/乐器类别固定stride与phase，
  删除起音不贡献奖励、不被LN代替，原两拍source group的密度检查和补鼓
  检查时刻保留。合成边界及相关检查通过。
  《樱花草》默认与持续覆盖、6–10K、完整/phase0/phase1共30组已完成，
  产物`test-data/expression-rhythm-v1`。完整方案逐Note重现；10K默认
  1496/382，phase0 1435/440，phase1 1391/532。phase1同时造成其他
  声部容量重排，不能解释成钢琴伴奏被安全抽稀；本轮无新增原声窗口、试听
  或音乐验收。结果见[固定节奏对照](expression-rhythm-results.md)与协议。
- 有限 LN 持续覆盖研究已完成并合并，原分支`research/sustain-coverage`。
  新参数`select_expression(..., sustain_coverage=.25)`默认0，只给通过现有
  尾音资格检查的`sustain` pitch 长音有限奖励，不改起止或尾音资格；
  资格字段不代表物理松键时刻已独立确认。
  固定《樱花草》6–10K候选重放的物件/LN分别为1490/368→1482/414、
  1493/378→1487/420、1496/382→1491/425、1496/382→1492/423、
  1496/382→1483/437。五种键数都伴随鼓与钢琴起音组损失；10K损失33个鼓、
  52个钢琴起音组，物件净减13个。结构、容量、落线和`.osu`回读通过，
  但尚未试听/试打、不能判音乐改善，所以保持研究用途、不接入默认。
  24个8秒原声窗口及click预览保存在ignored本地目录；
  216–224秒关注段无变化。详见[结果](sustain-coverage-results.md)与
  [固定协议](sustain-coverage-protocol.md)。专项174 passed，完整套件2044
  passed、85 skipped。没有重新推理或训练。
- 原AMT乐器评分恢复及完整评价已完成；入口PR #17已合并，结果分支
  `research/instrument-score-results`。
  [固定协议](instrument-score-evidence-protocol.md)先于本阶段推理/读取逐音标签。
  新入口`scripts/recover_tsumugi_instrument_scores.py`复用原区间，不重解码，
  保存36类logits、边界响应和拼接来源；排名逐项复现、音符不变才继续。
  16项新增控制与相关检查共52 passed。v1因配置tuple/list比较在首次前向前
  停止，保留目录；JSON规范化修正后v2全35份完成，session `54340`实际exit 0。
  315窗/5202区间/4790最终音，完整排名精确复现、边界最大差0；388音有多个
  来源，8音来源类别不一致。均值改变6音，其中2音在计分区；混奏带类起音
  F1仅82.44%→82.50%（多1匹配），完整区间69.82%不变，独奏均不变。
  全部推理完成后才读逐音标签，560组独立计数通过，评价实际exit 0。
  最高评分档仍有9个混奏唯一对应错类和99个无对应音，不默认平均类别或
  用分数筛音/指定主旋律。完整[结果](instrument-score-evidence-results.md)。
  产物`test-data/instrument-score-evidence-v2`及`test-data/instrument-score-evaluation-v1`。
  当前无运行中推理、研究队列、新训练或安装包。勿重复本批35份前向。
  已完成的评价命令：
  `.venv/Scripts/python.exe scripts/check_tsumugi_instrument_scores.py --evidence test-data/instrument-score-evidence-v2 --output test-data/instrument-score-evaluation-v1`。
  类别评分不能直接当音存在概率、力度或声部重要性。
- 相邻节奏组连贯取舍研究完成，分支`research/expression-coherence`。
  新可选研究参数`select_expression(..., coherence=.25)`惩罚相邻组的选中/
  落选切换，默认0，不接入应用设置。权重和边界规则先固定，未按结果调整。
  13项控制与相关检查共151 passed。固定6–10K默认逐音复现，新方案完整
  选择/定轨/导出回读通过；session `66354`实际exit 0，
  `test-data/expression-coherence-v1`。10K物件/LN仍1496/382，切换45→36，
  但10条消失中3条来自整组省略，另新增1条，不能称音乐更清楚。
  8张原声对照图已实际查看、24音频回读，session `22073`实际exit 0，
  `test-data/expression-coherence-audio-v1`。216–224秒持续层退出、孤立高音
  进入，不能凭更少切换启用默认。详见[结果](expression-coherence-results.md)。
  默认基线仍PR15的`chord-evidence-ties-v1`，勿将研究输出当发布行为。
  无运行中研究队列、推理、训练或新安装包；不再沿本版只调平滑权重。
- 同权重和弦起音排序修复完成，分支`fix/chord-evidence-ties`。
  Tsumugi的1.0为统一权重；原降层先按音高排除更贴近拍线的候选。
  现在同强度、有完整起音证据的音高行用原对齐评分打破平局；保留强度优先，
  无证据和瞬态行沿用原顺序，不统一选高音，不变造候选首尾。
  修复前7项新反例失败，修复后12项新检查及相关检查共138 passed。
  固定6–10K旧基线逐音复现，新选择、定轨与导出回读通过，session `28331`
  实际exit 0，产物`test-data/chord-evidence-ties-v1`。10K1499/373→1496/382，
  伴随12个鼓起音和1个贝斯起音落选；不能称只有和弦代表变化或音乐已改善。
  完整[结果与复现](chord-evidence-ties.md)，无新推理、训练、安装包。
  最新固定基线改用该目录；勿把旧1499/373当作修复后结果。
- 钢琴跨窗持续修复完成，分支`fix/piano-window-continuity`。
  原52.5秒突降定位为前后窗口的持续响应不一致；前窗包含起音且能继续跟踪。
  初版仅用frame延续造成5个踏板尾误差增加，未接入。最终还要求原尾恰在
  切换点、前窗以放音回归结束，保留起音与力度，接入可选钢琴专用检测器。
  缓存revision=2，保留`window_context_continuations`；无新安装包。
  SMD11处尾误差均更接近踏板；MAESTRO开发30处中29改善、1处3.91→14.77ms。
  物理按键参考有相反变化，1134标准按键区间1失1得，不能称松键识别改善。
  《樱花草》0变化，最初Liszt7个截断只恢复有回归支持的3个，不称全面解决。
  45项局部测试通过；22实录/550窗口应用重放、91单窗控制、160项开发计分
  独审通过，`test-data/piano-continuity-runtime-v1`。两版SMD各88数组/176计分
  独审通过。详见[完整结果与复现](piano-window-continuity-results.md)。
  采集session `51604`（SMD）、`32481`（Nova），评价`66157`/`97021`，
  开发重放`21351`、应用核查`41120`均实际exit 0。143+57不同原窗口已保存，
  正式队列复用定位的2窗，只新增198次网络前向，勿重跑。
  没有运行中推理或子代理；1155/736/687/841未读取。
- 钢琴低活动原声核查完成，分支`research/piano-gap-audio-review`。
  session `11495`实际exit 0，104处SMD和《樱花草》6处全部保存，
  `test-data/piano-gap-audio-v1`；预定10图已全部实际查看。独审exit 0、
  480文件哈希、110片段逐样本一致、34830项频带统计、7665帧直接FFT通过，
  `test-data/piano-gap-audio-audit-v1`。11新控制与相关检查共105 passed。
  2048/8192窗分别16/100处无完整间隔帧，记null；不能把跨界图当精确音尾。
  Liszt 52.500–55.140s在模型低活动期间仍见持续频谱成分；参考重叠中又有
  仅数毫秒的边界差异。不能统一称误休止，更不能把低活动等同于物理停音。
  完整[结果与复现](piano-gap-audio-results.md)。没有新推理、训练或默认行为。
  该分窗来源已进一步定位并修复其中有明确放音证据的情况，见上一条；
  保留全部原声核查产物，不重复本轮110处测量。
- 钢琴源模型低活动上下文验证完成，分支`research/piano-inactivity-context`。
  原Nova float PCM与应用缓存身份核对一致；1557候选起止音高一致，89处旧
  `frame_decay`与当前`frame_inactive`差异是既有语义修复，保留历史分析。
  原6个候选空隙中，第一个仍有持续激活，其余存在低活动段，不按空白直接
  推断静音。研究工具`scripts/piano_inactivity_context.py`，完整入口
  `scripts/check_piano_inactivity_context.py`。17项研究、5项独审控制与相关检查
  共122 passed。session `99800`实际exit 0，输出`test-data/piano-inactivity-context-v1`。
  《樱花草》新增5个模型支持边界、10个持续提案，6–10K各多5条钢琴LN，
  声部起音集合不变；6/8/10K另有1/2/1个other候选替换，不能说其他物件不变。
  11份SMD新增边界0、表达不变，不是有效休止准确率验证。随后补查104个
  原始候选空隙低活动区间，28/29个与按键/踏板参考持续重叠，20个全覆盖；
  不等于这些音已试听确认可闻，但不足以可靠自动判停音，因此不默认接入。
  全部重放后才读取SMD标签。6图实际查看、18音频回读；242文件及逐帧独审
  通过，`test-data/piano-inactivity-audit-v1`。完整[结果](piano-inactivity-context-results.md)。
  没有神经推理、新权重、安装包或运行队列，不重复本次已完成的16份重放。
- 连续起音表达的区间空隙保护验证完成；分支`fix/articulation-interval-gaps`，
  PR #11已合并，见上方提交与CI记录。
  旧实现两个反例失败，修复后20项专项与相关测试共216 passed。按原候选区间
  已结束至少80ms新增分组边界，未知音尾和更早仍持续的音阻止新增边界；
  不改变原起音、音高或音尾。完整协议见
  [区间空隙](articulation-interval-gap-protocol.md)。重放入口
  `scripts/check_articulation_interval_gaps.py`，session `36934`实际exit 0、输出
  `test-data/articulation-interval-gaps-v3`。最终6–10K选中音符、导出坐标均精确
  不变，11份SMD全部6122原候选/6070强候选的表达不变。没有新的音乐收益
  结论，完整[结果](articulation-interval-gap-results.md)。
  v1因检查脚本直接比较原JSON附加字段与Note失败，未完成
  生成；改为规范化Note比较后另开v2，不覆盖v1。v2全量重放完成，Beethoven
  9个原持续音被新短键化；增加失败反例后，v3限制只细分旧规则已短键化的组。
  原持续音不能变成新tap。初步《樱花草》表达没有变化，不把最小反例通过
  说成该曲LN改善。
- 音色分组35份已知乐器对照完成，shell session `66272`实际exit 0。旧首份
  复用，新34份串行耗时902.841秒；全部结束后读取本轮标签。不改原音高/起止。
  聚类后独奏类别起音/区间F1持平92.44%/87.74%；混奏82.44%/69.82%降至
  80.25%/67.76%。35份各为1个cluster；混奏唯一对应子集的声部分组成对
  precision仅27.94%，不能用作自动旋律身份。不接入、不对这版追加Nova
  全曲或按本批标签调阈值。完整[结果](instrument-family-validation-results.md)。
  入口`scripts/check_instrument_refinement.py`，产物`test-data/instrument-family-validation-v1`；
  21项查询/计分/分组测试及5项独立匹配控制通过。独审358文件、4790事件、
  840组匹配计数通过，回执`test-data/instrument-family-independent-v1`；未重建
  模型前向、聚类分区或等价匹配的误差均值。没有新默认行为或安装包。
- 修复同占用、同强度、同表达含义候选的去重顺序偏置，沿用原对齐评分择优，
  同分按ID；不同尾长、乐句代表和不同表达依据不作等价替换。25项专项及
  相关检查共114 passed。固定《樱花草》6–10K重放、定轨、导出回读通过：
  各键数160处候选代表变化，11对实际保留，全部other；物件/LN总数不变。
  6K/7K/9K分别额外2/1/2处对象取舍，8K/10K对象内容不变；不作音乐收益
  结论。详见[重复采音选择](expression-duplicate-selection.md)，新比较入口
  `scripts/check_expression_duplicates.py`，产物`test-data/expression-duplicates-v1`。
  119文件独审通过，回执`test-data/expression-duplicates-independent-v1`；
  8K/10K导出坐标精确不变，其他键数的重新定轨变化已在文档中列明。
- 原Nova other声轨已完成当前10K的三个固定规则选窗对照：200–208、
  96–104、16–24秒；3图实际查看、9份原声/起音click音频回读通过。
  186个other中65个量化移动超过30ms，62个向后、3个向前；这不是65个
  已确认错音。实际导出相对整数原候选的偏差>30ms为23个，中位+1ms；
  区分原候选、制谱提前、落线三种时间，不统一调Offset。
  无新推理或生产改动；见[原声对照](other-onset-current-review.md)，入口
  `scripts/review_current_other_onsets.py`，产物`test-data/sakura-other-onset-current-v3`。
- 鼓组整体落选时，只在剩余容量内补充一个已有瞬态起音，固定原方案全部
  ID及起止。89项局部检查通过；固定《樱花草》6–10K完整重放、定轨和导出
  回读通过，6K补5鼓、7–10K补4鼓，LN不变。最终产物
  `test-data/drum-group-fallback-v3`，见[结果与边界](drum-group-fallback.md)。
  不再运行旧 `check_expression_current_replay.py` 来生成“当前”基线；它冻结
  了改动前选择器AST。当时比较入口为 `scripts/check_drum_group_fallback.py`；
  本轮去重修改后的比较改用 `scripts/check_expression_duplicates.py`。
- Harmonica 第三方复现的固定对照入口：全部35份已有 URMP 输入及40秒静音。
  安全载入真实权重已通过，116个存储张量逐值核对、679444参数、6个解码
  控制通过。原bf16失败后显式FP32完成全部输入及独立复核；独奏起音F1
  89.32%／区间75.92%，混音56.99%／21.17%，均低于Tsumugi。萨克斯独奏
  有改善，不作整体替换。不是论文作者权重，未查到源码许可，未打包。
  见[协议](harmonica-reproduction-protocol.md)及[实测结果](harmonica-reproduction-results.md)。
- MuScriptor 公开 GGUF small 已下载并核 SHA，逆名称/布局转换为 safetensors；
  正向转换与回读逐值相同。原 Python 模型严格载入122个状态张量通过，
  102441984参数。15个纯符号事件控制和6个计分控制通过；保留完整token、
  人为封口、错误区间与原始类别。全部35真实输入及静音已完成并独审，
  独奏起音/区间F1为76.32%/58.70%，混音62.63%/41.17%，低于Tsumugi；
  最后一块缺EOS和异常事件完整保留，不接入、不追加这版Nova整曲推理。
  见[完整结果](muscriptor-small-results.md)。
  见[权重准备](muscriptor-gguf-preparation.md)和
  [比较协议](muscriptor-gguf-transcription-protocol.md)。
- Harmonica/MuScriptor均为研究工具，未改变应用默认，没有新训练或新 Release。

## 已完成的固定队列（勿重跑）

原 shell session `69438` 已完成 Mega53 分离及下游采音，但在Harmonica首个
静音控制的CPU bfloat16 LSTM处报错，已实际exit 1；当时MuScriptor尚未启动。
显式float32兼容入口已修复，新串行session `54376`已完成Harmonica全部输入
及MuScriptor全部36输入、评分，实际exit 0。音色分组session `66272`也完成
全部35输入及评分，实际exit 0；当前没有待运行的神经推理。
session/PID 仅用于本次会话定位，重开后先
确认命令行和启动时间，不能凭旧PID启动、结束或重复任务。

1. 分离已完成：原session `42354`、Python PID `15668`已退出。
   `scripts/check_mega53_long_context.py --output test-data/mega53-long-context-v1`
   使用作者20秒窗和拼接规则，共22份PCM、44次前向。日志：
   `test-data/mega53-source-review-v1/long-context-run.log`。独立波形复核通过，
   见[长窗结果](mega53-long-context-results.md)；不能据波形启用默认分离。
2. 下游采音已完成：应用 `.venv` 运行
   `scripts/check_mega53_transcription.py --separation test-data/mega53-long-context-v1 --output test-data/mega53-long-transcription-v1`。
   日志：`test-data/mega53-source-review-v1/long-transcription-run.log`。
   46个转录流程/64用途完成，两个完整来源的整体起音/区间仍低于原混音；
   铜管起音比旧短窗恢复但完整区间进一步下降。结果见上述长窗报告。
3. Harmonica原精度失败记录：音频 `.venv-audio` 运行
   `scripts/check_harmonica_transcription.py --output test-data/harmonica-urmp-v1`。
   日志：`test-data/harmonica-source-review-v1/inference-run.log`。
   错误为 `could not create a primitive descriptor for the LSTM forward propagation primitive`。
   保留原目录，未完成任何输入的完整转录，不能发布声学成绩。
   新运行：`--precision float32 --output test-data/harmonica-urmp-fp32-v1`，
   日志 `test-data/harmonica-source-review-v1/fp32-inference-run.log`。
   原源码未改，只在模型forward_spectrogram内禁用autocast，实际四头dtype
   逐batch保存；详见[精度修订](harmonica-reproduction-protocol.md)。
4. MuScriptor已完成：应用 `.venv` 运行
   `scripts/check_muscriptor_urmp.py --weights test-data/muscriptor-recovered-small-v1/muscriptor-small-recovered-f16.safetensors --output test-data/muscriptor-urmp-v1`。
   日志：`test-data/muscriptor-cpp-review-v1/urmp-run.log`。

上述完整结果均已生成。失败的bf16目录仍保留，不覆盖或当作成功。
两新模型均在各自全部推理结束后才读取逐音参考，原输入与预测没有按标签清洗。

唯一子代理 `/root/piano_stream_exports` 负责静态审核和独立复算，
不运行神经模型。已完成的独审目录：`test-data/mega53-long-independent-v1`、
`test-data/mega53-long-transcription-independent-v1`、
`test-data/muscriptor-independent-v4`、`test-data/harmonica-fp32-independent-v1`。
MuScriptor审计v1–v3假设问题与修订回执保留，最终v4通过；4个等价匹配
下的音尾均值差异另列，不能称全部均值精确相同。没有改动模型或原预测。

## 已知结果与下一步

- 原AMT分类评分完整对照已结束。跨窗类别本来大多稳定，均值只有极小
  类别起音收益且不修音尾，不作为声部显著度或LN依据。下一步回到实际
  Nova声轨的持续层/插奏取舍，特别是已看过的216–224秒持续层退出情况；
  不继续在本批调置信度阈值，也不追加本版全曲分类推理来替代表达改进。
- 连贯取舍研究已实跑并保留全部结果。更少切换可能来自省略短插奏，下一步
  要补声源证据与段落角色的依据，不能按模型类别连续性直接给声音重要性。
  原三窗other不变，所有变化other的分箱已看；勿重复本轮8图或重跑6–10K。
- 同权重排序已在完整固定输入验证。下一步仍检查声部连续表达和取舍；
  先看`chord-evidence-ties-v1/10k/windows.json`的三个原选窗及新旧落选起音。
  候选对齐更好不等于真实主旋律识别，12个鼓起音落选的音乐影响尚未验收。
- Mega53旧6秒与作者长窗均未支持默认接入；长窗44次分离前向及46个下游
  转录流程已完成，两个完整来源仍不及原混音。见[长窗结果](mega53-long-context-results.md)。
- 鼓组修复后的《樱花草》固定候选10K选择为1499物件/373LN，已完成定轨
  和`.osu`回读，仍不是试听验收。改动前1495/373的诊断留在
  [历史规则重放](expression-current-replay.md)。
- 新去重规则的10K仍为1499/373，11对来源代表替换，起止/音高/声部内容
  不变。6K/7K/9K另有少量全局取舍；保持原候选，不把代表评分提高说成补采音。
- MuScriptor small本轮结束；八类乐器汇总两项均不及Tsumugi，不继续调参
  或对这版跑Nova。Harmonica萨克斯正例也不支持全局替换。
- 下一步仍从原Nova other三个固定片段定位实际采音与段落表达问题。
  已证实的去重偏置已修，不重复模型搜索或继续用其解释全部省略；不能把
  65个大移动当作65个已确认错音，不能为减小移动量统一调Offset。
- 有限 LN 覆盖研究不接入默认。下一阶段先为表达层设计可解释的起音损失
  检查，按声部、段落和高价值候选审查与 LN 收益之间的取舍；冻结选择结果
  本身没有逐音真值，不能把全部基线候选硬性锁定。
- 音色分组验证已收束：混奏退步且全部单组，不沿这版默认分组继续调参。
  下一步以原候选和可解释的局部旋律/节奏连续性定位段内表达，避免把模型
  广义类别直接当独立声部；仍需原音和生成结果检查，不以标签稳定度代替。
- 区间空隙保护已完成固定重放。它解决可复现的分组影响，却没有改变
  《樱花草》的最长钢琴起音组；未知重弹尾仍使分组缺少完整休止证据。
  下一步需要结合原声和现有采音上下文定位实际分歧，不把`reattack`直接
  改成已观测音尾，也不重复跑本次已完成的16份对照。当前无神经推理队列。
- 已补查钢琴低活动上下文；不因《樱花草》多5LN就默认启用。SMD空隙与
  按键/踏板延续的冲突原声核查已完成，结果见本阶段新增。区别衰减后继续
  按住与实际可闻持续仍需谨慎，不能把上述频谱检查说成亲耳试听；
  不能把模型低活动或单一整轨RMS当作物理休止或独立旋律标签。
- 四排序列先验仍是独立研究入口，只在固定起音、音符和LN尾端上排序轨道，
  不负责采音、声部选择或决定LN长度，也未接入默认生成。新增窗口级LN/tap
  覆盖、失败负例和最大和弦诊断，并为缓存加入特征名SHA256与显式v2版本。
  验证：`.venv\Scripts\python.exe -m pytest -q tests/test_pattern_sequence.py tests/test_sequence_prior_training.py`
  共24 passed，`py_compile`通过。
  15谱smoke共抽样120窗、留存120窗、负例失败0；歌曲组预算后进入数据数组
  109窗，二次缓存读取的数据和报告完全一致。缓存位置
  `test-data/corpus-expanded/sequence-window-cache-v2-root-smoke-20261111`。
- 全量快照清点为7,907张6–10K谱、2,023个歌曲组；7K占5,777张，8K/9K
  分别只有251/236张，且8K验证/测试各8组、9K验证/测试6/3组，指标需单独
  谨慎解读。没有《樱花草》样本。默认窗口上限估算约596,236窗，数组约
  1.27GiB，拼接峰值估算约2.54GiB；运行时可用内存约6GiB、D盘余量28GiB。
- 全量 sequence-window v2 预处理在约3,000/7,907张谱时因 Loved 谱
  `1625602` 的异常红线 `beatLength=1,000,000,000ms` 产生 float16 溢出而中止；
  没有进入拟合或导出权重。此前 v2 缓存保留在本地，但后续不复用。
- v3 timing guard 已完成验证：专项测试28 passed，脚本和测试 `py_compile`
  通过，`git diff --check`通过。坏图 `1625602` 抽样80窗、留存80窗，忽略
  134条异常红线，正负特征均有限。15谱、15缓存文件的v3 smoke合计238窗、
  111个LN上下文窗，二次读取数组和报告完全一致；缓存留在
  `test-data/corpus-expanded/sequence-window-cache-v3-root-smoke-20261111`。
- 局部窗口特征忽略非有限、非正或超过10,000ms的红线周期，继续使用先前有效
  红线；若无有效红线则采用500ms fallback。该研究过滤不改写谱面Timing。
  v3修复仍待独立PR检查合并；完成后再启动全量预处理/训练。
- 下一步在核对当时可用内存后，使用2个CPU算子线程、1个跨算子线程运行全量
  研究训练，报告每键数、LN/tap和负例可行性；训练分数只用于决定是否值得做
  冻结候选整谱A/B。接入前还必须修正beam历史状态去重遗漏，并做具体片段
  审查与玩家试打。每个已验证阶段仍单独开 PR、检查、合并。已有源码与小阶段记录
  保存在 GitHub；本地 ignored 音频/权重仍需本机保留，GitHub源码不包含它们。
- 评估边界复核：四排模型仍只排序固定事件、固定起止下的四排局部窗口，训练
  JSON 不兼容当前 `manialab.pattern_prior` runtime 格式。即使测试窗口得分提高，
  也不代表整谱键型或 LN 质量；全量结果必须独立报 10K 的 LN/tap 窗口及独立
  歌曲组覆盖，8K/9K 作为小样本探索报告。下一阶段需先实现四排历史状态 scorer，
  再做冻结候选整谱A/B、《樱花草》片段审查及玩家试打。
