# 续接进度

更新：2026-11-12，Asia/Shanghai。本文件用于中断或额度耗尽后恢复工作；
先核对实际进程、Git 和产物，不能把这里记录的运行中状态当作已经完成。

## 当前研究

- 八拍LN乐句上下文仍是独立研究，不接入默认生成器。特征以累计红线拍位
  编码最近8拍，并保留末四个事件行的完整互动；新增了近放手起音的同手/跨手
  统计。由于特征和Timing guard变化，schema升为
  `sequence-window-ln-phrase-context-v3`，缓存升为v8，旧格式不会复用。
- 用户澄清《童话》和《樱花草》是不同写法的例子，目标是学习随段落、声部和
  乐句变化的条件化编排策略，不把任一曲的“单条/双条”或全曲LN比例固化成模板。
  已解析本地《童话》8K/9K/10K和《樱花草》Reference/成品：《童话》10K全曲
  LN比约78.6%，指定八段约55.5%–100%；《樱花草》10K成品约34.2%、Reference
  约8.5%。以上是表达分布差异，不是质量排名。当前v8仍只训练固定音符、起音和
  LN尾端下的轨道排序，不能学习段落声部、LN选择或LN延长。设计边界与验证方案
  见[编排设计](arrangement-design.md)。扩展语料检查到7,907张6–10K谱面、0个
  音频文件，故纯谱训练无法学习这些表达背后的声部身份与真实音尾。v8完整任务
  仍在运行（PID 55024，自08:41启动）；截至2026-11-12 11:23，隔离v8缓存已有
  4,082/7,083张可用谱的特征文件，不重复启动。另清点到40张已有10K原曲/谱面对照，
  其中按既有划分24张属train，
  但LN比均低于50%；另有14张跨键数holdout和10张未划分的6–10K选择对照。
  这些对照尚无Nova分轨和逐音LN证据，不能直接监督声部条件延长；范围和数据边界
  见[LN表达音频样本清点](expression-audio-pair-inventory.md)。本轮没有改生成器
  或训练数据。
- 新增研究入口`scripts/research_expression_actions.py`，从作者谱的8拍局部窗口
  提取起音骨架，预测LN比例、LN拍长、LN和弦、错开放手、Hybrid、反键与独立放手
  等图形动作；8拍只是采样上下文，不是主歌/副歌或固定乐句标签。采用歌曲组等权
  ridge，对比键数×键数内密度三分位×全曲位置三分位基线，test组在读取谱面前排除。
  基线在每个键数×密度×全曲位置格内重新按歌曲组等权，避免局部筛选后又被多谱歌曲支配。
  新研究不使用音频、没有乐器身份，不证明音尾有声学支持，也不接入默认生成器。
  定向测试8 passed，`py_compile`及`git diff --check`通过；尚未对全语料运行，避免
  与v8同时重复读取数千张谱。v8完成后运行：
  `.venv\Scripts\python.exe -m scripts.research_expression_actions --corpus test-data/corpus-expanded --output test-data/expression-action-predictability-v1/report.json`，
  并审查10K未见歌曲组结果。PR #40 的配对音频覆盖清点已合并（`3fc9b4e`）。
- 完整v7准备在谱面`2571106`处发现红线beatLength为`1E-61`，导致累计绝对拍位
  超出float64能分辨后续起音间隔的范围。sequence prior现在把小于10ms或大于
  10,000ms的红线周期都视为不可信并继续使用前一有效红线；这不修改谱面原Timing。
  失败缓存留在`sequence-window-cache-v7-ln-phrase-context-no-test-full-20261112`，
  不作为训练结果。修复后对谱面`2571106`重放得到80个窗口（40个LN），特征均
  有限；1016条红线中极端值按guard忽略，多行难负例成功80条、单行回退0次。
  定向测试`81 passed, 4 skipped`，完整套件`2244 passed, 89 skipped`，
  `py_compile`和`git diff --check`通过。修复后完整训练尚未重跑，下一次使用独立
  v8缓存，避免复用失败阶段的特征。
- LN难负例尝试改动连续的2–4个事件行；若窗口内没有相邻可改事件行，退化为
  单行候选。诊断分别记录两种候选的搜索次数和成功数；有限搜索找不到足够
  可行负例时，该窗口仍会被记为失败，不会伪造样本。
- `--exclude-test-split`现在先从manifest元数据确定歌曲组和split，再过滤test，
  然后才打开纳入的谱面核对哈希或读取缓存。回归测试以缺失的test `.osu`文件
  验证它不会被打开；此选项只允许与phrase-context schema和跳过test评分一起用。
- 2026-11-12烟测使用10K每split最多3张图，累计排除822张test图，处理3张train
  和3张validation图。2轮/4步训练与导出完成，导出误差`2.83e-7`。这组小样本
  validation没有LN窗口（24个tap负例对），所以只证明数据到模型的流程能跑通，
  不提供LN收益或音乐质量证据。产物留在ignored目录
  `test-data/sequence-prior-ln-phrase-context-v2-smoke-20261112`及相邻缓存目录。
- v7完整准备约处理3,850张谱后因上述异常周期中止，尚未进入训练。修复后还需
  重新生成完整v8的train/validation窗口并训练，再比较10K LN/tap分层结果与现有
  v5基线；如果validation没有稳定收益，保持关闭。当前特征固定起音和LN尾端，排序结果不
  等于真实歌曲写谱质量，也不能替代《樱花草》人工分段检查和实际试打。

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

- Beam未来状态去重修复已通过PR #26合入主线，merge commit
  `501b05404c30b08b15a0dc159f53d1bc156f0fb9`；两项Windows validation成功。
  原去重键只比较轨道结束、上次起音和当前行掩码，会合并未来评分上下文不同
  的路径；新签名保存`ends/last/load/source/recent/time`，不把累计cost或回溯
  指针当作未来状态。定向反例用旧签名会选错首排，修复后保留预期的`{0,2}`。
  独立审查未发现字段遗漏；`tests/test_layout.py`为13 passed，全套为2180
  passed、85 skipped，`py_compile`与`git diff --check`通过。每排重置`seen`
  保证去重只发生在同一候选行。本改动可能改变原beam布局结果，不代表整谱
  音乐质量改善；source映射排序的性能尚无独立基准。
- 四排序列先验运行时接入已通过PR #27合并：显式加载、manifest哈希校验、
  默认关闭，只用于完整beam定轨；训练/运行时共用异常Timing guard，窗口保留
  四排上下文与左缘前LN状态，累计分数按候选分差做有界排序影响。10K全谱
  validation三档均没有显示平衡的LN定轨收益，因此不默认启用；它不负责采音、
  改LN尾端或证明音乐质量。
- 新增`evaluate_sequence_prior_remap.py`作为固定输入全谱A/B入口，只从训练
  报告的validation/test歌曲组取图，逐图校验来源SHA、对象身份与轨道回读、
  TimingPoints、LN端点及可玩冲突；按键数分层、同图配对、歌曲组等权统计。
  test必须引用完整validation产物并匹配相同模型/报告/seed/键数/强度；
  输出留在ignored的`test-data`，支持原子检查点和配置一致的续跑。
- 上一轮完整测试为`2194 passed, 85 skipped`；序列运行时/训练/组评价相关
  32项通过，边界测试覆盖窗口左缘LN、beam序列历史签名、完整对象到轨道回读
  及test强度冻结契约。验证集有45张10K图、13个歌曲组，约45,019个物件；
  10K独立test为48张、12组。结构A/B仍不能替代《樱花草》分段/LN人工审查或
  实际试打，未验证前不能宣称键型质量提升。
- 本轮独立审查发现10K五音和弦在特定LN尾长顺序下会产生1008个release-order
  候选（`ends=(0,1,2,4,3)`），旧cap=256会截掉仍可能进入beam的布局。cap现调为
  1024，并以真实`release_choices()`反例及6–10K理论上限增加测试；定向测试
  `34 passed`，完整套件`2201 passed, 85 skipped`，`py_compile`与
  `git diff --check`通过。独立协议审查未发现A/B完整性阻断项，并确认强度
  应按歌曲组及多项结构指标并列比较；固定LN尾长的定轨结果不能证明LN表达
  质量。cap=1024的10K validation strength=0.15已完成45/45图、13组且结构检查
  全通过：cost/note变化为-0.001384（95%组级区间[-0.003085,0.000224]），
  LN inverse误差变化-0.032204（[-0.156698,0.072554]）；区间均跨0。
  原谱轨道偏移率增加0.080562，shield轮廓误差点估计增加0.112701。
  暂无明确收益，不冻结test；报告位于ignored的
  `test-data/sequence-prior-remap-v4-validation-015/report.json`。
  strength=0.35也完成45/45图、13组且结构检查全通过：cost/note下降0.004630
  （[-0.006457,-0.002829]），但LN inverse误差上升0.178843
  （[0.000000,0.501917]），LN相对原谱轨道变化增加0.261869
  （[0.213838,0.305008]）。平均先验定轨时间为10.07秒/图，对照baseline
  4.93秒/图。strength=0.60也完成45/45图、13组且结构检查全通过：cost/note
  下降0.006339（[-0.009652,-0.002920]），LN inverse误差增加0.221021
  （[0.018852,0.486119]），LN相对原谱轨道变化增加0.350249
  （[0.299511,0.398175]）；平均先验定轨12.83秒/图，对照baseline
  5.40秒/图。三档均不满足冻结test的条件。细项见
  [定轨验证报告](sequence-prior-remap-results.md)，完整报告均留在ignored的
  `test-data`。不能通过增大strength掩盖模型只学到局部定轨倾向的限制。
- PR #29已合并至`main`，merge commit `68e3ab1`；两项Windows validation均成功。
- 只读审查确认旧序列先验用作者轨道作正例、随机可行改轨作负例，每窗改
  1–3排且固定LN尾端；它学的是局部原谱排序，不是LN质量或长条时长。
  新增隔离的`ln-transition-hard`负例策略，仅在LN上下文窗改一排，并要求
  inverse、reverse-shield、shield、same-hand hybrid、split-release、
  release-press或near-release-press计数至少一项变化；候选仍须全窗可行。
  10K validation 13组45图抽样1461个LN窗，1360窗（93.1%）成功生成两条
  定向负例；101窗不足两条。可接受负例中的各模式变化次数见
  [LN难例探针](sequence-ln-hard-negative-probe.md)。
  专项测试27 passed；全量`pytest`为2206 passed、85 skipped；`py_compile`、
  `git diff --check`通过。PR #30已合并，merge commit `99a4e6c`。全量v4训练
  完成7,907张谱，按validation选中第20/24轮，导出误差`3.81e-5`。10K test
  共有12组、3,694窗；逐对准确率overall 78.11%、LN 77.71%、tap 78.41%。
  原序列先验、生成器默认行为和LN尾端均未改变。
- 新增`scripts/evaluate_sequence_prior_cross_model.py`，允许分别校验两模型的
  训练报告，再在同一份v3或v4候选集上作配对歌曲组bootstrap，并输出
  overall/LN/tap逐组差值；支持validation和test。PR #31已合并，merge commit
  `3622127`；两项Windows validation成功。完整`pytest`为2211 passed、85 skipped，
  `py_compile`及`git diff --check`通过。
- 10K配对结果和边界见
  [LN难负例实验报告](sequence-prior-ln-transition-results.md)。validation中，
  v4相对v3在v3候选上的10K LN差为-1.03个百分点
  （95% CI -4.45至+1.80），在v4候选上为-0.67个百分点
  （-2.67至+1.30）；test两者均跨零，未证明收益。模型训练每个键数轮流取
  相同优化步，所以7K样本多不是此差异的直接原因。
- 10K专训已完成，validation选择第4/24轮。它对比joint v4时，在相同10K
  validation hard-negative候选上的LN变化为+0.57个百分点
  （95%组级区间[-1.08,+2.47]），整体-2.30个百分点
  （[-3.58,-1.02]）、tap -4.08个百分点（[-6.06,-2.42]）；
  LN区间跨0且整体/tap下降，不接入生成器。独立评估产物保存在ignored目录
  `test-data/sequence-prior-cross-v1-validation-v4-10k-specialist.json`。
- 新增实验性`sequence-window-ln-interactions-v1`输入，在原382维前缀后添加
  4排×17个镜像不变的LN互动摘要，涉及持有密度/手间组合、hybrid、释放配合、
  inverse、reverse-shield、shield及近释放起音。训练、单窗/批量运行时和评估器
  支持新schema；采用独立v5缓存，旧模型和v1生成行为不变。v5难负例签名忽略
  运行时不可见的窗口外LN释放，避免训练目标依赖额外历史；配对评估器校验目标
  歌曲组与两模型训练组互斥。66项专项测试通过；真实模型报告的1626个训练组
  与195个validation组、202个test组均无重叠。2轮、每轮4步的小样本训练、
  导出及跨schema validation smoke跑通，导出最大误差
  `2.76e-7`。全量套件`2219 passed, 85 skipped`，`py_compile`与
  `git diff --check`通过。smoke数据只有每键每split最多2图，不能用于判断模型收益。
  全量v5训练已完成：7,907张谱，第18/24轮按validation选模，导出误差
  `3.81e-5`。v4/v5在相同v5候选上的配对validation显示，10K LN变化
  `+2.61pp`（95%组级区间`[+0.23,+5.20]`），overall `+1.14pp`
  `[-0.89,+3.06]`、tap `+0.04pp` `[-2.47,+3.02]`；6–9K没有相同的
  清晰信号。随后一次性10K held-out test覆盖12组（10组有LN），LN变化
  `-0.07pp` `[-2.61,+3.44]`、overall `+0.26pp` `[-1.67,+2.68]`、
  tap `+0.87pp` `[-1.31,+3.21]`，均未显示可靠改善；validation LN增益未
  复现。因此v5不接入生成器。完整覆盖、命令和限制见
  [LN互动特征实验结果](sequence-prior-ln-interactions-results.md)；结果仅为
  固定起音与LN尾端下的定轨排序，不是音乐质量验收。
- 10K random-initialized specialist在原validation上相对joint v5明显退步；
  从joint v5低学习率warm-start后，原validation曾有`+2.51pp` LN信号。
  为避免重复使用已查看的test，训练器新增兼容模型warm-start、学习率、
  确定性alternate split及可跳过test报告的显式参数，默认行为不变。
  使用新split seed `20261112`从头重训v4与joint v5，再从该joint权重微调10K；
  新validation覆盖10K的14组/150图。warm-start相对v4的LN变化
  `+1.17pp`（95%组级区间`[-0.76,+3.20]`），overall `+0.80pp`
  `[-0.61,+2.76]`、tap `-0.17pp` `[-1.50,+1.17]`；joint v5的LN变化
  `+1.35pp` `[-0.58,+3.37]`。均未显示可靠增益，未打开新test；
  新10K test含7组、6组有LN，与旧test组重叠1组，因此不能称为完全未见语料；
  且跳过test评分仍会准备/读取其窗口缓存，只是不运行test模型评分。
  warm-start不接入生成。复现及专家模型失败结果见
  [10K LN warm-start复验](sequence-prior-ln-warmstart-results.md)。
  warm-start现在要求来源报告与导出模型绑定，重算完整训练manifest SHA，
  并要求与当前各split完全相同；新模型记录来源model/report SHA。
  继续warm-start必须由model和report共同证明祖先也使用同一manifest，
  因此缺少来源或祖先证明的旧模型会拒绝。覆盖跨split、哈希篡改、
  单边/不一致祖先证明、model/report哈希绑定与同split joint-to-10K案例；
  定向和完整测试为2230 passed、89 skipped，`py_compile`及`git diff --check`
  通过。
- 当前研究仍未解决LN尾端/长度决策、整谱布局和实际可玩性。
- 该验证固定LN尾端，所以不能证明LN长度、音频声部角色、乐句分段或实际可玩性；
  strength排序指标不能代替《樱花草》共同bookmark人工检查与实际试打。
- `docs/full-remap-validation.json`中的既有谱不能用作本模型独立A/B：其中多张
  10K图属于sequence prior的train或validation组。优先报告10K的12个独立test
  歌曲组、48张图；6/7K作补充，8/9K因test组少只作探索。
- 当前项目`AGENTS.md`限制最多一个活动子代理。即使用户侧并发额度已升至3，
  仍按项目约束一次运行一个代理，优先用于独立只读审查；不并行修改共享的
  layout或研究脚本。

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
- timing guard 已通过 PR #23 合入，merge commit `ddb430b7cab8a6c8319e8fd4824edbcf9458b437`；
  两项 Windows validation 均成功。PR #24 随后记录了该阶段验证。
- 启动前资源快照为可用内存6.11GiB、D盘余量53.35GiB；按预处理估算峰值
  约2.54GiB，再加Python和训练开销；全量任务已结束，无活动训练进程。
- 全量v3预处理和四排窗口先验训练已完成，命令为
  `.\.venv-audio\Scripts\python.exe scripts\train_sequence_prior.py --threads 2 --cache test-data/corpus-expanded/sequence-window-cache-v3 --output test-data/sequence-prior-v1`。
  共7,907张谱，最佳轮次20，导出最大误差`2.29e-5`。10K逐窗口测试为
  7,672个排序对、77.05%准确率；LN为75.13%，tap为78.57%。训练报告和模型
  留在ignored本地目录，不提交源码或Release。
- 新增歌曲组等权评价入口`scripts/evaluate_sequence_prior_groups.py`，结果与协议
  见[四排序列先验结果](sequence-window-prior-results.md)。10K测试覆盖12个独立
  歌曲组（11组含LN）、48张图及3,836个窗口；歌曲组宏平均整体77.55%
  [74.57%,80.68%]、LN 77.61% [73.09%,83.02%]、tap 78.89%
  [75.97%,81.76%]，区间为固定种子的95%组级percentile bootstrap。
  9K测试只有3组、其中2组含LN，应视作探索性结果；8K也只有8组。
- 上述指标只说明模型能在固定起音、音符和LN尾端下，对局部可行轨道布局
  做高于随机水平的排序；不证明整谱键型、LN长度或音乐质量已提升。
  `sequence-window-v1`权重格式仍不兼容现有`manialab.pattern_prior` runtime。
  下一阶段先修复beam状态去重对未来评分上下文的保留，再设计可选scorer，
  做固定候选整谱A/B、《樱花草》片段审查、`.osu`回读和实际试打。
  目前没有默认行为变化或新Release。
