# 钢琴分析窗口切换导致的持续截断

2026-11-11。修复已接入可选的钢琴专用采音器：当窗口切换丢失持续状态、
而前窗有完整起音及明确放音回归证据时，保留前窗的持续和放音输出。
起音、音高、力度逐值保留。缓存版本升至2，重做分析时重算旧缓存，并保存
`window_context_continuations`。没有发布新安装包，也未改动其他采音器。

这解决一种可复现的分析截断，不代表声部规划或LN手感达标。《樱花草》
本轮没有候选变化。范围与结果后修订见[协议](piano-window-continuity-protocol.md)。

## 原因与规则

Liszt 52.5秒的模型最大持续概率，从前窗约.9972切到后窗约.004681。
原音没有在这个切换点消失；前窗的7个音高持续响应到54.68–54.70秒。
前窗包含48.6–49.6秒的起音，后窗音频从50秒开始。各窗原归属区间已与
历史输出核对，排除了读错原音或时间坐标的问题。

[INFERENCE] 起音上下文缺失可能解释后窗低活动；直接证实的是，同一段
PCM在相邻窗口的模型持续结果不连续，原拼接把这种不连续当成了音尾。

第一版用前窗frame下降接续，28项提案改变27音；对踏板参考22音误差减少、
5音增加。后者包括Bach约389ms、Liszt约557ms和Ravel约1760ms的新增尾误差。
因此没有启用这版，不能把仍活动的frame当成可靠持续。

最终规则还要求：前窗完整解码以`release_regression`结束，原拼接解码
恰在切换点以`frame_decay`结束。旧起音须早于新窗输入，前窗持续连续，
两窗都不能在恢复范围内出现同音高重弹，且放音上下文不能贴近窗尾。
任何条件缺失都保留旧结果，不从低活动或参考标签外推新的音尾。

## 实录对照

SMD完整143窗，加Nova《樱花草》57窗均保存原始四头输出并核对原归属帧。
正式队列执行198次网络前向，复用初次定位的2窗，共200个不同窗口；正式
队列记录的计算耗时约603秒。SMD与旧PyTorch输出最大绝对差4.59e-5，
在事前固定的1e-4绝对/1e-3相对容差内；Nova与旧ONNX缓存归属帧逐值相同。
后续350个MAESTRO开发窗口直接复用原权重缓存，没有再推理或训练。

|资料|修订后变化音数|踏板尾误差减少/增加|标准踏板区间匹配数，前→后|80ms尾容差匹配数，前→后|
|---|---:|---:|---:|---:|
|11份SMD，前60秒评分|11|11 / 0|4106→4111|4078→4085|
|6份MAESTRO train，前170秒评分|18|17 / 1|8685→8687|8668→8685|
|4份MAESTRO validation，前170秒评分|12|12 / 0|3805→3811|3739→3750|
|Nova《樱花草》283.84秒|0|无物理音尾标签|不评分|不评分|

两种起音容差分别是50ms和40ms；标准尾容差为max(50ms,参考长度×20%)。
所有资料都保留原起音及力度，未因音尾变化添加或删除候选。MAESTRO旧缓存
属于已读开发资料，不称独立盲测；1155/736/687/841未读取。

唯一踏板尾误差增加发生在1222、MIDI67、24.08055秒起音：原尾27.5秒，
新尾27.51867秒，参考27.50391秒；绝对误差由3.91ms增至14.77ms。没有把
这个退步隐藏到总分里。固定起音配对核查中，21份实录没有原正确踏板区间
变错，标准/80ms条件分别新增13/35个正确区间；全部变化配对均无同音高
起音歧义。

按键参考不能与踏板持续混用。SMD标准按键区间1875→1878，80ms条件
1925→1925；MAESTRO按键匹配总数均不变，但1134有1失1得。失去的一音
为MIDI57，原尾147.5秒→148.78493秒，按键尾147.19141秒、踏板尾
148.78385秒。这次更接近踏板，离物理松键更远，不应解释为松键识别改善。

仍有明确限制：Liszt最初7个跨窗持续音中，最终只修复有放音回归支持的
MIDI47/59/63；另外4个未恢复。1134的MIDI60虽然缩小了偏差，仍比踏板
参考早约2.1秒。不能把“误差减少”说成这些音都已正确，更不能直接转换为
玩家应使用的LN长度。

## 应用验证与保存

- 45项局部测试通过，涵盖切窗时钟、原声控制计分、重弹、无放音、窗尾、
  已观测的较早放音、缓存升级和损坏回执重算。
- 两版SMD结果分别独立重建88个输出数组，按另一种最大匹配算法复核
  176项计数；首版28提案和最终11提案都保留。
- 实际应用`infer`用原16kHz PCM及已核对网络输出重放22份完整录音，
  550个长曲窗口的输入逐样本核对，最终四头、解码、修正记录与离线方案
  逐值相同；不是再跑550次网络。
- 91份单窗声源控制的当前解码和输出全部不变。历史缓存未保存1001号
  帧，包装器给该必定丢弃的帧提供dummy值；它没有进入解码或修复判断。
  不把这项包装器核查说成新的神经前向或新增声学识别收益。
- MAESTRO两种尾语义、两种容差的160项计数再次独立复核。应用中仍需
  明确选择钢琴专用采音器；这次没有将它设为所有乐器的默认检测器。

完成入口（以下路径中的原音、模型与逐事件结果均在本地ignored目录）：

```powershell
.venv/Scripts/python.exe scripts/check_piano_window_overlap.py --output test-data/piano-window-smd-new --names bach bartok beethoven brahms chopin haydn liszt mozart rachmaninoff ravel skryabin
.venv/Scripts/python.exe scripts/evaluate_piano_window_continuity.py --source test-data/piano-window-smd-v1 --output test-data/piano-window-regression-new --require-release-regression
.venv/Scripts/python.exe scripts/check_piano_continuity_validation.py --output test-data/piano-continuity-development-new
.venv/Scripts/python.exe scripts/verify_piano_continuity_runtime.py --output test-data/piano-continuity-runtime-new
```

入口依赖本机既有冻结资料；验证入口使用文中v1产物，不是公共下载即跑基准。
源码和报告摘要提交Git，音源/标签/权重不提交。首次保存progress.json失败
和误填录音名称的情况见协议；都没有覆盖原输出。

|报告|SHA256|
|---|---|
|`piano-window-smd-v1/report.json`|`f149093312060907ac7911d092a2dd2f977e11e54848ddbdfae98ab8238e90b3`|
|`piano-window-sakura-v1/report.json`|`e547d9f70cd7d70dfadfabad427cf26acb9e7725029d6a217c01dee01837c934`|
|`piano-window-continuity-regression-v1/report.json`|`b4285fc730057ea46f112fedb0d9c209c789e4dff75080f81010f7b349523950`|
|`piano-window-regression-audit-v1/report.json`|`307a41bbab37fa1ef42ad50ea7caebdaa001c797f8d632a9ebd88dfc06ea552c`|
|`piano-continuity-development-v1/report.json`|`9ee2117c11f3f8f4972303dfdac2b8bfe197a209e049d234774a097e45002c3a`|
|`piano-continuity-runtime-v1/report.json`|`3800668b8574771991666e2881b59838db4dab0bbb4cbf1e80ecab641339523d`|

离线报告的`default_enabled=false`保留运行时状态；应用接入发生在这些
对照结束后，并由最后的runtime重放验证。没有改写历史报告追认部署。

下一步需要处理没有可靠放音的持续、重复起音与声部上下文；不能继续放宽
本规则来追求更多LN。其他乐器的采音、段内表达、自动Timing及6–10K实际
谱面质量仍需继续验证，本次不宣告完整交付或音乐验收。
