# 伴奏原始转录与实际编排

2026-11-09。本轮将 YourMT3 的全曲预测接入研究用的实际选音、声部表达、6–10K定轨及导出链路。**尚不作为默认采音，不宣称音乐效果已验收。** [固定协议](accompaniment-transcription-protocol.md)在查看全曲结果前保存；[公开统计](accompaniment-transcription-research.json)不含音频或逐事件数据。

## 输入与边界

使用《樱花草》原 Nova `other` 声轨283.84秒，22.05kHz float32，逐采样核对与应用已有声轨缓存相同。没有混入bass、guitar、人声或作者谱面，不归一化、不削掉超过1的波形峰值。前一次基于试听PCM的混合伴奏研究是不同输入，不能直接当本轮成绩对照。

固定 YourMT3 源码 `bd69fbac7da881df0dfb67d8f4068a3d473df347`、既有权重及官方解码器；2线程CPU的139个窗口推理用时612.53秒。运行时间只是成本，不是准确率。原始token、音符、输入PCM身份和运行代码快照在 `test-data/sakura-benchmark/other-float-yourmt3-v1`。

真实弦乐声学对照也修正了完整区间的分母：不再删除越过评分终点的预测长音；修订后仍有转录收益，见[URMP对照](instrument-transcription-benchmark.md)。这些录音只有两首，不能证明所有伴奏或合成音都适用。

## 采音与音尾

旧通用检测在other轨有3719个候选；本轮模型返回1252个事件。按已声明的伴奏声轨角色排除4个人声预测、2个鼓预测，保留1246个候选。所有排除都有来源记录，既有能量门槛没有再删除事件。数量减少本身不证明其余都是误检或新检测没有漏音。

类别分布包括弦乐ensemble（506）、cello（132）、violin（143）、viola（90），还有pad、钢琴、organ等。它们是模型推测的乐器标签，不是新分离出的独立声轨，也没有完整人工类别真值；不能按标签数量宣称能够识别所有音色。桥接时统一权重1，没有把velocity或类别当作置信度。

保存的token用官方解码器重建1252个事件，逐项一致。音尾来源计数：896带显式note-off、350带强制分块截断、137带同音高重触发、1个未决/后处理端点，另2个鼓固定短音；标签可重叠。解码器报告379条onset-not-found与421条segment-tie错误。这些错误不因最终成功导出而消失。

强制截断及未决音尾保留原始区间和起音，但不能自动形成LN；原有钢琴的时长上限和输入边界也适用同一保护。显式note-off仍是模型预测，不是音乐上的正确放手。该保护已覆盖逐段表达和原有LN选择两条路径。

## 全曲实际输出

对照只替换other候选，钢琴继续使用已经验证的专用采音，其他声部候选保持相同。沿用原13个自动分段边界，重新提出主次方案；给定74 BPM、Offset178、四细分、22.5ms制谱提前量，因此不是自动Timing验收。没有读取作者Reference或用匹配率选择输出。

10K的变化：

|项目|旧other采音|新other采音|
|---|---:|---:|
|物件总数|1591|1424|
|鼓物件|285|354|
|钢琴物件|471|807|
|other物件|769|180|
|bass物件|66|83|
|LN总数|308|698|
|保留起音移动超过30ms|550|66|

新结果不再将全部13段都以other为主：9段主声部改为钢琴、3段other、1段bass。钢琴650条LN、other28条、bass20条。**这并不证明新的主次和长条安排正确**：钢琴持续区间包含踏板延音，当前策略仍可能过度依赖延音覆盖来决定主声部和长短。伴奏取舍变少也可能遗漏应表现的乐句。需要继续检查真实声音与段落表达，不能以“更少离拍”或“LN更多”宣布解决。

|键数|物件|LN|起音移动超过30ms|
|---|---:|---:|---:|
|6K|1372|683|71|
|7K|1425|691|68|
|8K|1430|698|66|
|9K|1431|698|66|
|10K|1424|698|66|

六个对照输出均通过完整网格、同轨冲突、候选身份、未决音尾禁用与真实`.osu`回读检查，容量选择求解完成。没有为腾空间剪短已选音尾；其余声部原始候选逐项不变。这些是工程检查，不等于试听、试打或音乐验收。

## 复现与后续

```powershell
test-data\yourmt3\venv\Scripts\python.exe scripts/research_instrument_transcription.py --audio test-data/sakura-benchmark/raw-pcm-audit-v1/other.wav --start-ms 0 --seconds 283.84 --context-seconds 0 --output <新的推理目录>
test-data\yourmt3\venv\Scripts\python.exe scripts/audit_instrument_releases.py --runs <推理目录>
.venv\Scripts\python.exe scripts/check_instrument_expression.py --instrument <推理目录> --output <新的编排目录>
.venv\Scripts\python.exe scripts/summarize_instrument_expression.py --root <编排目录> --instrument <推理目录> --output <新的统计文件>
```

编排结果保存在 `test-data/sakura-benchmark/instrument-expression-v1`。首次接入脚本错误假定缓存没有内容哈希子目录，在推理已完成后、编排输出创建前停止；修正路径并核验缓存哈希后完成上述六个输出，没有重跑网络或覆盖成功结果。

后续重点是将同声部的重新发声、持续、停顿与乐句角色结合起来，避免只因持续覆盖率高便把整段主声部全部拉成LN；同时继续处理350个不可靠音尾、鼓音色与bass的时点、自动Timing及更广音源验证。新转录运行时尚未便携打包，这轮没有发行新的Release。
