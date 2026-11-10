# 音色分组模型的实际结果

2026-11-10。按照[固定协议](instrument-refinement-protocol.md)先运行URMP
第一项 `12-1-vn`，再运行《樱花草》整条Nova other。两次均使用作者原始
音色模型、8秒窗／4秒步长、0.25聚类距离、最少2音，不传乐器答案。

首个40秒小提琴片段的149个事件，原分类、逐音新分类和聚类后分类均为
strings，产生1组。真实前向34.922秒。此例没有显示分类改善。

283.84秒全曲的2207个原事件全部被实际窗口观察，产生12个音色组，
对应7种分类；真实前向274.953秒。1354个事件的分类改变。起音、音高、
音尾与事件数量完全保留，未将它们重新写入应用候选。

| 类别 | 原模型 | 分组后 |
|---|---:|---:|
| strings | 887 | 821 |
| synth_pad | 545 | 57 |
| chromatic_percussion | 284 | 508 |
| brass | 35 | 736 |
| piano | 150 | 0 |
| orchestral_woodwind | 75 | 0 |
| pizzicato_strings | 68 | 0 |
| orchestral_harp | 61 | 0 |
| flute_pipe | 58 | 9 |
| ethnic | 25 | 0 |
| choir | 13 | 0 |
| acoustic_guitar | 6 | 0 |
| organ | 0 | 9 |
| synth_fx | 0 | 67 |

最大组横跨8.29–280.75秒、包含736音，聚合分类为brass；作者softmax值
仅0.372，第二、三候选是synth_pad与strings。另有大组最高类别值约0.24。
这些值不是校准后的正确率。全曲没有逐音真音色标签，不能断言新分类正确，
尤其不能把1354次改名或更少的标签种类当成声部更稳定的证明。

因此该模型目前保留为研究工具，**没有启用自动重分类或段落主声部选择**。
它不会生成独立音轨，不能给每组复制同一other波形响度来假装显著性分析。
下一步应使用已知乐器混音检查分类和混组错误，再决定是否用于段内声部。

本地证据保留在 `test-data/instrument-refinement-first-v1/` 和
`test-data/sakura-instrument-refinement-v1/`，含原事件、逐音logits、embedding、
查询窗口次数与聚类记录。全曲decisions SHA256：
`c37b5ab635e364e754b488253e8d8f4cdba2f49c0f2f794cdcc7ced7498beeb0`；
clusters SHA256：`06194ad2bce69926bfe5a8edf6cff1c7d324d2cc380c6e694289537cdff559c2`。
