# 非钢琴乐器模型：当前接入选择（2026-11-10）

**当前值得继续放在应用里的选择是 Tsumugi `bass_v2` / `other_v1_5` 的可选采音路径，尚没有一个新候选具备直接替换默认采音的本地证据。** 贝斯、弦管乐已有真实录音比较和 Windows 便携程序一致性检查；这不代表它已解决同音重弹、复杂分离残留、声部主次或 LN 放手。[本地贝斯结果](bass-model-results.md)、[弦管乐结果](tsumugi-urmp-results.md)、[实际接入范围](instrument-worker.md)是该决定的依据。

**下一项最有价值的转录比较是 MuScriptor small，而不是先换上其大模型。** 官方源码已有 CPU 路径，但本机仍未取得受控权重、没有速度和准确率结果。轻量 PESTO 可以作为单声部音高/发声活动研究工具；它不能替代逐音转录。本次只审阅来源与源码，未安装依赖、下载新权重、加载 checkpoint、运行声学模型或读取 MAESTRO 保留测试集。

| 候选 | 对当前任务有用的输出 | 权重、许可与 Windows CPU 条件 | 本地证据与决定 |
|---|---|---|---|
| **Tsumugi**：bass / guitar / other / vocal_harmony | 复音起止、音高、宽乐器类别；other 可提出弦乐、铜管、木管等候选。同音高多 slot 与同音重弹需分别检查。 | 作者代码 MIT，HF 模型元数据 MIT、无受控访问；四个已测权重各约 57MB。已在本机 Windows CPU 实际运行；bass/other 已有便携 EXE。[S1] | 17份贝斯、35个弦管乐输入、60份吉他及40份人声已完成。**继续使用 bass/other 可选项**；吉他音尾未整体改善，人声两份标注结论不同，不全局切换。预测类别不是分离波形，物理音尾不是制谱放手。[本地完整比较](tsumugi-guitar-vocal-results.md) |
| **MuScriptor small** | 真实音乐混音的多乐器事件流；乐器条件可限制输出家族，适合检查跨段类别稳定性。输出音符，不输出各乐器分离波形。 | 代码 MIT；权重 CC BY-NC 4.0，另列访问条件，`gated=auto`。small 为102,968,832参数、411,888,600字节。官方推荐 CPU 用small，源码支持 Windows CPU/float32；**未在本机推理验证**。本次无凭据读取模型卡仍401。[S2–S4] | 最值得下一次独立比较；目前不能随包当作已验证引擎。不能把官网大模型试听推广成small成绩，也不能把条件列表当音色识别真值。其同乐器同音高重叠表示限制对重弹/延音尤其重要，见下文。 |
| **PESTO 当前推理实现** | 单声部逐帧F0、发声活动分数、能量与音高激活；可检查孤立贝斯、单线吉他、弦管乐或独唱的滑音/连续性。 | 仓库 LGPL-3.0；仓库列出`mir-1k.ckpt`与`mir-1k_g7.ckpt`，分别123,538/534,664字节，打包配置包含权重。默认CPU，官方CI仅Ubuntu；**未验证Windows运行或本地准确率**。本次只看文件清单，未下载权重；未发现单列的权重许可文本。[S5] | [INFERENCE] 值得做小规模辅助证据比较，接入成本可能小于新转录网络。每帧单F0不提供复音、同音重弹、音符起止或分源；发声活动分数也不是音高正确概率。不能据它的平滑轨迹合并重复音或直接拉LN。 |
| **ROSVOT / RWBD / RMVPE** | 独唱F0、歌唱音符和词边界，可区分音高变化与乐句候选。 | ROSVOT源码MIT；作者提供三个checkpoint的归档，已在本地核验。原入口依赖CUDA，本项目已有实际CPU适配；包中其他组件与权重身份需沿用既有归属记录，不能由主仓库MIT推定所有外部组件相同。[S6] | 已在Windows CPU跑完40份vocadito。A1/A2标准起音F1均低于当前应用，不能再把它列为“尚未测试、可能直接解决人声”的方案。每区间单音高，不能覆盖合唱；词尾不等于LN放手。[本地结果](vocal-transcription-results.md) |
| **SAM-Audio** | 按文字或时间提示输出目标声音及残差波形；在“other里再提取某种乐器”方面比只改标签更贴近需求。 | 源码和权重使用自定义SAM License，不能标成MIT。small权重元数据为5,100,547,943字节，`gated=manual`；官方推荐CUDA，示例使用`.cuda()`，依赖多个音频/多模态项目。**没有本机Windows CPU证明**，本次不下载权重。[S7] | 有研究价值但不适合当前便携CPU版本直接追加。输出波形并不自动获得完整起音、可靠音尾、同音高声部识别或段落主次；没有在Nova残留上对照过，不能声称会改善。 |

MuScriptor 的具体判断不能只看演示页。项目页将最佳版本写作1.3B，当前仓库表格四舍五入为1.4B；HF元数据为1,366,399,488参数、5,465,642,136字节。这不是small模型的性能证据。本机已有源码commit `8a1ef88672cc23c26619adaff37ec48d82c0b906`，本次官方API确认当前main仍为该commit。[S2–S4]

论文v2的§3.3/§4.2.3明确指出，其事件表示不能同时保留**同一乐器、同一音高**的多个活动音符，作者主要评测删除了其中较短的重叠参考音；保留它们时，论文1.3B模型的起音F1由60.4降至51.8、音尾F1由49.0降至41.9。这里不是说它不能表达不同音高的和弦，也不是说所有同音重弹都会失败。它说明该评测口径不足以证明重弹与前音延续已解决。[S4]

当前`muscriptor/transcription_model.py:749–750`在MIDI导出前调用`validate_notes(..., fix=True)`和`trim_overlapping_notes`，README另提供`fill-gaps`、量化和裁前导小节功能。若以后比较，应读取未经该导出整理的`NoteStartEvent`/`NoteEndEvent`，保留原事件、尾部来源、时钟和原波形，不先把音符吸到拍线、不补间隙，也不按乐器答案选最优结果。应用已知的“bass分离轨”等标签可以作为单独明确的条件实验，不能与无提示混音结果混用。[S2、S4]

PESTO 的官方实现比README透露了更具体的接入约束：实际函数位于`pesto/core.py`（README中`predict.py`路径已过时），CPU由`gpu=-1`选择；`ConfidenceClassifier`估计有声/无声，F0通过音高激活规约产生。loader使用`torch.load(..., weights_only=False)`，严格载入失败后允许`strict=False`，将来适配必须先固定权重和完整载入检查，不能沿用这种静默放宽并宣称模型完全一致。官方CPU速度表是作者设备的结果，本项目没有复现，故不填写本机运行耗时。[S5]

另外审阅了 **Jointist**。作者确实公开联合乐器识别、转录和分离的代码及Dropbox权重链接；但开发环境为CUDA10.2/Ubuntu18.04/Python3.8，当前仓库API无许可标识，完整文件树未见许可文件，本轮未下载或核验权重。因此暂不列为可发行接入项；它的“转录条件帮助分离”思路可参考，不能直接认作已可用Windows引擎。[S8]

已有 YourMT3 和 Basic Pitch 继续保留为对照，不重复下载。Tsumugi velocity、instrument_refinement、Melodia、libfmp 的既有失败与限制已经足够阻止直接拿它们决定主次：velocity受其他声音干扰，类别聚类会大幅改名，单F0/显著度轨迹也不等于完整声部。新模型不应绕过这些已观察到的限制。[力度混合对照](velocity-interference-results.md)、[音色分组结果](instrument-refinement-results.md)、[旋律轨迹结果](predominant-pitch-results.md)。本次HF清单还列出`vocal_harmony_v1_6`，但未验证它与已测v1_5的改动和效果，**没有仅按文件名升级模型**。

应用层此时能可靠利用的仍是：保持真实分轨时钟、保留原始音符证据，区分已观测起音与窗口延续、可追溯模型尾与人为封口，让段落表达选择与采音事实分开。主声部判断还缺少经过核验的乐器独立显著性证据；把同一other波形的RMS复制给每个预测类别不能补上这个缺口。具体哪一段应持续、重弹、只采起音或不采，应由可回听的段落方案约束；本轮没有发现能直接给出这种制谱答案的现成模型。[现有表达约束](instrument-expression-controls.md)

一手来源（2026-11-10重新读取；只取文本/API，未取音频或权重）：

- **S1**：[Tsumugi作者仓库](https://github.com/anime-song/tsumugi)，[作者权重API](https://huggingface.co/api/models/anime-song/instrument_agnostic_amt?blobs=true)。当前revision仍为`c6ee8a22626cd7185031ee71ce7cea27ec862781`，四份已测权重哈希见[既有清单](other-instrument-model-survey-2026-11-10.md)。
- **S2**：[MuScriptor固定源码](https://github.com/muscriptor/muscriptor/tree/8a1ef88672cc23c26619adaff37ec48d82c0b906)，尤其README、`transcription_model.py`、MIT LICENSE。
- **S3**：[small官方API](https://huggingface.co/api/models/MuScriptor/muscriptor-small?blobs=true)，revision `8c127f603b807520fa465c838e9bfee8a91ada4e`；[large官方API](https://huggingface.co/api/models/MuScriptor/muscriptor-large?blobs=true)，revision `8809fdfbed2affa7ade94a7059e746e3880720e7`。
- **S4**：[MuScriptor作者展示页](https://muscriptor.github.io/)，[论文v2](https://arxiv.org/html/2607.08168v2)，尤其§3.1、§3.3、§4.2.3；展示音频未下载或作为本地评估样本。
- **S5**：[PESTO固定源码](https://github.com/SonyCSLParis/pesto/tree/62bc0c9702558f19af4593752947fb9db1eadac9)，README、LICENSE.md、`pesto/core.py`、`model.py`、`loader.py`、CI与包配置；[作者论文v2](https://arxiv.org/html/2508.01488v1)。
- **S6**：[ROSVOT固定源码](https://github.com/RickyL-2000/ROSVOT/tree/3c8332bf43adae35f6e4d64971862f2f6139b310)，README的作者权重链接与MIT LICENSE；本地已有归档584,407,709字节，固定SHA与CPU适配见[既有记录](other-instrument-model-survey-2026-11-10.md)。
- **S7**：[SAM-Audio固定源码](https://github.com/facebookresearch/sam-audio/tree/bb4c6999d2677c7402360e426afc01ddfad6dce0)，README、LICENSE、pyproject与模型源码；[small官方API](https://huggingface.co/api/models/facebook/sam-audio-small?blobs=true)，revision `20b65f56888142eebe7c37448c6f6b3b32600e9b`。
- **S8**：[Jointist固定源码](https://github.com/KinWaiCheuk/Jointist/tree/1846975ec904c8b23ea653b9b7881f343574e7a6)，README、requirements、`pred_jointist_ss.py`及完整文件树。权重可用性仅确认作者提供链接，未把链接存在等同于本地可运行。

本次网络快照、HTTP状态、时间、SHA256与静态审阅记录保存在忽略目录`test-data/instrument-model-options-2026-11-10/`。网页工具未返回正文，使用官方公开HTTP接口取得同一手资料；401/404也保留。现有实验结果只从指定项目文档与已冻结清单读取，本轮没有新增声学成绩。
