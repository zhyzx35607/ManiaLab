# 补充模型：音高活动、逐弦重弹与跨窗记忆

另已实际核查并运行 MVSep Mega 53 的8种弦管乐分离控制。32次波形分离中
相对混音的指标改善，但后续64用途/46任务的Tsumugi采音并未改善整体结果，
未接入默认。见[分离结果](mega53-separation-results.md)和[完整采音结果](mega53-transcription-results.md)。

2026-11-10继续直接核查作者源码、权重和论文。新增以下三项，不把旧候选重复
算作新发现，也不以仓库热度替代本地比较。网络快照和源码在ignored
`test-data/instrument-search-continuation-2026-11-10/`，没有上传用户音频。

| 项目 | 与当前问题的关系 | 当前实际进度 |
|---|---|---|
| Timbre-Trap | 多乐器适用的复音活动，可研究某音高是否持续存在；没有乐器身份或逐音释放头。 | 作者11.35MB权重已实际运行全部35份弦管乐输入和零输入控制；独立审计通过。帧音高F1均未胜过现有Tsumugi，不作为删音/LN判据，见[完整结果](timbre-trap-results.md)。 |
| FretNet | 输出六弦离散指位、连续音高偏移及起音；逐弦表示可能帮助区分同音高跨弦重叠，但这是待测假说。 | 作者源码和本人issue回复提供的17.96MB权重归档已下载。只用pickletools静态查看checkpoint，没有加载模型或执行网络。 |
| MR-MT3 | 把前一音频片段的预测token传给下一片段，针对跨窗乐器归属；不是分离波形或歌曲段落规划器。 | 源码与作者HF全部13个checkpoint链接已核查。权重HEAD均200，未下载大权重；当前默认推理硬编码CUDA，未证实Windows CPU可用。 |

## Timbre-Trap

作者 `sony/timbre-trap` 的main为`7afe7e9b327929c099baeccd4b21973aedb94d9b`，
公开Space的requirements明确用updates分支，因此研究固定
`e645620938bff6ce1b69532f56addc31f16211ca`。作者Space revision
`84fbd38582435c863a2d65197a75edd794888f19`，`models/tt-orig.pt`的SHA256
为`1c3bafd33a831d61e8ee9051d6c5b4c5d483e6a7669ca9df85ac6ab304cb9fe3`。

`transcribe()`输出各频率bin的活动值，作者演示也可将转录系数重合成为波形，
但那不是目标乐器分离。帧级活动能同时包含多个音高；没有声部身份，同音齐奏
也不能据此分成多个演奏者。不能沿用“单F0拒绝多音和弦”的旧错误，也不能把
活动结束直接当物理松手。全部35份输入、相同帧时钟下的Tsumugi对照、静音控制、
默认0.5峰阈值与非盲限制见[固定协议](timbre-trap-protocol.md)。

完整实测独奏帧音高F1为81.027%，混音70.744%，对照Tsumugi分别95.248%
与93.471%。该结果不支持把活动峰直接作为现有音符的保留条件。六对共享参考
和样本依赖在[结果](timbre-trap-results.md)中列明，不能称35首独立歌曲。

## FretNet

作者源码 `cwitkowitz/guitar-transcription-continuous` 固定
`d481054f54184374c04b1cc27a487dc35c87f353`。README没有直接权重下载按钮，
但作者本人在issue1回复提供`FretNet.zip`，实测HTTP200；归档17,964,985字节，
SHA256 `87bebc4b24eae231c1d2fc084a7cb042fa4045149f12e9b2980bd5e3f778a2d0`。
其中只有`FretNet/models/fold-0/model-2000.pt`，19,492,608字节，SHA256
`3f3260c7130527884a35d341ec512782cf73bc03ae14cef168b96dc1ba83320d`。
这个SHA是本地实际下载计算值，作者未在回复另给校验值。

静态pickle结构包含完整FretNet模块、GuitarProfile、LogisticTablatureEstimator、
CBernoulliBank等对象，并非可直接当作纯state_dict的checkpoint。本次没有
`torch.load`或反序列化执行。依赖amt_tools、guitar-transcription-with-inhibition、
muda，需各自核查版本及前端。作者示例是22050Hz、hop512、六组谐波HCQT，
并会归一化输入；若比较必须记录这一增益，不能把结果强度当原曲显著性。

源码六折实验按演奏者留一，fold0按脚本对应第一个演奏者；作者提供文件名仅
表明fold0，未附完整训练配置，因此不能仅靠文件名认证全部GuitarSet未见。
无论后续比较哪一组，都须保留训练重叠解释，不能用训练内分数声称新曲效果。
模型代码MIT；作者上传归档未另列权重许可，暂只作为本地研究资产。

进一步追到示例所用amt_tools的解码路径后，确认FretNet没有独立offset head。
当前`StackedNoteTranscriber`调用的`multi_pitch_to_notes`不使用传入offset数组，
尾取frame活动消失、同音重弹或输入结束；这些不能自动视作独立放手证据。
保留逐弦stack时可容纳跨弦同音，但最终flatten会丢弦ID，须额外保存来源。
作者原函数的6项符号检查已通过，没有运行声学网络；现有依赖未锁定原训练版本，
这些检查也不能证明旧checkpoint的完整行为。暂不把它作为LN音尾接入路线。

## MR-MT3

源码固定`826ea84a933f93cd707d11e91af711f1d19c8d79`，HF
`gudgud1014/MR-MT3` revision `7bd24c3e1dc26468a6a9354c0a4f174405c9bad8`，
无访问gate。MR scratch/continual约582MB；GitHub自带184MB `mt3.pth`是
另一条基线，其LFS对象404不能推出HF权重不可用。代码主许可与模型卡MIT，
继承MT3/HF的Apache-2.0文件头也须保留。

现有YourMT3研究入口逐片段调用，没有MR的前片段预测token记忆通道；但
MR自定义generate忽略传入的部分kwargs，不能把`valid_programs`当成已经
生效的乐器限制。整首连续memory需要保留，普通分批会重置。
默认`.cuda()`、旧TF/t5/seqio依赖以及逐token前缀重算让CPU移植仍需验证。

独立提取作者解码函数的符号检查确认：活动槽仍以(pitch,program)唯一索引，
同音重弹、下一窗漏tie和末尾flush会人工封口；无EOS时
`tokens[:np.argmax(tokens == -1)]`还会截成空序列。这些都必须在适配时保留
异常和尾来源，不能把消失音符视为更少误采。没有本地声学成绩，暂不进入默认引擎。

一手来源：

- `https://github.com/sony/timbre-trap`，`https://huggingface.co/spaces/cwitkowitz/timbre-trap`。
- `https://github.com/cwitkowitz/guitar-transcription-continuous`，作者权重回复
  `https://github.com/cwitkowitz/guitar-transcription-continuous/issues/1#issuecomment-1680501895`。
- `https://github.com/gudgud96/MR-MT3`，`https://huggingface.co/gudgud1014/MR-MT3`，
  作者论文`https://arxiv.org/html/2403.10024v1`。

网页工具未返回正文时使用相同作者HTTP/API获取，原响应、状态及哈希留存本地。
