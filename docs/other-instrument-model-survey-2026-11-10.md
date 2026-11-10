# 其他乐器与人声的模型检索（2026-11-10）

目标是提高分离后音符的起音、音高、持续区间质量，并在混音中保留乐器身份。
分离模型输出波形，转录模型输出音符；模型自行预测的类别和边界都还需要对照。

本轮通过作者 GitHub 仓库、GitHub API、Hugging Face 模型卡和作者提供的下载
地址实际检查源码及权重。内置网页检索未返回正文时改用这些公开接口。下表中
“下载”只说明本地已有资料，不代表准确率已通过，也不代表已进入发行包。

| 候选及一手来源 | 用途 | 本轮处理与限制 |
|---|---|---|
| ROSVOT，`https://github.com/RickyL-2000/ROSVOT` | 单主唱的音符与词边界；内含 RMVPE 音高提取 | 已下载源码、ROSVOT/RWBD/RMVPE 三组权重与配置，写出 CPU 对照入口。作者原入口固定使用 CUDA，不能在本机直接照搬命令。 |
| MuScriptor，`https://github.com/muscriptor/muscriptor` | 多乐器转录，关注吉他、贝斯、弦乐、管乐、合成器候选 | 已下载源码。small 为约103M参数；代码 MIT，模型卡为 CC BY-NC 4.0 且需账号接受访问条件。本次权重请求返回401/GatedRepo，本机无已配置HF凭据，因此没有下载或推理。 |
| BassUNet，`https://github.com/jakobabesser/bassunet` | 贝斯帧音高及发声活动 | 已下载源码与仓库内权重。原环境 TensorFlow1.15；`note_formation` 明确不能区分无无声间隔的连续同音高起音。不能直接作为完整贝斯起音方案，尚未推理。 |
| MSST，`https://github.com/ZFTurbo/Music-Source-Separation-Training` | Demucs、BS/Mel RoFormer、SCNet 等分离模型 | 已下载训练/推理框架源码并检查预训练目录。具体权重输出类别必须逐个确认；架构支持分离不等于任一权重支持全部乐器。 |
| BS-RoFormer SW六声部，`https://huggingface.co/FractalSurfer/BS-RoFormer-SW-6stem` | bass/drums/other/vocals/guitar/piano 分离 | 已读取模型卡；它是镜像，声明上游许可 unknown。699MB权重尚未下载/推理/打包；不能声称它就是 Nova 的专用权重。其六声部仍未独立分出弦乐或合成器。 |
| SAM-Audio，`https://github.com/facebookresearch/sam-audio` | 文字/时间提示提取指定声音 | 已读官方README。需受控权重访问，作者推荐CUDA；本机只有Intel核显。保留为指定乐器分离候选，尚未下载权重或推理。 |
| MIROS，`https://github.com/amt-os/ai4m-miros` | 在YourMT3基础上引入预训练编码器 | 已读仓库列表与转录入口。本次未找到README、Release权重或许可文件，不能当成已就绪引擎。 |
| Tsumugi，`https://github.com/anime-song/tsumugi` | Transkun式Neural Semi-CRF泛乐器转录，含贝斯、吉他、人声及其他乐器专用权重 | 已下载源码和四组作者权重，准备CPU研究入口。源码和模型卡均标MIT。原README列出长音过度切分、失真吉他表现差及分类混淆；这些必须在实际音频上验证。 |

已有 YourMT3 的固定35输入比较继续覆盖弦乐、管乐、低音乐器及混音，并保留
模型乐器标签。它是先前接入的研究候选，不是本轮新下载的模型。合成器、失真
吉他、合唱及复杂流行混音仍缺充分的真实声音对照，不能用上述独奏成绩替代。

## 已固定的源码与文件

| 源码 | commit |
|---|---|
| ROSVOT | `3c8332bf43adae35f6e4d64971862f2f6139b310` |
| MuScriptor | `8a1ef88672cc23c26619adaff37ec48d82c0b906` |
| BassUNet | `b16ce8b7a6d0ff3ab4430bd95150543fe062e697` |
| MSST | `10b47380c0941226a7f1d5ba0cda63957429b7c2` |
| Tsumugi | `eba1c7d05e4d6a53cc4e8916d742d9623a3fdc38` |

本地资料：`test-data/other-instrument-model-survey-2026-11-10/`。
ROSVOT作者原归档为584407709字节，SHA256
`b6055e81315b93415c9bd7fc48e10a28a3da1bea960cab7385483bd7443ba852`。
每个权重/配置哈希保存在`download-identity.json`及研究入口常量中；未重发模型。

## ROSVOT 对照约定

使用固定GuitarSet/vocadito准备中的全部40份人声；A1/A2各自评分，不能择优。
使用完整原音频、原采样时钟、官方CLI的note阈值0.85/word阈值0.5、checkpoint
旁的配置。不会把歌词、作者谱、已知节拍或手工段落送入模型。

原始PCM不归一化。按官方`librosa`路径重采样到24000Hz；128采样hop，
`ceil(N/hop)`真实帧数，按16帧补齐。复用原始MelNet、RMVPE
`get_pitch_batch`（其min_gap=6）、RWBD、ROSVOT前向与解码；不使用不同
min_gap的单条RMVPE快捷方法。CPU两计算线程、一跨算子线程，始终串行。

最终区间端点是作者解码器在`real_frames-1`处的封口，不是观测到停音；标记
`input_boundary`。连续同音高但具有不同预测边界的音保留，不合并。没有歌词
边界时不调用作者的强制词对齐/连音修整。保存原始F0、音符边界logit、词边界、
音高logit及全部原始区间，音符统计权重1并不称为置信度。

模型输出pitch=0还包括被折叠的范围外类别，不能全部解释成真正静音。
保存折叠前类别，分开记录`zero_class`和`suppressed_out_of_supported_range`。
ROSVOT每区间只输出一个人声音高，不足以代表多声部合唱。

单条入口：`scripts/research_vocal_transcription.py`。
完整对照：`scripts/run_vocal_comparison.py`，要求100例通用基线先完成且全部
校验通过，然后串行计算40例新模型输出并分别汇总A1/A2。

准备阶段已通过源码/配置/权重身份校验、三个checkpoint的安全张量读取、
CPU依赖导入及端点纯函数控制。未以这些工程检查代替实际音频模型成绩；
声学推理在前一组URMP对照结束后依次执行。没有修改默认采音或发布新Release。

## Tsumugi的实际接入范围

正确作者仓库是`anime-song/tsumugi`；搜索过程中出现的
`guangyu-liu/Tsumugi`地址经GitHub API核查为404，不作为来源。
权重来自`https://huggingface.co/anime-song/instrument_agnostic_amt`，固定
revision `c6ee8a22626cd7185031ee71ce7cea27ec862781`，无需受控访问。

| 模型 | 大小（字节） | SHA256 |
|---|---:|---|
| bass_v2 | 57169753 | `807eedb409c77a710357d57c9e47d17821c4518db3b09c78d8dabd205a2b604a` |
| guitar_v1_5 | 57152017 | `59c8335517d1faf7d735fef29e061181d87b96385f2e9a08332791e2b64affd4` |
| other_v1_5 | 57151027 | `e836d5b30a29ad2301726aedb9ae348db927fbb4813359cdfb2e304ba3084501` |
| vocal_harmony_v1_5 | 57150903 | `9f068d21313039810c82ede5f01e30bf8bc25fe2ba42f2cad16f57a2095b5e97` |

四个文件均已验证实际长度和作者LFS哈希，并用`weights_only=True`读取配置。
不能把文件名中的bass_v2误认为Semi-CRF V2架构：这四组实际都进入V1解码。
other_v1_5有三个pitch slots，其余一个。这限制同音高的重叠容量；一个slot
仍能同时输出不同音高，不等于整个输入只能是单音。

`scripts/research_tsumugi_transcription.py`复用上游特征、模型和窗口解码。
原CLI使用`weights_only=False`并允许缺失权重；本入口使用安全张量读取，
任何缺失、额外或形状不符的权重都要求先检查。现有CPU运行时Torch2.9.1
不等于作者锁定的Torch2.13环境。

默认不限制乐器类别；只有输入为已知分离声轨时才显式提供`--stem`。
“其他”候选可含弦乐、管乐、合成器等，但类别标签不代表已得到独立音轨。
原始mono复制成双声道，按官方torchaudio默认路径重采样，不归一化，
不用歌词、谱面、节拍或标注修正时间。模型输出的代表GM音色不作识别真值；
本入口保存原始类别，`program=-1`。统计权重1不解释为置信度。
对照保留官方默认静音门控：先把立体声平均再算RMS，反相左右声道可能被
判为静音。因此本入口还不保证任意立体声相位情况下的召回。

审查发现上游`WindowNoteStitcher.finalize()`会强制设置每轨最后音的
`has_offset=True`；合并、裁切也可能改动边界。因此保留每窗拼接前快照及
全部最终原始note，最终音尾统一标`unverified_stitched_boundary`，不可直接
供LN放手使用。不经过MIDI导出的短音删除、重叠裁切和乐器数上限处理。

BassUNet仓库内两个h5另经文件头核对，分别1694696与9433912字节，均为
真实HDF5文件而非LFS指针。此检查不代表已完成TensorFlow1.15迁移或推理。

### CPU真实音频兼容性

四个权重已各完成一次实际前向，无缺失权重、额外权重或形状不符。
输入先按固定规则准备：BassUNet仓库附带片段、GuitarSet固定清单第一项、
URMP扩展清单第一项、vocadito第一项，均从0取最多20秒，保留原始PCM。
这是运行兼容性检查，不是挑选后的准确率比较，也不能代表整曲制谱效果。

| 权重 | 音频秒数 | 解码窗口 | 输出音符 | 前向与解码秒数 |
|---|---:|---:|---:|---:|
| bass_v2 | 7.7092 | 1 | 42 | 4.844 |
| guitar_v1_5 | 20 | 4 | 102 | 15.313 |
| other_v1_5 | 20 | 4 | 69 | 17.515 |
| vocal_harmony_v1_5 | 20 | 4 | 39 | 21.078 |

上述时间不含首次模型加载；机器为当前CPU，不能承诺其他电脑相同耗时。
逐窗与最终逐音证据在`test-data/tsumugi-smoke-v1/`。

完整吉他/人声比较入口为`scripts/run_tsumugi_guitar_vocal_comparison.py`。
须先核验60份GuitarSet和40份vocadito通用基线，再分别用固定guitar_v1_5、
vocal_harmony_v1_5权重处理同样的完整原音频。专用权重按已知输入乐器家族
选定，属于已知分轨类型的测试；不把逐音标签、歌词或GM类别交给模型，
`--stem all`保留所有分类输出及错误鼓音。A1/A2不择优；包含未知参考尾的
吉他案例继续保留起音、明确整例音尾不可评分。完整 100 例及起音回收消融
现已完成，见[全部结果](tsumugi-guitar-vocal-results.md)：吉他起音误采减少但
漏采增加、音尾未改善；人声两套标注结果不一致，未全局替换应用默认。

为保留LN相关原始边界，研究入口增加被动构造观察器，逐条记录窗口位置、
原始头尾帧、边界头输出、裁切后区间及实际pitch slot；最终仍不改变note。
四组相同真实音频重跑后，notes和raw-notes SHA256与无观察器版本完全一致。
构造记录数分别42、110、71、42，均与模型解码区间统计相等。
这使后续能追查人工封口、拼窗和真正的模型尾预测，不代表这些预测都正确。

URMP三方比较入口为`scripts/run_tsumugi_urmp_comparison.py`。固定使用other_v1_5
对全部35输入处理与已完成基线完全相同的0–40秒原始PCM，评分0–30秒起音。
保存裁取前后文件身份与真实样本数，不传乐器答案、不按乐器分数换权重。
完整起止区间的评分与前两方一致，不把端点不可信的预测从分母删除。

ROSVOT首轮在第15例校验处停止：librosa0.11按浮点`N*(24000/orig_sr)`再ceil，
会把894348个44100Hz样本变为486721个24000Hz样本，而不是整数有理数计算的
486720。已核对上游633–635行并用真实重采样控制验证；修正的是比较器长度
预期，未改音频或模型。新v2运行逐项核验原先15例后复制复用，旧目录保留，
其余25例继续相同推理。不能把这一个样本校验问题误称之前歌曲Timing的原因。
