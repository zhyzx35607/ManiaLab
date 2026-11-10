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
