# 两份公开吉他权重的固定对照协议（2026-11-10）

目标是检查独立吉他录音的起音、音高和物理持续区间是否优于现有两条采音路径。
这轮不训练模型、不改变应用默认引擎、不以物理音尾直接生成 LN，也不以孤立吉他
成绩证明复杂混音或完整谱面已经通过。协议在本轮神经推理开始前写定。

## 模型出处及已知边界

下载来源固定为 `xavriley/midi-transcription-models`，revision
`b7bec65a2b860aca72856b0feef58b5df407b777`，全部测试同时保留以下两份权重：

| 模型标识 | 文件 | 字节数 | checkpoint iteration | SHA256 |
|---|---|---:|---:|---|
| guitar_gaps | guitar-gaps.pth | 99178877 | 5000 | 65483e7c0e340a90415b15b520687587698c8c728f5fa470a205f13ee45c6513 |
| guitar_fl | guitar-fl.pth | 98916957 | 100000 | 50d93dba89bdd3401849bc735614478e83d9f46d21fa3f71d8aca5acc0a52028 |

权重大小与 SHA256 已在本地核验，iteration 已由 pickletools 只读查看，正式加载时还会
再次核验。iteration 只记录训练步数身份，不能据此推出训练集、表现或论文表格中的设置。

作者 wrapper 源码 `xavriley/hf_midi_transcription` commit
`96f6797881e9497cbfc8f8e5deccea9c1f2f7adc` 的 `instruments.json` 将两份文件分别描述为
GAPS 和 François Leduc 训练模型；`hf_midi_transcription/model.py` 默认 guitar 对应
`guitar-gaps.pth`。这些描述没有明确是否额外使用 GuitarSet，也没有逐权重训练配置或
文件哈希与论文具体实验行的对应表。本次核对过的模型卡、配置和上传代码同样未补足这个对应关系。

两篇作者论文提供的训练域信息如下：

- *High Resolution Guitar Transcription via Domain Adaptation*，arXiv `2402.15258v1`：
  François Leduc 数据为 79 份爵士吉他独奏的音频与谱对，约 4 小时、33 位吉他手，
  涵盖钢弦、尼龙弦、电吉他、原声吉他及不同录音条件。论文分别研究只用 FL 和
  GS+FL 的训练分支，不能把两种设置合并为一个“未见 GuitarSet”模型。
- *GAPS: A Large and Diverse Classical Guitar Dataset and Benchmark Transcription Model*，
  arXiv `2408.08653v1`：约 14 小时、300 首、200 多位演奏者的尼龙弦古典吉他。
  论文明确有 GAPS-only 与 GAPS+GuitarSet 两组，并比较不同初始化。
  GAPS 按作品划分 90:10；这些论文设置没有唯一绑定到当前 HF 文件。

两篇都处理复音吉他；不能因 wrapper 的单音优化描述，就删掉和弦标签或预测。
论文主要报告音高起音指标，其成绩不能移作当前权重完整音尾或 LN 成绩。
本轮 GuitarSet 使用麦克风钢弦原声吉他，不能据此证明失真电吉他、效果器或混音效果。

因此，本轮是固定输入的非盲实用对照，**无法认证上游训练无重叠**。
若后续得到权重与训练 split 的作者证据，再更新解释；不能预先把论文的 zero-shot
结果或具体百分比冠到本机两份权重上。

核对来源（2026-11-10）：

- `https://huggingface.co/xavriley/midi-transcription-models/tree/b7bec65a2b860aca72856b0feef58b5df407b777`
- `https://github.com/xavriley/hf_midi_transcription/tree/96f6797881e9497cbfc8f8e5deccea9c1f2f7adc`
- `https://arxiv.org/html/2402.15258v1`
- `https://arxiv.org/html/2408.08653v1`
- `https://xavriley.github.io/HighResolutionGuitarTranscription/`

## 固定队列与已有基线

复用 `docs/guitar-vocal-benchmark-protocol.md` 的全部 60 份 GuitarSet：6 位演奏者
× 5 种风格 BN/Funk/Jazz/Rock/SS × comp/solo，每格按原规则取字典序第一份。
使用原始完整 mono-mic 文件，不使用六弦分轨作为推理输入；六弦 JAMS 仅供评分，
保留同音高、同起音、不同弦的独立事件，不去重成对模型有利的标签。

固定文件身份：

| 文件 | SHA256 |
|---|---|
| test-data/guitar-vocal-benchmark-v3/manifest.json | 68932cb2b6aae626c83b1f725ec14ebb58bf7984edbd07208bfb37b52d6e76e9 |
| test-data/redecode-current-guitar-vocal-v1/evaluation.json | f36b6bb9ef0d04659a4ddb036bddb56807f78bf52cd40ee651fae7016592673f |
| test-data/tsumugi-guitar-vocal-comparison-v1/comparison.json | ff8e7a42d8dc92d18cd135e3b218fd9b3e02f75ef775acd0318e67ad7581047d |

`--prepare-only` 检查 60 份原始 PCM 和标签哈希，逐例核对采样率、样本数、长度、
单声道 int16 格式及标签计数；并读取 application/Tsumugi 原预测，重新计算每例两套
评分和完整 interval_scope，要求与冻结报告完全相等。准备阶段不加载 Torch 或网络。

两模型按 guitar_gaps、guitar_fl 的固定顺序分别完成全 60 份；不能逐曲选择更好的模型，
不能以某种风格成绩不佳而删例，也不根据结果调整阈值、移调、偏移、增益或片段。
准备阶段还检查 comp/solo 各 30 份及五种风格各 12 份。固定报告同时给出
comp/solo 和五种风格的全部分层，四引擎各用相同成员和评分规则；这些分层
不能替代全队列，不能逐层挑选模型。演奏者差异如需另行描述，也不得改变总队列。

## 推理与事件保留

运行器为 `scripts/check_hf_guitar_specialists.py`。它在进程内替换冻结
`scripts/research_hf_instrument.py` 的权重字典，复用其网络、推理窗及解码；
不修改已有 bass/sax 文件或历史结果。冻结 adapter SHA256 为
`b734e0806442ce028d1e16acf98e0808d0c96aaa1d123a9dc885d300a4b45bc1`。

网络源代码固定为 Kong `piano_transcription_inference` commit
`0226e74cbc805660e34bbd6a8fed2083890ebb88`，源归档 SHA256 为
`ac7062720a03feb965ffc9e48e6246497c415d90a60780ea98206fb89cbff7e7`。
先核对归档、全部 Python 文件及来源 manifest，随后用
`Regress_onset_offset_frame_velocity_CRNN`，`strict=True` 加载，逐 tensor 比对权重，
不忽略缺失项、不采用下载失败后的替代模型。

- 输入是完整原始 int16 PCM，转 float32 时除以 32768；这是单位转换，不是响度归一。
  不做逐首峰值/RMS 归一、静音裁剪或输入偏移。用既有 `scipy.signal.resample_poly`
  重采样至 16 kHz，核对 ceiling 样本数；这与上游 wrapper 的 librosa 重采样不同，
  因此不能宣称上游包装器逐样点复现。
- 使用既有 10 秒窗、5 秒 hop、100 fps 输出及作者式重叠丢弃拼接。窗输出连同补零帧
  全保存；解码只使用原始音频时钟内的物理帧，不能把末尾补零当实际停音证据。
- 固定 onset/offset/frame 阈值为 0.3/0.3/0.1，CPU、2 个计算线程、1 个跨算子线程；
  神经推理由主线程统一串行调度，不与其他网络测试争抢执行。
- 保存 `windows.npz`、`activations.npz`、`raw-notes.json`、`notes.json`，包括原始窗、
  拼接输出、物理帧掩码及 release_reason。来源解码器必须与作者 decoder 的事件数量、
  音高、力度、起尾时间逐项一致；不以未知尾、重起音或不利预测为由删除事件。
- 未观察到释放的数值边界仍保留为未知尾，不能升格为真实放手或 LN 终点。
  评分器比较其原数值误差，同时保留 release_reason 计数供另行审查。
- 不做 BPM/offset 网格吸附、应用的 22.5 ms 提前量、声部过滤或谱面采样预算。
  两模型输出 `program=-1`，不虚构精确 GM 乐器识别能力。

所有输入、权重、源码、decoder、评分器及结果留存 SHA256。重复运行只能恢复相同
plan 的未完成目录；复用逐例结果时再次校验模型/runtime 身份、原音频时钟、四类结果
哈希和事件计数。运行结束再次核对代码与冻结基准哈希，已完成总报告不能覆盖。

## 评分与交付门槛

起音评分范围固定为 `[0,min(20秒,音源时长−2秒))`，推理始终使用完整音频。
参考与预测均按起音入窗，跨评分终点的音尾保留。精确半音、一对一最大匹配：
主指标为 50 ms 起音，完整区间尾容差为 max(50 ms,参考音长的 20%)；
补报固定 40 ms 起音 / 80 ms 音尾。固定 40/80 ms 并非对所有音都更严格。
多余、重复和错误鼓预测仍计入预测分母；一个预测最多对应一份弦标注。

`guitar-05_BN1-129-Eb_solo` 存在评分内起音而参考尾达到输入终点的情况，
整例完整区间设为 unavailable，起音仍计分；不得只删除这一长音再计算区间分。
所以起音为完整 60 例，完整区间仅有 59 例可评。所有引擎必须一致执行此规则，
汇总同时显示可评案例、不可评案例及原因、全队列分母与可评子集分母。
micro 汇总匹配数/参考数/预测数，macro 等权平均案例；不能把 59 例成绩说成全 60 例。

最终并列 application、Tsumugi、guitar_gaps、guitar_fl 四组，完整展示两套容差、
起音与区间、precision/recall/F1，以及未知尾来源与计算耗时。
耗时不作为准确度证据。所有 120 次完整录音转写齐备后才写正常汇总；
无论哪份结果较好，都不自动改应用默认值。

本协议确认的是一次可复算的实用比较，既不验证上游训练隔离，也不验证混音内分离、
段落职责、持续乐器表现或最终 LN 键型。那些任务需要各自明确的输入和验收。
