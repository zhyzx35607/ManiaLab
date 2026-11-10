# Timbre-Trap 多音高活动辅助实验

2026-11-10，首次神经推理前固定。该模型只产生多音高活动，不预测乐器身份、
独立逐音起止或制谱放手。本轮不修改任何应用音符，不新增自动 LN 规则。

作者源码 `sony/timbre-trap` 的公开演示明确使用 `updates` 分支；固定为
`e645620938bff6ce1b69532f56addc31f16211ca`，不能把 main 源码身份冒充演示。
权重来自作者 Space `cwitkowitz/timbre-trap`，revision
`84fbd38582435c863a2d65197a75edd794888f19`，`models/tt-orig.pt` 共11,353,410字节，
SHA256 `1c3bafd33a831d61e8ee9051d6c5b4c5d483e6a7669ca9df85ac6ab304cb9fe3`，已下载核对。
代码MIT；Space元数据未另列权重许可，先仅本地研究。与官方来源相同的参数为
22050Hz、9个八度、每八度60个bin、3秒block、latent128、complexity2、无skip。
完整strict权重加载并逐tensor核对，CPU两计算线程、一跨算子线程，所有网络串行。

使用既有URMP全部35输入（28独奏、7混音），来自7个piece ID，不称35首独立歌曲。
其中不同编制存在共享参考和演奏声轨，7个piece ID也不代表7份独立演奏。
输入沿用已冻结Tsumugi对照的16k单声道int16片段，`/32768`转浮点，再以
`scipy.signal.resample_poly`转22050；不归一化、吸拍、调整offset，也不重分离。
这与作者演示的torchaudio重采样不同。保存原PCM和模型输入hash、全部活动输出、
频率、实际frame时钟及物理帧mask。作者内部重叠窗、Hann拼接保持原代码；
不把补零区当音尾。另有4秒全零控制。准备阶段不加载Torch或执行网络。

按作者后处理采用严格频率局部峰及固定0.5阈值；保留全部预测频率，不按参考
音域删候选。评分在原时钟[0,30秒)，仅物理有效帧，参考来自已有逐音区间的活动
半音集合。同音齐奏在**帧音高指标**中合并，参考逐音文件完全保留；预测的重复
频率峰不会合并。以±0.5半音最大一对一匹配累计precision/recall/F1，同时记录
静音误报和复音帧数。对照Tsumugi原区间投影到完全相同帧时钟后的音高集合。

该帧分数不能与先前逐音起音/完整区间F1混称，也不能认证重复同音、力度、音色、
物理松手或LN效果。逐音参考并非连续F0真值，揉弦/滑音会受半音容差影响。
模型有URMP训练/验证实验，未获得该权重的完整训练隔离证明，不能称盲测。
不因部分乐器表现较差提前截队列，不按结果调整阈值或挑选输入。

入口：`scripts/research_timbre_trap.py`。所有结果、音频及权重只存ignored目录，
没有将上游requirements整体安装进应用；新增cqt-pytorch0.0.5置于独立研究目录。
只有完成全组和核验后，才判断它是否值得继续做音尾支持实验。

一手来源：`https://github.com/sony/timbre-trap`、
`https://huggingface.co/spaces/cwitkowitz/timbre-trap`、
作者论文 `https://arxiv.org/abs/2309.15717`。
