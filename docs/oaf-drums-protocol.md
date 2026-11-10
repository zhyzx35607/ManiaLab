# OaF Drums 开发回归协议（2026-11-10）

目的：检验另一种鼓起音/音色识别模型，区分时间覆盖与类别混淆。
这不是作者 Reference 的复刻、盲测、LN 训练或成品验收。
计划在任何真实音频模型输出之前固定；零谱帧 graph smoke 已完成。

## 固定对象

- Magenta source `c15687ebd1c1cc12658cd468c737d3c24c02b918`，官方
  `e-gmd_checkpoint.zip`，`model.ckpt-569400`。ZIP SHA256
  `09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0`。
- AST 加载未修改的官方模型函数，全图建立并逐值核对全部 78 模型变量。
  不加载优化器，不训练；只使用 onsets/velocity。鼓 frame/offset 不作尾部真值。
- TF 2.15.1 CPU，两条 intra/一条 inter 线程，一个神经进程。
  TF Slim 使用自身 fused batch norm 推理分支；补回 Tensor 类型别名。
- 官方配置实际 `spec_hop_length=512`，采样率 44100；不能用未生效的
  `hop_length=441` 注释计算时间。整曲单次双向 LSTM，不切块、不补齐长度。
- 完整 Nova 鼓分轨（283.84 秒）和既有全部 23 份 MDB 原鼓音频。
  输入名单沿用已冻结的 ADT roster，原划分保留，仅开发回归。
  不读取 MAESTRO 保留 test，也不选择性丢弃表现差的 MDB 文件。

## 输入、输出和评价

- 官方纸版音频链：PCM16 除32767/float32原值→立体声均值→旧式
  kaiser_best 重采样→整曲峰值归一化→乘32767截断为int16→除32767。
  不在重采样前截断 Nova 超过1的浮点样本。
- librosa 0.10.2 显式复现旧版参数：FFT2048、hann、center、reflect、
  hop512、250 mel、HTK、fmin30、fmax22050、Slaney；整曲 power_to_db
  ref1、amin1e-10、top_db80。数值兼容验证另存，不能称原历史环境完全一致。
  在整曲转dB之后，按官方 `np.int32(N / 44100 * (44100 / 512))`
  裁剪最终帧数；不直接保留center mel多出的一帧，也不改写其浮点计算顺序。
- 概率严格大于0.5的每个frame/pitch各是一条事件，保存全88维概率与力度；
  不做峰值择优、重复折叠、落线、对照谱指导平移、阈值调参。
- 8-hit 模型类别粗糙：HH含Tambourine/Maracas，Tom含Conga，
  snare含clap，ride bell/cowbell合组；不可当精细音色真值。
  对MDB粗映射：36 kick、38 snare、48 tom、46 hi-hat、49/51 cymbal、
  53/75 other。其他MIDI输出显式unknown，不静默丢弃。
- MDB逐文件逐类20/40/50ms一对一匹配，保留重复，报告原时钟与误差。
  作者列1 kick、列2/3合为snare、列4以HH主指标及任意类覆盖辅助诊断；
  列4不是穷尽物理HH标注。报告原时钟与既有22.5ms映射时钟，后者不是本次拟合。
- 不以多预测覆盖原本无音处作为进步；保存额外事件和缺失事件供波形/频谱核查。
  模型固定50ms显示尾不写入LN，未做直接试听。

保存代码/source/checkpoint/input哈希、逐曲进度、完整概率、逐事件结果及
推理后变量一致性。内存/兼容失败则保存实际完成范围，不冒充整套完成。
没有可核查声学收益之前，不进入默认应用、不发布模型 Release。
