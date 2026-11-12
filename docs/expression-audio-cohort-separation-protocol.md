# 表达研究音频配对 cohort 分离协议

日期：2026-11-12

## 目标与范围

对冻结 `expression-audio-cohort-v3` 的训练歌曲组运行本地四声部 Hybrid Demucs，
为后续局部声学证据与作者谱动作对齐准备输入。只处理 `layout_training_split=train`
的39个歌曲组；不读取或处理 validation/test 音频，不训练模型、不调参，也不修改
生成器默认行为。

cohort 选择清单规范化 SHA-256 为
`d73f2ef642b8e16873ef36a43a2c641d0d22f2f21d7a17713c59379ae61b0604`。
72个已配对样本的混音与谱面哈希已经独立复核，但它们仍不是分轨或逐音声音标签。
作者谱音符只能作为 `chart_end` 和风格动作来源，不能被称作真实声部/声学尾音真值。

## 固定处理

- 输入：`test-data/expression-audio-cohort-v3/` 中哈希已核对的训练组完整混音。
- 解码：FFmpeg 输出双声道、44100 Hz、`pcm_f32le`；保留原起点，不截取、
  不归一化、不做响度处理。超过20分钟的输入拒绝处理，不静默截断。
- 分离：官方 Hybrid Demucs 权重
  `assets/deep/hdemucs_high_trained.pt`，SHA-256
  `a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816`；
  四个输出为 drums、bass、other、vocals。
- 执行：CPU 串行，一个歌曲组结束并验证后才开始下一个；PyTorch intra-op 2、
  inter-op 1。无 GPU 并行，无 beat 模型。
- 输出：忽略目录
  `test-data/expression-audio-cohort-v3-separated-train-v1/`。每首记录输入谱面/
  音频哈希、模型身份、样本数、四轨哈希和形状，输入 WAV 临时文件在成功后删除。
  完成目录可按 run identity 验证并续跑；已有不完整目录不覆盖，部分中间结果保留。

## 验证与限制

运行器在处理前重新核验冻结选择报告、训练 split、唯一歌曲组、谱面哈希和音频哈希。
每首分离后检查模型 SHA、采样率、输出帧数、float32、四轨有限值与逐轨 SHA-256；
ZIP/源码完整性或波形有限值不代表声部分离准确。

Hybrid Demucs 的 `other` 仍混有多种伴奏，四声部不等同真实乐器身份。分轨结果只用于
后续候选证据，不能直接监督“这条谱音对应某乐器”或 LN 音尾。音乐质量仍需真实音频
并听和实际 osu! 试打验收。

## 命令

```powershell
.\.venv\Scripts\python.exe scripts\separate_expression_audio_cohort.py --plan-only
.\.venv\Scripts\python.exe scripts\separate_expression_audio_cohort.py
.\.venv\Scripts\python.exe -m pytest -q tests\test_separate_expression_audio_cohort.py
```

成功后保留每个训练歌曲组的分离日志、`separation.json`、`complete.json` 和 NPY 哈希。
音频、谱面、权重与声轨不提交源码。
