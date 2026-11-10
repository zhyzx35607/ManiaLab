# FiloBass / FiloSax 固定对照协议

本实验只判断独立贝斯、萨克斯声轨的物理起音与持续区间是否改善。它不改变应用默认采音器、不做训练、不宣称解决 LN 写法，也不把与谱面 Reference 的一致性当作绝对真值。

## 冻结对象

HF 仓库 `xavriley/midi-transcription-models` 固定 revision：
`b7bec65a2b860aca72856b0feef58b5df407b777`。

| 模型 | 权重 | 字节 | SHA256 | checkpoint iteration |
|---|---|---:|---|---:|
| bass | `filobass_20000_iterations.pth` | 98,882,013 | `cbc792a1b6b6002a73ee0a433c0fe5e097ee728034fe2ea70b24bd06b5026e84` | 20,000 |
| sax | `filosax_25k.pth` | 99,341,469 | `448cf2c8ea6d4b77f7435f5b9a496211ea60300c5c17fa9c754da764f75f3a6a` | 25,000 |

运行只读取已下载的指定权重，不访问模型下载 wrapper，不使用仓库内占位 `model.safetensors`。两份 checkpoint 的 `model` 都有 316 个 state key；安全读取需要 NumPy ndarray 与 Int64 dtype 白名单，因为文件附带采样器索引。载入始终使用 `weights_only=True`，不使用通用反序列化回退。

网络直接复用本地核验过的 Kong 源码，commit：
`0226e74cbc805660e34bbd6a8fed2083890ebb88`，源归档 SHA256：
`ac7062720a03feb965ffc9e48e6246497c415d90a60780ea98206fb89cbff7e7`。
每个包内 Python 文件同时与归档原字节和核验表比对。模型是 `Regress_onset_offset_frame_velocity_CRNN(100, 88)`，MIDI 21–108，不加载原钢琴 checkpoint。`load_state_dict(strict=True)` 后再逐 tensor 确认相等。

此前对照 HF 所用 fork `7568dc7f78b625e40cf9776e2806d164006610e3`，网络唯一相关差异是 dropout 的 `inplace` 设置；本次始终 `eval()`。这属于本地兼容适配，**不是上游完整环境的精确复现**。HF wrapper、fork 源码与论文的静态证据保存在 ignored 的 `test-data/instrument-model-extra-search-2026-11-10/notes.md`。

## 输入、物理时钟与保留产物

CPU 单进程串行：先全部 17 条贝斯，再两条萨克斯；一个时刻只持有一份网络。Torch intra-op 2 线程、interop 1 线程；固定默认阈值 onset 0.3、offset 0.3、frame 0.1，不在这 19 例上搜阈值。

输入是既有基准的原始 mono int16 WAV，转换为 `float32 / 32768`，不峰值归一化、不响度调整、不吸拍。贝斯 44.1 kHz 用 `scipy.signal.resample_poly/default` 重采样至 16 kHz；萨克斯基准已是 16 kHz，直接使用。这里 **没有精确复现原 wrapper 的 librosa 重采样**，报告必须保留这个差异。

网络使用 10 秒窗、5 秒步长；每窗输出 1001×88。多窗丢弃中心 STFT 的额外末帧，保留首窗 0–750、内部窗 250–750、末窗 250–1000 帧，拼成连续 100 fps 采样时钟。单窗保留全部 1001 帧。所有原始窗与拼接后的补零激活均保存：

- `windows.npz`：四个原始输出头及窗口起点（16 kHz 样点）。
- `activations.npz`：完整拼接激活、秒时间轴、`valid_frame`。有效帧用整数不等式 `frame * original_rate < original_samples * 100` 定义，不把音频样点数当输出帧数。
- `raw-notes.json`：作者原解码器在**同一有效物理帧**上的全部事件。
- `notes.json`：现有 `manialab.piano.decode` 给出的同一组物理事件，加尾来源。与作者解码器校验事件数、音高、velocity，时间允许不超过 0.01 ms 的数值差异。
- `report.json`：输入文件/PCM、权重、源码、依赖版本、产物 SHA256、真实时长和耗时。

上面的 parity 仅针对相同物理激活上的解码结果，不能称整个上游推理流程已精确复现。补零区间的激活保留供审查，但不会作为真实录音的起音证据。

`release_regression`、`frame_decay` 与 `reattack`、`duration_limit`、`input_boundary`、`frame_inactive` 区分记录。后四类尾不被声称为观测到的真实 release，**音符仍完整进入起音及区间评分的预测分母**。不按音长、置信度、尾是否可信或音高范围进一步删选；不借此自动制作 LN。

## 固定基准与评分

贝斯沿用全部 IDMT-SMT-Bass-Single-Track 17 条原录音。每条完整推理，计分起音窗口为 `[0, min(20 秒, 原音频时长−2 秒))`。全录音参考共 948 音，计分参考 **818 音**。沿用 `run_bass_model_comparison.score_case`，保留名义音高、无音高起音、两种容差和基于整窗匹配的技法召回，不为未匹配预测编造技法标签。

萨克斯沿用 URMP 已固定的 `25-4-sax`、`39-4-sax`，使用此前 Tsumugi 的相同 0–40 秒 WAV，计分 `[0,30)` 秒的 **146 个参考起音**。使用原 URMP 比较器；不评分可能存在子型标签歧义的 GM instrument 项。历史 `strict` 仅重命名为 `fixed_40_80`，没有改变匹配规则。

| 口径 | 起音 | 音高 | 尾音 |
|---|---|---|---|
| standard | 50 ms | 精确半音 | max(50 ms, 参考音长的 20%) |
| fixed_40_80 | 40 ms | 精确半音 | 80 ms |

尾越过计分窗口的预测保留；所有参考尾必须在推理上下文中可观测，才能给完整区间成绩。区间不足时报告 unavailable 与原因，不剔除难音后补出成绩。微平均与宏平均分开，绝不将两个数据集的数值混成一个模型总分。

| 冻结文件 | SHA256 |
|---|---|
| `test-data/bass-benchmark-v1/manifest.json` | `5e575417b05674ba23a0d1614643197a97ace1a0d1d533fbdd689987fed3d46f` |
| `test-data/bass-model-comparison-v1/comparison.json` | `3bf7d0c9513b08396ef85c9b60209df662e8969222d66a9466361832360aa417` |
| `test-data/urmp/comparison-extension/manifest.json` | `e363a4e7b5f25628909eb13031251847492d9fa816e520b98fbc6d612bcda16a` |
| `test-data/tsumugi-urmp-comparison-v1/comparison.json` | `ff98a2e3c53671c1bcf4ec16a6c3cdff3215e17e239101dfe6883f884fbe2e75` |

每条样本及既有 Tsumugi 预测也核验 SHA256。只有 19 条按固定顺序全部完成才汇总；既有文件夹不覆盖。准备检查不会导入或加载神经模型。

```powershell
.venv-audio/Scripts/python.exe scripts/check_hf_instrument_specialists.py --prepare-only
.venv-audio/Scripts/python.exe scripts/check_hf_instrument_specialists.py --output test-data/hf-instrument-specialists-v1
```

单条兼容性检查可用 `scripts/research_hf_instrument.py --model bass|sax --audio <已核验WAV> --output <新目录>`。兼容性成绩不能替代完整固定队列。

可追加 `--reuse-root test-data/hf-instrument-compat-v1`。只有同名案例目录存在完整 `report.json`，且模型/源码/运行时身份、音频身份、四项产物 SHA256 与解码校验都通过才复用。不会仅根据目录存在推断成功，也不修改旧产物；新报告记录实际推理目录、复用标识与新增计算耗时。汇总仍要求全部 19 例。

## 解释边界

这些是公开独奏基准，不能排除上游训练接触过相关数据，不当作独立保留集。FiloBass 20K 的精确训练划分尚未确认；FiloSax 论文训练步数与本文件名不同，不把论文成绩直接套到本权重。两个萨克斯录音不能证明跨曲风、混音或复音表现。

贝斯技法中死音、滑音、泛音的名义音高存在解释空间，因此同时报告无音高起音。更高的物理区间分数仍不代表已经学会 mania 的段落表达、LN 放手或键型。此实验完成后才根据实际结果决定下一步，研究输出不直接进入发行包。IDMT 基准维持仅本地评估用途；第三方权重及代码的许可审查独立于模型评分。
