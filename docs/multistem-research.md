# 多声部与 Mapperatorinator 接入记录

2026-11-08，先检查本地 NovaMSS 安装包，后继续检查用户提供的 `D:\project\nova` 提取成果。`out/20_algorithms` 有可读的 Demucs 4.0.1 等第三方源码；`out/10_cython` 的应用层资料主要为还原出的符号和字符串，不能把这些报告当成完整可运行的应用源码。静态符号涵盖 HTDemucs、BSRoformer、MelBandRoformer、MelBandConformer 架构。

进一步读取 `out/30_models/WEIGHTS_ANALYSIS.md`、`load_weights.py` 和 `state_dict_keys.txt`：加载脚本需要已解密输入，结构清单仍明确标为「未生成（待解密）」。因此不能声称专用权重已恢复，也不能按文件体积或配置名字断言其公开模型身份。本轮没有执行解密脚本或复制 Nova 的提取物进项目。以下接入使用公开上游代码与可校验权重，不能声称与 NovaMSS 专用配置输出相同。

逐项参考了可读的 `demucs/separate.py`、`apply.py`、`htdemucs.py` 以及应用层 `core.infra_stream_gateway` 的符号报告。落到当前实现的是共享采样时钟、分块重叠拼接、有效长度检查、声部独立转录和模型身份缓存。六声部直接调用同版本公开 Demucs 的分块实现。四声部使用 torchaudio 的 HDemucs 实现，其网络内部已分别标准化频域和时域输入；不能仅因外层流程不同就认定它缺少标准化，或未经对照替换。

实际逐字节比较提取目录与本项目音频环境：`apply.py`、`separate.py`、`htdemucs.py`、`hdemucs.py`、`audio.py` 五个文件全部相同。其中 `apply.py` SHA256 为 `316278eeb028c66f99befa4201edca7f598166628aeec8e40a08544f00aa21df`，`htdemucs.py` 为 `3404ed68d35b946f9c54b54201851bb4171dd7861a498945ce53b1df1144b034`。这证明了可读 Demucs 核心的一致性，未证明 Nova 的模型选择、专用权重或处理配置一致。

## 可复现的六声部路径

- 上游：[Meta Demucs](https://github.com/facebookresearch/demucs)，`demucs==4.0.1`。
- 固定公开权重：`hybrid_transformer/5c90dfd2-34c22ccb.th`，54,996,327 字节，SHA256 `34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd`。
- 模型明确输出鼓、贝斯、其余伴奏、人声、吉他和钢琴。上游提示钢琴表现有限；增加声部不自动等于改善采音。
- 使用上游 `apply_model` 的分片、重叠拼接和有效长度处理；CPU 推理、关闭随机平移。模型包以 `weights_only=True` 读取，仅允许所需的 HTDemucs 和 Fraction 类型。
- 每个声部沿用同一采样时钟独立转录。四声部和六声部放在不同缓存目录；音高缓存同时核对音源哈希和分离模型哈希。
- 六声部保留为可选方案，四声部仍可比较。界面、波形、声部试听、编排权重和 API 助手均区分吉他／钢琴。

原版 40 秒合成样例包含 130 个已知音高的持续音。以相同 PCM 输入重新运行四／六声部，再检测音符；两者都没有用节拍模型辅助，比较不涉及定轨。头容差 40ms，尾容差 80ms，音高必须完全相等，使用一对一匹配。

| 指标 | 四声部 | 六声部 |
|---|---:|---:|
| 检测有音高事件 | 270 | 266 |
| 起音与音高匹配数 | 121 / 130 | 120 / 130 |
| 起音 F1 | 0.6050 | 0.6061 |
| 完整区间匹配数 | 69 / 130 | 67 / 130 |
| 完整区间 F1 | 0.3450 | 0.3384 |

这个结果没有证明六声部优于四声部，且显示持续区间仍是弱点。合成信号不代表自然吉他／钢琴或人声；不能凭模型输出类别给它们贴真实乐器标签。用户歌曲前60秒的声部对照另保存在本地，用于回听，不是独立真值。

## Mapperatorinator 路径

现有 V32 接口增加已确认的 Timing 输入：导出只含红线的参考 `.osu`，传 `beatmap_path`、`in_context=[TIMING]`、`output_type=[MAP]`。参考红线文件不能被误识别成模型生成结果。

真实 30 秒 CPU 推理用时约 567 秒，产生 238 个物件、92 个 LN。虽然红线正确，330 个端点中仍有 196 个偏离所选四分拍细分，最大约 34.86ms。由此确认：TIMING 是模型条件，不是硬解码约束。本轮增加生成后的完整量化与序列化校验，再交给 LN 连续搜索。该样例最终 330 个端点全部落线，物件与 LN 数量不变；其中三个起音移动超过 30ms，必须回听，不能当成采音改善的证据。

模型的原始文件保留在每次运行目录；候选记录原始起止、量化幅度和短 LN 合并成短键的数量。模型已经输出谱面时钟，不重复扣除制谱提前量。导出再次核对实际序列化的红线和音符，防止原始行覆盖内存中已量化的时间。此次上游还警告四个不完整 LN；没有凭空补造其尾端，警告随候选显示。

这是对既有模型推理流程的接入改进，尚无新增模型训练。Mapperatorinator 生成的是谱面事件，声部分离与音高转录提供另一种声音证据；两者的重合与分歧需比较，不能把模型预测当逐乐器真值。

## 进一步研究的候选

实际查阅了 [MVSep 2025 多乐器方案](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/mvsep-2025.md) 和 [BS-RoFormer-SW-6stem](https://huggingface.co/FractalSurfer/BS-RoFormer-SW-6stem) 的页面，并下载核对 YAML。后者公开权重约 699MB；配置明确是6个输出：bass、drums、other、vocals、guitar、piano，模型深度12、隐藏维度256。另一个 [enerjazzer 仓库](https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed) 的许可元数据是 unknown。不能据架构名称或网页宣传推断它与本地 NovaMSS 的模型相同。

这些候选目前只有资料检查，尚未推理验证、接入或随包发行。模型卡的指标与本项目采音、LN尾端和玩家评价是不同问题。
