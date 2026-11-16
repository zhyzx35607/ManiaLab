# 《樱花草》半速/倍速与 Offset 静态诊断协议

## 目的与边界

本诊断解释已冻结分析中相近的 74、148、296 BPM 候选，区分可观测的网格覆盖、
候选相位和局部窗口差异。它只复用现成候选，并将固定网格与 `timing_attacks`
比较；不调用 `estimate_timing`、不读取音频或 Reference、不重新运行模型，也不改
Timing 默认规则或稳定性门槛。

起音靠近某条节拍或细分网格，只表示候选事件的时间与该数学网格相容。它不能指出
强拍、听感拍感或正确的半速/倍速单位；内部相位接近也不能证明绝对 Offset 符合真实
音频或用户播放环境。

## 冻结输入

- `test-data/sakura-tsumugi-expression-onset-v3/analysis.json`
- SHA-256：`26a0250568b2ce149022c4aa1549af2d9dd49b045372959a6b26b5342ba5313a`
- 283.84 秒；1,365 条 `timing_attacks`；选择来源为 `drums`。

脚本在读取后立即核验文件哈希。只消费 JSON 的 `duration`、`timing_attacks` 和
`tempo`（含既存的 `comparisons`）；不会读取同目录音频或其他谱面/参考文件。

## 计算规则

- 74/148/296 BPM 与鼓轨、混音的得分和全曲覆盖率取自冻结的 `tempo.comparisons`，
  不重新估算 BPM 或 Offset。
- 对三种鼓轨候选，在相同 1,365 条起音上重新计算固定 BPM/Offset 网格的全曲拍点、
  半拍和四分细分加权覆盖；公差沿用 Timing 模块的 `min(25ms, grid_unit * 0.2)`。
- 使用 `grid_diagnostics` 对同一组冻结起音重算 12 秒局部覆盖与相位偏差。该步骤
  不从局部起音拟合新 Offset，也不挑选新 BPM。
- 候选相位差按两者较快候选的完整拍长折返，检查八度拍速候选是否相位等价；
  这只是坐标相容性，不是独立的拍感证据。鼓轨/混音的 Offset 也按相同拍长折返比较。
- 不读取 `analysis.candidates` 顶层候选列表、谱面 Reference、音乐文件、MAESTRO
  保留测试或其他本地研究输出；只使用 `tempo.comparisons` 中已经存下的 Timing
  候选。不运行推理、生成谱面、Timing 接受策略或任何改写候选的代码。

## 运行与产物

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_diagnose_sakura_timing_ambiguity.py
.\.venv\Scripts\python.exe scripts/diagnose_sakura_timing_ambiguity.py --output-dir test-data/sakura-timing-ambiguity-v1-20261116
```

输出目录必须不存在；脚本拒绝覆盖旧研究产物。机器报告写入
`report.json`，包含输入、脚本和 Timing 源码 SHA-256、候选相位关系、固定网格覆盖、
各来源窗口摘要及解释限制。

## 验收限制

报告可支持“本冻结输入的哪些候选在数值上相容、问题集中在哪些时间窗”的判断。
它不能判定真实物理 Offset、歌曲强拍、听感 BPM、起音是否采对或整体音乐质量；
不得据此宣布 Timing 稳定、改变默认值或绕过生成确认。
