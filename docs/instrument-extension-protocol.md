# 多乐器声学对照扩展协议（2026-11-10）

在运行扩展推理前固定范围，检查鼓和钢琴以外的起音、音高与完整音长。
这不训练编排主次，不把较响等同于应该采入谱面，也不评价LN键型。

沿用已核验的URMP作者16kHz归档，固定取上游测试划分中此前两首之外的
全部编号：`12,13,24,25,31,38,39`。每首的所有独奏轨和原混音都参与，
不按成绩筛选乐器或曲目。其他数据，包括MAESTRO保留测试不参与。
扩展集在独立`test-data/urmp/comparison-extension/`保存，旧研究结果不覆盖。

每个输入评分0–30秒的全部起音，推理上下文统一为0–40秒（若音频不足
40秒则取真实结尾）。若任一参考音尾超出可观察上下文，记录该限制，
不静默删除跨界预测或以音尾长短选择评分分母。模型不读取标注、谱面、
节拍、段落或乐器名；使用原始音频，不按声部归一化音量。

模型使用当前应用Basic Pitch及已核验的YourMT3冻结权重，各一次。
固定原始分块相位0，不试多个相位后挑最好结果；不调阈值或融合音尾。
神经推理串行，最多2个计算线程、1个跨算子线程。保留每次模型、源码、
输入和结果哈希，以及逐输入日志。YourMT3完成后复核token重建与端点来源。

使用既有一对一评分：起音50ms、精确半音；完整区间另外要求音尾误差
不超过max(50ms,参考音长20%)。同时保留固定40/80ms指标和逐乐器统计。
不使用制谱提前量或拍线吸附。错误鼓预测、重复音、错乐器和错误长音
都保留在相应分母；Basic Pitch不输出乐器标签，不补造标签得分。

这组数据曾用于上游验证，不能叫预训练模型完全没见过的盲测。
录音起止标注也不是玩家希望如何放手的真值。结果用于确定下一项模型
改进应针对漏音、误检、音尾还是混音干扰；不会凭总体均分直接替换应用。
本数据仍不覆盖流行人声、吉他、电子合成器的全部情况，不能据此宣称
所有乐器已解决。

```powershell
.venv/Scripts/python.exe scripts/prepare_instrument_benchmark.py --selection comparison-extension
.venv/Scripts/python.exe scripts/probe_instrument_baseline.py --root test-data/urmp/comparison-extension --context-seconds 10
```

对manifest的每条输入调用`research_instrument_transcription.py`，统一
`--start-ms 0 --seconds 30 --context-seconds 10 --segment-phase 0 --threads 2`，
输出到该目录`yourmt3/<name>`。所有案例结束后调用既有端点审计和
`evaluate_instrument_benchmark.py --root test-data/urmp/comparison-extension`，
以新文件名保留报告。原始音频、标签和逐音结果不随源码发布。
