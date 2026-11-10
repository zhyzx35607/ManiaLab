# 公开GGUF权重与原MuScriptor代码的采音对照

2026-11-10，在首次新模型声学推理前固定。此前原权重下载因访问条件未取得；
本轮找到 `DamRsn/muscriptor-gguf` 的公开格式转换，由NeuralNote作者提供。
small文件209,425,152字节，固定revision
`13a851294c5b5701966e850c7eba326f9da97e8d`，SHA256
`925f55af65a20ebc4f8b45ceaf095a12b72493d436cb112623cd0041a1af23d4`。
已真实下载并核对。代码MIT，权重CC BY-NC4.0及原署名/使用条件保留，
模型仅在本地研究目录；本轮不随发行包分发。

转换的大矩阵已经损失原FP32精度。恢复为safetensors仅逆名称映射及mel
filterbank转置，不声称恢复原始FP32或与原权重精确同效。原Python源码固定
`8a1ef88672cc23c26619adaff37ec48d82c0b906`，使用其现有CPU float32推理、
前端、流式token解码和跨窗tie前缀。先严格加载122张量并逐项核实；
加载/解码/零输入检查通过后才开始完整比较，不凭样本分数选择阈值。

本机已完成真实strict-load预检：122个状态张量全部逐值等于恢复权重转为
CPU float32后的值，102,441,984个参数；尚无声学前向。GGUF报告的
102,968,832个数值还包含buffer，不能与可训练参数数混用。预检记录在
`test-data/muscriptor-cpp-review-v1/strict-load-preflight.json`。

沿用全部35份URMP固定录音及Tsumugi基线，原16kHz单声道PCM、不归一化、
最多40秒上下文、0–30秒起音评分。先40秒全零，再全部35份；不读MAESTRO
保留测试。相同演奏在不同声轨/混音出现，训练重叠未经完整核验，不能称
35首独立歌曲或盲测。输入标签不参与选模型、输入类别、增益或筛音。

CPU两算子线程、一跨算子线程，模型严格串行；5秒音频块，batch1，greedy、
beam1、cfg1、prelude_forcing=true、max_gen_len2000、no_eos_is_ok=true。
不指定instruments，保留所有预测类别。使用原Python `transcribe`，没有
使用C++默认额外lead-in/静音门。缺少EOS不会悄悄清空该块，报告保留完整
token、缺EOS告警、强制prefix、窗口起点及实际补零样点哈希。

仅增加只读的生成器观察包装，原代码不改算法。保留完整原token、流式
事件、解码actions及未经过MIDI清理的原始音符。用同一原状态机离线重放，
逐项验证与实际start/end事件吻合，并标明自然预测note-off、同音重弹、
tie缺失、异常prologue、文件末尾10ms封口。人为封口不能视作独立放手
证据；全部音尾都不能直接当作已验证LN。
鼓类别沿用原作者的program命名函数，保留原始数字program（包括96和128），
不将不同编号归并，也不借此清理事件或改写音尾。

原作者论文v2第4.2.3节也明确，同一乐器类别、同一音高的重叠音不能由当前
token表示完整保留。本比较仍保留参考中的所有音，不为适配模型而删除重叠
参考；分离出来的一条声轨也不自动等于一个独立演奏者。

不使用自动节拍修正、吸拍、validate/trim重写、参考乐器类别或去重。
错误/越界事件均保留。若原预测时长非正，起音仍按原位置评分，完整区间
不允许匹配但保留原预测分母；不能通过删掉错误事件改善精度。所有推理
完成后才读取逐音参考，常规50ms/参考时长20%和既有40/80ms均保留。

需要本地冻结资产及隔离环境，输出必须使用新目录：

```powershell
.venv/Scripts/python.exe -B scripts/check_muscriptor_urmp.py --weights test-data/muscriptor-recovered-small-v1/muscriptor-small-recovered-f16.safetensors --output test-data/muscriptor-urmp-v1
```

这里验证音符与音色输出的可用性，不验证歌曲主次、分段方案、定轨或LN
键型。没有把论文/移植作者的分数当作本机成绩，也没有把small模型预定为
最终引擎。只有完整对照证明有价值后才讨论应用接入。
