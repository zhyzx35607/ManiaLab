# Harmonica 第三方复现的固定采音对照

2026-11-10，在声学推理前固定。论文为 Ou 等人的 arXiv:2609.04640；
本轮拿到的是 `z-huang/harmonica` 的**独立第三方复现**，明确不声称等同
作者实现、权重或论文分数。源码commit为
`6ad8feef7a540baf3a384712ae4b44abe4b7a734`，中等模型随源码提供，
2,750,697字节，SHA256
`ecd76a690d6b6e4bbd46d8ddb97e1d03c6e7a413ef9eeafdaa51b0b169cd586c`。
未找到仓库LICENSE或pyproject许可字段，暂只本地研究，不随应用发行。

复用此前完整35份URMP对照的原PCM和时钟：原16kHz单声道、不归一化，
最多40秒上下文、0–30秒起音评分。保留全部独奏和混音输入，后者是单乐器
设计模型的压力测试；没有给它目标乐器答案，也不声称输出含乐器身份。
来源包含重复演奏，不能当35首独立歌曲。复现源码声明的URMP测试作品集合
会逐项记录，但没有核验权重的完整训练清单，不能认证为盲测。

固定使用复现的默认medium、checkpoint自带阈值，不进行阈值搜索。
CPU两算子线程、一跨算子线程，所有神经推理串行；原实现CPU bfloat16
autocast、CQT float32、5秒CQT窗、半窗重叠保留中央半窗，batch8。
原checkpoint读取采用不受限pickle；适配器改为 `weights_only=True`，
其余调用未修改的模型/前端/推理/解码。除按配置重建的CQT核外逐tensor核验。
所有四个头的逐窗logits和拼接活动保留，起音/持续/力度拼接逐样点核对。

虽然网络有offset head，原解码只使用onset和sustain。音尾可由持续活动
跌落、同音重弹、输入结束产生，不能因此当作独立的物理放手或LN。
逐音保存端点原因，包括首帧峰、同音重弹与超出实际音频边界的尾；不修剪
这些失败来提高分数。输出program0只是评分占位，明确无已知乐器标签。

先运行静音、衰减、重弹、上下文末尾、末帧、平台峰六项解码控制，再运行
40秒全静音真实声学输入和全部35份真实输入。全部推理结束并记录完成清单
之后才读参考标签。与现有Tsumugi同原输入比较起音及完整区间，保留50ms/
时长20%的常规容差和已有40/80ms诊断，不将后者统称更严格。

原始执行入口如下，需要本地冻结来源/环境，复跑时使用新目录：

```powershell
.venv/Scripts/python.exe -B scripts/check_harmonica_transcription.py --prepare-only
.venv-audio/Scripts/python.exe -B scripts/check_harmonica_transcription.py --output test-data/harmonica-urmp-v1
```

本轮不会读取MAESTRO保留测试，不改变应用默认或谱面表达规则。真实波形、
逐音资料、模型均留ignored目录。能否接入仍由完整采音对照决定，不能由
模型大小、论文标题或库中自报分数决定。
