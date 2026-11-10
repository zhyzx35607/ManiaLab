# MVSep Mega 53 弦管乐分离对照

2026-11-10，在运行声学推理前固定。本轮检查从混音提取更具体的乐器波形，
不把分离输出类别当自动乐器识别，也不修改已发行 dev.4。

来源为 ZFTurbo/Music-Source-Separation-Training 作者的 v1.0.21 Release，
发布于2026-04-20。作者明确说明53个输出可能重叠，例如 vocal 包含
lead-vocal 与 back-vocal，因此不能相加、不能重复计算音符或各自响度。
作者提醒完整模型显存开销大，并且单声部效果可能低于专用模型。

- 源码 commit `1d4b3c431b37de696b3795ed2018493aacba946f`，代码 MIT。
- 作者配置 SHA256：`7e198062a251587088adb91215a4f44ab59e67bd62fcc805cf54d6e7dfc51103`。
- 作者权重1,368,919,887字节，SHA256：
  `c62820893bbf86d4e734f966bd142d9157cfc8bb8e79e9d8f9ea553f3ff3519f`。
- 实际从 HF `oulianov/mvsep_mega_53` 固定 revision
  `911881e9f5c306e3f4f021fc6c4f61890ef32955` 下载，字节数与哈希和
  GitHub作者Release资产digest完全一致。模型卡/Release未单列权重许可；
  后续核到作者 ZFTurbo 于2026-09-25在官方 issue 245 明确说明53-stem
  权重采用 MIT，不能继续把许可未知当作接入障碍。该声明只针对作者的
  53-stem模型，不自动覆盖仓库列出的其他模型。

许可补核一手记录为[作者官方回复](https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/245#issuecomment-5834608838)。
2026-11-10从官方API保存原回应到 `test-data/mega53-license-clarification-v1/`；
完整评论响应SHA256为 `f3e7e5778f7daadca323968c3a232f22d49cc9444338e389a1ffebeaebcd5e3f`。
本轮仍只保留本地研究，是因为后续采音对照未支持默认接入；许可补核不改变
已完成的分离或转录结果。

复用已完成的 Banquet 实验8种乐器、20–26秒浮点输入，逐个核验保存的
`inputs.npz`。固定小提琴、大提琴、中提琴、萨克斯、小号、圆号、长号、
低音提琴8组，来自5首作品；每组混音、仅目标、目标缺席、全零四种，全部
32次串行运行。目标类别由已知声轨给定，属于理想条件，不是假装模型已能
自动识别曲中有哪些乐器。不读参考音符、不用测试输出改选片段或类别。

保留原共同增益，不归一化、不量化。输入由16kHz单声道升采样到44.1kHz，
双声道是复制，未恢复空间或高频信息。使用作者网络单次6秒前向，与既有
Banquet窗口一致；**不是作者默认20秒窗/重叠拼接的整曲结果**。分别报告
整窗和中央2–4秒的SNR、去均值SI-SDR、增益、RMS和MAE。静音参考的
SNR/SI-SDR不定义；缺席目标和仅目标控制也必须保留。

CPU float32，两计算线程、一跨算子线程。安全张量读取，完整严格载入共同
网络与预定8个独立输出头，逐tensor验证；另外45个头不分配。原作者forward
已有 `active_stem_ids`，每次只运行对应头，没有改写网络/滤波算法。原权重
为float16，运行时转float32；依赖 beartype0.14.1、rotary_embedding_torch0.3.5
按作者requirements装进隔离目录，不修改应用环境。

比较波形收益只决定是否值得继续做采音验证，不能当作起音、音尾、段落主次
或LN质量成绩。公共语料训练重叠未知，也不是用户Nova分离声轨的盲测。

原始运行入口如下。需要本地已有并通过哈希核对的研究源码、权重、隔离依赖、
URMP来源与Banquet基线产物；这些不随源码仓库分发。脚本拒绝覆盖既有目录，
再次运行时应把输出改为新的目录，例如 `test-data/mega53-separation-v2`。

```powershell
.venv/Scripts/python.exe -B scripts/check_mega53_separation.py --prepare-only
.venv-audio/Scripts/python.exe -B scripts/check_mega53_separation.py --output test-data/mega53-separation-v1
```

一手出处：作者GitHub仓库及 `releases/tag/v1.0.21`，作者 `models/bs_roformer/`
实现与LICENSE。原响应、源码、配置、镜像及官方digest对照在忽略目录
`test-data/mega53-source-review-v1`；用户音源未上传到网络。
