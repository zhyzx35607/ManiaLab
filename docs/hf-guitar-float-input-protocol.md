# FL 吉他浮点声轨验证协议（2026-11-10）

本轮检查两件不同的事：同一批60份GuitarSet经过应用式22.05k浮点中间采样后，
相对于原FL结果发生什么变化；以及两条真实完整吉他声轨在同一固定模型下的运行证据。
前者有标签，后者没有独立吉他逐音真值，不把两条真实曲目加入准确率分母。
脚本为 `scripts/check_hf_guitar_float_inputs.py`，默认不改应用配置或模型。

## 运行前提

必须先得到 `scripts/verify_hf_guitar_onnx.py` 的完整60例原输入报告，且完整队列的
真实音频帧、逐音事件身份与音尾来源、两容差评分及范围全部通过。
入口复核完整成员及顺序、每例记录和产物哈希；部分报告、缺例、重复、旗标与逐例不符均拒绝。
该门禁在导入ORT、创建session之前执行，`--prepare-only` 也要求完整前置报告。

原全窗口协议失败仍保留：不得把被丢弃上下文的差异隐藏为“原便携协议已通过”。
本轮是独立的浮点输入开发探针，不回溯更改原协议。使用原容差的前置四项状态均写入计划。
首例诊断显示的补零差异只能说明该例，不能预先推定全队列。

固定模型为 `guitar_fl`，ONNX图 SHA256：
`e02ee3b540717bcae07678764847f5e078213be12b1cba98400f091019df73f9`。
冻结原120份总报告 SHA256：
`fec9fa8feb07330f9dd51c35deaa5bf8f36c074aec9bece7a0b5c6a8d99136dc`。
运行环境必须为应用 `.venv/Scripts/python.exe`，CPUExecutionProvider、计算线程2、跨算子线程1；
记录Python路径及NumPy/SciPy/ORT版本。神经推理由主线程串行安排。

## 60份派生采样链

固定原60份全部完整mono-mic录音，保持原顺序、comp/solo各30、五风格各12，不挑片段或删失败案例：

1. 原44.1kHz int16用 `/32768` 转float32。
2. `scipy.signal.resample_poly` 降为22.05kHz float32。
3. 同样方法从22.05kHz转16kHz float32。
4. 原10秒窗、5秒hop、100fps四头、.3/.3/.1阈值、原解码器与音尾来源策略。

禁止音量归一、限幅、int16往返、裁静音、提前量、BPM/offset吸拍、标签引导阈值或音尾选择。
保留每级ceil样本，不通过裁剪制造对齐。原44.1k样本时长仍作为研究物理帧掩码和参考评分时钟；
记录两次重采样的数值边界，避免把ceil延长部分当作真实停音。

评分沿原起音窗 `[0,min(20秒,原长度−2秒))`。同音跨弦标签不去重，未知预测尾及多余事件不删。
起音60例；原参考尾达到EOF的 `guitar-05_BN1-129-Eb_solo` 整例区间不可评，完整区间59例。
标准50ms起音及 max(50ms,20%参考音长)音尾、固定40/80ms均重算，完整列P/R/F1、micro/macro、
分母、不可评原因、comp/solo与五风格全部分层。不预设新采样链应与原FL完全相同。

## 应用runtime回放

每条输入的ORT窗口仅运行一次。随后用RecordedSession将保存的同一批窗输出交给
`manialab/guitar.py:infer`；回放不再执行神经网络。每次调用逐样点核对输入float32、窗序、
补零与输出头名，若应用改变输入、归一、量化、增加或漏掉窗则失败。

应用看到的是22.05k样本长度，可能比原44.1k时长多半个22.05k采样；在边界情况下会多一物理帧。
两套物理掩码及帧数分别保存。应用额外帧正常参与其解码，禁止截掉后假称事件完全一致。
共同物理帧必须逐值一致；应用raw事件另存，并使用原标签/原评分时钟单独计算指标。

同时调用实际 `guitar.candidates` 验证默认仅起音模式和显式模型尾模式，保存候选及逐音trace。
候选起止只做应用既定毫秒取整，原浮点事件保留；默认模式不让预测尾自动变为LN，
未知尾在两个模式都不能晋级。即使允许模型尾，也写 `authored_ln_verified=false`。
记录新runtime文件hash，不能用本实验声称完整应用路由、UI、缓存、打包已通过。

## 两条无标签的真实完整声轨

| 输入 | 长度/采样数 | 来源与限制 | NPY SHA256 |
|---|---|---|---|
| sakura-nova-guitar | 283.84秒 / 6258672 | 用户确认用于Reference的Nova声轨；吉他极弱，适合检查误报和幅度保留；无独立吉他逐音真值 | `b0f42b37b15623492068e4b8021c1cfd0f733298b762b7da68fb8c46226a115f` |
| youdiantian-htdemucs6-guitar | 235.200590秒 / 5186173 | HTDemucs6对用户歌曲的分离输出；目录名含nova-reference，实际不是Nova输出；无独立吉他逐音真值 | `1bc609bda3a1da0111dd3642f82eb8f1d612b031f1696c4457a0e11aca03ba5a` |

两者均是22050Hz、单声道float32。直接读取已核NPY及separation.json，禁止读取其预览WAV。
《樱花草》来源谱包2644803，原OSZ SHA256：
`fee43757b5ea94869abc9c2626708de022da4506382bb055f0f90ec51e88fe9c`；
包内吉他MP3 SHA256：`6bb1103126f4f7d7bdbef7429e17a05c5a696ed455266eadf173dcc6e65d98b5`。
来源链只用于身份复核，推理不读取MP3或试听WAV。Nova模型及分离参数未知。
《有点甜》原曲SHA256：`08b6aff5db874fca7718c6bab0b7c97e01b59648531bfd0c53d5b1628346010d`；
分离模型SHA256：`34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd`。

这两条只报告原输入幅度、时钟、全部模型事件、尾来源、应用回放与候选来源。
不计算假想准确率、不对照Reference中未标出的吉他音高、不把事件更多当作更准。
它们是开发材料，《樱花草》不是未见泛化验证。

## 保存与执行

所有62条均保存22.05k输入、16k模型输入、完整原始窗口、拼接激活、原始物理掩码、
应用物理掩码与激活、两种raw事件、默认及显式音尾候选trace、来源及产物哈希。
60份标签评分与2份无标签诊断分开目录和总报告字段。输出目录必须全新，不覆盖原证据。
无标签输入的 `accuracy_metrics=null`，`musical_acceptance=false`。

准备检查：

```powershell
.venv/Scripts/python.exe scripts/check_hf_guitar_float_inputs.py --verification <完整60例ORT验证目录> --output <新准备目录> --prepare-only
```

前置门禁通过后由主线程串行执行，输出使用另一个全新目录：

```powershell
.venv/Scripts/python.exe scripts/check_hf_guitar_float_inputs.py --verification <完整60例ORT验证目录> --output <新实测目录>
```

准备与符号测试不执行NN。程序完成只表示本协议证据齐备，不代表已达到LN或成品谱面验收。
