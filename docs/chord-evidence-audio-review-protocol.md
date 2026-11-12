# 樱花草和弦候选差异听审协议

2026-11-13。此包为已经冻结的10K新旧候选差异准备原混音听审材料，不重跑采音、
分离、转录或排序模型，不读取Reference谱，也不修改旧实验产物。

## 构建与范围

运行：

```powershell
.\.venv\Scripts\python.exe scripts\build_chord_evidence_audio_review.py
```

默认输出为忽略目录`test-data/sakura-chord-evidence-audio-review-v1/`。输出目录必须
不存在；脚本逐项核对和弦排序报告、执行计划、10K候选/选择文件、基线已选结果、
《樱花草》分析文件和原混音PCM身份。不会复制整首原音频，只输出复核窗口片段。

六个窗口为8–16、16–24、40–48、96–104、136–144和200–208秒，覆盖冻结
`changes.json`中的全部27条旧/新增删起音记录。8–16、96–104和136–144秒包含
12个落选鼓起音中的全部12个；16–24、96–104和200–208秒保留原先选定的候选
取舍窗口。攻击位置没有变化的窗口不会伪造old/new click差异，而会展示新旧候选
音高、原始起音和决策原因。

## 音频与标注

来源为已审计的原混音浮点PCM：
`test-data/sakura-benchmark/raw-pcm-audit-v1/mix.wav`，SHA-256
`dfd2e1581124db8255a347c8414411d1978e279d06116603df64dcfa60e8745e`，
22050Hz、6,258,672样本。每个窗口输出原混音、旧版落选起音click和新版新增起音
click（仅在该窗确有起音集合变化时生成）。所有片段统一使用音乐增益0.50和click
增益0.12；不逐片响度归一、滤波、重采样或时间移动，且输出不会触发峰值归一化。

用本地HTTP服务打开`index.html`。先听“原混音，无定位click”，再听click预览定位。
逐条标记原混音附近是否能听到对应起音：听到、未听到或不确定；可补充文字备注。
导出的`*-labels.json`留在本机，页面暂存只属于当前浏览器。声部名称、MIDI音高、
模型strength与candidate source仅是候选元数据，不是声学真值；试听和click也不
证明鼓点的音乐价值、声源身份、LN音尾或谱面正确性。

标签导出后运行：

```powershell
.\.venv\Scripts\python.exe scripts\summarize_chord_evidence_audio_review.py `
  --manifest test-data\sakura-chord-evidence-audio-review-v1\manifest.json `
  --labels "$env:USERPROFILE\Downloads\sakura-chord-evidence-audio-review-v1-labels.json" `
  --output test-data\sakura-chord-evidence-audio-review-v1\summary.json
```

汇总器核验review key、清单SHA、窗口与标签枚举，只输出覆盖和分布，不输出备注正文，
不计算准确率，`musical_acceptance`固定为false。人工声音判断之后仍需实际osu!试打，
才能评估布局、声部和LN表达。
