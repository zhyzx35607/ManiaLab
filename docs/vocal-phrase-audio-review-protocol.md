# 《樱花草》人声乐句候选音频复核协议

## 目标

逐条检查冻结分析提出的 97 条连续人声乐句候选：它们是否确实对应可听见的人声，以及分析给出的起点和活动结束是否大致吻合。结果只用于定位候选声源和边界问题，不把音频活动自动解释为歌词乐句、物理松键或可制成 LN 的音乐依据。

## 输入与时间轴

- 候选来自 `test-data/sakura-benchmark/piano-full-pipeline-v1/general-analysis.json` 的 `phrase_candidates`。
- Nova 人声轨和原混音来自 `test-data/sakura-benchmark/raw-pcm-audit-v1/` 的 float32 WAV。
- 人声原始来源 SHA-256、PCM、分析、激活缓存及冻结几何须与 `test-data/vocal-phrase-release-review-v1/report.json` 完全匹配。
- 两条 WAV 均为 22050 Hz、单声道、283.84 秒。分析时长和声部分离身份必须与样本数及原混音 SHA 一致。
- 每条候选切片含起止各 1.5 秒上下文，片段不重采样、不移动，不做逐片段响度归一。
- 所有片段用相同的全局线性增益，将原 float32 PCM 转为 16-bit PCM，目标峰值不超过 0.95。增益和输入峰值记录在 `manifest.json`。

## 复核记录

每条候选分别记录：

- 声源：明确人声、疑似人声、非人声或疑似串音、无法判断。
- 边界：起止基本吻合、起点需复核、音尾需复核、两端需复核、无法判断。
- 可选备注：写出听到的具体声部、乐句上下文或需要人工修正的时间。

页面中的音高、支持音数量、coverage 和分离来源只是候选元数据，不是置信度或独立声学证据。Reference 谱没有人声音符不代表原曲没有人声，也不用于判定候选。

审核页面默认将标注保存在浏览器本地；导出 JSON 后应保留原件，后续分析不得覆盖。标签尚未导出或人工试听尚未完成时，不报告人声准确率、LN收益或音乐质量改善。

## 标签导入与完整性检查

导出文件只保存原件，不要重命名成摘要或覆盖。使用汇总器验证冻结审核包、导出 `review_key`、97 个候选 ID 及声源/边界标签枚举，并检查每项产物哈希：

```powershell
.\.venv\Scripts\python.exe scripts\summarize_vocal_phrase_review.py `
  --labels test-data\sakura-benchmark\vocal-phrase-audio-review-v1\vocal-phrase-review-v1-labels.json `
  --output test-data\sakura-benchmark\vocal-phrase-audio-review-v1\vocal-phrase-review-v1-summary.json
```

摘要保留缺少声源或边界标签的候选 ID，并列出需回听的边界标签；它不会输出备注正文，也不会把标签覆盖完整等同于听审质量或音乐验收。验证失败时先保留标签原件，修复输入来源后再另存摘要。

## 复现

```powershell
.\.venv\Scripts\python.exe scripts\build_vocal_phrase_review.py
.\.venv\Scripts\python.exe -m pytest -q tests\test_vocal_phrase_review.py
```

默认产物为忽略目录 `test-data/sakura-benchmark/vocal-phrase-audio-review-v1/`。脚本拒绝覆盖已有目录；需要重新生成时应显式指定一个新的输出路径。
