# 真人评审回收与解盲顺序

本协议验证评分表身份、完整性与解盲前留档，不判断音乐质量，也不能证明评审者此前没有查看解盲信息。评分 CSV 和锁定回执留在 ignored `test-data/`，不得提交音频、谱包、评分备注或解盲映射。

## 钢琴盲听

旧包保持原样。用既有参与者文件创建带本地草稿恢复的新包；脚本复制已渲染的短片段与模板，不解码源音频、不运行模型、不读取或复制解盲表：

```powershell
.\.venv\Scripts\python.exe scripts\refresh_piano_blind_review.py `
  --source-participant test-data\piano-onset-blind-review-csv-20261118\participant `
  --output test-data\piano-onset-blind-review-csv-20261119
```

刷新包先在输出父目录生成临时副本，校验及文件写入全部成功后才发布为目标目录；失败时清理临时内容，不留下半成品目标。

打开新包的 `participant/index.html`。本机浏览器草稿便于中断后恢复，但浏览器配置或本地文件隔离可能使其不可用；页面会提示保存是否成功，仍应定期导出 CSV。填完后把 `blind-review-results.csv` 存到新的 ignored 目录，再先运行只汇总状态的检查：

```powershell
New-Item -ItemType Directory -Force test-data\human-review-20261119 | Out-Null
.\.venv\Scripts\python.exe scripts\check_blind_review_scores.py `
  --kind piano `
  --template test-data\piano-onset-blind-review-csv-20261119\participant\review-template.csv `
  --scores test-data\human-review-20261119\blind-review-results.csv `
  --template-sha256 98ec483e0c8864876be2652b13b128dd253aff7fdb80b887fe35eef177b6764a
```

状态摘要须显示 `lockable: true`。这表示 84 个片段均填写了合法听感和把握度；不要求判断全部为“听到”。随后用同样参数增加 `--lock-dir test-data\human-review-20261119\piano-lock`。锁定目录保存提交 CSV 原始字节和只含哈希/计数/时间的 `receipt.json`；目标必须是此前不存在的新目录。保存回执后，才可查看旧包外的 `test-data/piano-onset-blind-review-csv-20261118/deblind-key.json`。刷新包沿用同一片段编号和顺序，所以映射身份不变。

## 定轨试打

逐档只记录实际游玩的键数。已准备的 2026-11-18 包中的 `scores-template.csv` 是在填写前从空白表保留的基线；今后构建的包会自动同时提供基线和可填写的 `scores.csv`。不要用填写后的表替换基线。

```powershell
.\.venv\Scripts\python.exe scripts\check_blind_review_scores.py `
  --kind layout `
  --template test-data\layout-only-playtest-20261118\participant\scores-template.csv `
  --scores test-data\layout-only-playtest-20261118\participant\scores.csv `
  --template-sha256 9f16276eea8148ee20017ffcf1fdaf14f0e44ea165281ff3afc5d7d458503a14
```

至少一档的三项偏好都填写且没有半填行，即可为实际试过的键数组合上锁；剩余空行是未测档位，不会算作失败，也不会被计为已覆盖。评分表保留了谱包哈希；允许反转某档的首轮 A/B 顺序。只检查和锁定参与者评分的身份，不读取 `deblind-after-scoring.json`。

在创建 `test-data\human-review-20261119\layout-lock` 前先确认状态摘要的 `lockable: true`，再用相同命令增加 `--lock-dir test-data\human-review-20261119\layout-lock`。锁定后才打开包外解盲表。若玩家采用不同播放顺序，保留独立的评分副本和对应空白基线；不要把多位玩家的评分合并成一份。

## 独立歌曲复听

两首外部歌曲需要真人分别记录原片、试听和实际试打体验，不用模型报告或自动 Timing 门槛替代。先把[独立歌曲复听表](independent-audio-review-template.csv)复制到 ignored `test-data/`，不要直接修改仓库里的空白模板。原音频和谱包仍留在 ignored 评审目录。`original_listened` 和 `chart_audition_listened` 用 `yes/no`；`keys_playtested` 只记实际游玩的键数，例如 `6;8;10`。记录具体时间及可复查的声音或手势。`onset_timing`、`voice_clarity`、`phrase_continuity`、`ln_expression`、`playability` 使用 `clear`、`uncertain` 或 `issue`；无代表时间时 `concern_time_ms` 留空。评价不自动推导模型准确率或项目验收结论。

来源、试听链接及目前需要复听的位置见 `test-data/independent-audio-20261118/listening.md`。
