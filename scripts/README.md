# 脚本索引

`scripts/` 只保留当前构建、worker、发布证据和人工验收所需入口。历史研究代码可从 `archive/research-202611` 恢复；不要将音频、谱包、权重、缓存或逐音结果加入源码提交。

## 构建与运行时

- `fetch_audio_models.py`、`fetch_piano_model.py`：获取并校验构建所需模型。
- `build_audio_worker.py`、`build_instrument_worker.py`：生成独立 Windows 音频 worker。
- `build_windows.py --profile core`：构建 Windows 便携包；其他 profile 的本地构建不等于公开支持。
- `package_assets.py`、`worker_inventory.py`、`build_provenance.py`：按能力清单暂存、盘点并绑定源码、模型和运行文件身份。
- `separate_audio.py`、`separate_six.py`、`transcribe_instruments.py`、`worker_lifetime.py`：worker 运行入口和进程生命周期。
- `research_tsumugi_transcription.py`、`tsumugi_candidate_adapter.py`、`audit_tsumugi_releases.py`、`recover_tsumugi_attacks.py`、`evaluate_guitar_vocal_benchmark.py`、`evaluate_instrument_benchmark.py`、`probe_mapper_adapter.py`：Extended worker 的固定来源身份闭包。虽然名称含研究/评价字样，仍是打包运行文件，不得单独删除。

## 自动化与便携验证

- `pytest_provenance.py`：pytest JUnit 报告的源码与输入身份 sidecar。
- `verify_portable.py`、`verify_portable_guitar.py`、`verify_portable_instruments.py`：便携包合成流程和 profile 能力验证。
- `verify_shutdown.py`、`verify_core_media.cjs`：worker 关停与 Core 浏览器媒体检查。
- `collect_release_validation.py`：核对 ZIP、manifest、测试与便携报告的身份链。
- `verify_official_decoder/`：固定版本 osu!mania 官方规则集，对照导出谱面对象、Timing 和音频引用。
- `replay_frozen_generation.py`：对固定来源的忽略目录分析和计划执行真实生成/导出回读，用于定位代码变化影响；不重跑模型，输入与逐音产物不进入 Git。

Windows CI 入口为 `.github/workflows/windows.yml` 和 `.github/workflows/extended-release.yml`。本地常用命令：

```powershell
.\.venv\Scripts\python.exe -m compileall -q manialab launcher.py
.\.venv\Scripts\python.exe -m pytest -q
npm run test:ui
git diff --check
```

## 人工音乐验收

- `verify_independent_audio.py`：在新目录用指定真实音频执行分析、6–10K 生成与导出往返。
- `verify_chart_workspaces.py`：以固定哈希 `.osz` 检查多难度导入、编辑、重开和导出。
- `build_layout_playtest.py`：从指定谱包生成 6–10K 同音符定轨盲测包和评分表。
- `build_chord_evidence_audio_review.py`、`summarize_chord_evidence_audio_review.py`：生成并校验原混音差异听审包。
- `build_piano_onset_blind_review.py`、`refresh_piano_blind_review.py`：构建及刷新钢琴起音盲听材料。
- `build_vocal_phrase_review.py`、`summarize_vocal_phrase_review.py`：生成人声长条候选听审材料并校验回收标签。
- `check_blind_review_scores.py`、`check_music_issue_ledger.py`：校验盲听/试打结果及逐问题处置记录。它们不证明参与者身份、完整覆盖或音乐验收通过。

运行真实音频流程前使用新的 ignored `workspace_exp/` 子目录，并确认输入文件身份；真实 `.osz`、音频和评分材料不得提交。
