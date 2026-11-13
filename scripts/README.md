# 脚本索引

脚本保留在 `scripts/` 根目录，避免改变历史研究命令和续接记录中引用的路径。不要将本地音频、下载语料、模型权重或实验产物加入提交。

## 构建与便携验证

- `fetch_audio_models.py`：获取并校验打包需要的模型资源。
- `build_audio_worker.py`：构建便携音频工作进程。
- `build_windows.py`：构建隔离的 Windows 包，产物位于 `build/portable/<run-id>/`。
- `verify_portable.py`：用干净 PATH 和离线数据目录验证实际 EXE。
- `verify_shutdown.py`：验证关闭应用时音频子进程结束。
- `collect_release_validation.py`：核对 ZIP、测试报告和实际 EXE，并生成去除音频/逐音数据的发布验证摘要。

Windows Release CI 的正式入口为 `.github/workflows/windows.yml`；完整流程及版本状态见 [`docs/README.md`](../docs/README.md) 和 [`docs/PROGRESS.md`](../docs/PROGRESS.md)。

## 研究脚本命名

- `check_*.py`、`review_*.py`、`analyze_*.py`：固定输入核查、人工复核辅助或报告生成。
- `prepare_*.py`、`train_*.py`、`evaluate_*.py`：数据准备、训练或评价流程。
- 运行前阅读对应的 `docs/*-protocol.md`、`docs/PROGRESS.md` 和脚本参数；测试通过不等于音乐质量验收。

## 《樱花草》候选差异听审

- `build_chord_evidence_audio_review.py`：从冻结候选生成本地原混音听审包；不会运行模型或读取 Reference。
- `summarize_chord_evidence_audio_review.py`：校验导出标签并汇总覆盖，不计算准确率或音乐质量。
- 输入身份、固定播放范围和人工复核边界见 [`docs/chord-evidence-audio-review-protocol.md`](../docs/chord-evidence-audio-review-protocol.md)。

## 《樱花草》表达取舍静态审计

- `summarize_expression_tradeoffs.py`：从冻结的分析与选择 JSON 定位省略的持续音资格候选；不运行模型或选择器，不读音频或 Reference。
- 输入哈希、候选折叠、区间重叠语义和验收限制见 [`docs/expression-tradeoff-audit-protocol.md`](../docs/expression-tradeoff-audit-protocol.md)。

## 常用验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q manialab scripts tests
node --check static/structure.js
git diff --check
```

研究推理按项目续接规则串行执行，CPU 算子线程数设为 2；不要重复已完成队列，也不要修改正在运行或排队的研究脚本。
