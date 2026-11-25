# 架构说明

ManiaLab 由本地 HTTP 应用、音乐领域逻辑、音频/模型组件、浏览器界面和便携构建工具组成。运行时职责以产品路径为准；离线研究不属于生产依赖。

| 层 | 入口 | 职责 |
| --- | --- | --- |
| 启动与 HTTP | `launcher.py`、`manialab/app.py`、`manialab/*_routes.py` | 本机服务、项目/API 路由与任务组合 |
| 任务和持久化 | `manialab/task_service.py`、`manialab/storage.py`、`manialab/provided_stems.py` | 任务状态、取消、原子保存、输入与声轨来源 |
| 制谱领域 | `manialab/generation.py`、`domain.py`、`timing.py`、`arrange.py`、`layout.py`、`solver.py`、表达与候选模块 | Timing、证据选择、长条表达、容量约束、定轨和导出 |
| 音频组件 | `manialab/audio*.py`、`separation.py`、`piano.py`、`guitar.py`、`instrument_detector.py` | 解码、起音分析、分离、专用模型适配与缓存 |
| 浏览器 | `static/` | 导入、Timing 检查、候选试听和编辑 |
| 构建与验证 | `scripts/`、`.github/workflows/` | Windows 构建、worker 身份、便携验证、CI 和人工音乐验收入口 |

## 音频时钟与制谱时钟

采音、段落和来源证据使用原始音频时钟。制谱提前量、量化和导出使用制谱时钟。来源时间和候选时间分别保存；量化不改写原始证据。改变 BPM、Offset、键数或采音证据后，过期候选和诊断不能冒充当前结果。

## 候选与保存

生成、编辑和导出以候选身份及版本化来源快照为边界。任务只有在成功且未取消时提交项目；失败与取消保留上一状态。上传、缓存和导出使用独立路径及来源校验，清理失败必须留下可诊断的残留路径。

## 构建身份

`scripts/build_windows.py` 按 profile 选择资源；`package_assets.py` 和 `worker_inventory.py` 核对资源清单；`build_provenance.py` 将源码、worker、模型、许可和包清单绑定。便携检查只说明受检 profile 和输入范围内的工程行为，不代表真实歌曲质量。

## 验收原则

`pytest`、浏览器回归和便携烟测不能替代真实音频试听、声部与段落判断、Timing 复核及 6–10K/LN osu! 实打。每项结果必须区分已实现、已运行验证和待人工验收。
