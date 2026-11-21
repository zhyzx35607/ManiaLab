# 文档索引

研究记录按实验保留原文件名与路径，便于复现已有命令和交叉引用。本索引提供入口，不代表所有研究结论都进入产品默认行为。

## 使用 ManiaLab

- [快速开始](quick-start.md)：安装后导入、分析、编排与导出。
- [乐器采音选择](instrument-choices.md)：各声部入口、默认值与音尾限制。
- [已有声轨导入](provided-stems.md)：导入分轨并保留原音频时钟。
- [段落与编排设计](arrangement-design.md)：段落边界提示、声部和表达控制的范围。
- [音乐质量目标](music-quality-target.md)：项目长期验收标准。

## 项目状态

- [全项目验收计划](full-project-acceptance-plan-20261119.md)：工程、音乐、支持 profile、许可和最终停止条件。
- [全项目验收问题账本](full-project-acceptance-ledger-20261119.md)：当前问题、覆盖缺口、外部依赖、证据和关闭条件。
- [2026-11-18 工程与音乐验收记录](acceptance-20261118.md)：真实谱包往返、两首外部歌曲及真人试打入口。
- [真人评审回收与解盲顺序](human-review-integrity.md)：校验并锁定盲听/试打评分后再解盲；不替代真人音乐判断。
- [osu!mania 官方解码对照](../scripts/verify_official_decoder/README.md)：固定 ppy 版本交叉校验导出谱包的物件、拍号、滚速和音频引用。
- [架构与开发入口](architecture.md)：模块职责、音频/制谱时钟、提交边界与开发顺序。
- [开发与验证](development.md)：锁定浏览器回归、构建 profile 与历史模型共存。
- [2026-11-17 检查与结构优化](project-review-20261117.md)：问题处理、验证证据及尚未完成的音乐目标。
- [冻结多曲生成回归](frozen-generation-regression-results.md)：容量边界修复、66例对照与声部取舍变化。
- [生成矩阵、键型与采音效果审计](generation-effect-audit-2026-11-18.md)：钢琴采音 A/B、6K–10K 实际列映射及 LN/Timing 证据边界。
- [Timing诊断与波形时钟](timing-diagnostic-clock-results.md)：固定音频窗口、来源追踪、证据不足及界面对音修复。
- [LN来源与完整歌曲复核](ln-evidence-review-results.md)：检测/表达/谱面尾端分层、不可变来源快照及6–10K整曲验证。
- [局部乐句边界保护](regional-phrase-boundary-results.md)：防止范围外新乐句错误代替范围内起音，含冻结候选对照与扩大范围后的导出检查。
- [历史开发与研究导航](research-history.md)：从首页归档的版本与研究记录；不是当前包的功能清单。
- [当前续接进度](PROGRESS.md)：已完成阶段、真实验证结果、产物位置和下一步。
- [dev.19 发布说明](releases/0.1.0-alpha.6-dev-19.md)：Core profile 缓存恢复更新与验证边界。
- [dev.18 发布说明](releases/0.1.0-alpha.6-dev-18.md)：已发布包的变更范围与验证边界，不含后续主线修复。
- [dev.17 发布说明](releases/0.1.0-alpha.6-dev.17.md)：历史包的研究范围与验证边界。
- [dev.16 发布说明](releases/0.1.0-alpha.6-dev.16.md)：历史包的研究范围与验证边界。
- [dev.15 发布说明](releases/0.1.0-alpha.6-dev.15.md)：历史包及验证范围。
- [起音保留账本协议](onset-retention-ledger-protocol.md)：冻结输入上的静态起音去留核对。
- [起音保留账本结果](onset-retention-ledger-results.md)：《樱花草》10K候选、定轨与选择器去留的静态汇总。
- [表达取舍静态审计协议](expression-tradeoff-audit-protocol.md)：按段落、声部定位被省略的持续音候选，不估算音乐收益。
- [表达取舍静态审计结果](expression-tradeoff-audit-results.md)：《樱花草》10K默认候选的LN资格与现有覆盖关系。
- [《樱花草》Timing 歧义静态诊断协议](sakura-timing-ambiguity-protocol.md)：固定候选网格、半速/倍速相位和局部覆盖。
- [《樱花草》Timing 歧义静态结果](sakura-timing-ambiguity-results.md)：74/148/296 BPM 候选的网格相容与未解决边界。
- [dev.7 发布说明](releases/0.1.0-alpha.6-dev.7.md)：Windows 便携构建修复及验证范围。
- [dev.6 发布说明](releases/0.1.0-alpha.6-dev.6.md)：模型下载回退及历史变更。
- [开发测试版本记录](alpha5-validation.md)：早期发布验证历史。

## 研究入口

- 采音与声部：`*instrument*`、`*guitar*`、`*vocal*`、`*drum*`、`*piano*`。
- LN 与表达：`*ln-*`、`*expression-*`、`*gesture-*`、`*sequence-*`。
- 分离与转录：`*separation*`、`*transcription*`、`*mapper*`、`*tsumugi*`。
- 数据、协议和复核：相同主题的 `*-protocol.md` 记录冻结条件；`*-results.md` 或 `*-research.md` 记录实际结果、退步和未验证项。
- [《樱花草》候选差异听审协议](chord-evidence-audio-review-protocol.md)：本地原混音复核和标签边界。

## 目录约定

- `manialab/`：应用服务端、领域逻辑和运行时组件。
- `static/`：浏览器界面。
- `assets/`、`licenses/`、`vendor/`：随源码管理的资源、来源说明和固定第三方模板。
- `scripts/`：可复现的构建、验证、研究和数据处理入口，见[脚本索引](../scripts/README.md)。
- `tests/`：自动化测试。
- `docs/`：使用说明、设计、研究协议与进度；研究文件保留原路径。
- `build/`、`dist/`、`release/`、`test-data/`：忽略的本地构建或研究产物。音频、参考谱、模型权重和逐音结果不进入源码提交。

## 本地目录清理规则

这些目录虽然不进入 Git，但不都属于缓存：

| 路径 | 用途 | 清理规则 |
| --- | --- | --- |
| `build/portable/<run-id>/` | 隔离的构建、验证和历史 Release ZIP | 保留已验证产物；新构建使用新 run ID，不覆盖旧目录。 |
| `release/` | 早期便携包、校验记录和发布辅助材料 | 作为版本对照保留，不批量删除。 |
| `test-data/` | 用户音频、参考谱、下载语料、逐曲研究输入和输出 | 保留；不得加入源码提交，也不要按缓存清理。 |
| `assets/deep/`、`assets/instruments/`、`assets/runtime/` | 本地模型和便携运行依赖 | 保留。当前构建按明确 profile 选取并核验资源；目录中仍可能保存历史研究所需文件，不能据目录名推断内容可丢弃。 |
| `.venv/`、`.venv-audio/` | 项目运行和音频研究环境 | 保留，除非已计划重建对应环境。 |
| `__pycache__/`、`.pytest_cache/` | Python 字节码和 pytest 缓存 | 可再生；没有相关任务运行时可以删除。 |
| 根目录 `dist/` | 旧式或手动 PyInstaller 输出位置 | 当前 `scripts/build_windows.py` 使用 `build/portable/<run-id>/`。若根目录 `dist/` 非空，先检查具体产物再决定；空目录可移除。 |

清理 Git 分支时，只删除已确认包含于 `origin/main` 的本地分支；保留当前工作分支、`main` 和所有未合并分支。清理本地分支不会删除远端分支。

打包按明确的组件与 worker 文件清单核验；发布前应核对对应报告中的能力及模型身份，不能只看源码里有对应功能入口。当前操作见[开发与验证](development.md)，历史问题见[钢琴专用模型的发布范围审计](PROGRESS.md#钢琴专用模型与发布包范围审计)。
