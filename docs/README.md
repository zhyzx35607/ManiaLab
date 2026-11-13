# 文档索引

研究记录按实验保留原文件名与路径，便于复现已有命令和交叉引用。本索引提供入口，不代表所有研究结论都进入产品默认行为。

## 使用 ManiaLab

- [快速开始](quick-start.md)：安装后导入、分析、编排与导出。
- [乐器采音选择](instrument-choices.md)：各声部入口、默认值与音尾限制。
- [已有声轨导入](provided-stems.md)：导入分轨并保留原音频时钟。
- [段落与编排设计](arrangement-design.md)：段落边界提示、声部和表达控制的范围。
- [音乐质量目标](music-quality-target.md)：项目长期验收标准。

## 项目状态

- [当前续接进度](PROGRESS.md)：已完成阶段、真实验证结果、产物位置和下一步。
- [dev.13 发布说明](releases/0.1.0-alpha.6-dev.13.md)：当前主线对齐包及验证范围。
- [起音保留账本协议](onset-retention-ledger-protocol.md)：冻结输入上的静态起音去留核对。
- [起音保留账本结果](onset-retention-ledger-results.md)：《樱花草》10K候选、定轨与选择器去留的静态汇总。
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
