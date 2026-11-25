# ManiaLab 2.0

ManiaLab 是面向 osu!mania 6–10K 的本地 Windows 制谱工作台，重点关注 10K、长条（LN）、清晰采音与可读的段落表达。自动化测试用于验证工程行为，不代替真实歌曲试听和 osu! 实打。

## 当前版本

公开试用版本为 Windows x64 Extended profile。版本、下载与校验文件见 [GitHub Releases](https://github.com/zhyzx35607/ManiaLab/releases)。当前包包含声部分离及专用采音组件；具体能力以包内 `release-manifest.json` 为准。Extended 仍是试用范围，合成便携检查不构成真实音乐质量验收。

## 开始使用

下载并完整解压 ZIP，运行 `ManiaLab.exe`。导入歌曲后检查 BPM / Offset，试听采音和拍线，生成并检查候选，最后导出 `.osz` 并在 osu! 中实打。已有谱包也可导入后严格重新定轨。详细流程见[快速开始](docs/quick-start.md)。

## 质量目标与边界

- 以真实音频为依据检查起音、声部、段落和 LN 尾部。
- 参考谱用于定位差异，不要求逐音符复刻，也不假设人工参考绝对正确。
- Timing 与轨道合法只是必要条件；最终听感和手感需要人工验收。
- 用户音频、谱包、模型权重和逐音研究数据保留在本地忽略目录，不提交到源码。

## 开发

主程序和音频 worker 使用 `.venv/`、`.venv-audio/` 两个环境。安装、pytest、浏览器回归和 Windows 包构建命令见[开发与验证](docs/development.md)。

项目状态见[续接进度](docs/PROGRESS.md)，代码职责见[架构说明](docs/architecture.md)，可复现构建和人工验收入口见[脚本索引](scripts/README.md)。阶段任务与停止条件见[路线图](ROADMAP.md)。
