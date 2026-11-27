# ManiaLab 2.0

ManiaLab 是面向 osu!mania 6–10K 的本地智能制谱工作台，以 10K 为核心。V32 Full 为旗舰，`v32-mini` 为默认快速版；NVIDIA GPU 自动使用 CUDA，无 GPU 时回退 CPU。谱面结构与自动指标用于检查工程结果，最终质量仍需试听并在 osu! 中实打。

## 当前公开试用版

[下载 ManiaLab 2.0 Workbench Beta 2（Windows x64）](https://github.com/zhyzx35607/ManiaLab/releases/download/v0.2.0-beta.2/ManiaLab-0.2.0-beta.2-windows-x64-workbench.zip)（单文件 ZIP，首次使用 V32 时在界面点击「准备 V32 运行环境」）。

## 工作流程

1. 导入一首歌曲或已有 `.osu/.osz`；也可一次选择最多 20 首歌曲。
2. 检查采音证据并确认 BPM / Offset；批量任务按单首记录进度，刷新后可继续未完成歌曲。
3. 选择 6–10K、难度与风格：切指偏米 `rice_stream`、叠键偏米 `rice_jack`、综合随歌 `hybrid_adaptive`、精致 LN `ln_focus`，或已有谱 Torosu 改造 `torosu_remap`。可选择本地 LoRA。
4. 使用竖屏下落预览检查短键、LN、拍线与键型摘要，然后导出 `.osu/.osz`、逐首谱包或整套 ZIP。

## Windows 便携版

下载并完整解压 Windows x64 `workbench` 包，运行 `ManiaLab.exe`。首次使用 V32 时按提示准备独立运行环境与模型权重；本地包提供 V32 按需安装器、FFmpeg 和 Torosu CLI。用户自行训练的 LoRA 可放到 `assets/lora/<名称>/`，目录内需包含 `adapter_config.json` 与 `adapter_model.safetensors`，重启或刷新后会出现在选择列表中。包内 `使用说明.txt` 提供简要操作步骤；组件清单见 `release-manifest.json`。

旧 Core、Standard 和 Extended profile 仍保持各自声明的能力范围；自动构建与验证结果不代表真实音乐质量验收。

## 开发

主程序和音频 worker 使用 `.venv/`、`.venv-audio/`。测试、浏览器回归及 Windows 便携构建入口见[开发与验证](docs/development.md)；代码职责见[架构说明](docs/architecture.md)，人工验收工具见[脚本索引](scripts/README.md)，阶段路线见[ROADMAP.md](ROADMAP.md)。
