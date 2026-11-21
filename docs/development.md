# 开发与验证

从 [README](../README.md) 的锁定环境步骤开始。应用主环境和音频 worker 环境分开，研究推理串行，CPU 算子 2 线程、跨算子 1 线程。日常软件测试不运行真实歌曲模型队列。

## 日常检查

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q manialab scripts tests
npm ci --ignore-scripts
npx playwright install chromium
npm run test:ui
git diff --check
```

浏览器测试由 `playwright.config.cjs` 启动隔离服务，创建临时合成音频与已冻结的虚拟检测结果，覆盖生成、键数改变、Timing 改变、音符编辑、保存重开、导出及原谱严格保时；不调用神经采音。Python 路径可用 `MANIALAB_TEST_PYTHON`，端口可用 `MANIALAB_TEST_PORT`（默认 8977）设置；本机已安装 Edge 时可设置 `MANIALAB_TEST_BROWSER=msedge`。不会复用占用该端口的旧服务，避免测试误改真实项目。失败截图和 trace 在忽略目录 `tests/artifacts/browser/`。

`scripts/ui_*.cjs` 保留为历史研究入口；常规 CI 使用新的合成测试，不依赖私人 `test-data` 音频或已安装 Edge。

已有冻结分析可通过 `scripts/replay_frozen_generation.py` 回归真实生成流程，无需重新运行模型。
计划固定输入SHA、Timing与参数；修改前后使用同一计划、不同输出目录，并用 `--baseline`
保存逐ID对照。协议与局限见[冻结回归](frozen-generation-regression-protocol.md)，本轮结果见
[多曲对照](frozen-generation-regression-results.md)。逐音产物不加入Git。

## 模型与构建

`scripts/build_windows.py --profile core` 仅包含许可与来源记录齐全的 Basic Pitch ONNX、应用自身谱面配置及通用分析/制谱代码；不包含内置 Demucs 分离器、节拍模型、钢琴/键型权重、uv 或原模型安装器。dev.20 是当前最新公开 Windows 包，且仅支持 Core。Core 包携带 `assets/model-manifest.json`；固定版本、来源 URL 与 Basic Pitch SHA-256 在暂存、EXE 和最终 ZIP 中逐层核对。混音采音、已有谱面重定轨和用户导入声轨仍可使用。`standard` 描述历史本地构建的四/六声部分离、节拍、钢琴、键型先验及原模型安装器范围；`extended` 再要求吉他 FL 模型和 Tsumugi worker。这些显式源码构建配置不表示相应完整包已公开发布或通过当前发布验收。Demucs 权重的单独再分发授权仍未核实，不能把这些本地显式构建配置称为可公开发行。GitHub 便携验证只构建 `core`，并核对最终归档实际文件；公开旧版包不会因此自动更新。`scripts/package_assets.py` 按清单暂存资源，排除本地历史模型、额外实验文件及未声明组件。

包内 `release-manifest.json` 记录 profile、capabilities、模型身份与源码摘要；`开始使用.txt` 从同一能力声明生成。发布验证核对实际组件，不能仅凭文件存在称模型可用。开发时钢琴状态区分 missing、mismatched、verified 与文件读取异常；状态查询缓存至多 30 秒，推理前强制校验。

两个 worker builder 在构建完成时记录完整 `worker_files`（相对路径、字节数与 SHA）。暂存、包后校验和实际 EXE 验证拒绝多余、缺失或变化的运行文件，并拒绝符号链接与目录联接。没有清单的历史 worker 不能直接用于新包，应保留历史产物并重新构建；不得给未经核验的旧目录补清单绕过检查。Tsumugi 的运行验证报告也绑定该清单，依赖文件变化后须重新验证。此前的本地 Standard worker 资源检查不等于完整 Standard 包的构建、运行或发布验收；历史 Tsumugi 音乐实验也未重新运行。

已有历史 `assets/deep/piano-regression.onnx` 时不覆盖它。可将已验证的当前模型复制到 `assets/deep/piano-regression-<当前完整SHA256>.onnx`；运行和构建优先使用该版本文件。SHA 和大小从 `assets/piano-model.json` 读取并严格核对。需要重新准备时，可把该版本文件路径传给 `scripts/fetch_piano_model.py --output <路径>`；其默认工作目录按时间生成，已有输出不覆盖。发布包中统一命名为 `piano-regression.onnx`，历史文件不进入包。

正式流水线见 `.github/workflows/windows.yml`。PR 始终运行 Python 和浏览器测试；普通功能分支 push 跳过重复的验证 job。`main`、tag、手动触发或带 `[portable]` 的 push 仍运行检查；tag、手动触发和 `[portable]` push 继续执行模型准备、EXE、离线 clean PATH、Unicode 数据路径、worker 退出及 ZIP 身份验证。检查通过和构建完成只证明对应的软件契约，音乐质量仍单独记录。
