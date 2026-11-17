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

## 模型与构建

`scripts/build_windows.py --profile standard` 明确包含通用音高、四／六声部分离、节拍及钢琴组件。`--profile extended` 额外要求通过来源与运行验证的吉他 FL 模型、Tsumugi worker；缺失时构建失败，不默默降级。`scripts/package_assets.py` 按清单暂存资源，排除本地历史模型、额外实验文件及未声明组件。

包内 `release-manifest.json` 记录 profile、capabilities、模型身份与源码摘要；`开始使用.txt` 从同一能力声明生成。发布验证核对实际组件，不能仅凭文件存在称模型可用。开发时钢琴状态区分 missing、mismatched、verified 与文件读取异常；状态查询缓存至多 30 秒，推理前强制校验。

两个 worker builder 在构建完成时记录完整 `worker_files`（相对路径、字节数与 SHA）。暂存、包后校验和实际 EXE 验证拒绝多余、缺失或变化的运行文件，并拒绝符号链接与目录联接。没有清单的历史 worker 不能直接用于新包，应保留历史产物并重新构建；不得给未经核验的旧目录补清单绕过检查。Tsumugi 的运行验证报告也绑定该清单，依赖文件变化后须重新验证；本阶段只完整验证 standard，未重新运行历史 Tsumugi 音乐实验。

已有历史 `assets/deep/piano-regression.onnx` 时不覆盖它。可将已验证的当前模型复制到 `assets/deep/piano-regression-<当前完整SHA256>.onnx`；运行和构建优先使用该版本文件。SHA 和大小从 `assets/piano-model.json` 读取并严格核对。需要重新准备时，可把该版本文件路径传给 `scripts/fetch_piano_model.py --output <路径>`；其默认工作目录按时间生成，已有输出不覆盖。发布包中统一命名为 `piano-regression.onnx`，历史文件不进入包。

正式流水线见 `.github/workflows/windows.yml`。普通 PR 运行 Python 和浏览器测试；tag、手动触发或带 `[portable]` 的构建提交继续执行模型准备、EXE、离线 clean PATH、Unicode 数据路径、worker 退出及 ZIP 身份验证。检查通过和构建完成只证明对应的软件契约，音乐质量仍单独记录。
