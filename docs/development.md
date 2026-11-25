# 开发与验证

主程序和音频 worker 使用独立的 `.venv/` 与 `.venv-audio/` 环境；不要运行已完成或正在排队的研究推理。CPU 算子线程数设为 2，跨算子线程数设为 1。所有构建和测试使用新的 `build/` run 目录，不覆盖旧产物。

## 日常检查

```powershell
.\.venv\Scripts\python.exe -m compileall -q manialab launcher.py
.\.venv\Scripts\python.exe -m pytest -q
npm ci --ignore-scripts
npx playwright install chromium
npm run test:ui
git diff --check
```

浏览器回归通过 `playwright.config.cjs` 启动隔离服务，使用合成夹具，不调用神经采音。失败截图和 trace 写入忽略目录 `tests/artifacts/browser/`。可通过 `MANIALAB_TEST_PYTHON` 指定测试解释器、`MANIALAB_TEST_PORT` 指定端口；不要复用占用端口的旧服务。

## Windows 构建

按固定锁文件准备虚拟环境与模型资源：

```powershell
uv venv .venv --python 3.12
uv pip sync --python .venv\Scripts\python.exe requirements-lock.txt
.\.venv\Scripts\python.exe scripts\fetch_audio_models.py
uv venv .venv-audio --python 3.12
uv pip sync --python .venv-audio\Scripts\python.exe --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match requirements-deep-lock.txt
.\.venv-audio\Scripts\python.exe scripts\fetch_piano_model.py
.\.venv-audio\Scripts\python.exe scripts\build_audio_worker.py
.\.venv\Scripts\python.exe scripts\build_windows.py --profile core
```

使用新的 `MANIALAB_BUILD_ID` 构建到 `build/portable/<run-id>/`。当前 Windows CI 固定验证 Core；其他 profile 的本地构建不等于公开支持或完整便携验收。构建器会核验模型、源码、worker 文件清单、许可和 ZIP manifest。合成 smoke 只验证对应软件路径，不证明真实音频质量或授权状态。

## 音乐验收

真实歌曲、谱包、模型和逐音产物保存在忽略的 `workspace_exp/` 或用户指定的本地目录。新实验使用唯一目录；禁止覆盖历史实验。人工试听与 osu! 试打须单独记录音源身份、Timing、键数、问题位置和结论，详见[音乐质量目标](music-quality-target.md)与[人工评审记录](human-review-integrity.md)。
