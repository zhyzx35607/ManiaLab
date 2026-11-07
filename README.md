# ManiaLab · 6–10K 制谱工作台

面向 **6–10K osu!mania，优先 10K 和 LN** 的 Windows 桌面工作台。以 Mapperatorinator 为上游，增加可检查的音频证据、持续音编排、LN 占轨约束和严格重新定轨。

**当前版本：0.1.0-alpha.1。可运行的实验版；尚未训练专用 10K 模型。** 本版建立工作流和测试基础，不能保证输出达到可发布谱面的音乐表达与手感水平。

![ManiaLab 界面](docs/screenshot.png)

## Windows 直接运行

1. 在 [Releases](https://github.com/zhyzx35607/ManiaLab/releases) 下载 `ManiaLab-0.1.0-alpha.1-windows-x64.zip`。
2. **完整解压**后双击 `ManiaLab.exe`。程序会自动打开本机浏览器界面；保持控制台窗口运行，关闭窗口即退出。
3. 导入音频 → 分析音频 → 检查采音证据 → 生成候选 → 下载 `.osz` → 拖入 osu! 测试。

基础功能包含 Python 运行时、ONNX 音高模型和音频解码器，不需要安装 Python、Docker、CUDA 或配置服务。面向 Windows 10/11 x64；本机 Windows 10 已验证，其他电脑兼容性仍待测试。默认在 CPU 上运行。

所有项目自动保存到 `%LOCALAPPDATA%\ManiaLab\projects`。可从「最近项目」继续工作；迁移项目时复制该文件夹。日志在 `%LOCALAPPDATA%\ManiaLab\ManiaLab.log`。`.osu` 单文件导出需要自行配套音频；`.osz` 包含音频和支持的图片、音效文件。

## 已有功能

- **音频生成**：Basic Pitch ONNX 检测有音高的起音和持续区间，频谱变化检测瞬态；展示波形、采音证据、音符起止和来源。
- **LN 编排**：LN 只能来自已检测的持续区间；长条偏好决定选择，不任意延长声音。控制密度、同时 LN 数、起音组合、最短 LN 和采音侧重。
- **6–10K 定轨**：检查轨道占用，考虑放键间隔、同轨连续敲击、双手负担与音高连贯。它是约束和启发式编排，尚不具备成熟谱师的完整键型判断。
- **严格重定轨**：保留原谱每个物件的数量、整数起止时间、类型、音效和原始 TimingPoints，只改变轨道及相关谱面标识。无法满足占轨约束时报告时间，不删音或缩短 LN。
- **候选与局部重写**：保留最近 8 个版本；改变种子比较布局，按起音时间局部重写并锁定范围外物件。
- **可选 API 助手**：接入兼容 `/chat/completions` 的服务，读取本地段落统计和文字要求，产生经过范围校验的参数与段落偏好。不会向文本 API 发送音频，也不让 API 编造音符时间。密钥不落盘。
- **可选原模型**：内含 Mapperatorinator 固定版本源码快照，通过按钮安装隔离的 CPU 环境并下载固定版本 V32 mania 权重。无需手动配置；首次下载需要网络和数 GB 空间。可与基础采音编排结果比较。
- `.osu` / `.osz` / 分析 JSON 导出，以及本地训练语料审计脚本。

## 首版边界

Basic Pitch 不是乐器分离模型。复杂混音、滑音、混响和音尾会误判；声部连续性、跨音高长 LN、段落音乐表达和高阶 10K 键型尚待改进。检测激活强度不是经过校准的置信概率。

BPM / Offset 是粗略估计，可能有倍速或半速问题。本版不自动把采音时间吸附到估计拍线上；已有谱面的重定轨保留原始变速。生成谱面的节拍线需人工检查。

手感参数是软偏好；严格重定轨优先保持原谱时间。结构校验通过表示没有同轨重叠、轨道越界等错误，**不表示手感、难度或音乐性已合格**。LN 偏好值不是精确的 LN 数量占比。

API 目前验证了兼容协议和本地模拟服务，未用用户的真实供应商密钥验证。不同供应商可能需要调整模型 ID 或 Base URL。原模型引擎仍为上游通用 mania 权重，不是专训后的 10K 权重。原模型生成会把输出设为当前项目的新原谱并替换候选列表。

支持不超过 20 分钟音频、单次上传 512MB。谱包导入保留音频、`.osu` 和常见图片；不复制视频和外部 `.osb` 文件。基础版目前没有手工拖动音符编辑器、自动分乐器、稳定的变速识别或 osu! 星级评估。

## 与上游的关系

- 上游：[OliBomby/Mapperatorinator](https://github.com/OliBomby/Mapperatorinator)，固定 commit `5884d1bcb57494f09fe3d74f3a6b58ca0d2960aa`，MIT。
- 复用其 `Event` / `EventType` 表示和 `.osu` 模板；原模型运行器直接调用该版本 `inference.py`。
- 新增音频证据、严格时间不变量、LN 占轨分配、API 参数助手和 Windows 打包工作流。**没有修改或替换上游原目录。**
- Basic Pitch 模型来自 Spotify `v0.4.0`，Apache-2.0；版本与哈希见 `assets/model-manifest.json`。
- 自有代码 MIT；第三方组件各自保留原许可证，见 `licenses/` 和 `THIRD_PARTY_NOTICES.md`。

## 开发与验证

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt
.\.venv\Scripts\python.exe launcher.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\build_windows.py
```

测试覆盖 6–10K 读写、LN 占轨冲突、严格保时、原始变速、跨边界局部锁定、模型多窗口对齐、持续音长度、ZIP 路径、API 参数约束及 HTTP 接口。浏览器测试脚本 `scripts/ui_smoke.cjs` 需要另外安装 Playwright，并通过 `MANIALAB_URL` 指定运行中的程序；详见 `docs/validation.md`。

训练准备：

```powershell
.\.venv\Scripts\python.exe scripts\audit_corpus.py D:\YourManiaMaps --output corpus-audit.jsonl
```

该命令只做本地筛选与分割审计，**不会启动训练**。后续训练重点和验收标准见 [训练计划](docs/training-plan.md)。
