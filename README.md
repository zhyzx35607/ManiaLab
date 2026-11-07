# ManiaLab · 6–10K 制谱工作台

面向 **6–10K osu!mania，优先 10K 和 LN** 的 Windows 桌面工作台。以 Mapperatorinator 为上游，增加可检查的音频证据、持续音编排、LN 占轨约束和严格重新定轨。

**当前版本：0.1.0-alpha.3。可运行的实验版；尚未训练专用 10K 模型。** 本次修复 BPM 数值整理、完整拍线量化及导出落线校验；实际歌曲自动 Offset 的准确性仍待验证。仍不能保证输出达到可发布谱面的音乐表达与手感水平。

![ManiaLab 界面](docs/screenshot.png)

## Windows 直接运行

1. 在 [Releases](https://github.com/zhyzx35607/ManiaLab/releases) 下载 `ManiaLab-0.1.0-alpha.3-windows-x64.zip`。
2. **完整解压**后双击 `ManiaLab.exe`。程序会自动打开本机浏览器界面；保持控制台窗口运行，关闭窗口即退出。
3. 导入音频 → 分析音频 → 校准 BPM / Offset、试听节拍器并确认 → 生成候选 → 下载 `.osz` → 拖入 osu! 测试。

基础功能包含 Python 运行时、ONNX 音高模型和音频解码器，不需要安装 Python、Docker、CUDA 或配置服务。面向 Windows 10/11 x64；本机 Windows 10 已验证，其他电脑兼容性仍待测试。默认在 CPU 上运行。

所有项目自动保存到 `%LOCALAPPDATA%\ManiaLab\projects`。可从「最近项目」继续工作；迁移项目时复制该文件夹。日志在 `%LOCALAPPDATA%\ManiaLab\ManiaLab.log`。`.osu` 单文件导出需要自行配套音频；`.osz` 包含音频和支持的图片、音效文件。

## 已有功能

- **音频生成**：Basic Pitch ONNX 检测有音高的起音和持续区间，频谱变化检测瞬态；展示波形、采音证据、音符起止和来源。
- **拍线校准**：自动 BPM 整理、整数毫秒 Offset、固定 BPM 重新拟合偏移、粘贴参考 Timing、倍速/半速切换、细分网格、节拍器和全曲漂移检查。生成与导出均逐个检查短键与 LN 起止落线。
- **LN 编排**：LN 只能来自已检测的持续区间；目标比例定义为 LN 数 ÷ 总物件数，以整数约束求解可达比例；保持量化后的选中起音；LN 尾端就近量化，可能向前或向后移动半个细分。控制密度、同时 LN 数、起音组合、最短 LN 和采音侧重。
- **6–10K 定轨**：比较 6 个完整布局，检查轨道占用、放键间隔、同轨连续敲击、双手短时负担与音高连贯；7K/9K 中间键按共享拇指处理。它是约束和启发式编排，尚不具备成熟谱师的完整键型判断。
- **严格重定轨**：保留原谱每个物件的数量、整数起止时间、类型、音效和原始 TimingPoints，只改变轨道及相关谱面标识。无法满足占轨约束时报告时间，不删音或缩短 LN。
- **候选与局部重写**：保留最近 8 个版本；改变种子比较布局，按起音时间局部重写并锁定范围外物件。
- **可选 API 助手**：接入兼容 `/chat/completions` 的服务，读取本地段落统计和文字要求，产生经过范围校验的参数与段落偏好。不会向文本 API 发送音频，也不让 API 编造音符时间。密钥不落盘。
- **可选原模型**：内含 Mapperatorinator 固定版本源码快照，通过按钮安装隔离的 CPU 环境并下载固定版本 V32 mania 权重。无需手动配置；首次下载需要网络和数 GB 空间。可与基础采音编排结果比较。
- `.osu` / `.osz` / 分析 JSON 导出，以及本地训练语料审计脚本。

## 当前边界

Basic Pitch 不是乐器分离模型。复杂混音、滑音、混响和音尾会误判；声部连续性、跨音高长 LN、段落音乐表达和高阶 10K 键型尚待改进。检测激活强度不是经过校准的置信概率。

自动检测的 BPM：接近 .25 / .333 / .5 / .667 / .75（误差不超过 0.01 BPM）时保留该分数，其余四舍五入到整数。例如 210.00073 → 210。随后固定 BPM 重新拟合整数毫秒 Offset，并重新计算全曲诊断；原始拟合值保存在分析 JSON 中。手动输入的 BPM 不强制取整，适用于特殊速度。

确认 Timing 后，生成会把所有短键起点、LN 起点和尾端量化到所选细分；短 LN 若量化为零长则变为短键。报告单独显示超过 30ms 的起音移动。导出回读真实 `.osu`，按其中的红线检查每个端点，只允许整数毫秒存储产生的不超过 0.501ms 误差。**落线是必要条件，不代表采音正确。**

自动拍线不稳定时会停止生成，需输入参考 Timing 或手动校准后试听确认。可粘贴 `BPM: 210` 和 `Offset: 91`；更改 Timing 后需重新生成整谱，局部重写不能混用拍线。旧版项目需重新分析（音高缓存可复用），旧候选保留供查看，但不能直接导出谱面。分析Timing.exe 不随包分发。当前不自动生成多段变速红线；严格重定轨保留原始 TimingPoints。

手感参数是软偏好；严格重定轨优先保持原谱时间。结构校验通过表示没有同轨重叠、轨道越界等错误，**不表示手感、难度或音乐性已合格**。LN 目标按物件数计算，受持续音证据和占轨约束限制；界面显示目标、实际和当前约束下的可达比例。

API 目前验证了兼容协议和本地模拟服务，未用用户的真实供应商密钥验证。不同供应商可能需要调整模型 ID 或 Base URL。原模型引擎仍为上游通用 mania 权重，不是专训后的 10K 权重。原模型生成会把输出设为当前项目的新原谱并替换候选列表。

支持不超过 20 分钟音频、单次上传 512MB。谱包导入保留音频、`.osu` 和常见图片；不复制视频和外部 `.osb` 文件。基础版目前没有手工拖动音符编辑器、自动分乐器、稳定的变速识别或 osu! 星级评估。

## 与上游的关系

- 上游：[OliBomby/Mapperatorinator](https://github.com/OliBomby/Mapperatorinator)，固定 commit `5884d1bcb57494f09fe3d74f3a6b58ca0d2960aa`，MIT。
- 复用其 `Event` / `EventType` 表示和 `.osu` 模板；原模型运行器直接调用该版本 `inference.py`。
- 新增音频证据、严格时间不变量、LN 占轨分配、API 参数助手和 Windows 打包工作流。**没有修改或替换上游原目录。**
- Basic Pitch 模型来自 Spotify `v0.4.0`，Apache-2.0；版本与哈希见 `assets/model-manifest.json`。
- 定轨设计参考 [Manolo-dl/Torosu](https://github.com/Manolo-dl/Torosu) 的手部布局、释放状态与键型代价思路，固定 commit `b283791f46622b0dd1f47f53afa10bd62d700e66`；本项目独立实现多方案选择。Torosu 的 Inverse 转换不是音频持续音识别。
- 自有代码 MIT；第三方组件各自保留原许可证，见 `licenses/` 和 `THIRD_PARTY_NOTICES.md`。

## 分析为什么可能很快

默认运行的是小型 Basic Pitch 音符检测模型，然后做拍线拟合、LN 约束求解和多方案定轨。它不是已经训练好的 10K 制谱大模型。首次分析运行神经网络；同一音频调整参数时会复用缓存。界面显示实际分析/编排耗时和缓存状态，不用等待动画模拟深度分析。模型仍可能漏音或错判长条。

`SHA256SUMS.txt` 是校验文件；正常使用只需下载 Windows ZIP 并完整解压。

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
