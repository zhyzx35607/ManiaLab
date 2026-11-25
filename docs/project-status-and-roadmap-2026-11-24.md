# ManiaLab 当前状态与重规划

更新：2026-11-25。状态依据本地 `main`/精确 release tag、GitHub Release、随包验证说明和源码清单核对。本次只对生成的合成音运行便携/worker 校验，没有读取 ignored 用户音频、谱面或逐曲研究数据。

## 结论

ManiaLab 已经是可体验的 6–10K osu!mania 本地制谱工作台，不是尚未成形的原型。最新的 `0.1.0-alpha.6-dev.27` Extended Windows x64 预发布包已于 2026-11-25 发布，提供单个 ZIP 直接下载；dev.26 Release 与资产保持不变。工程主流程和便携启动检查覆盖较完整；真实音乐质量、跨机器安装体验和扩展模型权利仍未闭环，所以目前应称为“公开试用、等待真实反馈”，不能称为稳定完成。

当前重点是收回真实听审/试打结果、确认扩展模型权利，并在另一台干净 Windows 电脑验证直接下载和首次启动。dev.27 的单 ZIP 已免去 dev.26 下载器的 12 片组装步骤；完整 Extended 包仍约 1.15 GiB，清理测试文件数量不会显著缩小它。

## 目前已经具备的产品流程

用户可以导入音频或已有 osu!mania 谱面，运行可用的声部分离与采音，核对 BPM/Offset 和音符来源，选择预设或编排方案生成候选，编辑候选，再导出 `.osu`/`.osz`。应用还支持区域重写、候选历史与恢复、局部音频试听/波形定位、6K–10K 键型和 LN 设置，以及一次导入 2–10 首、最多并行分析 2 首的批量流程。

已完成的重点是可用工作台、数据和候选身份保护、任务取消与资源清理、导出一致性、浏览器工作流，以及 Core/Extended 便携包的构建验证路径。模型研究脚本仍存在很多，但研究结果不会自动进入默认产品行为。

## 当前完成度

| 范围 | 状态 | 真实边界 |
| --- | --- | --- |
| dev.27 发布 | 已发布 Extended 公开试用预发布版 | tag `v0.1.0-alpha.6-dev.27` 指向 `32368e685a102e7aaf334e8bd6d1808200aa30ee`；ZIP `1,235,335,195` bytes，SHA-256 `dec6206c7b35c29c7345bbdd9179d3db14012ecc0eef80d13771ba5c08afe3bc`；不是正式稳定版。 |
| 工程回归 | dev.27 精确 tag 的 Python/API `3667 passed, 161 skipped, 1 warning`；Edge Playwright `82 passed` | Windows 自动工作流默认只构建 Core；合成测试不能证明音乐质量。 |
| Extended EXE | 20 项便携检查、worker 关停检查通过 | 缺少吉他和 Tsumugi 的冻结音频样本，因此没有重跑对应 EXE 真实音频专项，也没有完整聚合报告。 |
| 真实音乐 | 未验收 | 84 段钢琴起音盲听、6–10K/LN 实际试打、Sweet Waltz 与 Choice 独立歌曲复听仍待真人结果。 |
| 发布下载 | dev.27 提供 ZIP 直链 | dev.27 Release 单文件直链已返回 HTTP 200；仍需在另一台干净 Windows 电脑实测下载、解压和首次启动。dev.26 的 12 片附件保留作历史资产。 |
| 模型权利 | 未闭环 | Demucs checkpoint 再分发权利、Tsumugi 训练录音权利尚未独立确认。 |

## 体积从哪里来

dev.26 组装 ZIP 为 `1,310,888,430` bytes，即 1.31 GB（十进制）或约 1.22 GiB；按 ZIP 目录统计，解压后约 `2,714,858,795` bytes（2.53 GiB）。组装期间，分片和临时 ZIP 会同时存在，约需 2.44 GiB 空间；保留 ZIP 并解压后，建议预留至少 4 GiB。

本地已验证 ZIP 的压缩体积约 85.3% 来自随包 `assets`。最大的内容是 Demucs 高质量权重、钢琴 ONNX、吉他 FL、Beat This、Tsumugi 贝斯/其他伴奏权重和六声部模型。2026-11-25 已在构建器加入安全去重：只有 app 根目录的 `torch_cpu.dll` 与音频 worker 内副本大小及 SHA-256 完全相同时，才从新构建 app 目录删除根副本；隔离 worker 保留，Core profile 为 no-op。匹配删除、不匹配保留、worker 缺失保留、根 DLL 缺失 no-op 和链接拒绝均有回归测试。该改动已随 dev.27 精确 tag 发布；Extended 包完成 portable verifier、four-piano 合成分析/导出、worker 关停、CRC 和 archive manifest 清单校验。dev.27 ZIP 为 `1,235,335,195` bytes，SHA-256 `dec6206c7b35c29c7345bbdd9179d3db14012ecc0eef80d13771ba5c08afe3bc`；相对 dev.26 已发布 ZIP 减少 `75,553,235` bytes（5.764%）。dev.26 ZIP 未被替换。

dev.26 的 `.ps1` 下载器会同时启动 12 个隐藏的 `curl.exe`，窗口只显示已完成分片数，未完成的 `.download` 文件不能续传。dev.27 已提供一个 ZIP 的直链，不需要分片组装；本机对该 ZIP 直链的 HEAD 返回 HTTP 200，但这不等于另一台电脑已完成下载，仍要做实际跨机器验证。

## 项目“臃肿”的实际构成

| 内容 | 数量/大小 | 判断 |
| --- | ---: | --- |
| Git 跟踪 `tests/` | 瘦身前最新 main：264 个文件（238 个 Python、21 个 CJS、5 个 PEM 夹具） | 没有按文件数删除 pytest；把一个旧 Timing 导航 runner 的有效断言迁入现有 Playwright spec，测试总文件数不变。 |
| Git 跟踪 `scripts/` | 瘦身前最新 main：421 个文件（395 个 Python、22 个 CJS、1 个 C#、1 个 csproj、2 个 Markdown）；本次退役后 420 个文件，CJS 为 21 个 | 研究、数据准备和报告工具混在一起；本次只移除已有替代覆盖且无调用方的单个旧浏览器 runner，不批量移动或删除其他脚本。 |
| Git 跟踪 `docs/` | 瘦身前最新 main：382 个文件（301 个 Markdown、77 个 JSON） | 研究证据有价值，但导航入口太多。`docs/PROGRESS.md` 约 365 KB / 2,204 行，把当前状态与长期历史混在一页；它只占 docs 体积约 2.3%，主要是阅读负担。 |
| 本地 ignored `build/` | 8,687 个文件，约 7.68 GB | 主要是不同 run 的构建、发布 ZIP、检查报告和缓存；需按 run/保留策略盘点，不能整目录清空。 |
| 本地 ignored `assets/` | 4,993 个文件，约 2.20 GB | 模型和运行资源；不能按缓存清理。 |

以上文件数依据本轮开始时的最新 `origin/main` `17381f9` 重新统计；该基线已包含此前增加的状态路线图及 `scripts/ui_smoke.cjs` 退役。当前本轮只移除 `scripts/ui_timing_navigation_smoke.cjs`，并在已存在的浏览器测试与 fixture server 中迁入其唯一行为覆盖，因此合并后预期为 docs 382、scripts 420、tests 264。

2026-11-25 本次测试工具收敛将旧 `scripts/ui_timing_navigation_smoke.cjs` 的 Timing 可疑区间跳转、试听 A、Offset 改动后诊断失效、结尾定位行为迁到 `tests/browser/timing.spec.cjs`，并在 `tests/browser/fixture_server.py` 添加 30 秒 disposable synthetic fixture。旧 runner 不能在干净 checkout 运行，依赖 ignored Playwright、固定 Edge 路径、外部 server 与未跟踪 WAV；其窄屏检查原已由 `tests/browser/workflow.spec.cjs` 覆盖。没有删除 pytest 用例；`ui_grid_smoke.cjs`、`ui_timing_smoke.cjs`、`ui_waveform_smoke.cjs` 和其他有独有场景的 runner 保留。Windows workflow 中 Python/API、浏览器、Core EXE、媒体/来源切换、关停与 release evidence collector 分属不同验证层，本次未合并或移除。

## 重规划顺序

### 1. 验证直接下载与首次启动

dev.27 已通过 Release 页面提供单 ZIP 直链，免去分片下载和组装。下一步在另一台干净 Windows 电脑从该链接完整下载、校验、解压并首次启动，记录实际网络、磁盘占用和首个失败点。若直链在特定网络仍不稳定，再决定是否为 dev.26 分片下载器增加低并发、续传、字节进度和磁盘预检；先用真实失败证据确定需求。

### 2. 把首次下载缩到轻量 Core

项目已经有 `core`、`standard`、`extended` 三种打包 profile，资源由 `scripts/package_assets.py` 按清单挑选。Core 主要提供 Basic Pitch；Standard 增加分离、节拍、钢琴及 Mapperatorinator 安装组件；Extended 再加入吉他和 Tsumugi。dev.26 精确 tag 的 Core 包已在本地构建验证，但 GitHub 上传曾遇到 HTTP 408，未发布；评估其作为轻量下载入口时需重新准备独立 Release 附件并验证下载。继续保留完整离线 Extended 包供愿意一次下载所有资源的用户。

dev.27 已验证并发布只移除 app 根目录与音频 worker 内大小和 SHA-256 完全相同的 `torch_cpu.dll`；两个隔离 worker 的运行时继续保留，Core profile 不执行去重。同步审计没有进入当前 profile 的文件，绝不从全局 `assets` 直接删除模型。第三方权利未核实的权重仍是后续正式发布门槛，不把“包能构建”当作再分发授权。

### 3. 冻结新增模型功能，回收真实音乐证据

保持当前“导入→分析→校准→表达/定轨→编辑→导出”主流程稳定，暂停没有真人反馈支撑的新模型堆叠。按既有协议回收 84 段钢琴盲听、6–10K/LN 玩家试打、Sweet Waltz 与 Choice 两首独立歌曲的 Timing/声部/段落/LN 评审。先锁定盲听结果再解盲；每个 issue 都要有复测或明确接受结论。工程测试通过不能代替该阶段。

### 4. 让文档与研究代码有明确入口

把本页作为“我现在做到哪”的首要入口；`PROGRESS.md` 继续记近期阶段和可复现命令，历史阶段逐步迁入归档文件，迁移前先盘点 Markdown 锚点并保留兼容链接。给研究脚本补“用途、协议/结果、输入目录、是否进入产品、最近复现状态”索引。历史路径先用兼容 wrapper 过渡，不一次性重命名数百个文件。

### 5. 用覆盖证据继续精简

把自动验证分成三层：每次 CI 的 API/领域与浏览器关键用户流程；发布时运行的 profile/真实 EXE/工件身份检查；仅在有冻结输入和协议时运行的研究/听审工具。只有在确认无人调用、无独有断言、无历史复现价值，并由替代测试覆盖后，才删除一个测试入口。保留故障注入、身份/回滚/取消、OSU/OSZ 往返、真实窄屏和候选来源保护等高风险测试。

### 6. 安全处理本地工作区占用

将 `build/` 视为分 run 的证据仓，而非可随手清空的缓存：保留当前 release 验证、仍被进度记录引用的 artifact 与用户研究输出；只对明确过期、可重新生成且没有引用的目录建立清单后清理。`assets/`、`.venv/`、用户音频、参考谱和模型不随源码瘦身任务删除。

## 何时可以说项目完成

需分别通过工程与音乐两道门槛：工程没有未解决的高严重度状态/导出/关停问题，公开 profile 在干净 Windows 机器可下载、安装和运行且许可范围清晰；真人音乐材料全部完成并且高影响问题已修复复测或明确接受。任一道仍未完成时，都标注为“公开试用/等待验收”，不因 CI 绿灯或 Release 已发布宣布整体完成。

## 关联文档

- [dev.27 发布说明](releases/0.1.0-alpha.6-dev-27.md)
- [dev.26 发布说明](releases/0.1.0-alpha.6-dev-26.md)
- [全项目验收计划](full-project-acceptance-plan-20261119.md)
- [全项目验收问题账本](full-project-acceptance-ledger-20261119.md)
- [开发与验证](development.md)
- [脚本索引](../scripts/README.md)
- [文档索引](README.md)
