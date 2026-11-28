# ManiaLab 2.0 续接进度

更新：2026-11-28（Asia/Shanghai）。阶段 #01–#06 已验收，`v0.2.0-beta.1` 与热修版 `v0.2.0-beta.2` 已发布；当前仅收尾阶段 #07，不进入阶段 #08。

## 当前基线

- `main` / `origin/main` 阶段基线：`83d5b43`（PR #284，tag `v0.2.0-beta.2`）。当前唯一阶段分支为 `feat/phase3b-6-10k-lora-kit`，唯一 worktree 为 `D:\project\ManiaLab`。
- 阶段 #07 先前工作已从恢复点 `stash@{0}` 应用到 beta.2 基线；stash 保留作恢复副本。实验产物仅在忽略目录 `workspace_exp/lora_6_10k_kit/`。
- 阶段 #03 用户实打冠军：FullMix → v32-mini 7K → Torosu dense 10K；去人声不再作为主推理输入。
- 阶段 #07 停止线按用户最新指示为 C 盘可用 5 GiB；最终复核空间见收尾记录。下载、缓存、临时目录和测试输出全部必须留在 D 盘。

## Workbench Beta 已发布

- `v0.2.0-beta.1` 已作为公开 Workbench 预发布版发布；`v0.2.*` 精确 tag 与标记 `[portable]` 的 PR 均构建/校验 `workbench`，旧 Core/Standard/Extended profile 策略保留。
- beta.2 热修经 PR #284 合并并发布；修复全曲 Timing 上下文、V32 吸附后 AI 同刻碰撞、Whisper 离线配置缓存、安装超时/镜像、任务错误摘要与批量全局故障提前停止。CI 与 tag Workbench 资产校验通过。

## 阶段 #07（Phase 3B）收尾

- 只读扫描本机 `osu!\Songs` 得到 `zhyzx35607` 6–10K 谱面 59 张：自动 `KEEP` 56 张、`REVIEW_INVERSE` 3 张；键数为 6K 0、7K 30、8K 2、9K 2、10K 25。审核文件位于忽略目录 `workspace_exp/lora_6_10k_kit/`，59 行 `ReviewDecision` 均留空，未代替用户作决定。
- 最终 provisional 语料 `workspace_exp/lora_6_10k_kit/dataset_provisional_phase07_deep_tier2/`：740 张谱 / 212 个歌曲泄露组 / 1,184,537,245 bytes；Tier mapset 为 18/15/187，Train/Val 为 180/32 组（约 85/15），音频哈希/歌曲别名/mapset 零泄露，6K–10K 五键数均覆盖两个 split。中文定向搜索上限加深到每查询 50 页。
- Tier 1 有 56 张正常谱及 3 张自动标红谱；未人工判定。加深检索后 Tier 2 仍为 15 个 Sayobot 中文 mapset，LN mapset 数：6K 1/4、7K 9/7、8K 0/2、9K 0/2、10K 0/4；50 页内未找到能补缺的匹配中文 LN mapset，短缺已写入 manifest，不以 Tier 3 冒充。Tier 3 为 187 Ranked/Loved mapset。
- 最新真实 CPU smoke `smoke/run-20261128-123132/` 针对加深搜索后的数据完成冷启动 1 步、`--resume-latest` 到 2 步；V32-mini CPU 加载 LoRA，step 2 固定抽到 10K LN 与 10K Hybrid。两个 `.osz` 均通过 CRC、10K 键数及结构检查；Hybrid 有 3 条窗口内未闭合 LN 警告，LN 样本 0 条。Peak RSS 1,083.08 MiB，训练 step 约 15.14 秒，评测约 241 秒；未运行长时间训练。
- 数据器/训练器专项 `29 passed` / `47 passed`；全量 pytest `2692 passed, 76 skipped, 1 warning`（231.40s），Playwright 最终复跑 `103 passed`（4.0m）。LN-only 键配额、最终 payload 预算、固定 LN+Hybrid 验证抽样、内容哈希推理缓存及 V32 边界修复均有回归测试。
- 最终复核 C 盘可用约 9.99 GiB、D 盘约 73.82 GiB，高于 C 盘 5 GiB 停止线；缓存、临时目录、浏览器副本和 smoke 产物均在 D 盘或忽略目录。阶段 #07 代码与本记录统一进入单一 PR，等待 Windows CI；不发 Release/tag，不进入阶段 #08。

## 阶段 #01 历史基线

- 归档标签 `archive/research-202611` 指向瘦身前完整 `origin/main`：`fb213565951178006f488caae3f44b9b1b9efa14`。
- 阶段 #01 分支 `refactor/phase0-cleanup` 起点为归档标签提交。
- 清理前跟踪文件与文本行：`manialab/` 88 / 14,631；`scripts/` 420 / 66,871；`docs/` 383 / 448,682；`tests/` 266 / 47,250。
- 清理前 59 个 worktree、78 个除 `main` 外的本地分支；C 盘可用 9.23 GiB，D 盘可用 50.78 GiB。

## 历史与本地数据保护

- 瘦身前所有 refs 已备份至 `D:\ManiaLab_UserBackup\ManiaLab-pre-phase0-all-refs.bundle`；`git bundle verify` 通过，包含 191 个 refs。另有 5 份未提交 worktree 差异及清理清单，见 `D:\ManiaLab_UserBackup\`。
- 素材扫描覆盖 ManiaLab 工作树、同名前缀目录及主仓库 `build/`；发现 0 个 `test-data/` 目录、0 个用户音频/谱包，99 个 `.osu` 候选均为模板或合成测试数据，无需复制用户素材。
- 根目录 `GPT_PLAN.md` 和本地参考工具、模型权重、虚拟环境及 `node_modules/` 保留；`GPT_PLAN.md` 与 `workspace_exp/` 已加入 `.gitignore`。

## 阶段 #01 执行记录

- 已将根 `AGENTS.md`、`ROADMAP.md` 按 `GPT_PLAN.md` 附录 A/B 落地。
- 已移除 58 个非主 worktree，worktree 列表收敛为主仓库 1 条；已快进 `main` 到归档提交并新建唯一分支 `refactor/phase0-cleanup`。
- 已移除 `D:\project\ManiaLab-*`、`D:\ManiaLab-*` 中确认属于旧 ManiaLab 的 worktree/发布缓存，清空并重建主仓库 `build/`。C 盘可用空间从 `9,923,108,864` bytes 增至 `11,043,991,552` bytes；D 盘从 `54,529,728,512` bytes 增至 `135,933,820,928` bytes。Codex worktrees 下无内容的 ManiaLab 残留壳已移除，其他项目目录保留。
- 根 `GPT_PLAN.md`、`分析Timing.exe`、`assets/`（含 `deep`、`instruments`）、`.venv/`、`.venv-audio/` 与 `node_modules/` 均保留。
- 素材扫描结果为 0 个 `test-data/` 目录、0 个用户音频/谱包；99 个 `.osu` 候选是模板或合成夹具。备份包、5 份 dirty worktree diff 与素材清单均留在 `D:\ManiaLab_UserBackup\`。
- 同一行数口径的瘦身对比（文件数 / 文本行数）：`manialab/` 88 / 14,631 → 79 / 13,622；`scripts/` 420 / 66,871 → 43 / 11,382；`docs/` 383 / 448,682 → 11 / 350；`tests/` 266 / 47,250 → 161 / 35,060。忽略的 reference 脚本和测试产物未纳入统计。
- `manialab/` 9 个经依赖复核的死研究模块已删除；`scripts/` 删除 377 个研究/审计/旧 smoke 文件；研究脚本专属测试及其他死模块测试已退役，产品/HTTP/导出/浏览器与人工音乐验收入口保留。
- 阶段规格存在不可兼得项：构建/pytest/worker 的必要 Python 闭包为 25 个文件；再加 `scripts/README.md`、`verify_core_media.cjs` 和官方解码器 4 文件，最低 31 个，还未计算必须保留的真实音乐验收工具。当前保留 43 个脚本文件，不通过删除活动入口伪造 `≤25`。
- `.venv\Scripts\python.exe -m compileall -q manialab launcher.py` 通过；Flask 启动冒烟 `GET /api/status` 返回 200。
- `.venv\Scripts\python.exe -m pytest -q` 最终为 `2364 passed, 75 skipped, 1 warning`，耗时 `247.60s`。首次运行发现 provenance 固定输入缺失；协议已从归档恢复，SHA-256 与 worker 常量一致，完整重跑通过。
- `npm run test:ui` 为 `87 passed`；`git diff --check` 通过。
- 单一提交 `bcb4dfe` 已推送，单一 PR #278 已创建；未发布 Release/tag。首轮 Windows run `45607010561` 的 2366 passed/72 skipped 中，只有 Node 控件子进程在 15 秒限时内未退出；同一用例本机 1.61 秒通过，测试超时已放宽至 30 秒并推送到同一 PR，等待新 CI。CI 绿灯后再合并 `main` 并删除临时分支。
- `scripts/≤25` 与活动 worker/build/decoder/验收闭包冲突已在 PR 明确披露；该项不伪报通过。不得发布 Release 或进入阶段 #02。

## 验收边界

- 阶段 #02 使用独立 Rust CLI，复用 `D:\project\Torosu` 固定提交的 MIT 源码；Torosu 工作树保持只读。主仓库保留单一 `feat/phase1-torosu-engine` 分支与一个 worktree。
- 严格桥接保留源物件字段、起止和 LN 类型；Torosu 密度扩展只增加物件。原始 6–10K 满宽谱按 Torosu 原生规则严格转换；无轨音频候选先生成较小源型，必要时扩至目标宽度，超过目标和弦容量时明确失败。局部重写不回退到旧引擎。
- 真实本机歌曲 `Qian Ying A_ - Tan Gu Zhi` 的 7K 谱（1366 物件 / 582 LN）已导出 10K A/B `.osz`：严格 Torosu 约 133 ms；Dense 约 140 ms，1925 物件 / 760 LN。包位于忽略目录 `workspace_exp/ab_torosu/`，内部解析与结构校验通过；仍需用户在 osu! 内实打验收。
- Rust CLI 使用 D 盘 `stable-x86_64-pc-windows-gnu` 与 self-contained linker 编译；CARGO_HOME / RUSTUP_HOME / 临时目录均在 D 盘，未安装 Visual Studio。Windows CI 使用 MSVC runner toolchain 构建、测试并打包 CLI；portable verifier 以干净 PATH 实际运行包内 convert/analyze。
- 本机完整验收：`.venv\Scripts\python.exe -m pytest -q` 为 2410 passed / 75 skipped / 1 warning（241.33 秒）；`npm run test:ui` 为 87 passed；GNU CLI `cargo check`、15 个核心测试 + 7 个 CLI 集成测试、Release 构建均通过。
- 阶段 #02 用户验收已通过；后续只执行阶段 #03，不发布 Release 或版本 tag。

## 阶段 #03（Phase 2A）执行记录

- 单一分支 `feat/phase2a-chimera-v32mini` 基于阶段 #02 `main`；唯一 worktree 为 `D:\project\ManiaLab`。只读复用 `D:\project\Mapperatorinator`；固定 `OliBomby/Mapperatorinator-v32-mini` revision `7807f0dc70cab671be012e1f5ddf945b0b8b7278`，权重与 HF/Torch/TMP 缓存均位于 D 盘。`torosu-cli.exe` 与 `D:\rust-build\...` 二进制 SHA-256 相同，文件被 `.gitignore` 忽略。
- 新增 `manialab/chimera_v32.py`：六声部 HTDemucs 重混、v32-mini CPU 推理、RMS 空气 LN 裁尾、已确认拍格吸附、Torosu 10K 严格验证与四谱 `.osz` 导出；`manialab/separation.py` 增加可选跳过 Beat This 的分轨模式，默认产品路径不变。
- 真实歌曲 `Tan Gu Zhi` 201.6 秒。30 秒 v32-mini FP32 6 线程实测：生成 115.873 秒、进程 132.337 秒、峰值 RSS 904.742 MiB、214 物件；整曲推算 795.131 秒（13分15秒），因此只导出自动选定的高潮 45 秒（114000–159000ms），未跑整曲。
- 线程对照：FP32 2 线程生成 110.887 秒、峰值 899.895 MiB，整曲推算约 12分25秒；CPU BF16 autocast 已中止（6/19 窗口持续慢于 FP32，CPU 不支持原生 BF16）；动态 INT8 67 个线性层量化后因上游 `self.proj_out.weight.dtype` 不兼容失败，未改上游。
- HTDemucs 六声部真实音频 30 秒预检：端到端 25.901 秒、worker 21.63 秒、峰值 RSS 1575.320 MiB、整曲线性估算 174.055 秒。45 秒高潮分轨 worker 29.71 秒。去人声增益为 vocals=0、drums=1.1、piano=1.2、bass/other/guitar=1.0；四个 `.osz` 难度共用同一原混音片段播放音频。
- 四难度包 `workspace_exp/ab_v32/tan_gu_zhi_four_way_final2.osz`：Old BasicPitch+MILP 219/58；v32-mini 原生10K 769/62；7K→Torosu dense FullMix 491/264；7K→Torosu dense NoVocal 523/211（物件/LN）。四谱均为合法10K、零同轨冲突、拍线校验通过、媒体引用齐全、ZIP CRC 通过；四个难度共用同一原混音片段播放音频。
- Torosu 统计：旧版 0 minijack/0 pill/手均衡0.856；原生10K 8/29/0.962；FullMix dense 21/42/0.911；NoVocal dense 6/34/0.909。原生10K一处冲突只重排轨道、不增删物件；v32-mini 一个零时长LN保留起音并降为tap。
- 30秒单模型测速输出与各路耗时/峰值、Torosu键型统计及LN裁切记录在忽略目录 `workspace_exp/ab_v32/`。相关测试 `tests/test_chimera_v32.py`、`tests/test_separation.py`、`tests/test_torosu_bridge.py` 为 46 passed；全量 `.venv\Scripts\python.exe -m pytest -q` 为 2426 passed / 75 skipped / 1 warning（483.28s）。所有实验音频、模型输出和 `.osz` 均未入库。
- 阶段 #03 曾由用户在 osu! 实打验收；后续阶段 #04 与本阶段变更由 PR #280 一并合入 `main`。

## 阶段 #04（Phase 2A+）执行记录

- 唯一特性分支 `feat/phase2-v32-gpu-engine` 从阶段 #03 提交 `582dd33` 起步；PR #280 将阶段 #03 与 #04 同一 PR 合入。合并后 `main` / `origin/main` 为 `f8d2331c77f67be2099044806a9b817994520be7`，临时分支已删除，worktree 仅主目录一条。
- 重构已将 `manialab/chimera_v32.py` 从 1967 行收敛为 34 行兼容入口；音频、拍格/LN/键型处理与谱包实现分别位于 `v32_audio.py`、`v32_charts.py`、`v32_packaging.py`。旧去人声四路 runner、固定盘符默认值和 mini 下载流程已退役。
- 新增可移植 `v32_device.py`、`v32_process.py`、`v32_engine.py`、`v32_runtime.py` 与 `/api/projects/<pid>/v32-generate`、`v32-install`；首启用发行包固定源码 SHA 校验、独立 Python 3.10.20 和 uv 安装 CPU/CUDA PyTorch；NVIDIA 机器选 CUDA wheel、无卡回退 CPU。HF/Torch/Python/TMP 缓存遵循环境变量或应用数据目录。V32/V32-mini 共用固定 revision、自动 BF16/FP16 + 6 GB 安全 batch=1、CPU FP32 回退、LoRA 原生传递、取消及状态 API。路由复用六声部分轨 RMS 和原始全混音输入，生成候选绑定音频 SHA 并复用现有导出。
- 完整版权重仅缓存 `gamemode=3/*` 于 D 盘：`OliBomby/Mapperatorinator-v32` revision `74f22583400d259bf424819e11027c17933efe54`，四个必要文件共 870,880,343 字节，SHA-256 清单验证通过；`v32-mini` 固定 revision `7807f0dc70cab671be012e1f5ddf945b0b8b7278`，四文件共 228,108,558 字节。
- Full 真实歌曲 30 秒 CPU FP32 测速：293.75 秒、峰值 RSS 1,623.96 MiB、379 物件 / 266 LN；整曲 201.6 秒线性估算 1,974 秒（32分54秒），因此未运行整曲。
- 同一 `Tan Gu Zhi` 114000–159000ms 全混音片段输出四难度至 `workspace_exp/ab_v32_full/tan_gu_zhi_v32_full_comparison.osz`：mini 冠军 491/264；Full 7K→dense 668/197；Full 原生 10K LN55 457/316；Full 7K→more-dense 693/196（物件/LN）。四谱结构、媒体引用、统一播放音频与 ZIP CRC 检查通过。
- 首次 Windows CI 发现冻结包优先选择 `CARGO_TARGET_DIR` 中测试二进制的阶段 #03 回归；已修复为显式 CLI 覆盖优先、冻结包内 CLI 次之，并补回归测试。另为发行包增加了首启运行时安装入口。
- 最终 Windows CI：pytest `2476 passed, 72 skipped, 1 warning`；Playwright `90 passed`。本地全量 pytest 曾以 `2473 passed, 75 skipped, 1 warning` 通过；本地 Playwright `90 passed`。
- 4050 首次启用需使用包含 Mapperatorinator 安装器的 `standard` / `extended` Windows 包；磁盘空间有限时，启动前把 `MANIALAB_V32_RUNTIME_ROOT` 与 `MANIALAB_CACHE_ROOT` 指向大容量盘，再点 UI「准备 V32 运行环境」。本机无 CUDA，真实 4050 性能仍待目标机实测。
- 本机没有 CUDA GPU；RTX 4050 的 BF16 / fast decoder / batch=1 由 mock 设备测试覆盖，需在目标机实际确认运行表现。未发布 Release/tag，未进入阶段 #05。

## 阶段 #05（Phase 2B）执行记录

- 本阶段下载、数据构造、训练、pytest 与 Playwright 的 `TMP`/`TEMP`/`TMPDIR`、HF/Torch/pip/uv 缓存均钉在 D 盘。清理 131 个旧 Playwright profile（30,031,035 bytes）及 36 个由 `tests/browser/fixture_server.py` 创建、含合成 WAV/状态/谱面的过期 `manialab-ui-*` 目录（2,021,845,108 bytes）；未碰用户音频或其他程序临时目录。C 盘用户报告起始约6.95 GiB，本轮清理前实测 7,131,627,520 bytes，CI/验证结束后 9,144,037,376 bytes；D 盘 119,617,241,088 bytes。
- 单一分支 `feat/phase2b-10k-lora-kit`，只读复用 `D:\project\Mapperatorinator` 固定 commit `5884d1bcb57494f09fe3d74f3a6b58ca0d2960aa`。
- 新增 Sayobot API/无视频 OSZ 构造器，按 embedded `BeatmapID` 索引谱面，严格复核 Mania/CS/TIMING/LN、完整解码音频，并按 mapset/曲名/音频 SHA 连通组划分。最终忽略数据集 `workspace_exp/lora_10k_kit/dataset/`：34 Ranked/Loved 10K mapset + 2 个精选 7K LN 辅助 mapset，173 charts / 36 leakage groups / 247,569,055 bytes（0.231 GiB）；10K 167（LN≥45% 15、Hybrid 22、Rice 130），辅助 7K LN 6。Train/Val 31/5 组，86.11%/13.89%（整数分组下最接近85/15）；音频 SHA 与歌曲组均零泄露。跨窗口活动 LN 547 个窗口边界，至少一条跨窗状态的谱面 173 张。
- 训练器默认 V32-mini，支持 V32；LoRA `r=16/alpha=32`、micro-batch 1、默认梯度累积16、bf16、gradient checkpointing、可选8-bit AdamW、checkpoint resume。难度与 hold 比例保留真实值；采样权重 LN 4x、Hybrid 2x、Rice 0.25x、辅助 LN 1x。针对底座 tokenizer 的非负时间移位词表，仅注入窗口边界时间0的活动 LN 前缀；Windows worker 会重装该 patch 和随包 FFmpeg。
- 纯 CPU 真实端到端 smoke 使用 Sayobot 音频和 v32-mini 固定 revision `7807f0dc70cab671be012e1f5ddf945b0b8b7278`：冷启动 step1 保存完整模型/optimizer/scheduler/RNG/tokenizer checkpoint，续训 step2 后导出 LoRA，并由 `manialab/v32_engine.py` 实际加载生成合法 10K 谱、调用 Torosu `analyze`。更新后的验证抽样实测选中两个不同歌曲组：10K LN（目标 LN 70.6%，推理 35.6 秒）与 10K Hybrid（目标 LN 29.0%，推理 101.3 秒）；两谱均合法、LoRA 已加载、无 incomplete-LN 警告。两步 smoke 尚未学出 LN（实际均为 0%），只验证链路，不代表训练质量。
- 训练器默认 V32-mini（约 228 MB），支持 V32；LoRA `r=16/alpha=32`、bf16、gradient checkpointing、micro-batch 1、默认梯度累积16、可选8-bit AdamW、断点续训及逐轮 NLL/hold/LN/Torosu 指标。CPU 实测约17.35秒/optimizer step，peak RSS 报告约932 MiB。没有本机 CUDA/RTX 4050，不能给出可信的 4050 实测训练时长；脚本会在目标机首轮按实际 step 与评测耗时生成预计总时长。
- 最终本地全量 `pytest -q`：`2498 passed, 75 skipped, 1 warning`（238.44秒）；Playwright `90 passed`。跨窗LN、泄露组去重抽样、CUDA配置、LoRA加载等专项测试通过；PowerShell解析、Python编译与 `git diff --check` 通过。单一 PR #281 的 Windows `verify` CI run `45876826692` 已通过，代码与进度记录在同一 PR；不发 Release/tag，不进入阶段 #06。

## 阶段 #06（Phase 3A）执行记录

- 清理了 `workspace_exp/lora_10k_kit/` 下 10 个阶段 #05 重复 smoke/audited 副本，保留唯一正式 `dataset/`；其 `manifest.json` SHA-256 为 `685D5AE89BD3A100B6AD17AF03B8A69383D64358E773E00DA44D2D2E27D68830`。更新了 `ROADMAP.md` 的 #06 → #07 → #08 顺序。
- 本阶段唯一分支 `feat/phase3-full-workbench`，唯一工作区为 `D:\project\ManiaLab`。C 盘遗留 pytest 临时目录的 19,386 个文件已逐一校验 SHA-256 后复制到忽略的 `workspace_exp/phase06/c-drive-test-temp/`，再清理 C 盘副本；本阶段 pytest 使用显式 D 盘 `basetemp`。
- 新增统一 `manialab/presets.py` 五预设、6–10K/Torosu/LoRA/V32 参数映射；批量 20 首上限、持久化任务、V32 串行生成、候选恢复 lineage、批次终态恢复下载已实现并有 API 测试。新增 Canvas 下落预览、原音频同步控制与工作台集成。
- V32 独立 Python 3.10.20 + PyTorch 2.9.1 CPU 运行时成功安装到 `workspace_exp/phase06/v32-runtime/`；HF/Torch/uv/TMP 均指向 D 盘。真实安装发现原 SciPy 1.17.0 和 NumPy 2.3.5 锁定版本不兼容 Python 3.10，已改为 SciPy 1.15.3 / NumPy 2.2.6。运行时专项测试 57 passed；便携构建脚本/验证专项此前 71 passed，后续 CI/verifier 增量 93 passed。
- 本机真实歌曲《探故知》30 秒片段来自整曲 `114000–144000ms`，Basic Pitch 分析通过（22,050Hz，音频 SHA-256 `884a1bc04c20a2c5228f37b5b250fb34339adeaa7e3a029c03bc680d57b57f0a`）；复用真实 7-TimingPoint reference，BPM 150 / offset `-110819ms`。CPU `v32-mini` + `hybrid_adaptive` 工作台路由推理 370.078 秒、峰值 RSS 912,179,200 bytes（869.92 MiB），生成合法 10K 560 物件 / 472 LN（84.3%），Torosu dense；30 秒推理本身已超过纯 CPU 3 分钟提示阈值。
- 完整 201.6 秒歌曲工作台推理在 360 秒 smoke 等待上限未完成，未生成候选，不计作成功；未继续盲目挂机。有效短片 A/B 包 `workspace_exp/phase06/real-music/Tan Gu Zhi - Phase06 V32 Mini vs Old Engine.osz` 为 5,211,689 bytes，ZIP CRC、媒体哈希、10K 键数及谱面结构均通过：V32-mini 560/472，旧引擎 138/35（物件/LN）。待用户在 osu! 实打验收音乐质量。
- 最终本机全量 pytest：`2624 passed, 76 skipped, 1 warning`（368.17s）；Playwright `103 passed`（3.6m）。本机 `workbench` 便携 ZIP `build/portable/phase06-workbench-20261127-final1/release/ManiaLab-0.1.0-alpha.6-dev.28-windows-x64-workbench.zip` 为 130,003,207 bytes，SHA-256 `a7d78d88561cdf530b11c804bedbe678f5d160396551efeca3d6fa76f71c6ec4`；解压目录 332,225,831 bytes。`verify_portable.py` 全部通过：EXE 启动/重启、6–10K 生谱导出、FFmpeg 9.0.2、包内 Torosu `convert/analyze`、V32 按需安装器状态与服务关停；未安装 V32 runtime/模型。
- 最终本地测试、模型缓存、便携构建与 smoke 的临时/输出目录均显式置于 D 盘。C 盘阶段起始可用 `7,754,723,328` bytes（约 7.22 GiB）；完成清理后实测 `10,281,119,744` bytes（约 9.58 GiB），净增 `2,526,396,416` bytes（约 2.35 GiB），超过阶段起始基线。
- 阶段 #06 本地 pytest `2624 passed, 76 skipped, 1 warning`，Playwright `103 passed`；真实歌曲 A/B `.osz` 与本机 workbench 便携包 smoke 均通过。Windows CI run `45963032904` 的 workbench 构建、EXE 验证与关停检查通过；代码及本最终进度记录统一包含在 PR #282。未发布 Release/tag，未进入阶段 #07。
