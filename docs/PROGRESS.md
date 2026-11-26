# ManiaLab 2.0 续接进度

更新：2026-11-26（Asia/Shanghai）。阶段 #02 已由用户验收通过；阶段 #03 自动验收已完成，等待用户 osu! 实打，不进入阶段 #04。

## 当前基线

- 归档标签 `archive/research-202611` 指向瘦身前完整 `origin/main`：`fb213565951178006f488caae3f44b9b1b9efa14`。
- 阶段分支：`refactor/phase0-cleanup`，起点为归档标签提交。
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
- 本阶段通过唯一特性分支快进合入 `main` 并删除临时分支；`git worktree list` 保持仅主目录。未发布 Release/tag，未进入阶段 #04。
