# ManiaLab 2.0 续接进度

更新：2026-11-25（Asia/Shanghai）。当前任务仅执行 `GPT_PLAN.md` 的阶段 #01；完成后停止，不进入阶段 #02。

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

- 本阶段不改算法行为、不新增产品功能、不发布 Release、不打版本 tag。
- 自动化测试仅证明工程行为，不代表真实歌曲采音、声部表达、Timing、6–10K/LN 手感或 osu! 实打验收通过。
- 目标若与活动构建、用户数据或真实音乐验收保护冲突，保留保护项并在完工汇报中明确记录冲突；不进入下一阶段。
