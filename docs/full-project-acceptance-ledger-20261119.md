# ManiaLab 全项目验收问题账本

更新：2026-11-19。此账本区分已复现缺陷、覆盖缺口和外部验收门槛；测试通过不代表音乐质量通过。审计未读取 ignored 音频、谱包、评审/解盲文件、模型权重或研究产物，也未运行模型。

## 当前结论

- 第一轮静态架构与工程路径审查没有发现 P0，也没有发现已确认的 P1/P2 数据破坏问题；阶段 1–3 仍须按计划收口，不能据此宣布整体工程验收完成。
- 后续路径审计确认一个 P2 资源清理缺口：分离 worker 启动抛出 `OSError` 时会遗留预转码 WAV。第七十四阶段以合成故障注入复现并修复；PR #181/main run `44468079196`、`44468606711`通过，Python/API `2720 passed, 89 skipped`、Edge `38 passed`，Core便携构建按条件跳过；阶段3仍待集成审查。
- 第七十五阶段复现了分离算法修订不进入stem缓存和下游activation identity的P2缓存身份缺口。当前未提交分支已修复四/六声部版本传递、worker输出校验、下游identity，并将六声部实现纳入打包源码fingerprint；定向测试17 passed，PR/main CI待补。
- 导入、项目保存、谱面切换、候选编辑、证据快照、试听与导出已有大量合成 API/浏览器覆盖。独立只读子审计对编辑、证据、试听和导出的定向测试为 29 passed。
- 真人音乐验收与非 Core profile 的许可/构建边界仍是最终停止条件，不由 CI 代签。

## 待闭合事项

| ID | 类型 / 级别 | 当前状态与证据 | 关闭条件 |
| --- | --- | --- | --- |
| ENG-01 | 导入身份 / 已修复 | PR #175 已合并：没有有效 mania 谱且包含多个音频时拒绝任意挑选，并清理本次新项目。PR run `44452212429`、main run `44452857369`通过；对应测试在 `tests/test_import_reliability.py`。 | 保持回归覆盖；最终主线回归通过。 |
| ENG-02 | 音轨恢复提示 / P2 / 已闭合 | 后端在缺音频时保留 `audio=None`，不会沿用上一难度音轨；API 行为由 `tests/test_project_routes.py` 覆盖。PR #177 加入持续提示与重新导入指引；PR Windows run `44457739696` 和合并后 main run `44458199401` 均为 Python/API `2712 passed, 89 skipped`、Edge `37 passed`。浏览器用例确认播放器来源被清除且恢复步骤可见。项目内没有单独补配入口；当前恢复方式是将 `.osu` 与对应音频重新导入为新项目。 | 已闭合。若以后增加项目内补配入口，需另行定义音频身份变更和缓存/声部失效契约。 |
| ENG-03 | 文件生命周期 / 阶段 1 | #173 OSZ 临时文件、#174 状态保存临时文件、#175 导入清理均有针对性合成测试和 Windows CI 证据。第七十三阶段复现并修复嵌套谱面 `../song.wav` 被误拒的问题：仅放行最终仍位于 `files` 根内的谱面相对资源路径，ZIP 成员仍由原 `safe_path` 严格校验；PR #180 run `44465032776`与main run `44466336310`均通过，Python/API `2719 passed, 89 skipped`、Edge `38 passed`。 | 继续完成剩余跨路径审查；确认导入、编辑、保存、导出失败时旧项目/候选保持有效且新资源清理，并取得合并版本 CI 证据。 |
| ENG-04 | 任务与关停 / 阶段 3 | PR #172 合入统一 15 秒协作 deadline，Core 便携 EXE、离线、Unicode 路径和关停验证通过。第七十四阶段以合成故障注入复现分离 worker 启动 `OSError` 后临时 WAV 残留，并补上仅在尚无进程对象时清理、已启动进程仍需确认退出的处理；PR #181/main run `44468079196`、`44468606711`通过，main为Python/API `2720 passed, 89 skipped`、Edge `38 passed`，Core便携步骤按触发条件跳过。已知边界：deadline 不是任意卡死 Python/native 调用的硬退出 SLA，线程池非守护线程可能阻止解释器退出。 | 完成剩余取消、事务提交和子进程回收集成审查；把协作式限制留在用户文档，不宣称硬退出保证。 |
| ENG-05 | profile 支持范围 / 发布门槛 | GitHub Windows workflow 默认验证 `core`；它不证明 `standard`/`extended` 的完整包可运行。开发文档与工作流需作为能力矩阵一起维护。 | 明确公开支持的 profile；仅对拟公开支持的配置执行匹配的构建、离线/Unicode/导入/生成/导出/关停验证。 |
| ENG-06 | 权重再分发许可 / 条件性发布阻塞 | Demucs 四/六声部 checkpoint 的独立再分发授权尚未核实。源码 MIT、模型来源或文件哈希均不能代替权重许可。 | 获得可核验授权，或从公开 profile 移除这些权重并准确降级能力；逐个核查历史 Release 的包内内容。 |
| ENG-07 | 分离缓存身份 / P2 / PR待验证 | 旧逻辑仅检查音频、模型、输入处理版本及stem文件存在；算法修订变化时仍可命中旧结果，下游`separation_identity`也不区分算法版本。当前工作分支已为四/六声部增加`algorithm_revision`，将其写入worker清单并纳入缓存校验与activation identity；同时将`scripts/separate_six.py`纳入worker源码fingerprint。定向测试19 passed，独立只读复核未发现阻断；PR/main CI待补。 | PR Windows CI通过，并在合并版本确认四/六声部过期清单触发重算、对应下游身份变化；更新本账本与进度。 |
| MUS-01 | 真人音乐验收 / 外部阻塞 | 84 段钢琴起音盲听、实际 6K–10K/LN 试打、Sweet Waltz 与 Choice 两首独立歌曲复听仍没有人工回收结果。流程见 `docs/human-review-integrity.md` 与 `docs/acceptance-20261118.md`。 | 评分身份/完整性校验后锁定再解盲；提供真实玩家键数与 LN 反馈；两首歌曲记录可定位的试听分歧、处置结论和复测。 |
| COV-01 | 浏览器删除流程 / 已闭合 | `tests/browser/evidence.spec.cjs` 新增合成 Edge 流程，验证画布选择与删除会创建带来源记录的新候选、可切回父候选、重开后删除版仍保留。PR #179 run `44461711637`与合并后 main run `44462358541`均成功；Python/API `2712 passed, 89 skipped`，Edge `38 passed`。本次只补覆盖，未发现产品行为缺陷。 | 已闭合。 |

## 已核对的验证基线

- PR #176 验收计划合并提交：`abf7c05ca8e9b15f26418daf61bc51a15417c889`。main Windows run `44454674214`成功；Python/API `2712 passed, 89 skipped`，Edge UI `36 passed`。本次按触发条件跳过 Core 便携构建步骤。
- PR #177 与文档复验 Windows runs `44457046075`、`44457739696`均通过，Python/API `2712 passed, 89 skipped`，Edge UI `37 passed`；便携构建按触发条件跳过。合并提交 `b69acf7f1db7a3ae3aafde4ca3e744c64c466be9` 的 main run `44458199401`成功，测试计数相同。
- 本地开发验证（合并前）Python/API `2711 passed, 90 skipped`，Edge UI `37 passed`；`compileall`、所有 `static/*.js` 的 `node --check` 与 `git diff --check`通过。当前该变更已由 main CI 重新验证。
- PR #179 合并提交 `ef4e788d59c696b48af241ed40652549923449c6`；PR run `44461711637`与main run `44462358541`均成功，Python/API `2712 passed, 89 skipped`、Edge UI `38 passed`，便携步骤按条件跳过。
- PR #180 合并提交 `6888912147e5deed81545cb2376ef2c5066cb65c`；PR run `44465032776`与main run `44466336310`均成功，Python/API `2719 passed, 89 skipped`、Edge UI `38 passed`，便携步骤按条件跳过。
- PR #181 合并提交 `a9e5844632d3db10b2cba6fea4185f9314420f97`；PR run `44468079196`与main run `44468606711`均成功，main Python/API `2720 passed, 89 skipped`、Edge UI `38 passed`，便携步骤按条件跳过。
- 第七十五阶段尚未提交的缓存身份修复本地定向测试：`tests/test_separation.py tests/test_audio_cache.py tests/test_build_provenance.py`为19 passed；`compileall`、静态JS语法与`git diff --check`通过。未运行全量测试或模型。
- 下一步继续阶段 1–3 的跨层时钟、采音证据、Timing/段落/LN 候选映射及取消/资源生命周期集成审查；不重复模型实验。任何新增缺陷需有最小复现、回归测试、来源和适用范围。
