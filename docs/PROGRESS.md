# 续接进度

更新：2026-11-21，Asia/Shanghai。本文件用于中断或额度耗尽后恢复工作；
先核对实际进程、Git 和产物，不能把这里记录的运行中状态当作已经完成。

## 当前状态（先读）

+ 第一百零五阶段通用子进程取消升级（本地验证通过，PR/CI待验证）：ENG-04复审发现`stop_process()`在`terminate()`抛`OSError`且子进程仍存活时立即报清理失败，不再尝试后续`kill()`；合成故障注入先确认旧代码在该用例失败。现记录terminate信号失败后继续共享deadline内的等待及kill升级；若kill也失败仍报告清理失败并由任务门禁阻止后续计算。原进程门禁夹具同步改为显式拒绝terminate和kill，保留失败边界。分支已rebase到当前`origin/main` `b2c5124`；精确命令`D:\project\ManiaLab\.venv\Scripts\python.exe -m pytest -q --tb=short tests/test_cancellation.py tests/test_task_service.py tests/test_task_cleanup_errors.py tests/test_process_cleanup_gate.py tests/test_worker_process_tree.py tests/test_task_lifecycle_flow.py`为`77 passed in 6.31s`；`D:\project\ManiaLab\.venv\Scripts\python.exe -m compileall -q manialab tests`和`git diff --check`通过。PR尚待创建，未运行模型/研究脚本或读取ignored素材。

+ 第一百零四阶段状态保存提交边界（本地验证通过，PR #221检查中）：复现`os.replace(state.tmp,state.json)`成功后，`Projects.save`仍在`finally`清理已移动的临时路径；注入该清理异常会让任务报告error并回滚其已保存状态引用所指资源。修复为仅在原子替换失败时清理临时文件；storage故障注入与真实`/provided-stems`生命周期回归验证任务完成、状态引用对应的新解码缓存保留、旧缓存不变且worker可复用。精确命令`D:\project\ManiaLab\.venv\Scripts\python.exe -m pytest -q --tb=short tests/test_storage.py tests/test_task_service.py tests/test_task_lifecycle_flow.py`结果`28 passed in 4.47s`；`D:\project\ManiaLab\.venv\Scripts\python.exe -m compileall -q manialab tests`及`git diff --check`通过。PR #221已创建，更新head的Windows PR CI待验证；测试仅使用临时目录和合成音频，未运行模型/研究脚本或读取ignored用户素材。

+ `main` 当前HEAD为`063dfabff4c7a5886372731e8da30b5311981a61`：PR #218于`523b4d7`合并；PR #219 head run `44907441797`通过并于2026-11-21 `13:17:38 UTC`合并为`063dfab`。PR #219合并后main run `44908128277`成功：Python/API `3561 passed, 156 skipped, 1 warning`，Chromium `69 passed`；便携/EXE、Core媒体、shutdown和collector按条件跳过。PR #220已rebase到该main，重新验证待完成。公开Release仍为`v0.1.0-alpha.6-dev.19`，tag固定指向`3bd0dfef89c25710c043f120a7290da5adb8dc84`；后续main与PR提交均不包含在dev.19中，文档提交不移动release tag。

+ dev.19已发布（2026-11-21）：PR #211合并修复声部缓存格式号问题；PR #212于`2026-11-21 09:54:17 UTC`合入`main`，merge commit `3bd0dfef89c25710c043f120a7290da5adb8dc84`，合并后main Windows run `44891768664`成功。annotated tag `v0.1.0-alpha.6-dev.19`精确指向该提交；tag Windows run `44892302625`成功（8分24秒）：pytest `3547 passed, 156 skipped, 1 warning`，合成浏览器 `69 passed`，Core包构建、离线Unicode EXE、媒体切换、关停、collector和工件上传均通过。
  CI Core ZIP为`ManiaLab-0.1.0-alpha.6-dev.19-windows-x64-core.zip`，111,912,214字节，SHA-256 `db16fc81c265e0a02c3da58337a5616f63f2e32710f099d64af73db12f687c28`；独立复核1,225个ZIP条目CRC通过、无重复路径，Core payload、manifest、EXE和固定Basic Pitch身份通过。验证JSON SHA-256为`f048c8099a62cd09152242f008d5b27c586de5329d64beaa55cf27a103fee953`，SHA256SUMS及GitHub三个资产摘要均与本地一致。GitHub Pre-release：[ManiaLab alpha.6 dev.19](https://github.com/zhyzx35607/ManiaLab/releases/tag/v0.1.0-alpha.6-dev.19)。
  本版仅为Core profile；Standard/Extended未验证。CI夹具为合成音频，工程/便携验证不构成真人音乐、Timing或osu!试打验收；Demucs checkpoint未随包分发，单独再分发授权仍未核实。版本准备未纳入主工作区`D:\project\ManiaLab`阶段98/99未提交改动；用户素材、模型和研究产物均保留。发布产物保存在ignored目录`D:\project\ManiaLab\build\portable\verified-dev19-44892302625\`；复核后C:约14.4 GiB、D:约40.1 GiB可用，无需清理。

+ 第一百零一阶段Windows状态文件短暂读取锁（已由PR与main验证）：Windows真实读取句柄保持打开`state.json`时，`os.replace(state.tmp, state.json)`可抛`PermissionError`/`WinError 5`；`Projects.save`现在只在Windows对该异常按10/20/40/80/160 ms有界重试，并保留原子临时文件清理及`pack_state`序列化行为。真实文件句柄回归在首次拒绝后关闭reader并确认保存成功。PR #218 Windows run `44905185931`通过，PR于2026-11-21合并为`523b4d7f9e22cd8f65197e4a9b7d5b71e74be02a`；合并后main run `44906384408`通过：Python/API `3559 passed, 156 skipped`、Chromium `69 passed`，便携/EXE、Core媒体、shutdown与collector按条件跳过。基于main `a7a8c4e`定向pytest `tests/test_storage.py tests/test_project_routes.py tests/test_task_service.py tests/test_task_lifecycle_flow.py`为`36 passed`。本地与CI均不覆盖永久ACL/占用、超过重试窗口的锁或并发写者。未运行模型/研究脚本，未读取ignored用户素材。

+ 第一百零二阶段推荐过滤后的结构乐句证据（PR及main验证通过）：结构化生成将`plan_evidence()`得到的候选与乐句池同时传入自动方案建议和表达执行；表达执行按当前候选池协调乐句support，防止旧保存分析中的ghost support ID进入新variant来源快照。新增真实API合成回归覆盖弱候选/弱乐句不进入推荐生成，以及旧项目JSON导出只保留仍存在的support ID。定向pytest为`59 passed`；`compileall`与`git diff --check`通过。PR #219最终head Windows run `44907441797`通过，合并为`063dfabff4c7a5886372731e8da30b5311981a61`；合并后main run `44908128277`成功（Python/API `3561 passed, 156 skipped, 1 warning`，Chromium `69 passed`）。便携/EXE、Core媒体、shutdown和collector按main工作流条件跳过。验证仅为合成工程回归，不代表真实音频、音乐质量或osu!试打；未运行模型/研究脚本，未读取ignored用户素材。

+ 第一百零三阶段候选编辑日志哈希完整性（PR #220验证中）：候选编辑日志读取时验证事件ID为规范小写SHA-256、节点字段与parent/event形状有效，并重算`SHA-256(parent,event)`以拒绝被篡改的历史；旧版内嵌`edits`仍可读取并在保存时迁移为日志。新增损坏读取拒绝、保存失败不覆盖原状态/不遗留临时文件及旧格式迁移回归。定向命令`D:\project\ManiaLab\.venv\Scripts\python.exe -m pytest -q --tb=short tests/test_candidate_edit_journal.py`在更新后的worktree中为`6 passed`；变基`origin/main` `063dfabff4c7a5886372731e8da30b5311981a61`无冲突，代码head为`c62a0ef4924a80c469cbb1b61946675119996424`。旧基线head run `44906889742`虽通过，但不计入此次变基后验证；当前文档更新将触发新的PR Windows检查。阶段仍待新head CI、合并及必要的合并后main验证。没有运行模型/研究脚本，未读取ignored用户素材。

+ 历史状态更正（2026-11-21 08:26:38 UTC，已由后续状态取代）：PR #209最终head `4fcbac8`及合并后main run `44883194111`、`44884080485`成功，详情见下方阶段99最终状态；当时“当前推进阶段100”的指引已过期。当前继续阶段2–3剩余工程审查：ENG-24乐句来源一致性、ENG-26候选编辑历史与区域删除、ENG-04任务/资源生命周期；三条独立worktree审查正在进行。ENG-05、ENG-06和MUS-01仍开放。

+ 资源边界：阶段99的256帧STFT与4096窗口RMS限制单次输入块（分别最多66,304、90,112个采样），不表示全程或内部库临时量有硬内存上限；times/flux/energy/amplitude/rms/smooth摘要仍按音频时长增长。当前解码音频限制20分钟，float32输入加节奏摘要核心数组估算约117.9 MiB，不含FFT工作区、其它分析状态及事件对象。SciPy/NumPy单次内核和estimate_timing内部运算不能在调用期间协作取消。
+ 快照并发边界：应用导入在新项目/随机上传路径用`xb`创建文件，任务串行；未发现受支持应用流程会原地改写源音频。外部程序在复制期间直接原地改写文件仍可能令副本混合版本；分析SHA与解码均绑定副本中的相同字节，因此这属于外部编辑边界，不是应用内竞态阻断项。未读取ignored素材或运行模型。

- 第九十七阶段CI修正（2026-11-21）：此前run `44839324200`中Python/API、Edge浏览器、Core包构建、离线Unicode EXE、Core媒体探针和worker关停通过，collector因`basic_analysis.engine`分类字段错误拒绝证据。修正以提交`5330b8c`推送；Windows run `44841323526`的API、Edge、Core包、离线Unicode EXE、媒体探针、关停、collector和报告收集全部通过（11分11秒）。PR #207于`2026-11-21 00:17:21 UTC`合并到`main`，merge commit为`1ad69d710d4df5829a984bc4d60ed1f8103569e4`；合并后main run `44842816110`成功，Core便携验证通过。ENG-05、ENG-06与真人音乐验收继续开放。
+ 第九十八阶段来源与候选编辑（本地、PR及合并后main检查已通过）：Basic/deep与Provided导入绑定私有快照来源摘要和解码字节；来源身份缺失/冲突时拒绝局部重写；OSZ引用要求唯一归档成员，audio模式校验SHA、remap校验音频扩展名。Provided缓存事务与清理失败会恢复旧目录或报告备份路径。定向pytest `123 passed`，波形端点Edge流程本地`2 passed`。PR #208 head run `44863244643`通过后于`2026-11-21 04:04:29 UTC`合并，merge SHA `c3c1f407d0f4c2018b69618afa11107a93543af6`；合并后main run `44864272448`成功（pytest `3509 passed, 156 skipped`，Chromium `69 passed`）。`BUILD_PORTABLE=false`，portable/EXE、Core媒体、shutdown和collector步骤跳过，Edge未运行。合成工程结果不构成音乐验收。
+ 第九十九阶段音频分析内存与取消响应（已由PR/main验证）：长输入节奏STFT最多256帧/块，RMS最多4096窗口/块；单块输入分别最多66,304和90,112个采样，不能外推为全程或NumPy/SciPy内部临时量的硬内存上限。取消回调从Basic/deep调用路径传入，并在分块与后处理边界检查；`estimate_timing`的同步调用期间仍无法协作取消。定向pytest `56 passed`；隔离完整pytest `3532 passed, 159 skipped`（363.07秒）。PR #212 Windows run `44890894327`及合并后main run `44891768664`通过，发布tag run `44892302625`为`3547 passed, 156 skipped, 1 warning`。新增一ULP峰阈值跨块回归保留旧归约判定；现有随机float32/float64和click对照不构成任意输入逐位等价保证。全长摘要及其它分析状态仍随时长增长；20分钟float32输入及已知核心节奏数组约117.9 MiB，不含FFT工作区和事件对象。未运行模型/研究脚本，未读取ignored素材；工程验证不代表音乐验收。
- 第九十九阶段最终状态更正：PR #209最终head `4fcbac8`的Windows run `44883194111`成功并合并为`54c094f4fd7b98c98551d4b6d3eba0552dd60396`；合并后main run `44884080485`成功，Core包、离线Unicode EXE、媒体切换、关停及collector均通过。模型下载、钢琴专项、独立音频worker构建按条件跳过。
- 第一百阶段激活缓存损坏恢复与取消发布（ENG-25，PR/main已验证）：整曲、声部和钢琴NPZ loader将CRC损坏视为缓存miss；整曲与声部缓存通过同目录唯一pending文件发布，失败/取消时保留旧canonical并清理暂存文件，deep声部缓存转发任务取消回调。PR #210修复缓存恢复与原子发布，PR Windows run `44885703013`通过；PR #211恢复声部缓存格式4兼容并加入v3安全miss回归，Windows run `44887352125`通过。PR #212 main验证`44891768664`及dev.19 tag验证`44892302625`均成功。本地缓存组`29 passed`，扩展缓存组`59 passed`；序列化压缩期间不能协作取消。未运行真实模型/研究脚本或读取ignored用户素材；缓存恢复不代表音乐质量改善。
- 第九十七阶段中间状态更正（历史，已由 PR #207 最终结果取代）：推送期间曾因 OAuth token 缺少`workflow` scope受阻；随后 Node 路径、Core media 终态 sidecar、collector 和迟到编辑修正均进入最终 head `5330b8c`，Windows run `44841323526`通过并合入`main`。PR #207 的最终 head/main 状态见本节首条；上述阻塞已解除。
- 第九十六阶段状态更正（2026-11-21）：PR #206 最终head `d919c3e0955c42f61c8ead36036ac3f6fe290eed` 的 Windows run `44775376309`成功，随后于2026-11-20 16:46:24 UTC合并到`main`，merge commit为`b4c445fa19cb7feca53ab68b08fc8ff4229c70ec`。合并后的`main` Windows run `44776711898`于2026-11-20 16:52:49 UTC成功；时序/API与合成浏览器检查通过，模型下载、EXE、worker关停和portable collector步骤按main工作流条件跳过，不计入验证。PR本地验证为`tests/test_release_validation.py` 166 passed、`tests/test_release_evidence_identity.py` 184 passed/67 skipped、MUS-01台账33 passed/2 skipped及`git diff --check`通过。阶段97本地结果见下项；ENG-05、ENG-06和真人音乐验收仍开放。
- 第九十七阶段中间实现记录（历史，已由顶部PR #207最终状态取代）：该阶段补充FFmpeg/Worker/Mapper清理gate、portable ZIP闭包、Core媒体CI探针与collector绑定。旧run `44825562350`曾因AST helper缺字段失败；run `44827561030`的Core媒体步骤曾因Node路径处理失败，关停与collector因前置失败跳过。这些失败均已在最终提交`5330b8c`修正；PR #207 run `44841323526`及合并后main run `44842816110`均成功，详见顶部记录。阶段97未运行模型/研究脚本，也未读取ignored用户素材。
- 第九十六阶段PR中间CI记录（历史失败，已由本节上方最终head/main记录取代）：head `de128c9`的Windows run `44773574751`于2026-11-20 16:31:19 UTC失败，唯一失败为`tests/test_release_evidence_identity.py::test_core_rejects_optional_piano_audio_rows`仍匹配旧错误文本`unsupported by Core`；collector已按profile allowlist提前拒绝并返回新的“不适用于当前profile/capability”错误。随后本地更新对应断言并新增Standard合法`bundled_model`身份行正向用例，最终本地定向验证为`tests/test_release_validation.py` 166 passed、`tests/test_release_evidence_identity.py` 184 passed/67 skipped、MUS-01台账33 passed/2 skipped及`git diff --check`通过；后续最终head与合并后main结果见本节上方记录。
- 第九十六阶段后续状态（历史中间状态，已由本节上方最终head/main记录取代；2026-11-21）：PR #206 首个head `35aab870`的Windows run `44763958795`于2026-11-20 15:38:06 UTC成功；独立复审随后发现Standard报告可夹带不适用或未绑定本次run的已知检查行。当时的本地修正按profile/capability限制检查集合，并校验所有普通报告行的本次运行身份；新增三项回归，`tests/test_release_validation.py`为`165 passed`，MUS-01台账为`33 passed, 2 skipped`，`git diff --check`通过。台账明确哈希是用户声明且工具不读取素材核对，单条`accepted`不等于整体真人音乐验收通过。当时修正尚未提交或推送、未经过新head CI，PR仍开放且未合并；后续最终验证和合并状态见本节上方记录。真实音乐验收继续开放。
- 第九十六阶段（历史中间状态，已由本节上方最终head/main记录取代；本地合成验证通过，当时待PR/main验证）：collector为所有必需portable检查校验本次profile、能力、EXE/manifest摘要和run ID，严格检查boolean/integer、shape及Standard钢琴证据字段，并拒绝未知check名；新增报告行只绑定当前运行身份。Standard `--piano-audio`无可选音频行时明确报告`not_run`/`optional_audio_fixture_not_supplied`，Core报告为`not_run`/`profile_has_no_piano_audio_capability`；完整且有效的可选音频行才为`passed`，所有状态仍标记`musical_acceptance: false`。本阶段未提供真实钢琴音频fixture，未运行钢琴音频检查。区域LN API回归使用BPM 120、offset 103 ms：来源端点3600/4350 ms配`audio_shift_ms=22.5`经拍线量化为谱面3603/4353 ms，配`audio_shift_ms=-75`则为3728/4478 ms；同时检查试听事件位置、区域/锁定来源证据、JSON/OSU/OSZ及重开切换一致。MUS-01逐问题CSV锁定CLI现在只接受解析后仍位于仓库忽略`test-data/`下的目标；回执仅证明CSV结构、声明字段和字节摘要，不证明真人听审/试打、身份、盲法或完整覆盖。冻结定向验证：发布证据组`362 passed, 67 skipped`（含portable core 12项），区域/试听/Timing组`78 passed`，台账`33 passed, 2 skipped`；台账的两项symlink逃逸测试因Windows `WinError 1314`缺少创建链接权限而跳过。`.venv\Scripts\python.exe -m compileall -q manialab scripts tests launcher.py`与`git diff --check`通过。均为合成工程检查，未运行模型、EXE或研究脚本，未读取ignored用户素材。ENG-05、ENG-06、MUS-01仍开放。
- 历史验证故障（已修复）：PR #201旧head run `44688448229`的34项失败源于合成manifest缺runtime许可inventory字段；修正夹具后身份测试为`36 passed`。旧head run `44691961334`的12个浏览器失败是共享fixture项目被产品“最近30项”限制挤出；修复只调整合成浏览器server，并新增独立断言保留产品端31项时只返回最新30项。本地聚焦组`254 passed`及regional Edge `5 passed`属于该次较早验证记录。
- 当前状态：PR #204已于2026-11-20合并到`main`，合并提交`d45ddb3f710edd6c0c8fb0557c49334eb61a5a69`；PR head run `44740972271`及main run `44742135848`均成功。两次CI Python/API为`3062 passed, 89 skipped`、Edge UI `66 passed`；main Core构建、离线/Unicode EXE、shutdown与collector成功。Core仍是唯一公开支持profile；standard/extended实包、Demucs权重再分发许可与MUS-01真人音乐验收保持开放。未运行模型/研究或读取用户素材。
- PR #201历史head run `44703167319`验证：Python/API `3020 passed, 89 skipped`，Edge UI `63 passed`，Core PyInstaller构建、离线EXE/Unicode检查、shutdown与collector均成功，便携和关停报告`passed=true`。较早head的`44702861247`及`44703158868`状态不代表最终head；最终head检查已全部通过。
- main失败用例复核：单独运行`playback-state.spec.cjs`为`10 passed`；后续本机整套重放中该目标用例通过，但套件结果为`61 passed, 2 failed`，失败在`regional-audit.spec.cjs:85`和`timing.spec.cjs:36`，与目标用例无关。CI未将失败trace作为artifact上传，尚不能确认根因；attempt 2成功证明本次main检查通过，不证明该时序风险已被源码修复。
- 第九十三阶段（吉他缓存）：缓存NPZ或current marker发布失败/取消时，只清理本次独占创建的文件；已完成marker替换后即使异步异常到达，也保留新缓存。`.venv\Scripts\python.exe -m pytest -q --tb=short tests/test_guitar_runtime_review.py`为`32 passed`。PR #202 head run `44713508506`通过后于2026-11-20 09:53 UTC合并，merge commit `6e4e2d7674a5b4e9fe1e8bf4b682a458e5532256`；main run `44714530783`于09:58:04 UTC成功，Python/API `3028 passed, 89 skipped`、Edge `63 passed`。本次main的Core构建、EXE、关停和collector步骤因`BUILD_PORTABLE=false`跳过，不算Core build验证。合成故障注入，未运行模型或读取用户音频。
- 第九十四阶段（计划请求取消与DNS）：替换`asyncio`默认执行器里的系统DNS解析为单个有界daemon resolver；取消可立即释放任务worker，挂起解析完成后同一线程可继续复用，关闭旧event loop后的结果会被丢弃。另修复请求异常恰与取消/超时竞争时HTTP错误覆盖取消状态的路径，并在取消优先返回前回收已完成请求异常。合成回归覆盖线程构造/启动失败、挂起解析、loop关闭、代理TLS配置、worker释放及HTTP 503取消竞态。七个相关模块本地为`122 passed`；新取消竞态本地为`19 passed, 28 deselected`。PR #203最终head `447cbfce`的Windows run `44719481641`成功：Python/API `3035 passed, 89 skipped`、Edge `63 passed`；于2026-11-20 10:34:49 UTC合并为`5fb8e4b63c37d4cfd3ff34093ed266c555334823`。合并后main run `44720438728`成功，Python/API `3035 passed, 89 skipped`、Edge `63 passed`；Core构建、EXE、关停、collector与模型相关步骤均按条件跳过。单次系统DNS若永久挂起，resolver会保持busy直至返回或进程重启；真实HTTPS代理TLS-in-TLS握手也未验证。未运行模型、研究或读取用户素材。
- 第九十五阶段（候选来源、事务与发布验收；本地及PR/main验证通过，阶段整体仍有开放门槛）：Mapper安装事务`tests/test_upstream.py`为`26 passed`，任务生命周期定向组`5 passed`，保存提交后并发HTTP cancel回归`1 passed, 43 deselected`。本机双层TLS测试`.venv\Scripts\python.exe -m pytest -q --tb=short tests/test_server.py -k real_tls_over_tls`为`1 passed, 42 deselected`。候选来源/区域报告定向组`16 passed`，区域报告API`6 passed`，候选删除后来源历史保留、导出仅含当前音符的集成为`3 passed`；Edge evidence/regional-audit定向组`17 passed`。许可打包门禁`11 passed`，许可staging/source snapshot`2 passed`，Core collector定向门禁`3 passed`。本地全量pytest为`3061 passed, 90 skipped`且JUnit provenance校验通过；完整Edge UI`66 passed`，compileall、静态JS语法与差异检查通过。PR #204 head run `44740972271`和合并后main run `44742135848`均通过；CI Python/API均为`3062 passed, 89 skipped`、Edge `66 passed`，main run 的Core build、离线/Unicode EXE、shutdown及collector均成功。合并提交`d45ddb3f710edd6c0c8fb0557c49334eb61a5a69`。standard/extended实包、Demucs权重再分发许可与MUS-01真人音乐验收仍开放；未运行模型/研究或读取用户素材。
- 第九十五阶段当时计划（已由后续工作推进）：完成该阶段提交、PR Windows检查及合并后，优先对ENG-04/阶段3做只读覆盖差距审计，核对任务终态、协作取消、资源清理、子进程回收和共享关停deadline；只对可复现缺口补最小synthetic回归。ENG-05非Core实包、ENG-06权重/数据权利及MUS-01真人听审和试打继续开放，不由工程或CI验证代替。
- 历史状态（阶段98–99，已由本节顶部dev.19发布和PR/main验证记录取代）：当时独立代码审查、定向Python/API/Edge、第二轮全量pytest、compileall和差异检查已完成；阶段98最终由PR #208及main run `44864272448`验证，阶段99由PR #212及main run `44891768664`验证。首轮全量单项句柄计数失败在隔离复验和第二轮全量中通过，根因未确认，历史记录保留。当前剩余工程检查点见阶段2–3账本ENG-24、ENG-26、ENG-04；ENG-05、ENG-06和MUS-01继续开放。
- 第九十二阶段本轮续接：浏览器失败根因为共享fixture项目被产品“最近30项”限制挤出，修复仅作用于合成浏览器server；完整Edge UI `npm run test:ui`为`63 passed`。重装开始改动环境前删除旧`ready.json`，失败/取消保持标记失效、成功原子替换新标记；`tests/test_upstream.py`为`16 passed`。许可staging、ZIP根目录与`_internal`完整许可清单校验及身份夹具同步；复审发现并拒绝`require_bytes=True`时manifest将许可身份置为`null`绕过哈希验证，三种许可清单均有collector集成负例。`.venv/Scripts/python.exe -m pytest -q --tb=short tests/test_release_validation.py tests/test_portable_capability_audit.py tests/test_release_evidence_identity.py tests/test_build_provenance.py`为`186 passed`；另`.venv/Scripts/python.exe -m pytest -q --tb=short tests/test_build_provenance.py tests/test_release_evidence_identity.py tests/test_upstream.py tests/test_browser_fixture_server.py tests/test_storage.py`为`64 passed`。这些均为本地合成验证；未在本机运行EXE或下载/读取权重正文。
- PR #201代码提交`2d485e81aef0a9cd61136d6c6fcd963c48414c81`的Windows run `44701670390`通过：pytest `3020 passed, 89 skipped`，Edge `63 passed`，Core PyInstaller构建、离线EXE/Unicode检查、shutdown和collector均通过，portable/shutdown报告`passed=true`，验证与发布artifact上传成功。Build ID为`44701670390-1`；Core ZIP为111,821,616 bytes，SHA-256 `792885fb416d0d51f3e5d49c0af65cbbb454c42c3eacaec811133f05ed31ce74`；EXE SHA-256 `5963e87579a2b1aa8a16427817f9eddf63e3700a1050bc5dbc8f79a4faecdf01`；manifest SHA-256 `9df36f2804939c4f9e51a36c461b2f77af27f15da62ee47704144ee832441cd4`。本次仅验证Core。
- ENG-06权重证据补充（2026-11-20）：Demucs上游issue #508于2023-06-09提出HTDemucs v4权重商用问题；维护者于2023-08-02答复“Weights are provided for research purpose!”。该答复直接涉及v4模型系列，但不是对本项目具体checkpoint的再分发授权。对20个带ZIP资产的release以HTTP Range检查central directory及嵌入notice：17份ZIP含`hdemucs_high_trained.pt`（334,697,255字节），其中16个已发布版本（`alpha.4`、`alpha.5`、`dev.1`、`dev.2`、`dev.4`、`dev.7`–`dev.13`、`dev.15`–`dev.18`）及draft `dev.3`；其中16份（除`alpha.4`外）还含`htdemucs_6s.th`（54,996,327字节）。`alpha.4`、`alpha.5`、`dev.1`、`dev.2`、draft `dev.3`、`dev.4`的内嵌notice把Hybrid权重标为MIT；除`alpha.4`无六声部权重外，其余五份也把Six-source权重标为MIT。`dev.7`–`dev.13`及`dev.15`–`dev.18`共11份内嵌notice已明确披露checkpoint专属授权未核实。`alpha.1`–`alpha.3` ZIP不含上述checkpoint，`dev.5`、`dev.6`、`dev.14`没有ZIP资产。范围为17份ZIP内单个notice条目及目录的Range读取，未下载ZIP正文。源码声明更正不会改写历史ZIP；历史资产存在不能替代授权证据，ENG-06仍开放。
- 历史声明处置审计（2026-11-20）：5个公开误标版本（`alpha.4`、`alpha.5`、`dev.1`、`dev.2`、`dev.4`）的tag源码仍把Hybrid权重标成MIT，且对应Release正文均无权重许可纠错；除`alpha.4`外还把HTDemucs标成MIT。draft `dev.3`未发布且无可解析Git tag。纠错Release正文不能改动ZIP内容；审计建议授权未核实期间撤下所有16个含权重公开ZIP、保留tag和验证记录，必要时另发无权重版本。尚未对外修改Release/资产；该处置属于外部发布操作，需先形成可审查决定。
- Tsumugi 权重证据（2026-11-20）：draft `dev.3`和已发布`dev.4`的ZIP均含`bass_v2`与`other_v1_5`。Range检查确认ZIP central directory列出的模型条目大小与固定Hugging Face revision的LFS size一致，包内manifest声明的SHA-256与同一revision的LFS SHA-256一致；未读取ZIP权重正文，也未复算ZIP内权重哈希。对应固定revision的repo metadata及README声明MIT；这和`eba1c7d`源码仓库的MIT代码许可证是分开的依据。模型卡没有列出训练集；作者在上游issue #1发布过通用训练集/微调说明，但没有固定数据集版本/manifest与这两个最终checkpoint的映射，不能确认其精确训练沿袭。数据集与录音适用权利仍未核实；当前`THIRD_PARTY_NOTICES.md`已限定为仓库声明并保留上述边界。未下载权重正文或修改历史资产。

- 第九十二阶段续接（2026-11-20）：候选持久化`generation_origin`，连续编辑继承；旧候选父链缺失、重复`created`键或非有限浮点键均保守显示`unknown`，历史裁剪不误标。新增手动逐段表达贯穿整谱生成、区域重写、编辑、重开和JSON导出的集成回归，以及保存已提交但cleanup阻塞时HTTP cancel仍以`done/result`收尾、worker可复用的事务回归。
- 发布证据继续扩充：collector对所有profile将`licenses/runtime/`及`runtime-inventory.json`的逐文件路径/大小/SHA绑定manifest，并验证ZIP根目录与`_internal`两份副本；extended另校验吉他/Tsumugi资源及专项许可。ZIP路径检查拒绝大小写、尾随点/空格规范化冲突、文件/目录冲突、Windows非法字符与保留设备名。README/脚本索引明确源码和Windows workflow当前唯一公开支持profile为`core`。
- 首轮本地聚焦命令`.venv\Scripts\python.exe -m pytest -q --tb=short tests/test_candidate_source_export.py tests/test_task_lifecycle_flow.py tests/test_release_validation.py tests/test_verify_portable_core.py tests/test_process_cleanup_gate.py tests/test_upstream.py tests/test_regional_selection_report.py tests/test_generation_origin_ambiguity.py -k 'not real_model'`为`174 passed`；regional Edge `npm run test:ui -- tests/browser/regional-audit.spec.cjs`为`5 passed`；`compileall`、全部`static/*.js`的`node --check`及staged/unstaged `git diff --check`通过。均为合成输入，未运行模型、EXE或研究脚本，也未读取用户素材。
- PR #201较早run `44677307367`于`2026-11-20 05:47:56 UTC`结束为`cancelled`，pytest step 6未完成且GitHub未提供日志，不能视为通过或失败；后续最终head run `44703167319`通过并随PR合并，详见本节上方。音乐质量、standard/extended真实构建运行、权重许可与真人验收仍开放。

- 2026-11-20当前整合：PR #196/#197/#198均已合并至main，合并提交分别为`256ecf8`、`75179ae`、`bee69ca`；对应PR/main Windows runs `44544668731/44546117834`、`44547867366/44549577592`、`44549463878/44552057267`均成功。PR #198 main回归为Python/API `2857 passed, 89 skipped`、Edge `62 passed`；Core步骤按触发条件跳过，已另按冻结main执行一次本机Core证据链。Core run `core-evidence-20261119-bee69cad232a-20261119T232220475-32408`的全量pytest为`2856 passed, 90 skipped`、Edge `62 passed`；Core build、portable EXE、Basic Pitch合成分析、6K–10K导出、Unicode重启/坏音频恢复、worker-free shutdown及collector都通过。原PowerShell 5.1 runner在末尾把portable根数组保留为嵌套数组，错误报EXE身份不匹配；复核确认实际EXE、manifest、portable、shutdown摘要一致后，用同一报告/产物修正数组解包并完成所有最终断言，没有重建或重跑EXE。PR #199已合并为`e4415e3`且main run `44588267476`成功；COV-04随PR #200合并为`25fb40a`，PR Windows run `44588614212`与main Windows run `44646025472`均成功，本地COV-04定向组`54 passed`。阶段92增加真实regional generation来源回环、保存失败回滚与重试，以及发布专项证据身份绑定；完整聚焦组`122 passed`，另通过`compileall`和`git diff --check`。这些是合成工程验证，不代表真人音乐验收、non-Core构建或权重许可完成；阶段92 PR/main检查待运行。完整Core记录见第九十一阶段。
- extended collector按capability强制吉他与Tsumugi专项证据，并将报告行绑定到本次EXE、manifest和`verification_run_id`；没有该profile capability时会拒绝夹带专项检查行。负例门禁不表示extended已构建或运行。Windows workflow仍固定验证`core`，没有准备hosted specialist fixtures或向完整`verify_portable.py`传入`--guitar-probe`和`--instrument-fixtures`；ENG-05保持开放。

- 2026-11-19历史续接记录：第八十一阶段候选声轨试听来源修复由PR #188合并为0e1b7d687fbf503339c0b7175fca57ad9324d5d8；PR/main Windows runs 44525372079、44526302353均成功，Python/API 2751 passed, 89 skipped、浏览器44 passed。ENG-12已闭合；阶段88–91的后续工程验证见本文件末尾。整体音乐验收仍未完成：真人钢琴盲听、6–10K/LN试打、两首独立歌曲听审及非Core发布profile/权重许可保持开放。
- 2026-11-19历史续接：PR #186 已合并为`06e1d1d34df9e920742bd5f0a33c6bfa97b94cce`；合并后 main run `44500510492`成功，Python/API `2743 passed, 89 skipped`，Edge `38 passed`，Core便携与模型步骤按触发条件跳过。第七十九阶段生命周期跨路径审查未发现产品状态破坏；两处失败路径补测已进入 main，阶段1工程审查收口。第八十阶段的手动参考 Offset/波形对齐 ties-to-even 修复随后由PR #187及main run `44508222088`验证；自动估计保留与服务端默认一致的 half-up 规则。
- PR #183 已于2026-11-19合并为`3427f179e17068f4b4581c70fa37d96583e5816e`，修复Basic Pitch整曲/声部activation缓存及ONNX session未绑定模型身份/推理修订的问题；PR run `44476736699`与合并后main run `44477712665`均通过，Python/API为2,728 passed、89 skipped，Edge UI为38 passed，Core便携步骤按条件跳过。独立只读复审通过，没有加载真实模型或读取用户音频。
- 第七十七阶段复现并修复 `/api/import` 创建项目时子目录失败会留下不可见UUID目录的问题；`Projects.create()` 现在清理本次半创建目录后重抛原异常。PR #184 已合并为`d5d1e374e9d315c9d39662da796e14008ee193c7`；PR Windows run `44479294481`及合并后 main run `44480110018`均成功，Python/API为2,729 passed、89 skipped，Edge UI为38 passed，Core便携步骤按触发条件跳过。定向故障注入和相关回归为37 passed。
- 第七十八阶段修复计划 API 同步 HTTP 请求阻塞唯一任务 worker、取消不能及时终止的问题，并补充 1xx、HTTP/HTTPS 代理和非法端口边界。PR #185 已合并为`25460346bf87e847b68ae1aad00300b22d4590a3`；PR runs `44493364596`、文档复验`44495044806`及合并后main run `44496133010`均通过，Python/API为`2741 passed, 89 skipped`、Edge为`38 passed`，Core便携与模型步骤按触发条件跳过。90秒只约束 asyncio 网络任务，不覆盖被取消的系统 DNS `getaddrinfo` 与 `asyncio.run()` 默认执行器收尾；该限制及真实 HTTPS 代理握手测试缺口已列入 ENG-10。
- 第七十九阶段只为已识别的候选发布/状态保存失败回滚和提供声轨长度不匹配目录清理缺口增加合成集成断言，没有确认产品逻辑缺陷。PR #186 Windows runs `44498461836`、最新 head 文档复验`44499686610`成功，Python/API为`2743 passed, 89 skipped`、Edge为`38 passed`，Core便携与模型步骤按触发条件跳过；已合并为`06e1d1d34df9e920742bd5f0a33c6bfa97b94cce`，main run `44500510492`同样成功。阶段1生命周期跨路径审查收口。
- PR #182 修复分离算法修订未纳入缓存身份的问题，已于2026-11-19合并为`58c48b9966da2b8a798fe5baa32d0fdc932bfaf7`；PR run `44471599763`与合并后main run `44472118121`均通过，Python/API为2,725 passed、89 skipped，Edge UI为38 passed，Core便携构建按条件跳过。第七十五阶段还将`scripts/separate_six.py`纳入worker源码fingerprint。PR #181 修复分离worker启动失败后的临时WAV泄漏，合并为`a9e5844632d3db10b2cba6fea4185f9314420f97`；PR/main run `44468079196`、`44468606711`成功，Python/API为2,720 passed、89 skipped，Edge UI为38 passed。此前PR #180修复嵌套谱面父目录音频引用，合并为`6888912147e5deed81545cb2376ef2c5066cb65c`；PR/main run `44465032776`、`44466336310`的Python/API为2,719 passed、89 skipped，Edge UI为38 passed。
- PR #176 全项目验收计划已于2026-11-19合并为`abf7c05ca8e9b15f26418daf61bc51a15417c889`；main Windows run `44454674214`成功，Python/API为2,712 passed、89 skipped，Edge UI为36 passed，Core便携构建按触发条件跳过。第一轮工程路径审计未发现P0，阶段1–3仍在收口；问题与证据入口见[全项目验收问题账本](full-project-acceptance-ledger-20261119.md)。
- 第七十一阶段增加缺失 `AudioFilename` 的可见状态和重新导入说明；浏览器回归确保不借用上一难度音轨。本地全量Python/API为2,711 passed、90 skipped，Edge UI为37 passed；`compileall`、全部静态JS语法及`git diff --check`通过。PR #177 的 Windows runs `44457046075`、文档复验`44457739696`及合并后 main run `44458199401`均通过，Python/API为2,712 passed、89 skipped，Edge UI为37 passed；便携构建按触发条件跳过。PR 已合并为`b69acf7f1db7a3ae3aafde4ca3e744c64c466be9`。
- PR #178 将第七十一阶段结果和验收账本同步到 main，PR Windows run `44458933001`与合并后 main run `44460183632`均成功；合并提交`0be4b24dea71aee4f604ebc6be7dd922d5e6090b`。第七十二阶段补上浏览器删除候选、切回父版本与重开持久化回归；PR #179 Windows run `44461711637`及合并后 main run `44462358541`均成功，Python/API为2,712 passed、89 skipped，Edge UI为38 passed；便携构建按触发条件跳过。PR #179 合并提交为`ef4e788d59c696b48af241ed40652549923449c6`，COV-01已闭合。阶段1–3跨路径审查、profile/许可处置与真人音乐验收仍未完成。
- 2026-11-19最新续接核验：第六十九阶段 PR #174 已合并为`d5d1cc70201bbb6556a849587cc52d7261e82b37`，PR Windows run `44448768663`与main run `44449536131`成功，Python/API为2,709 passed、89 skipped，Edge UI为36 passed，Core便携EXE/离线/Unicode/关停验证通过。第七十阶段 PR #175 已合并为`32564c4ae08382dc29e802cd8704bb0689d1febf`；PR Windows run `44452212429`成功，main run `44452857369`成功，Python/API为2,712 passed、89 skipped，Edge UI为36 passed；两次按触发条件跳过便携构建。新增[全项目验收计划](full-project-acceptance-plan-20261119.md)覆盖端到端工程、配置与许可、真人音乐验收和停止条件。公开Pre-release仍为dev.18；真人钢琴盲听、6K–10K/LN玩家试打及两首独立歌曲听审仍未取得人工结果，音乐质量目标未结项。
- 最新状态（2026-11-19）：PR #164 任务取消与进程关停已合并为`6d779209561b6b767b5f6886d75e9713bf371227`；PR Windows run `44403326416`及合并后 main run `44405337764`成功，Python/API均为2,682 passed、89 skipped，Edge UI均为36 passed；便携构建按触发条件跳过。PR #165 补录验证结果并合并为`265414e7f59d5875079b86e040ae4cfd6a457a8b`，PR run `44406379629`通过；合并后 main run `44407019369`被后续 main 工作流取消，不记作成功。PR #166 结构分析分块 STFT 已合并为`e684551abd5071d75f33febefe76e0bd7b182135`；PR runs `44410686981`和进度同步后 `44411505246`均通过，合并后 main run `44412123852`成功，Python/API为2,683 passed、89 skipped，Edge UI为36 passed，便携构建按触发条件跳过。PR #167 音频身份哈希流式化已合并为`76c3dc00b6466682f4bd3bbbacd5e50535b3a370`；PR runs `44415178836`和进度同步后`44416022976`均通过；合并后 main run `44417472152`成功，Python/API为2,686 passed、89 skipped，Edge UI为36 passed。第六十四阶段本地全量回归为2,686 passed、90 skipped，Edge UI为36 passed，compileall、JS语法和差异检查通过；PR #168 的首轮 `44418933013`、文档复验 `44420164907`和 EOCD 字节测试复验 `44420946296`均通过，最后一轮 Python/API为2,687 passed、89 skipped，Edge UI为36 passed。当前公开 Pre-release 仍为 dev.18。真人钢琴盲听、6K–10K/LN试打与两首独立歌曲听审仍缺人工证据，音乐质量目标未结项。
- PR #168 已合并为`87f168cca73d2937a879aa38178655769606b02f`；合并后 main run `44422409442`通过，Python/API为2,687 passed、89 skipped，Edge UI为36 passed。PR #169 迁移`actions/setup-node`至 Node 24 action runtime 的 Windows runs `44424536788`和进度同步后`44426309442`均通过，Python/API为2,687 passed、89 skipped，Edge UI为36 passed；Node 20 弃用告警已消失。PR #169 已合并为`4c9fdae8fb9a1f9eb36b0c0df1e34419042028f`，合并后 main run `44427013041`成功，测试计数相同。PR #170 的 Windows runs `44430965427`、`44431613071`和最终文档复验`44432358696`均通过，Python/API为2,691 passed、89 skipped，Edge UI为36 passed。PR #170 已合并为`3cf043aa7c027a905aaa1aa88ca551e84c9557d5`；合并后 main run `44432819183`成功，Python/API为2,691 passed、89 skipped，Edge UI为36 passed；Core EXE步骤按触发条件跳过。

- 2026-11-19 当前续接：从 PR #157 合并提交 `19132940` 开始，公开 Release 仍为 dev.18，近889MB standard 构建仅为 Actions artifact。本阶段经 PR #158 合并为 `6200891e`；PR Windows run `44366735851`及合并后 main run `44368922837`的 Core 构建、实际 EXE、shutdown、Python/API 与 Edge UI 检查均通过。没有创建新 Release。人工钢琴盲听、同音符 6K–10K 定轨试打、两首独立歌曲听审仍为空，长期音乐质量目标保持未结项。
- 2026-11-18 续接核验：PR #151 已合并为 `fc20c92`，Windows PR run `44320958903` 和 main run `44322594836` 均成功。第 49 阶段修复编辑后导出的段落书签，复现与验证见文末。音乐验收还需要玩家盲听、试打及独立歌曲证据。
- 已发布的 Windows 包为 `v0.1.0-alpha.6-dev.18`；后续主线修复不自动进入该已发布包。
- 2026-11-17 工程整理、跨段落容量、Timing诊断、LN来源、草稿保护、局部报告与乐句边界已分九阶段合并（PR #101–#109）；检查与结果见 [project-review-20261117.md](project-review-20261117.md)，分阶段记录在本文件末尾。
- PR #109已合并到`main`，九阶段代码在2026-11-17完成完整Windows便携验证；run `44057633497`通过，源码树与合并提交`3dc0705`一致。报告保存在ignored `build/structure-20261117-validation/ci-44057633497/`，没有创建新Release。
- 已有66例[冻结片段对照](frozen-generation-regression-results.md)及10例[完整歌曲来源对照](ln-evidence-review-results.md)。第十阶段完成整合便携验证，另完成声部取舍静态复核与协议哈希更正；没有重复已完成的模型实验。
- PR #110已合并为`035816c4d048e5743ddf6807b7c4650104c2ba60`，包含集成便携验证记录与冻结输入哈希更正。
- PR #111已于2026-11-17合并到`main`，合并提交`cca4a21f831c3aa5b2cad989b6c287b7dbe50f01`。普通功能分支push run `44064879215`为成功跳过，PR run `44064888429`通过，合并后的main run `44065394291`通过；现有分支已从该main提交续接。
- PR #112修复了未知任务查询/取消的API状态码，已合并为`c386eac8c423ee47707c63e24743e78541a40492`；PR run `44068195710`与合并后main run `44068922008`成功，普通push run `44068133938`为成功跳过。
- PR #113修复最近项目快速切换时的旧响应覆盖，已合并为`dd83a1b0ffd0e6c9f4bca141c99ef9899bc1d6c9`；PR run `44070209976`与合并后main run `44070802839`成功，普通push run `44070113643`为成功跳过。
- PR #114将播放、独立声部读取、试听与波形HTTP路由从`server.py`提取到`playback_routes.py`，已合并为`7b9c7ef23e7aff45f5f5bb099d97415b6d6f8fd9`。PR run `44071878909`成功、重复push run `44071832433`跳过，合并后main run `44073069913`成功。
- PR #115 将任务状态、串行执行、取消与项目提交边界提取到`task_service.py`，已合并为`cfddeeb`；PR run `44075974260`与合并后main run `44076507773`成功，普通push run成功跳过。
- PR #116 将导入、项目读取、谱面切换和候选切换路由提取到`project_routes.py`，已合并为`d7a6f7db1cc4c515acfcd0985a53c93c8c356064`；PR run `44079580314`与合并后main run `44080283011`的Python/API和合成浏览器验证均成功，功能分支push run `44079484962`成功跳过。
- PR #117 将 JSON、OSU 与 OSZ 导出处理提取到`export_routes.py`，已合并为`374caf536c269632df2967de641ae792d2fba171`；PR run `44082425764`与合并后main run `44082995836`的Python/API和合成浏览器验证均成功，功能分支push run `44082374845`成功跳过。
- PR #118 将 Timing 检查与Offset重拟合路由提取到`timing_routes.py`，已合并为`fbd47e6f5d5c7009bda4dd921fc34a7009adeb43`；PR run `44085269448`与合并后main run `44085905987`的Python/API和合成浏览器验证均成功，功能分支push run `44085216977`成功跳过。
- PR #119–#123 的路由边界整理均已合入`main`；PR #123 run `44098248603`与合并后 main run `44098766111`通过，普通 push run `44098175931`成功跳过。当前基线为`b80dad8`，本次求解器结构阶段从该提交续接。
- PR #132–#136 已依次合并到`main`；当前结构整理基线为`3f3879f`。PR #133 run `44128674193`、进度更新后 run `44129552127`及 main run `44130292999`均通过；PR #134 run `44133239396`、进度更新后 run `44134458472`及 main run `44135099313`均通过。阶段35完整 Python/API 为2,527 passed、89 skipped，Edge UI为31 passed。阶段36 PR run `44137877217`、main run `44138712410`均通过：Python/API为2,531 passed、89 skipped，Edge UI为31 passed。阶段37最终 PR run `44217665635`与 main run `44218152214`均通过：Python/API为2,533 passed、89 skipped，Edge UI为31 passed。
- PR #137 已合并为`b483895eefafe2f5c432a4c73b73e7e274f847b1`。PR Windows run `44225629152`通过，合并后 main run `44226100575`成功；详见本文件末尾。
- PR #142 Windows run `44248668628`通过并已合入`main`；当前基线为`f35b22f6c50ddfe9a1aa0926ab2121087dce6451`。精确 BPM 候选的重算和 UI/API 往返验证在下一阶段记录。
- 开发导航见 [architecture.md](architecture.md)，旧版与研究摘要见 [research-history.md](research-history.md)。下方历次“当前”表述属于当时记录。
- 真实音乐质量、Timing、声部主次与 LN 音乐性仍未验收；保留历史结果，不重复已完成模型实验。

## 项目整理与开发测试版发布记录

- 2026-11-12 从 `origin/main` 的 `e17fc1b` 开始整理，导航索引已通过 PR #48
  合并到 `main`（`ba17760416dabace44a693653624ba69ade809e9`）。没有搬迁
  `docs/` 或 `scripts/` 文件，因为研究记录、测试导入、历史命令和协议哈希依赖
  现有路径；忽略的 `test-data/`、模型、音频、便携包和历史构建产物均保留。
- dev.5 的 `/api/import` 半上传清理和 Windows 唯一构建 ID 隔离已合并并验证；
  原 tag `v0.1.0-alpha.6-dev.5` 指向上述 main 提交，不改写该 tag。
- dev.5 tag CI run `43189215747` 的测试与 tag/source 版本检查通过，但在
  `Fetch verified audio models` 因 `dl.fbaipublicfiles.com` TLS 证书过期失败，
  Windows EXE、离线便携检查、关停验证和发布验证报告均未运行。GitHub Release
  尚未创建，因此不能称 dev.5 已发布。
- 本机 `assets/deep/htdemucs_6s.th` SHA-256 与清单的
  `34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd` 一致。
  本轮 dev.6 修复只在原源下载失败时尝试固定 revision 镜像，并对每个来源继续
  校验字节数与 SHA-256；不关闭 TLS。本轮未运行本地下载/构建流程，未修改本机
  模型、音频或研究产物，且不改变应用行为。
- 完整检查发现 `THIRD_PARTY_NOTICES.md` 先前把 Demucs 代码 MIT 延伸到了预训练
  权重；现已改为明确区分软件与 checkpoint。上游维护者说明预训练权重不由代码
  MIT 覆盖、限科学用途；本项目没有核实两个打包 checkpoint 的单独再分发授权。
  固定镜像 SHA 只证明字节一致，不证明授权；该权利状态是本版发布的已知风险。
- dev.6 正式 tag CI 前的本地验证：`tests/test_fetch_audio_models.py` 为4 passed；全量
  `.\.venv\Scripts\python.exe -m pytest -q` 为2272 passed、89 skipped；
  `compileall -q manialab scripts tests`、全部 `static/*.js` 的 `node --check`、
  `assets/deep-models.json` JSON 解析与 `git diff --check`通过。这组本地结果不覆盖
  Windows 便携 EXE、在线模型镜像下载、离线运行和关停验证。
- PR #49 已合并到 `main`（`300b22deed77fabf6277afa5845d6fdf09684fae`），
  tag `v0.1.0-alpha.6-dev.6` 指向该提交。tag CI `43194516944` 的测试、模型下载、
  音频 worker 构建均通过，但主程序打包在 PyInstaller 的 `--add-data static;static`
  上失败：生成的 spec 位于独立临时目录，`static` 被错误解析为 spec 旁边的路径。
  因此 EXE、完整 ZIP、离线运行、关停检查和发布验证报告没有生成，dev.6 没有 Release。
- PR #50 已合并到 `main`（`9f43081cdb3fef80ac6838fea8acfa5d50135d28`），
  将 PyInstaller 数据源与 `launcher.py` 改为仓库绝对路径，并把应用版本和发布文档
  同步到 dev.7。README 下载步骤改为从 Releases 选最新已发布 ZIP，避免把 Tags 页面
  自动生成的源码归档误当 Windows 安装包。保留 `v0.1.0-alpha.6-dev.6` 原 tag，
  不移动或重写。
- Windows `workflow_dispatch` run `43196428192` 对代码提交 `7603c3f` 完整通过，
  验证了路径修复的模型下载、worker/主程序构建、ZIP、离线 EXE、关停与报告生成；
  该运行的应用版本还是 dev.6，仅作为修复验证，不作为 dev.7 发布文件。最终 dev.7
  tag CI `43199094591` 对合并提交完整通过，耗时8分04秒：tag/source 版本检查通过，
  全量测试 `2272 passed, 89 skipped`，模型下载、worker 和 Windows ZIP 构建、clean
  PATH 与 Unicode 路径下离线 EXE 检查、worker 关停、验证报告及工件上传均成功。
  验证报告记录2361项测试（89 skipped、零 failures/errors）、4213个 ZIP 条目 CRC
  通过；报告明确标注没有音乐或 Timing 质量验收。
- 从 tag CI 下载并复核的产物留在忽略目录
  `build/portable/verified-dev7-43199094591/ManiaLab-windows-x64/`。Windows ZIP 大小
  698,916,155 字节，SHA-256 为
  `2f91d87c30661977e9eef9ef2e377caa8ef9af936b08995347109422811082f1`；校验文件及验证
  JSON 的 SHA 均匹配，独立 ZIP `testzip()` 通过。GitHub Pre-release 已发布：
  [ManiaLab alpha.6 dev.7](https://github.com/zhyzx35607/ManiaLab/releases/tag/v0.1.0-alpha.6-dev.7)，
  附有 Windows ZIP、SHA256SUMS 和机器可读验证报告。Demucs checkpoint 单独再分发
  授权仍未核实，见 `THIRD_PARTY_NOTICES.md`。
- 现有长期目标仍是用 Nova 人声音轨逐段核验连续人声 LN 候选，再由真实音频听检和
  osu! 试打验收段落、声部与 LN 表达；研究权重保持关闭，不能重复已完成的模型实验
  或把 CI 通过当作音乐质量完成。
- dev.7 tag 固定在 `ad6a398`，而 PR #52 合并后 `main` 前移到 `0b5ddfe`；差异仅为
  复核进度、协议、研究脚本和测试。PR #53 将源码版本升至 dev.8 并合入 `main`，
  合并提交 `1fb9ef964f04919e02a12559f6b8a598db7fe602`。annotated tag
  `v0.1.0-alpha.6-dev.8` 精确指向该提交；打 tag 时 `HEAD` 与 `origin/main` SHA 相同。
  本次只同步版本元数据和发布说明，没有应用运行逻辑、模型或用户数据变化。PR 的两项
  Windows validation 均通过；本地全量测试2275 passed、89 skipped。
- dev.8 tag CI run `43209073390` 于2026-11-12完成，全部步骤成功，耗时6分46秒：
  tag/source 版本一致、2364项测试（89 skipped、零 failures/errors）、模型校验下载、
  音频 worker 和 Windows 包构建、clean PATH 与 Unicode 路径离线 EXE 检查、worker
  关停、验证报告和工件上传均通过。报告记录4215个 ZIP 条目 CRC 检查通过，并明确
  标注没有音乐或 Timing 验收。
- CI 产物下载到 ignored 目录
  `build/portable/verified-dev8-43209073390/` 并独立复核：ZIP 大小698,920,560字节，
  SHA-256 `0c7e7844cdb77f81cef8ed58e19b54bdb5d0b584ea4b7c478f495113b6f8baa3`；
  本地完整 CRC 检查和4,215条目重复名检查通过，包内版本和 release manifest 与报告相符，
  ZIP 与验证 JSON 的 `SHA256SUMS` 2/2 匹配。GitHub Release 返回的 ZIP digest、
  文件大小及两个小型资产 digest 均与本地结果相符；验证 JSON SHA-256 为
  `8297467152e3b10fa5701573f2378d899235758f59a712c50d3d0872468f9ca3`。Release 已发布：
  [ManiaLab alpha.6 dev.8](https://github.com/zhyzx35607/ManiaLab/releases/tag/v0.1.0-alpha.6-dev.8)，
  附 Windows ZIP、SHA256SUMS 和验证报告。音乐质量、Timing 和 Nova 人声候选人工听审
  仍未完成；Demucs checkpoint 单独再分发授权仍未核实。
- PR #55 将人工人声标签校验工具合入 `main`，merge commit 为
  `78335ad0e773e502903b2d9236290916e799323e`，两项 Windows `verify` 均通过。
  这使 dev.8 tag (`1fb9ef9`) 不再等于当前主线；因此只同步版本元数据和发布说明准备
  dev.9，应用运行行为不变。
- dev.9 版本准备仅变更 `manialab.__version__`、README 当前版本/下载入口和发布说明。
  本地全量 `.\.venv\Scripts\python.exe -m pytest -q` 为
  `2282 passed, 89 skipped`，`py_compile`、`git diff --check` 通过。tag
  `v0.1.0-alpha.6-dev.9` 精确指向当时的 `main` 提交
  `8857ba9c2bc9c46b34dd14e7224dfc545ef39ecf`。
- dev.9 Windows tag CI run `43225453896` 于2026-11-12成功，耗时7分20秒：
  tag/source版本检查、全量测试、模型校验下载、worker和Windows ZIP构建、clean
  PATH与Unicode路径离线EXE、worker关停及报告/工件上传全部通过。JUnit记录
  2,371项测试、89项跳过、零failures/errors；报告记录4,216个ZIP条目CRC通过。
- CI工件下载并独立复核后保留在ignored目录
  `build/portable/verified-dev9-43225453896/`。ZIP大小698,921,115字节，
  SHA-256为`1180e5242033dfccf57d10e8cb11bdd5ef74025383127b2f069e51730fd55b8e`；
  本地完整CRC检查和无重复条目检查通过，包内版本、EXE和manifest哈希相符，
  ZIP与验证JSON的`SHA256SUMS`匹配。GitHub上传资产的大小和digest均与本地结果相同。
- GitHub Pre-release已发布：
  [ManiaLab alpha.6 dev.9](https://github.com/zhyzx35607/ManiaLab/releases/tag/v0.1.0-alpha.6-dev.9)，
  附Windows ZIP、SHA256SUMS和机器可读验证报告。验证和发布资产不构成音乐验收；
  音乐质量、Timing、Nova人声候选人工听审与osu!试打仍未完成，checkpoint单独再分发
  授权仍未核实。
- 2026-11-13 准备并发布dev.10。版本变更仅同步`manialab.__version__`、README下载入口
  和发布范围说明，使tag对齐已合并PR #63/#64后的`main`；不改变应用运行逻辑、默认行为
  或模型。训练集听审包是ignored本地产物，不进入Release。本地全量`pytest`为2295
  passed、89 skipped，`compileall`和`git diff --check`通过。PR #65两项Windows verify
  通过并合入；tag `v0.1.0-alpha.6-dev.10` 精确指向`0b3ac2fc91b7b9aa9b4e5a3ff22b540164b0c264`。
  Windows tag CI run `43256384273` 于2026-11-13完成：2,295 passed、89 skipped、零失败/错误；
  模型校验、worker/Windows ZIP构建、clean PATH与Unicode路径离线EXE、worker关停及报告上传
  全部通过。CI包独立复核：ZIP 698,930,796字节，SHA-256
  `affeeb90518477168bcc38322c3889c39770523e9ad2fa3c7b7e1b9cdc2474f6`；4,219个条目CRC
  全过、无重复路径，GitHub资产digest与本地匹配。GitHub Pre-release已发布：
  [ManiaLab alpha.6 dev.10](https://github.com/zhyzx35607/ManiaLab/releases/tag/v0.1.0-alpha.6-dev.10)，
  附Windows ZIP、SHA256SUMS和验证JSON。报告未声称音乐/Timing验收；本地39组听审仍待人工完成。
- 2026-11-13 准备并发布dev.11，使新tag/Windows便携包源码快照精确对齐包含PR
  #67/#68/#69及版本准备PR #70的`main`。应用版本/README/发布范围仅同步版本；
  stem-review标签校验器是研究辅助CLI，不改变应用运行、默认采音或模型。本地全量
  `pytest`为2310 passed、89 skipped；`compileall`、11个静态JS `node --check`及
  `git diff --check`通过。PR #70两项Windows verify通过后合入，merge commit为
  `635dd382758916902df4d876bdc4bbb44e46d97d`；annotated tag
  `v0.1.0-alpha.6-dev.11`精确指向该提交。Windows tag CI run `43271869370`于
  2026-11-13完成、耗时8m10s：2,399项测试中2,310 passed、89 skipped、零失败/错误；
  模型校验下载、worker/Windows归档构建、clean PATH与Unicode数据路径离线EXE、worker
  关停、验证报告和工件上传全部成功。ZIP 698,933,352字节，SHA-256
  `4cabc4a8b7879afd83f4fc688dad29848dfb8d7f7e1bcb82ba82ae26401a6e06`；4,220个条目
  CI与本地独立CRC检查通过、无重复路径，包内版本/manifest/EXE哈希匹配；验证JSON
  SHA-256为`53e266794dfd3f9a7e9b6dd3ef1e1d155c1717b46d6cf11f731b52f9ffa1fddd`。
  GitHub Pre-release已发布，ZIP、SHA256SUMS和验证JSON资产digest/大小与本地一致。
  CI使用合成音频只验证软件/便携运行，不代表音乐或Timing验收；本地39组人工听审仍待完成。
- 2026-11-13 发布后复核发现GitHub Release正文把2,399项测试总数误写为
  `2,399 tests passed, 89 skipped`；已更正为`2,310 passed, 89 skipped (2,399 total)`。
  使用`gh release view`及带no-cache参数的GitHub API/Release页面复核，更新后的正文可见；
  tag仍指向`635dd382758916902df4d876bdc4bbb44e46d97d`且仍为Pre-release。3个上传资产
  名称、大小和digest均未变化：Windows ZIP 698,933,352字节、
  `4cabc4a8b7879afd83f4fc688dad29848dfb8d7f7e1bcb82ba82ae26401a6e06`；
  SHA256SUMS 226字节、`6b7e4dc10b53104ef1edf6a744bba7c9077eed86876220fef208ff0a2849f482`；
  validation JSON 1,992字节、
  `53e266794dfd3f9a7e9b6dd3ef1e1d155c1717b46d6cf11f731b52f9ffa1fddd`。
  仅修改Release正文，没有移动tag、重建或重传资产。
- 2026-11-13 dev.12已发布，使Windows便携包源码快照重新对齐当前`main`。
  dev.11 tag早于PR #71–#74；版本准备PR #75合并后，tag
  `v0.1.0-alpha.6-dev.12`精确指向`main`提交
  `5bc3972ba79cf8d59f684361233edfa5aefb72cb`。此次仅同步版本、README、发布说明和索引，
  不改变应用运行逻辑、默认采音行为或模型。源码验证为2,315 passed、89 skipped；
  `compileall -q manialab scripts tests`及`git diff --check`通过。
- Windows tag CI run `43291046336`成功：2,404项测试中2,315 passed、89 skipped、零失败/错误；
  模型校验、worker/Windows ZIP构建、clean PATH与Unicode数据路径离线EXE、worker关停、
  验证报告上传均通过。工件下载并复核于ignored目录
  `build/portable/verified-dev12-43291046336/`。ZIP 698,937,720字节、
  SHA-256 `0eadf4911a062f7d5a6929605f0a9ad7686d14ac0772f4ec494f41d228969bff`；
  4,222个条目CRC通过且无重复路径。
- GitHub Pre-release已发布：
  [ManiaLab alpha.6 dev.12](https://github.com/zhyzx35607/ManiaLab/releases/tag/v0.1.0-alpha.6-dev.12)。
  三个资产的名称、大小及digest均已与本地复核结果比对；SHA256SUMS资产自身
  SHA-256为`7ca14900c26e0db47825002c52e625f01b8e5bb60ad8fc15867cbabbb56cac93`，
  验证JSON为1,992字节、SHA-256
  `025ed271f5ef922028519410d480da098fcbe7d6c9a19aa6372dc69f2dae0725`。
  CI使用合成音频，只验证软件与便携运行，不构成音乐或Timing验收；人工听审、
  标签复核及osu!试打仍未完成，随包Demucs checkpoint单独再分发授权仍未核实。
- dev.13 通过 PR #78 合入，merge commit、tag `v0.1.0-alpha.6-dev.13` 与发布时
  `main` 均为 `1b8995f89d73403e9cdb678d83d6dd7ff4637208`。本次只同步版本元数据、
  README 和发布说明，以包含 PR #76/#77 的当前源码快照重新打包；不改应用行为、
  默认采音或模型。本地全量测试为2,315 passed、89 skipped，
  `compileall -q manialab scripts tests`、版本导入及`git diff --check`通过；
  PR 两项 Windows `verify` 均通过。
- Windows tag CI run `43338859633` 于2026-11-13成功，耗时7分42秒：2,404项测试中
  2,315 passed、89 skipped、零失败/错误；模型校验、worker/Windows ZIP构建、clean PATH
  与Unicode数据路径离线EXE、worker关停、验证报告和工件上传全部通过。CI报告明确
  未做音乐或Timing验收。
- 工件下载复核于ignored目录`build/portable/verified-dev13-43338859633/`。
  ZIP大小698,940,661字节，SHA-256
  `4bfec96f3b2f1d1209dd57814a76615fb3c884ae9b997cc35ef29d769dfdf67b`；
  4,223项全量CRC通过、无重复路径；包内版本、EXE及release manifest哈希与报告相符。
  GitHub Pre-release 已发布：
  [ManiaLab alpha.6 dev.13](https://github.com/zhyzx35607/ManiaLab/releases/tag/v0.1.0-alpha.6-dev.13)，
  Windows ZIP、SHA256SUMS、验证JSON三项资产的名称、大小和digest均与本地匹配。
  音乐质量、Timing、人工听审与osu!试打仍未完成；Demucs checkpoint单独再分发授权
  仍未核实。当前后续代码阶段先做冻结输入上的可解释起音保留账本，不重跑模型；
  人工听审结果仍须用户实际试听并导出。
- PR #81 合入 `main` 后准备dev.14，tag
  `v0.1.0-alpha.6-dev.14` 精确指向 `4e9cb485d56a18a67dd5ebd7fcb538c3736afa70`。
  Windows tag CI run `43348232249` 于2026-11-13成功，耗时5分15秒：2,407项测试中
  2,318 passed、89 skipped、零失败/错误；模型校验、worker/Windows ZIP构建、clean PATH
  与Unicode数据路径离线 EXE、worker关停、报告和工件上传均通过。ZIP保留于 ignored
  目录 `build/portable/verified-dev14-43348232249/`，大小698,949,530字节，SHA-256
  `a3842716fd20feaf47723297360edb1b36c49844c12b143df3e0d2baad627b23`；本地完整CRC
  校验通过，4,226个条目无重复路径，验证报告、JUnit、便携报告、关停报告及包内版本、
  EXE、manifest哈希一致。**没有创建dev.14 GitHub Release**：发布前复核发现 README
  将模型概括为不随便携包分发，但 ZIP 实际包含应用所需固定模型。tag不移动或重写，
  该工件只保留作历史验证记录。
- 2026-11-13 准备dev.15修正打包范围说明，使 README 明确区分随包固定应用模型与
  不进入源码/便携包的用户音频、参考谱、下载语料、本地研究权重和逐曲产物。相对dev.14
  仍只改版本与文档，不改变应用运行逻辑、默认采音或模型。本地全量
  `.\.venv\Scripts\python.exe -m pytest -q` 为2,318 passed、89 skipped；
  `compileall -q manialab scripts tests`、版本导入和`git diff --check`通过。PR #82
  两项Windows `verify`通过并合入；merge commit与`main`为
  `ead17d5e07f2936025272463dc1a5d56678e4923`，annotated tag
  `v0.1.0-alpha.6-dev.15`精确指向该提交。
- Windows tag CI run `43352716618` 于2026-11-13成功，耗时6分54秒：2,407项测试中
  2,318 passed、89 skipped、零失败/错误；模型校验、Windows构建、clean PATH与Unicode
  数据路径离线EXE、便携启动/分析/导出、重启与取消行为、worker关停、验证报告和工件
  上传均通过。CI夹具使用合成音频，不构成音乐或Timing验收。
- 工件及便携包独立复核保留在ignored目录
  `build/portable/verified-dev15-43352716618/`。GitHub Actions工件
  `14004881832`为698,952,215字节，SHA-256
  `1a84d7397fadac0fd621d6a9f029a4016df2eaa6d27572489a6d76b43e45f612`；外层ZIP含3个文件、
  无重复路径且完整CRC通过。正式Windows ZIP为698,949,407字节，SHA-256
  `b5a4257172c0b1febe827c8967e766544853824da5931e32504bbc4a5784c1aa`；本地4,227项CRC
  全过、无重复路径，包内版本、EXE与release manifest哈希和验证JSON相符。固定
  `htdemucs_6s.th`随包模型SHA-256为清单值
  `34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd`；未发现用户音频、
  参考谱、下载语料、本地研究权重或逐曲产物。
- GitHub Pre-release已于2026-11-13发布：
  [ManiaLab alpha.6 dev.15](https://github.com/zhyzx35607/ManiaLab/releases/tag/v0.1.0-alpha.6-dev.15)。
  Windows ZIP、SHA256SUMS与验证JSON均已上传；GitHub资产大小和digest分别与本地文件
  一致。验证JSON为1,992字节、SHA-256
  `bfa9d4604dbefc7b99facbb040fe9fcec4c54295f95c6d320b822ca1bdbe243b`；
  SHA256SUMS为226字节、SHA-256
  `2c2616290a1c5a8dc258006b805eaf760e6045ae257477d3bb0f3d3997a8b075`。
  本版不代表采音准确率、Timing或音乐质量提升；人工听审、标签复核和osu!试打仍未完成，
  Demucs checkpoint单独再分发授权仍未核实。

## Nova 人声乐句候选复核包

- 2026-11-12 新增 `scripts/build_vocal_phrase_review.py` 和
  [复核协议](vocal-phrase-audio-review-protocol.md)，从已冻结分析制作本地听审材料，
  不重跑模型、不读取谱面真值、不改候选。构建前逐项校验冻结人声复核报告中的6个输入
  SHA、分离来源、时长、样本数和采样率：Nova vocals 原始来源
  `7b1f57f1c5756c51a1da14b0f60980a01d74b8478dbd869dc8280005d84706e7`，
  对应 raw float WAV `638f56f5f3f25ff82fc0f6222dc1b4003239c6f167c80e53eaec96dfd51535b7`；
  原混音来源 `2a82d46c43e43acb4f8e21de71573be1ae6b71b8cf5485364c9ed1d5d3b12765`，
  对应 raw float WAV `dfd2e1581124db8255a347c8414411d1978e279d06116603df64dcfa60e8745e`。
  两者均为22,050 Hz、6,258,672样本、283.84秒；同一分析时钟和来源哈希不证明分离轨内部
  无偏移，仍须逐段试听核验。
- 产物留在 ignored 目录
  `test-data/sakura-benchmark/vocal-phrase-audio-review-v1/`：冻结97条候选，每条含
  人声与原混音各一段，共194个PCM16音频片段，绝对时间前后各留1.5秒上下文；共13段，
  第4–13段含候选。对两轨使用同一固定线性增益 `0.6262192773170971`，无逐片响度归一、
  滤波、压缩、重采样或时间移动。`index.html` 支持回放、声源/边界人工标记、备注、
  浏览器本地暂存和 JSON 导出；清单与机器报告记录输入/产物哈希。独立逐样本复核194/194
  通过、200个产物哈希匹配、PCM峰值31129/32767，未削波。报告 SHA-256：
  `1f61709cfde80f7a2211acaebfc00218f94bd710f7fe9cd08b50eb5f138f9eec`。
- `tests/test_vocal_phrase_review.py` 为3 passed；本轮全量
  `.\.venv\Scripts\python.exe -m pytest -q` 为2275 passed、89 skipped，
  `compileall -q manialab scripts tests`、审核页内联 JavaScript 语法检查和
  `git diff --check` 通过。
  尚未由人试听或填写标签，不能称97条候选为真声，也未做 osu! 试打或音乐质量验收。
  下一步需复核并导出标签，再基于人工声学证据检查连续 LN 的起止、声部和段落表达。
- 2026-11-12续接复查确认主线当前与 `origin/main` 同为 `78335ad`；未发现 ManiaLab
  研究/推理进程。审核包有97个冻结候选和194段音频，磁盘上未发现导出的
  `vocal-phrase-review-v1-labels.json`。Edge 存在审核页窗口，但桌面采集连续两次超时；
  没有据此推断浏览器本地暂存为空，也未触碰标注。
- 新增 `scripts/summarize_vocal_phrase_review.py`：接收导出的 JSON，核验审核包的
  输入身份、`review_key`、冻结候选集合和全部登记产物哈希，汇总声源/边界标签覆盖、
  缺项及边界待回听 ID；不输出备注正文，不计算准确率，`musical_acceptance` 保持
  false。实际审核包 200 项登记产物哈希和 97 个候选 ID 均通过核验；定向测试
  `10 passed`，全量 `.\.venv\Scripts\python.exe -m pytest -q` 为
  `2282 passed, 89 skipped`，新脚本/测试 `py_compile` 与 `git diff --check` 通过。
  此工具不能代替人工试听，仍需用户导出标签后才能进行声部和边界复核。

## 当前研究

- 八拍LN乐句上下文仍是独立研究，不接入默认生成器。特征以累计红线拍位
  编码最近8拍，并保留末四个事件行的完整互动；新增了近放手起音的同手/跨手
  统计。由于特征和Timing guard变化，schema升为
  `sequence-window-ln-phrase-context-v3`，缓存升为v8，旧格式不会复用。2026-11-12
  13:17完整准备和训练已结束，输入7,083张谱（6,282 train谱、801 validation
  谱；test split排除），最佳第19/24轮。最佳轮的10K验证排序准确率为74.21%，
  LN窗口75.19%、tap窗口73.25%，覆盖10个验证歌曲组；第24轮退至73.49%、
  74.89%、72.12%。这些是固定音符与LN尾端下的候选轨道排序，不是整谱生成
  质量或LN写法收益，也没有独立test结果。模型和报告在ignored目录
  `test-data/sequence-prior-ln-phrase-context-v3-no-test-full-20261112`，
  特征缓存有7,083份；当前没有训练/下载后台进程。
- 用户澄清《童话》和《樱花草》是不同写法的例子，目标是学习随段落、声部和
  乐句变化的条件化编排策略，不把任一曲的“单条/双条”或全曲LN比例固化成模板。
  已解析本地《童话》8K/9K/10K和《樱花草》Reference/成品：《童话》10K全曲
  LN比约78.6%，指定八段约55.5%–100%；《樱花草》10K成品约34.2%、Reference
  约8.5%。以上是表达分布差异，不是质量排名。当前v8仍只训练固定音符、起音和
  LN尾端下的轨道排序，不能学习段落声部、LN选择或LN延长。设计边界与验证方案
  见[编排设计](arrangement-design.md)。扩展语料检查到7,907张6–10K谱面、0个
  配套音频文件，故纯谱训练无法学习这些表达背后的声部身份与真实音尾。另清点到
  40张已有10K原曲/谱面对照，
  其中按既有划分24张属train，
  但LN比均低于50%；另有14张跨键数holdout和10张未划分的6–10K选择对照。
  这些对照尚无Nova分轨和逐音LN证据，不能直接监督声部条件延长；范围和数据边界
  见[LN表达音频样本清点](expression-audio-pair-inventory.md)。
- 已加入参考谱 Kiai 状态切换导入：与书签分开选择，换算到音频时钟后作为可编辑
  段落边界提示；不把 Kiai 或书签宣称为主歌／副歌识别。TimingPoints 中红线、
  绿线的状态变化、重复状态、非有限时间、偏移和越界均有覆盖。
  `tests/test_structure_expression.py` 定向验证 `27 passed`，Python 编译、
  `node --check static/structure.js` 与 `git diff --check` 通过。PR #42 已通过
  两项 Windows CI 并合入 `main`，merge commit 为 `5c17e50`。
- 为表达研究新增的音频配对 cohort 已完成冻结与下载：从7,902张可用谱、2,020个
  歌曲组中排除80个已有音频配对组，再按6–10K、LN-heavy/Hybrid/tap-heavy和
  train/validation/test选取74组（63 Ranked、11 Loved）；目前成功配到72份音频。
  10K LN-heavy仅有2个test组、没有train/validation组；8K LN-heavy没有新样本，
  9K LN-heavy只有3个train组。两项失败是谱面音频文件名不匹配或无支持扩展名，
  未强行配对。一个113.7MB谱包通过HTTP Range只读取3.82MB。
  配对样本是原曲与谱面，不带分轨或逐音乐器标签；只能为下一步对音与弱监督
  提供候选，不能直接当作声部条件LN真值。实现已通过PR #44合入`main`
  （`f7764ae`）；`tests/test_expression_audio_cohort.py`、
  `tests/test_fetch_paired_maps.py`共6 passed，`py_compile`与`git diff --check`通过。
- 2026-11-12 对冻结目录`test-data/expression-audio-cohort-v3`完成只读独立复核：
  选择清单规范化SHA-256与报告相符，74个所选歌曲组中72组配对成功、2组按
  文件名/扩展名规则失败；72个成功样本的音频哈希和谱面哈希均与登记值一致，
  歌曲组及音频哈希均无重复。71份保存的完整`.osz`哈希相符，1,631个成员CRC
  全部通过且无重复成员；另1个成功样本为HTTP Range取音频，不保存完整归档。
  该审计未重下载、解包写入或运行分离/转录模型。目录大小约731MB，音频仍为
  与谱面按文件名配对的混音，不包含声部身份、逐音标签或LN音尾真值；只证明
  本批次字节与配对清单完整，不构成音乐质量验收。
- 新增`scripts/separate_expression_audio_cohort.py`及
  [分轨协议](expression-audio-cohort-separation-protocol.md)，仅面向冻结cohort的
  39个train组，validation/test不读音频；CPU串行，线程2/1，输出在ignored目录，
  完整输入、模型和逐轨哈希后才允许续跑。首次`--plan-only`发现新入口的规范JSON
  哈希错误使用原样Unicode，而冻结选择器采用默认ASCII转义；已统一为选择器规则并
  加Unicode回归测试。定向测试6 passed，`py_compile`与`git diff --check`通过。
  `--plan-only`核对出6K/7K/8K/9K/10K分别9/9/6/9/6组，Hybrid Demucs模型SHA
  匹配，PyTorch CPU 2.9.1、intra/inter-op线程2/1，`inference_started=false`。
  PR #59已通过两项Windows verify并合入`main`（`54cb27b`）。随后串行运行39组
  CPU分离，`run_identity_sha256=2ce6c31a6ee8c8ee1850a7f5994af5eaefb7b62bc8b68496de52b88b3ddb9903`；
  共97.44分钟原音频、156条四轨NPY，输出约1,967.15 MiB，保留于ignored目录
  `test-data/expression-audio-cohort-v3-separated-train-v1`。重新核验39/39完成记录、
  输入身份、模型SHA、输出采样率/帧数/float32、有限值及全部声轨SHA，均匹配；
  汇总清单与39个训练ID完全一致。结果只提供后续局部声学候选，不证明乐器身份、
  谱音对应关系或LN真实音尾，也不是音乐质量验收。下一步仅基于train分轨和谱面
  作者动作做局部证据对齐，避免把弱分轨当真值；仍需原曲听审与osu!试打验收。
- 新增`scripts/build_expression_cohort_stem_review.py`及
  [听审协议](expression-audio-stem-review-protocol.md)，为每个训练歌曲组确定性选择
  一个作者谱动作较丰富的8拍窗口，配对原混音和四条同时间分轨；不构造逐音标签，
  不读取validation/test，不重跑模型。原混音使用与分轨相同的解码/重采样时钟，
  所有片段和轨道共享单一增益，审核页支持人工判断和JSON导出。首次直接CLI启动发现
  项目包搜索路径问题，在任何音频读取前修复；新增仓库外启动`--help`回归测试。
  真实首次构建在首组`3833020`的选窗阶段报`Training chart metadata mismatch`，
  尚未解码音频或创建输出目录。根因是构建器误要求谱面原始`AudioFilename`与
  冻结清单存储名逐字相同；下载器按谱面名取音频后会规范保存为`audio.<ext>`。
  只读核对39个训练组，12组文件名不同但39组扩展名全相同。已修正为校验键数及
  非空、大小写不敏感的音频扩展名；测试覆盖`track.wav`映射到`audio.wav`可构建、
  扩展名不一致拒绝。定向测试`7 passed`，`py_compile`与`git diff --check`通过。
  PR #63 的两项Windows verify通过后已合并，merge commit为`4fbbe10`。
  随后只基于冻结train输入构建审核包
  `test-data/expression-audio-cohort-v3-stem-review-v1`：39组、每组原混音加四条
  分轨，共195段PCM16片段、合计268.063秒；报告SHA-256为
  `7296238d3c77f32fcb304a8e099a1d076a23733893356cd30e73dcdd0260c034`。
  独立核对清单登记的200项产物哈希全部匹配，195段均为22050Hz单声道PCM16，
  页面内联JavaScript语法检查通过。机器报告确认未读取validation/test、未重跑模型；
  人工听审和音乐验收仍为false，Demucs声道名称不代表真实乐器或LN音尾真值。
  下一步打开本地`index.html`完成人工听审并导出标签，再据实际声学证据复核声部、
  起音与LN尾端；不得将此目的性选窗样本用于准确率估算。2026-11-13通过仅绑定
  `127.0.0.1:8765`的静态HTTP服务实际打开审核页：页面显示39组、`已填写 0 / 39`，
  对全部195个登记WAV资源的本机HEAD检查均返回200；只确认界面渲染和资源可达，未播放音频、
  修改标签或导出。页面首次加载时尚未选择音频，选择轨道按钮会调用播放。人工
  听审仍是下一步，未把页面检查或HTTP可达性算作听审结果。
- 新增`scripts/summarize_expression_cohort_stem_review.py`，为上述导出JSON校验冻结
  review key、case集合、字段和标签枚举、时区时间戳，以及包内200项登记产物SHA和195段
  PCM16 WAV头；输出标签覆盖与分布、待复核case ID，不输出备注正文、不算准确率，
  `musical_acceptance`固定为false。实际本地包只读校验通过：39组、39个case、195段、
  200项产物匹配。新测试15 passed；与builder和既有人声审核器相关测试合计29 passed，
  `py_compile`、CLI `--help`通过。当前 Downloads 中尚无
  `expression-audio-stem-review-v1-labels.json`；导出后可运行
  `.\.venv\Scripts\python.exe scripts\summarize_expression_cohort_stem_review.py --labels "$env:USERPROFILE\Downloads\expression-audio-stem-review-v1-labels.json" --output test-data\expression-audio-cohort-v3-stem-review-summary-v1.json`。
- 新增研究入口`scripts/research_expression_actions.py`，从作者谱的8拍局部窗口
  提取起音骨架，预测LN比例、LN拍长、LN和弦、错开放手、Hybrid、反键与独立放手
  等图形动作；8拍只是采样上下文，不是主歌/副歌或固定乐句标签。采用歌曲组等权
  ridge，对比键数×键数内密度三分位×全曲位置三分位基线，test组在读取谱面前排除。
  基线在每个键数×密度×全曲位置格内重新按歌曲组等权，避免局部筛选后又被多谱歌曲支配。
  全语料运行已完成：6428张train谱/1626组、735张validation谱/195组，test组未读取。
  10K validation有13组；ridge相对分层基线在LN和弦比例上改善（MAE
  0.3314→0.2623），但LN总比例略退步（0.2167→0.2201）且反键比例退步
  （0.0267→0.0366）。报告不含组级置信区间，不使用音频或乐器身份，也不接入默认
  生成器；这些结果只表明部分图形动作可从谱面上下文预测，不能证明听音、声部条件
  或音乐质量收益。结果位于ignored目录
  `test-data/expression-action-predictability-v1/report.json`。
  《樱花草》单曲逐段表达报告由`scripts/analyze_sakura_expression_profile.py`
  生成，核验Reference/成品身份并按共同书签统计LN长度、交错放手、Hybrid及同时持条。
  除开头7物件引子外，成品各段LN比为10.7%–52.3%；此范围仅描述一张作者谱，
  不作为生成配额。
  报告留在ignored目录`test-data/sakura-benchmark/expression-profile-v2/report.json`；
  研究边界和代表区间见[编排设计](arrangement-design.md)。PR #40 的配对音频
  覆盖清点已合并（`3fc9b4e`）。
- 完整v7准备在谱面`2571106`处发现红线beatLength为`1E-61`，导致累计绝对拍位
  超出float64能分辨后续起音间隔的范围。sequence prior现在把小于10ms或大于
  10,000ms的红线周期都视为不可信并继续使用前一有效红线；这不修改谱面原Timing。
  失败缓存留在`sequence-window-cache-v7-ln-phrase-context-no-test-full-20261112`，
  不作为训练结果。修复后对谱面`2571106`重放得到80个窗口（40个LN），特征均
  有限；1016条红线中极端值按guard忽略，多行难负例成功80条、单行回退0次。
  定向测试`81 passed, 4 skipped`，完整套件`2244 passed, 89 skipped`，
  `py_compile`和`git diff --check`通过。修复后已使用独立v8缓存完成全量准备与训练；
  其验证范围和限制见本节首条记录。
- LN难负例尝试改动连续的2–4个事件行；若窗口内没有相邻可改事件行，退化为
  单行候选。诊断分别记录两种候选的搜索次数和成功数；有限搜索找不到足够
  可行负例时，该窗口仍会被记为失败，不会伪造样本。
- `--exclude-test-split`现在先从manifest元数据确定歌曲组和split，再过滤test，
  然后才打开纳入的谱面核对哈希或读取缓存。回归测试以缺失的test `.osu`文件
  验证它不会被打开；此选项只允许与phrase-context schema和跳过test评分一起用。
- 2026-11-12烟测使用10K每split最多3张图，累计排除822张test图，处理3张train
  和3张validation图。2轮/4步训练与导出完成，导出误差`2.83e-7`。这组小样本
  validation没有LN窗口（24个tap负例对），所以只证明数据到模型的流程能跑通，
  不提供LN收益或音乐质量证据。产物留在ignored目录
  `test-data/sequence-prior-ln-phrase-context-v2-smoke-20261112`及相邻缓存目录。
- v7完整准备约处理3,850张谱后因上述异常周期中止，尚未进入训练。修复后还需
  重新生成完整v8的train/validation窗口并训练，再比较10K LN/tap分层结果与现有
  v5基线；如果validation没有稳定收益，保持关闭。当前特征固定起音和LN尾端，排序结果不
  等于真实歌曲写谱质量，也不能替代《樱花草》人工分段检查和实际试打。

## 已保存的应用

- 可下载版本：`v0.1.0-alpha.6-dev.4`，含声轨导入、逐段表达、可选钢琴、
  Tsumugi 贝斯/其他伴奏及 FL 吉他采音。使用方式见
  [乐器采音选择](instrument-choices.md)。
- 开发基线提交 `63fcc08baff631d642e1bc8ae69620a83cc7a1cb` 已推送，
  Windows validation `42839955554` 成功。此前本地完整检查为
  1866 passed、85 skipped；不是本轮重新运行的结果。
- 之前81个开发提交已通过 PR #1 归入主线，合并提交
  `9b11b10d913acb19070f228397bb7cbf8bc5233c`；PR检查 `42846636872` 成功。
- 模型恢复、比较入口和续接说明已通过 PR #2 合入主线，合并提交
  `0a43911477f7c11eea58a31e57bfdfa88c753e3c`；PR检查 `42847322903` 成功。
- 鼓组剩余容量补充已通过 PR #3 合入主线，合并提交
  `40e1a061c35caf89ad7b004a858b9619892e1fb5`；PR检查 `42852204830` 成功。
- Mega53长窗和采音结果已通过 PR #4 合入主线，合并提交
  `a7b2478c2df09e0f6dc2a3d0dc409ea356dabdac`；PR检查 `42854329970` 成功。
- Harmonica显式FP32对照及完整结果已通过 PR #5 合入主线，合并提交
  `28c9a51613d33e991f1b7bb888e6c1940a627cf9`；PR检查 `42856082126` 成功。
- 当前Nova other原声与落线对照已通过 PR #6 合入主线，合并提交
  `3a1aa6d139d24b41508143e82773973d3789f5f6`；PR检查 `42859344954` 成功。
- MuScriptor完整结果及独审已通过 PR #7 合入主线，合并提交
  `2510e103ffe83a38ae3fb4963d35a9fe1b498ecf`；PR检查 `42860878982` 成功。
- 等价候选去重修复已通过 PR #8 合入主线，合并提交
  `3b11f6d1b0c513b9e83fa2d3c4bc4c9004018933`；PR检查 `42863989466` 成功，
  GitHub检查1901 passed、85 skipped。
- 已知乐器音色验证入口已通过 PR #9 合入主线，合并提交
  `279bf6e1e9cdca9d906301de7b12781481a60d38`；PR检查 `42865447918` 成功，
  GitHub检查1905 passed、85 skipped。
- 音色分类与声部分组完整结果已通过 PR #10 合入主线，合并提交
  `cc3e82817a9940008c0e2703e34cda777c776430`；PR检查 `42868880993` 成功，
  GitHub检查1908 passed、85 skipped。
- 区间空隙保护已通过 PR #11 合入主线，合并提交
  `3dfb85da843fbaddd46a8d9c3b53949a3498e7b6`；PR检查 `42872567983` 成功，
  GitHub检查1928 passed、85 skipped。
- 钢琴模型低活动上下文已通过 PR #12 合入主线，合并提交
  `ff4d7fed161bbe30ab9d43b0cda42ab60f57b4a9`；PR检查 `42875415387` 成功，
  GitHub检查1950 passed、85 skipped；研究规则未默认启用。
- 钢琴低活动原声核查已通过 PR #13 合入主线，合并提交
  `46cb32647c1fc62b3ee99c33e774013eac8f117c`；PR检查 `42877636337` 成功，
  GitHub检查1961 passed、85 skipped。
- 钢琴跨窗持续修复已通过 PR #14 合入主线，合并提交
  `ad9c8c05ed69848d5bfc0538c834c7729ee4b2f8`；PR检查 `42882871715` 成功，
  GitHub检查1980 passed、85 skipped；没有新Release。
- 同权重和弦起音排序已通过 PR #15 合入主线，合并提交
  `12bb01c01f5daa6cec0a454c7e504071f94b1f80`；PR检查 `42885147076` 成功，
  GitHub检查1992 passed、85 skipped；没有新Release。
- 相邻节奏组连贯取舍研究已通过 PR #16 合入主线，合并提交
  `abf8e31547cb71d7cf23330eebb8b8ac278967ad`；PR检查 `42887691860` 成功，
  GitHub检查2005 passed、85 skipped；研究参数仍默认0，没有新Release。
- 原区间乐器评分恢复入口已通过 PR #17 合入主线，合并提交
  `2179391207ecaadcd6bb966dd0bdf3ef04533d2d`；PR检查 `42890087106` 成功，
  GitHub检查2021 passed、85 skipped。后续完整结果另开PR；没有新Release。
- 乐器评分完整结果已通过 PR #18 合入主线，合并提交
  `97f9fc1`；GitHub检查 `42892770435`、`42892755994` 均为2021 passed、
  85 skipped；没有新Release。
- 有限持续覆盖研究已通过 PR #19 合入主线，合并提交
  `1f90d1c06530ed59124d7882ccdbda46b0ec2fd9`；检查 `42969262042`、
  `42969130741`成功。研究权重仍默认0，没有新Release。
- 自动 Timing、真实混音的漏采/串音、声部连续性和 LN 音乐性尚未最终验收。
  不宣称已超越 Mapperatorinator。质量标准见[目标](music-quality-target.md)。

## 本阶段新增

- 段落级人声乐句选择已贯通表达、服务端和结构界面：只对覆盖充分、音尾可观测的跨音高人声候选排序，按段落选取比例替换 LN 锚点；未选中的起音仍可进入后续短键选择。用户可在每段启用“连续人声乐句”并设置本段比例。此比例是候选选取上限，不是全谱 LN 配额或音频识别置信度。冻结《樱花草》分析（9,138个候选、97个人声乐句）回放中，自动方案在9/13段启用该模式并选出31条人声乐句 LN；用户说的是《樱花草》谱面没有人声音符，不能据此推断原曲没有人声，也不能据此判定分离标签泄漏。现有候选是否为人声仍需对照 Nova 人声音轨与人工核验，阶段未通过音乐质量验收。人声模式改变共享密度/占轨后的选择：对照基线，10K最终非人声音符新增57、移除41；变更前非人声表达候选完全一致。补充候选 ID 前置校验，避免无效 ID 在字典构造中变为 `TypeError`；修复 Windows Mapper 取消后残留子进程的问题，终止启动器后再枚举并回收仍存活的后代。完整测试 `.\.venv\Scripts\python.exe -m pytest -q` 为2265 passed、89 skipped；`py_compile`、`node --check static/structure.js`、`git diff --check`通过。此结果验证代码路径，不代表音频采音和音乐质量已验收。
- Beam未来状态去重修复已通过PR #26合入主线，merge commit
  `501b05404c30b08b15a0dc159f53d1bc156f0fb9`；两项Windows validation成功。
  原去重键只比较轨道结束、上次起音和当前行掩码，会合并未来评分上下文不同
  的路径；新签名保存`ends/last/load/source/recent/time`，不把累计cost或回溯
  指针当作未来状态。定向反例用旧签名会选错首排，修复后保留预期的`{0,2}`。
  独立审查未发现字段遗漏；`tests/test_layout.py`为13 passed，全套为2180
  passed、85 skipped，`py_compile`与`git diff --check`通过。每排重置`seen`
  保证去重只发生在同一候选行。本改动可能改变原beam布局结果，不代表整谱
  音乐质量改善；source映射排序的性能尚无独立基准。
- 四排序列先验运行时接入已通过PR #27合并：显式加载、manifest哈希校验、
  默认关闭，只用于完整beam定轨；训练/运行时共用异常Timing guard，窗口保留
  四排上下文与左缘前LN状态，累计分数按候选分差做有界排序影响。10K全谱
  validation三档均没有显示平衡的LN定轨收益，因此不默认启用；它不负责采音、
  改LN尾端或证明音乐质量。
- 新增`evaluate_sequence_prior_remap.py`作为固定输入全谱A/B入口，只从训练
  报告的validation/test歌曲组取图，逐图校验来源SHA、对象身份与轨道回读、
  TimingPoints、LN端点及可玩冲突；按键数分层、同图配对、歌曲组等权统计。
  test必须引用完整validation产物并匹配相同模型/报告/seed/键数/强度；
  输出留在ignored的`test-data`，支持原子检查点和配置一致的续跑。
- 上一轮完整测试为`2194 passed, 85 skipped`；序列运行时/训练/组评价相关
  32项通过，边界测试覆盖窗口左缘LN、beam序列历史签名、完整对象到轨道回读
  及test强度冻结契约。验证集有45张10K图、13个歌曲组，约45,019个物件；
  10K独立test为48张、12组。结构A/B仍不能替代《樱花草》分段/LN人工审查或
  实际试打，未验证前不能宣称键型质量提升。
- 本轮独立审查发现10K五音和弦在特定LN尾长顺序下会产生1008个release-order
  候选（`ends=(0,1,2,4,3)`），旧cap=256会截掉仍可能进入beam的布局。cap现调为
  1024，并以真实`release_choices()`反例及6–10K理论上限增加测试；定向测试
  `34 passed`，完整套件`2201 passed, 85 skipped`，`py_compile`与
  `git diff --check`通过。独立协议审查未发现A/B完整性阻断项，并确认强度
  应按歌曲组及多项结构指标并列比较；固定LN尾长的定轨结果不能证明LN表达
  质量。cap=1024的10K validation strength=0.15已完成45/45图、13组且结构检查
  全通过：cost/note变化为-0.001384（95%组级区间[-0.003085,0.000224]），
  LN inverse误差变化-0.032204（[-0.156698,0.072554]）；区间均跨0。
  原谱轨道偏移率增加0.080562，shield轮廓误差点估计增加0.112701。
  暂无明确收益，不冻结test；报告位于ignored的
  `test-data/sequence-prior-remap-v4-validation-015/report.json`。
  strength=0.35也完成45/45图、13组且结构检查全通过：cost/note下降0.004630
  （[-0.006457,-0.002829]），但LN inverse误差上升0.178843
  （[0.000000,0.501917]），LN相对原谱轨道变化增加0.261869
  （[0.213838,0.305008]）。平均先验定轨时间为10.07秒/图，对照baseline
  4.93秒/图。strength=0.60也完成45/45图、13组且结构检查全通过：cost/note
  下降0.006339（[-0.009652,-0.002920]），LN inverse误差增加0.221021
  （[0.018852,0.486119]），LN相对原谱轨道变化增加0.350249
  （[0.299511,0.398175]）；平均先验定轨12.83秒/图，对照baseline
  5.40秒/图。三档均不满足冻结test的条件。细项见
  [定轨验证报告](sequence-prior-remap-results.md)，完整报告均留在ignored的
  `test-data`。不能通过增大strength掩盖模型只学到局部定轨倾向的限制。
- PR #29已合并至`main`，merge commit `68e3ab1`；两项Windows validation均成功。
- 只读审查确认旧序列先验用作者轨道作正例、随机可行改轨作负例，每窗改
  1–3排且固定LN尾端；它学的是局部原谱排序，不是LN质量或长条时长。
  新增隔离的`ln-transition-hard`负例策略，仅在LN上下文窗改一排，并要求
  inverse、reverse-shield、shield、same-hand hybrid、split-release、
  release-press或near-release-press计数至少一项变化；候选仍须全窗可行。
  10K validation 13组45图抽样1461个LN窗，1360窗（93.1%）成功生成两条
  定向负例；101窗不足两条。可接受负例中的各模式变化次数见
  [LN难例探针](sequence-ln-hard-negative-probe.md)。
  专项测试27 passed；全量`pytest`为2206 passed、85 skipped；`py_compile`、
  `git diff --check`通过。PR #30已合并，merge commit `99a4e6c`。全量v4训练
  完成7,907张谱，按validation选中第20/24轮，导出误差`3.81e-5`。10K test
  共有12组、3,694窗；逐对准确率overall 78.11%、LN 77.71%、tap 78.41%。
  原序列先验、生成器默认行为和LN尾端均未改变。
- 新增`scripts/evaluate_sequence_prior_cross_model.py`，允许分别校验两模型的
  训练报告，再在同一份v3或v4候选集上作配对歌曲组bootstrap，并输出
  overall/LN/tap逐组差值；支持validation和test。PR #31已合并，merge commit
  `3622127`；两项Windows validation成功。完整`pytest`为2211 passed、85 skipped，
  `py_compile`及`git diff --check`通过。
- 10K配对结果和边界见
  [LN难负例实验报告](sequence-prior-ln-transition-results.md)。validation中，
  v4相对v3在v3候选上的10K LN差为-1.03个百分点
  （95% CI -4.45至+1.80），在v4候选上为-0.67个百分点
  （-2.67至+1.30）；test两者均跨零，未证明收益。模型训练每个键数轮流取
  相同优化步，所以7K样本多不是此差异的直接原因。
- 10K专训已完成，validation选择第4/24轮。它对比joint v4时，在相同10K
  validation hard-negative候选上的LN变化为+0.57个百分点
  （95%组级区间[-1.08,+2.47]），整体-2.30个百分点
  （[-3.58,-1.02]）、tap -4.08个百分点（[-6.06,-2.42]）；
  LN区间跨0且整体/tap下降，不接入生成器。独立评估产物保存在ignored目录
  `test-data/sequence-prior-cross-v1-validation-v4-10k-specialist.json`。
- 新增实验性`sequence-window-ln-interactions-v1`输入，在原382维前缀后添加
  4排×17个镜像不变的LN互动摘要，涉及持有密度/手间组合、hybrid、释放配合、
  inverse、reverse-shield、shield及近释放起音。训练、单窗/批量运行时和评估器
  支持新schema；采用独立v5缓存，旧模型和v1生成行为不变。v5难负例签名忽略
  运行时不可见的窗口外LN释放，避免训练目标依赖额外历史；配对评估器校验目标
  歌曲组与两模型训练组互斥。66项专项测试通过；真实模型报告的1626个训练组
  与195个validation组、202个test组均无重叠。2轮、每轮4步的小样本训练、
  导出及跨schema validation smoke跑通，导出最大误差
  `2.76e-7`。全量套件`2219 passed, 85 skipped`，`py_compile`与
  `git diff --check`通过。smoke数据只有每键每split最多2图，不能用于判断模型收益。
  全量v5训练已完成：7,907张谱，第18/24轮按validation选模，导出误差
  `3.81e-5`。v4/v5在相同v5候选上的配对validation显示，10K LN变化
  `+2.61pp`（95%组级区间`[+0.23,+5.20]`），overall `+1.14pp`
  `[-0.89,+3.06]`、tap `+0.04pp` `[-2.47,+3.02]`；6–9K没有相同的
  清晰信号。随后一次性10K held-out test覆盖12组（10组有LN），LN变化
  `-0.07pp` `[-2.61,+3.44]`、overall `+0.26pp` `[-1.67,+2.68]`、
  tap `+0.87pp` `[-1.31,+3.21]`，均未显示可靠改善；validation LN增益未
  复现。因此v5不接入生成器。完整覆盖、命令和限制见
  [LN互动特征实验结果](sequence-prior-ln-interactions-results.md)；结果仅为
  固定起音与LN尾端下的定轨排序，不是音乐质量验收。
- 10K random-initialized specialist在原validation上相对joint v5明显退步；
  从joint v5低学习率warm-start后，原validation曾有`+2.51pp` LN信号。
  为避免重复使用已查看的test，训练器新增兼容模型warm-start、学习率、
  确定性alternate split及可跳过test报告的显式参数，默认行为不变。
  使用新split seed `20261112`从头重训v4与joint v5，再从该joint权重微调10K；
  新validation覆盖10K的14组/150图。warm-start相对v4的LN变化
  `+1.17pp`（95%组级区间`[-0.76,+3.20]`），overall `+0.80pp`
  `[-0.61,+2.76]`、tap `-0.17pp` `[-1.50,+1.17]`；joint v5的LN变化
  `+1.35pp` `[-0.58,+3.37]`。均未显示可靠增益，未打开新test；
  新10K test含7组、6组有LN，与旧test组重叠1组，因此不能称为完全未见语料；
  且跳过test评分仍会准备/读取其窗口缓存，只是不运行test模型评分。
  warm-start不接入生成。复现及专家模型失败结果见
  [10K LN warm-start复验](sequence-prior-ln-warmstart-results.md)。
  warm-start现在要求来源报告与导出模型绑定，重算完整训练manifest SHA，
  并要求与当前各split完全相同；新模型记录来源model/report SHA。
  继续warm-start必须由model和report共同证明祖先也使用同一manifest，
  因此缺少来源或祖先证明的旧模型会拒绝。覆盖跨split、哈希篡改、
  单边/不一致祖先证明、model/report哈希绑定与同split joint-to-10K案例；
  定向和完整测试为2230 passed、89 skipped，`py_compile`及`git diff --check`
  通过。
- PR #35已合并，merge commit `b111855`。全谱10K A/B评估器现接受
  `sequence-window-ln-interactions-v1`及旧v1模型，读取地图前验证报告、
  模型schema、清单与模型SHA；test冻结同时绑定feature schema。
  仅增加v5 run identity字段，保持旧v1续跑身份不变。synthetic v1/v5与
  不匹配控制35 passed，`py_compile`、`git diff --check`通过。
  后续验证中发现v5运行时逐个候选窗重算特征，导致一张真实10K图约237秒。
  v5现复用v1批量前缀并向量化LN互动特征；单窗训练编码仍作基准。3,000个
  合成窗用时从1.472秒降至0.081秒、特征最大误差0；6–10K随机状态各120窗
  与单窗逐值一致。全套测试2247 passed、89 skipped，`py_compile`与
  `git diff --check`通过。真实验证图4993903的v5定轨从236.710秒降至49.167秒，
  所有非耗时输出完全一致。
  alternate-split validation按相同seed每歌曲组抽1图，v4/v5相同14图、
  14组均通过结构检查。按歌曲组配对后没有平衡的LN键型收益：逆LN模式误差
  点估计增加0.093（95%组区间[0,0.206]），release-shield与release-press
  等若干误差点估计下降但区间跨0；短jack略增。更多数值与限制见
  [v5全谱验证](sequence-prior-remap-v5-alternate-validation.md)，完整结果
  留在ignored的`test-data/sequence-prior-remap-alt-v{4,5}-validation-pergroup-20261112.json`。
  本阶段不读test分数、不冻结强度、不接入默认生成。下一步应以《樱花草》
  实际bookmark分段和人工轨道角色为依据建立更直接的键型检查，不能继续把
  lane熵或与原谱的相似度当作音乐质量。
- 2026-11-13续接时发现原PR #36仍以`b111855`为基线，和后续新增的
  `LN_PHRASE_CONTEXT_SCHEMA`运行时及测试发生冲突；已将批量LN互动编码移植到最新
  主线并保留phrase-context路径。基于`47cb43c5d5b95433266b54c982bae6e2de39a13b`
  的当前工作分支验证：定向序列测试112 passed、4 skipped；全量pytest
  2,340 passed、89 skipped；`compileall -q manialab scripts tests`、
  定向`py_compile`和`git diff --check`通过。此次仅验证代码集成，不重跑14组模型
  对照或读取test split，不读取音频/模型/本地研究产物，也不接入默认生成。
- PR #87的两项Windows `verify`均通过并合入`main`，merge commit为
  `fe8e728d97976d7e68e25e3a7985ad73437db270`；旧PR #36因基线过期关闭并注明由
  #87替代。该集成仅优化研究评估运行时间，不代表LN键型或音乐质量改善。
- 当前研究仍未解决LN尾端/长度决策、整谱布局和实际可玩性。
- 该验证固定LN尾端，所以不能证明LN长度、音频声部角色、乐句分段或实际可玩性；
  strength排序指标不能代替《樱花草》共同bookmark人工检查与实际试打。
- `docs/full-remap-validation.json`中的既有谱不能用作本模型独立A/B：其中多张
  10K图属于sequence prior的train或validation组。优先报告10K的12个独立test
  歌曲组、48张图；6/7K作补充，8/9K因test组少只作探索。
- 当前项目`AGENTS.md`限制最多一个活动子代理。即使用户侧并发额度已升至3，
  仍按项目约束一次运行一个代理，优先用于独立只读审查；不并行修改共享的
  layout或研究脚本。

- 表达求解精度对照完成，当前分支`research/expression-tradeoffs`。
  旧持续覆盖9K方案在10K完整约束中可行且目标更高；原0.005相对间隙允许
  这部分损失。固定10组对照全部gap=0，主选择2.60–5.20秒，原15秒/5秒
  时间上限不增加。默认6–10K及持续覆盖6–8K逐物件不变；持续覆盖9K和
  10K均1491物件/425LN。只修正数值取舍，持续覆盖仍不默认启用。
  生产两处gap改0，`attack_optimal`要求两个阶段status0且gap0；不表示
  音乐验收或联合最优。新增9项及相关检查共97 passed。
  生产修复后完整套件2053 passed、85 skipped；72份case产物、14份源码
  与85份输入独审通过，默认selected/placed全部字段与顺序一致。
  入口`scripts/check_expression_precision.py`，产物
  `test-data/expression-precision-v1`，session `49169`已exit 0，勿重跑。
  完整[结果](expression-precision-results.md)与[协议](expression-precision-protocol.md)。
  无推理/训练。后续研究段内节奏骨架必须用确认过的全曲拍格，不能从每个
  短组首重置相位，也不能把全部support声部误认为重复伴奏。
- 固定节奏骨架研究已完成，当前分支`research/rhythm-policy`。新增显式恒速
  拍格策略接口，默认不启用；每个段落/声部/乐器类别固定stride与phase，
  删除起音不贡献奖励、不被LN代替，原两拍source group的密度检查和补鼓
  检查时刻保留。合成边界及相关检查通过。
  《樱花草》默认与持续覆盖、6–10K、完整/phase0/phase1共30组已完成，
  产物`test-data/expression-rhythm-v1`。完整方案逐Note重现；10K默认
  1496/382，phase0 1435/440，phase1 1391/532。phase1同时造成其他
  声部容量重排，不能解释成钢琴伴奏被安全抽稀；本轮无新增原声窗口、试听
  或音乐验收。结果见[固定节奏对照](expression-rhythm-results.md)与协议。
- 有限 LN 持续覆盖研究已完成并合并，原分支`research/sustain-coverage`。
  新参数`select_expression(..., sustain_coverage=.25)`默认0，只给通过现有
  尾音资格检查的`sustain` pitch 长音有限奖励，不改起止或尾音资格；
  资格字段不代表物理松键时刻已独立确认。
  固定《樱花草》6–10K候选重放的物件/LN分别为1490/368→1482/414、
  1493/378→1487/420、1496/382→1491/425、1496/382→1492/423、
  1496/382→1483/437。五种键数都伴随鼓与钢琴起音组损失；10K损失33个鼓、
  52个钢琴起音组，物件净减13个。结构、容量、落线和`.osu`回读通过，
  但尚未试听/试打、不能判音乐改善，所以保持研究用途、不接入默认。
  24个8秒原声窗口及click预览保存在ignored本地目录；
  216–224秒关注段无变化。详见[结果](sustain-coverage-results.md)与
  [固定协议](sustain-coverage-protocol.md)。专项174 passed，完整套件2044
  passed、85 skipped。没有重新推理或训练。
- 原AMT乐器评分恢复及完整评价已完成；入口PR #17已合并，结果分支
  `research/instrument-score-results`。
  [固定协议](instrument-score-evidence-protocol.md)先于本阶段推理/读取逐音标签。
  新入口`scripts/recover_tsumugi_instrument_scores.py`复用原区间，不重解码，
  保存36类logits、边界响应和拼接来源；排名逐项复现、音符不变才继续。
  16项新增控制与相关检查共52 passed。v1因配置tuple/list比较在首次前向前
  停止，保留目录；JSON规范化修正后v2全35份完成，session `54340`实际exit 0。
  315窗/5202区间/4790最终音，完整排名精确复现、边界最大差0；388音有多个
  来源，8音来源类别不一致。均值改变6音，其中2音在计分区；混奏带类起音
  F1仅82.44%→82.50%（多1匹配），完整区间69.82%不变，独奏均不变。
  全部推理完成后才读逐音标签，560组独立计数通过，评价实际exit 0。
  最高评分档仍有9个混奏唯一对应错类和99个无对应音，不默认平均类别或
  用分数筛音/指定主旋律。完整[结果](instrument-score-evidence-results.md)。
  产物`test-data/instrument-score-evidence-v2`及`test-data/instrument-score-evaluation-v1`。
  当前无运行中推理、研究队列、新训练或安装包。勿重复本批35份前向。
  已完成的评价命令：
  `.venv/Scripts/python.exe scripts/check_tsumugi_instrument_scores.py --evidence test-data/instrument-score-evidence-v2 --output test-data/instrument-score-evaluation-v1`。
  类别评分不能直接当音存在概率、力度或声部重要性。
- 相邻节奏组连贯取舍研究完成，分支`research/expression-coherence`。
  新可选研究参数`select_expression(..., coherence=.25)`惩罚相邻组的选中/
  落选切换，默认0，不接入应用设置。权重和边界规则先固定，未按结果调整。
  13项控制与相关检查共151 passed。固定6–10K默认逐音复现，新方案完整
  选择/定轨/导出回读通过；session `66354`实际exit 0，
  `test-data/expression-coherence-v1`。10K物件/LN仍1496/382，切换45→36，
  但10条消失中3条来自整组省略，另新增1条，不能称音乐更清楚。
  8张原声对照图已实际查看、24音频回读，session `22073`实际exit 0，
  `test-data/expression-coherence-audio-v1`。216–224秒持续层退出、孤立高音
  进入，不能凭更少切换启用默认。详见[结果](expression-coherence-results.md)。
  默认基线仍PR15的`chord-evidence-ties-v1`，勿将研究输出当发布行为。
  无运行中研究队列、推理、训练或新安装包；不再沿本版只调平滑权重。
- 同权重和弦起音排序修复完成，分支`fix/chord-evidence-ties`。
  Tsumugi的1.0为统一权重；原降层先按音高排除更贴近拍线的候选。
  现在同强度、有完整起音证据的音高行用原对齐评分打破平局；保留强度优先，
  无证据和瞬态行沿用原顺序，不统一选高音，不变造候选首尾。
  修复前7项新反例失败，修复后12项新检查及相关检查共138 passed。
  固定6–10K旧基线逐音复现，新选择、定轨与导出回读通过，session `28331`
  实际exit 0，产物`test-data/chord-evidence-ties-v1`。10K1499/373→1496/382，
  伴随12个鼓起音和1个贝斯起音落选；不能称只有和弦代表变化或音乐已改善。
  完整[结果与复现](chord-evidence-ties.md)，无新推理、训练、安装包。
  最新固定基线改用该目录；勿把旧1499/373当作修复后结果。
- 钢琴跨窗持续修复完成，分支`fix/piano-window-continuity`。
  原52.5秒突降定位为前后窗口的持续响应不一致；前窗包含起音且能继续跟踪。
  初版仅用frame延续造成5个踏板尾误差增加，未接入。最终还要求原尾恰在
  切换点、前窗以放音回归结束，保留起音与力度，接入可选钢琴专用检测器。
  缓存revision=2，保留`window_context_continuations`；无新安装包。
  SMD11处尾误差均更接近踏板；MAESTRO开发30处中29改善、1处3.91→14.77ms。
  物理按键参考有相反变化，1134标准按键区间1失1得，不能称松键识别改善。
  《樱花草》0变化，最初Liszt7个截断只恢复有回归支持的3个，不称全面解决。
  45项局部测试通过；22实录/550窗口应用重放、91单窗控制、160项开发计分
  独审通过，`test-data/piano-continuity-runtime-v1`。两版SMD各88数组/176计分
  独审通过。详见[完整结果与复现](piano-window-continuity-results.md)。
  采集session `51604`（SMD）、`32481`（Nova），评价`66157`/`97021`，
  开发重放`21351`、应用核查`41120`均实际exit 0。143+57不同原窗口已保存，
  正式队列复用定位的2窗，只新增198次网络前向，勿重跑。
  没有运行中推理或子代理；1155/736/687/841未读取。
- 钢琴低活动原声核查完成，分支`research/piano-gap-audio-review`。
  session `11495`实际exit 0，104处SMD和《樱花草》6处全部保存，
  `test-data/piano-gap-audio-v1`；预定10图已全部实际查看。独审exit 0、
  480文件哈希、110片段逐样本一致、34830项频带统计、7665帧直接FFT通过，
  `test-data/piano-gap-audio-audit-v1`。11新控制与相关检查共105 passed。
  2048/8192窗分别16/100处无完整间隔帧，记null；不能把跨界图当精确音尾。
  Liszt 52.500–55.140s在模型低活动期间仍见持续频谱成分；参考重叠中又有
  仅数毫秒的边界差异。不能统一称误休止，更不能把低活动等同于物理停音。
  完整[结果与复现](piano-gap-audio-results.md)。没有新推理、训练或默认行为。
  该分窗来源已进一步定位并修复其中有明确放音证据的情况，见上一条；
  保留全部原声核查产物，不重复本轮110处测量。
- 钢琴源模型低活动上下文验证完成，分支`research/piano-inactivity-context`。
  原Nova float PCM与应用缓存身份核对一致；1557候选起止音高一致，89处旧
  `frame_decay`与当前`frame_inactive`差异是既有语义修复，保留历史分析。
  原6个候选空隙中，第一个仍有持续激活，其余存在低活动段，不按空白直接
  推断静音。研究工具`scripts/piano_inactivity_context.py`，完整入口
  `scripts/check_piano_inactivity_context.py`。17项研究、5项独审控制与相关检查
  共122 passed。session `99800`实际exit 0，输出`test-data/piano-inactivity-context-v1`。
  《樱花草》新增5个模型支持边界、10个持续提案，6–10K各多5条钢琴LN，
  声部起音集合不变；6/8/10K另有1/2/1个other候选替换，不能说其他物件不变。
  11份SMD新增边界0、表达不变，不是有效休止准确率验证。随后补查104个
  原始候选空隙低活动区间，28/29个与按键/踏板参考持续重叠，20个全覆盖；
  不等于这些音已试听确认可闻，但不足以可靠自动判停音，因此不默认接入。
  全部重放后才读取SMD标签。6图实际查看、18音频回读；242文件及逐帧独审
  通过，`test-data/piano-inactivity-audit-v1`。完整[结果](piano-inactivity-context-results.md)。
  没有神经推理、新权重、安装包或运行队列，不重复本次已完成的16份重放。
- 连续起音表达的区间空隙保护验证完成；分支`fix/articulation-interval-gaps`，
  PR #11已合并，见上方提交与CI记录。
  旧实现两个反例失败，修复后20项专项与相关测试共216 passed。按原候选区间
  已结束至少80ms新增分组边界，未知音尾和更早仍持续的音阻止新增边界；
  不改变原起音、音高或音尾。完整协议见
  [区间空隙](articulation-interval-gap-protocol.md)。重放入口
  `scripts/check_articulation_interval_gaps.py`，session `36934`实际exit 0、输出
  `test-data/articulation-interval-gaps-v3`。最终6–10K选中音符、导出坐标均精确
  不变，11份SMD全部6122原候选/6070强候选的表达不变。没有新的音乐收益
  结论，完整[结果](articulation-interval-gap-results.md)。
  v1因检查脚本直接比较原JSON附加字段与Note失败，未完成
  生成；改为规范化Note比较后另开v2，不覆盖v1。v2全量重放完成，Beethoven
  9个原持续音被新短键化；增加失败反例后，v3限制只细分旧规则已短键化的组。
  原持续音不能变成新tap。初步《樱花草》表达没有变化，不把最小反例通过
  说成该曲LN改善。
- 音色分组35份已知乐器对照完成，shell session `66272`实际exit 0。旧首份
  复用，新34份串行耗时902.841秒；全部结束后读取本轮标签。不改原音高/起止。
  聚类后独奏类别起音/区间F1持平92.44%/87.74%；混奏82.44%/69.82%降至
  80.25%/67.76%。35份各为1个cluster；混奏唯一对应子集的声部分组成对
  precision仅27.94%，不能用作自动旋律身份。不接入、不对这版追加Nova
  全曲或按本批标签调阈值。完整[结果](instrument-family-validation-results.md)。
  入口`scripts/check_instrument_refinement.py`，产物`test-data/instrument-family-validation-v1`；
  21项查询/计分/分组测试及5项独立匹配控制通过。独审358文件、4790事件、
  840组匹配计数通过，回执`test-data/instrument-family-independent-v1`；未重建
  模型前向、聚类分区或等价匹配的误差均值。没有新默认行为或安装包。
- 修复同占用、同强度、同表达含义候选的去重顺序偏置，沿用原对齐评分择优，
  同分按ID；不同尾长、乐句代表和不同表达依据不作等价替换。25项专项及
  相关检查共114 passed。固定《樱花草》6–10K重放、定轨、导出回读通过：
  各键数160处候选代表变化，11对实际保留，全部other；物件/LN总数不变。
  6K/7K/9K分别额外2/1/2处对象取舍，8K/10K对象内容不变；不作音乐收益
  结论。详见[重复采音选择](expression-duplicate-selection.md)，新比较入口
  `scripts/check_expression_duplicates.py`，产物`test-data/expression-duplicates-v1`。
  119文件独审通过，回执`test-data/expression-duplicates-independent-v1`；
  8K/10K导出坐标精确不变，其他键数的重新定轨变化已在文档中列明。
- 原Nova other声轨已完成当前10K的三个固定规则选窗对照：200–208、
  96–104、16–24秒；3图实际查看、9份原声/起音click音频回读通过。
  186个other中65个量化移动超过30ms，62个向后、3个向前；这不是65个
  已确认错音。实际导出相对整数原候选的偏差>30ms为23个，中位+1ms；
  区分原候选、制谱提前、落线三种时间，不统一调Offset。
  无新推理或生产改动；见[原声对照](other-onset-current-review.md)，入口
  `scripts/review_current_other_onsets.py`，产物`test-data/sakura-other-onset-current-v3`。
- 鼓组整体落选时，只在剩余容量内补充一个已有瞬态起音，固定原方案全部
  ID及起止。89项局部检查通过；固定《樱花草》6–10K完整重放、定轨和导出
  回读通过，6K补5鼓、7–10K补4鼓，LN不变。最终产物
  `test-data/drum-group-fallback-v3`，见[结果与边界](drum-group-fallback.md)。
  不再运行旧 `check_expression_current_replay.py` 来生成“当前”基线；它冻结
  了改动前选择器AST。当时比较入口为 `scripts/check_drum_group_fallback.py`；
  本轮去重修改后的比较改用 `scripts/check_expression_duplicates.py`。
- Harmonica 第三方复现的固定对照入口：全部35份已有 URMP 输入及40秒静音。
  安全载入真实权重已通过，116个存储张量逐值核对、679444参数、6个解码
  控制通过。原bf16失败后显式FP32完成全部输入及独立复核；独奏起音F1
  89.32%／区间75.92%，混音56.99%／21.17%，均低于Tsumugi。萨克斯独奏
  有改善，不作整体替换。不是论文作者权重，未查到源码许可，未打包。
  见[协议](harmonica-reproduction-protocol.md)及[实测结果](harmonica-reproduction-results.md)。
- MuScriptor 公开 GGUF small 已下载并核 SHA，逆名称/布局转换为 safetensors；
  正向转换与回读逐值相同。原 Python 模型严格载入122个状态张量通过，
  102441984参数。15个纯符号事件控制和6个计分控制通过；保留完整token、
  人为封口、错误区间与原始类别。全部35真实输入及静音已完成并独审，
  独奏起音/区间F1为76.32%/58.70%，混音62.63%/41.17%，低于Tsumugi；
  最后一块缺EOS和异常事件完整保留，不接入、不追加这版Nova整曲推理。
  见[完整结果](muscriptor-small-results.md)。
  见[权重准备](muscriptor-gguf-preparation.md)和
  [比较协议](muscriptor-gguf-transcription-protocol.md)。
- Harmonica/MuScriptor均为研究工具，未改变应用默认，没有新训练或新 Release。

## 已完成的固定队列（勿重跑）

原 shell session `69438` 已完成 Mega53 分离及下游采音，但在Harmonica首个
静音控制的CPU bfloat16 LSTM处报错，已实际exit 1；当时MuScriptor尚未启动。
显式float32兼容入口已修复，新串行session `54376`已完成Harmonica全部输入
及MuScriptor全部36输入、评分，实际exit 0。音色分组session `66272`也完成
全部35输入及评分，实际exit 0；当前没有待运行的神经推理。
session/PID 仅用于本次会话定位，重开后先
确认命令行和启动时间，不能凭旧PID启动、结束或重复任务。

1. 分离已完成：原session `42354`、Python PID `15668`已退出。
   `scripts/check_mega53_long_context.py --output test-data/mega53-long-context-v1`
   使用作者20秒窗和拼接规则，共22份PCM、44次前向。日志：
   `test-data/mega53-source-review-v1/long-context-run.log`。独立波形复核通过，
   见[长窗结果](mega53-long-context-results.md)；不能据波形启用默认分离。
2. 下游采音已完成：应用 `.venv` 运行
   `scripts/check_mega53_transcription.py --separation test-data/mega53-long-context-v1 --output test-data/mega53-long-transcription-v1`。
   日志：`test-data/mega53-source-review-v1/long-transcription-run.log`。
   46个转录流程/64用途完成，两个完整来源的整体起音/区间仍低于原混音；
   铜管起音比旧短窗恢复但完整区间进一步下降。结果见上述长窗报告。
3. Harmonica原精度失败记录：音频 `.venv-audio` 运行
   `scripts/check_harmonica_transcription.py --output test-data/harmonica-urmp-v1`。
   日志：`test-data/harmonica-source-review-v1/inference-run.log`。
   错误为 `could not create a primitive descriptor for the LSTM forward propagation primitive`。
   保留原目录，未完成任何输入的完整转录，不能发布声学成绩。
   新运行：`--precision float32 --output test-data/harmonica-urmp-fp32-v1`，
   日志 `test-data/harmonica-source-review-v1/fp32-inference-run.log`。
   原源码未改，只在模型forward_spectrogram内禁用autocast，实际四头dtype
   逐batch保存；详见[精度修订](harmonica-reproduction-protocol.md)。
4. MuScriptor已完成：应用 `.venv` 运行
   `scripts/check_muscriptor_urmp.py --weights test-data/muscriptor-recovered-small-v1/muscriptor-small-recovered-f16.safetensors --output test-data/muscriptor-urmp-v1`。
   日志：`test-data/muscriptor-cpp-review-v1/urmp-run.log`。

上述完整结果均已生成。失败的bf16目录仍保留，不覆盖或当作成功。
两新模型均在各自全部推理结束后才读取逐音参考，原输入与预测没有按标签清洗。

唯一子代理 `/root/piano_stream_exports` 负责静态审核和独立复算，
不运行神经模型。已完成的独审目录：`test-data/mega53-long-independent-v1`、
`test-data/mega53-long-transcription-independent-v1`、
`test-data/muscriptor-independent-v4`、`test-data/harmonica-fp32-independent-v1`。
MuScriptor审计v1–v3假设问题与修订回执保留，最终v4通过；4个等价匹配
下的音尾均值差异另列，不能称全部均值精确相同。没有改动模型或原预测。

## 已知结果与下一步

- 原AMT分类评分完整对照已结束。跨窗类别本来大多稳定，均值只有极小
  类别起音收益且不修音尾，不作为声部显著度或LN依据。下一步回到实际
  Nova声轨的持续层/插奏取舍，特别是已看过的216–224秒持续层退出情况；
  不继续在本批调置信度阈值，也不追加本版全曲分类推理来替代表达改进。
- 连贯取舍研究已实跑并保留全部结果。更少切换可能来自省略短插奏，下一步
  要补声源证据与段落角色的依据，不能按模型类别连续性直接给声音重要性。
  原三窗other不变，所有变化other的分箱已看；勿重复本轮8图或重跑6–10K。
- 同权重排序已在完整固定输入验证。2026-11-13只读复核其冻结JSON：
  三个原选窗的other候选选择变化分别为2/20/4项；96–104秒只覆盖12个落选鼓起音中的2个。
  全部12个落选鼓候选当前均因`group_capacity_or_density`省略，分布在9.908–12.340、
  100.719–101.124及138.016–139.637秒；其中5个起音时间与新other起音重合，
  2个与新bass起音重合。六个8秒窗可覆盖`changes.json`全部27条增删起音记录；
  见[静态定位及候选表](chord-evidence-ties.md)。没有播放或解码音频样本，
  未读取Reference或运行模型；制作之前任何取舍音乐影响均未验收。
- 本地原混音听审包已生成于ignored目录
  `test-data/sakura-chord-evidence-audio-review-v1/`：6个固定8秒窗口、27条增删起音、
  63条候选选择变化及14个PCM16片段。`review_key=13f7bf27e2445f8a69a1`，
  manifest SHA-256为
  `8bf989eacb607cb9a86c4c7e8a943a7d143a7c05cdbc42dd394ce32a4e24a6b2`；
  原混音PCM SHA-256仍为
  `dfd2e1581124db8255a347c8414411d1978e279d06116603df64dcfa60e8745e`。
  页面逐窗提供原混音和适用的差异click，使用固定增益，不滤波、不重采样、
  不移动时间、不峰值归一化。听审包构建不运行神经推理、不读取Reference。
  `tests/test_chord_evidence_audio_review.py`为5 passed；三个新增Python文件
  `py_compile`、页面内联JavaScript语法和14个片段的PCM16格式/采样数/哈希校验通过。
  这些仅验证听审材料完整性，不代表已开始人工听审；声音判断、导出标签复核和
  osu!试打仍待完成，不以click或类别标签代替音乐判断。
- 2026-11-13只读续接核查：本地`127.0.0.1:8766`听审页仍可访问，页面显示
  `已标记 0 / 27 条起音`；预期导出文件
  `sakura-chord-evidence-audio-review-v1-labels.json`在审核目录和Downloads均不存在。
  `127.0.0.1:8765`训练集审核服务仍在监听；该页此前实际显示`0 / 39`，本轮未改动
  标签。三个预期导出文件（含`vocal-phrase-review-v1-labels.json`）均未出现在Downloads。
  人声候选审核页使用本地`file://`地址，当前界面工具拒绝读取该页；浏览器本地暂存状态
  未知，不能据此推断未标注。此次未播放音频、操作标记或导出，音乐验收仍未完成。
- Mega53旧6秒与作者长窗均未支持默认接入；长窗44次分离前向及46个下游
  转录流程已完成，两个完整来源仍不及原混音。见[长窗结果](mega53-long-context-results.md)。
- 鼓组修复后的《樱花草》固定候选10K选择为1499物件/373LN，已完成定轨
  和`.osu`回读，仍不是试听验收。改动前1495/373的诊断留在
  [历史规则重放](expression-current-replay.md)。
- 新去重规则的10K仍为1499/373，11对来源代表替换，起止/音高/声部内容
  不变。6K/7K/9K另有少量全局取舍；保持原候选，不把代表评分提高说成补采音。
- MuScriptor small本轮结束；八类乐器汇总两项均不及Tsumugi，不继续调参
  或对这版跑Nova。Harmonica萨克斯正例也不支持全局替换。
- 下一步仍从原Nova other三个固定片段定位实际采音与段落表达问题。
  已证实的去重偏置已修，不重复模型搜索或继续用其解释全部省略；不能把
  65个大移动当作65个已确认错音，不能为减小移动量统一调Offset。
- 有限 LN 覆盖研究不接入默认。下一阶段先为表达层设计可解释的起音损失
  检查，按声部、段落和高价值候选审查与 LN 收益之间的取舍；冻结选择结果
  本身没有逐音真值，不能把全部基线候选硬性锁定。
- 音色分组验证已收束：混奏退步且全部单组，不沿这版默认分组继续调参。
  下一步以原候选和可解释的局部旋律/节奏连续性定位段内表达，避免把模型
  广义类别直接当独立声部；仍需原音和生成结果检查，不以标签稳定度代替。
- 区间空隙保护已完成固定重放。它解决可复现的分组影响，却没有改变
  《樱花草》的最长钢琴起音组；未知重弹尾仍使分组缺少完整休止证据。
  下一步需要结合原声和现有采音上下文定位实际分歧，不把`reattack`直接
  改成已观测音尾，也不重复跑本次已完成的16份对照。当前无神经推理队列。
- 已补查钢琴低活动上下文；不因《樱花草》多5LN就默认启用。SMD空隙与
  按键/踏板延续的冲突原声核查已完成，结果见本阶段新增。区别衰减后继续
  按住与实际可闻持续仍需谨慎，不能把上述频谱检查说成亲耳试听；
  不能把模型低活动或单一整轨RMS当作物理休止或独立旋律标签。
- 四排序列先验仍是独立研究入口，只在固定起音、音符和LN尾端上排序轨道，
  不负责采音、声部选择或决定LN长度，也未接入默认生成。新增窗口级LN/tap
  覆盖、失败负例和最大和弦诊断，并为缓存加入特征名SHA256与显式v2版本。
  验证：`.venv\Scripts\python.exe -m pytest -q tests/test_pattern_sequence.py tests/test_sequence_prior_training.py`
  共24 passed，`py_compile`通过。
  15谱smoke共抽样120窗、留存120窗、负例失败0；歌曲组预算后进入数据数组
  109窗，二次缓存读取的数据和报告完全一致。缓存位置
  `test-data/corpus-expanded/sequence-window-cache-v2-root-smoke-20261111`。
- 全量快照清点为7,907张6–10K谱、2,023个歌曲组；7K占5,777张，8K/9K
  分别只有251/236张，且8K验证/测试各8组、9K验证/测试6/3组，指标需单独
  谨慎解读。没有《樱花草》样本。默认窗口上限估算约596,236窗，数组约
  1.27GiB，拼接峰值估算约2.54GiB；运行时可用内存约6GiB、D盘余量28GiB。
- 全量 sequence-window v2 预处理在约3,000/7,907张谱时因 Loved 谱
  `1625602` 的异常红线 `beatLength=1,000,000,000ms` 产生 float16 溢出而中止；
  没有进入拟合或导出权重。此前 v2 缓存保留在本地，但后续不复用。
- v3 timing guard 已完成验证：专项测试28 passed，脚本和测试 `py_compile`
  通过，`git diff --check`通过。坏图 `1625602` 抽样80窗、留存80窗，忽略
  134条异常红线，正负特征均有限。15谱、15缓存文件的v3 smoke合计238窗、
  111个LN上下文窗，二次读取数组和报告完全一致；缓存留在
  `test-data/corpus-expanded/sequence-window-cache-v3-root-smoke-20261111`。
- 局部窗口特征忽略非有限、非正或超过10,000ms的红线周期，继续使用先前有效
  红线；若无有效红线则采用500ms fallback。该研究过滤不改写谱面Timing。
- timing guard 已通过 PR #23 合入，merge commit `ddb430b7cab8a6c8319e8fd4824edbcf9458b437`；
  两项 Windows validation 均成功。PR #24 随后记录了该阶段验证。
- 启动前资源快照为可用内存6.11GiB、D盘余量53.35GiB；按预处理估算峰值
  约2.54GiB，再加Python和训练开销；全量任务已结束，无活动训练进程。
- 全量v3预处理和四排窗口先验训练已完成，命令为
  `.\.venv-audio\Scripts\python.exe scripts\train_sequence_prior.py --threads 2 --cache test-data/corpus-expanded/sequence-window-cache-v3 --output test-data/sequence-prior-v1`。
  共7,907张谱，最佳轮次20，导出最大误差`2.29e-5`。10K逐窗口测试为
  7,672个排序对、77.05%准确率；LN为75.13%，tap为78.57%。训练报告和模型
  留在ignored本地目录，不提交源码或Release。
- 新增歌曲组等权评价入口`scripts/evaluate_sequence_prior_groups.py`，结果与协议
  见[四排序列先验结果](sequence-window-prior-results.md)。10K测试覆盖12个独立
  歌曲组（11组含LN）、48张图及3,836个窗口；歌曲组宏平均整体77.55%
  [74.57%,80.68%]、LN 77.61% [73.09%,83.02%]、tap 78.89%
  [75.97%,81.76%]，区间为固定种子的95%组级percentile bootstrap。
  9K测试只有3组、其中2组含LN，应视作探索性结果；8K也只有8组。
- 上述指标只说明模型能在固定起音、音符和LN尾端下，对局部可行轨道布局
  做高于随机水平的排序；不证明整谱键型、LN长度或音乐质量已提升。
  `sequence-window-v1`权重格式仍不兼容现有`manialab.pattern_prior` runtime。
  beam去重上下文修复已合并：`manialab.layout._state_signature`现纳入
  `sequence_rows`；`tests/test_sequence_prior_runtime.py`覆盖不同历史布局必须保留为
  不同状态。固定候选整谱A/B也已按歌曲组验证，未显示足以默认启用的平衡收益；
  不重复这些状态修复或已完成的强度对照。当前下一步是完成人工原声听审包的标注，
  再结合段落和声部角色复核候选，并做`.osu`回读与实际osu!试打。序列先验仍默认关闭，
  当前没有因此启用默认行为；便携Release按dev.12当前main对齐任务单独验证。

## 起音保留账本

- 2026-11-13新增`scripts/summarize_onset_retention.py`及
  [协议](onset-retention-ledger-protocol.md)、[结果](onset-retention-ledger-results.md)，
  只读四份冻结的《樱花草》10K JSON，不运行模型、选择器或音频处理。专项测试
  `tests/test_onset_retention.py`为3 passed；全量pytest为2,318 passed、89 skipped，
  `compileall`、输入SHA、scope候选覆盖和`git diff --check`通过。最终报告在ignored目录
  `test-data/onset-retention-ledger-final-20261113/report.json`，SHA-256
  `94b85ca856eccd3555fa02d218b520fd819c5979ac331889e93a5d5ccad2f96d`。
- 6,053条分析候选中4,623条进入定轨，1,496条被选中；定轨候选内3,127条由
  `duplicate_pitch`、`represented_by_chord`、`group_capacity_or_density`省略。
  另有1,430条在进入定轨前被`weak_detection`或`voice_not_selected`排除。
  按段落×声部×原始起音时刻去重后有5,276组；`other`的2,164条类别行对应
  1,533组，其中232组带多个模型类别标签。类别、强度与省略原因都不是声源真值、
  置信度、可闻性或音乐价值；未计算准确率或质量指标。
- v1–v4中间输出均保留在ignored目录，不覆盖；同毫秒跨段落归组及ID一致性断言均由最终
  报告覆盖。没有读取音频、参考谱、MAESTRO保留测试，
  没有音乐/Timing验收或生产默认变化。下一步仍待人工听审导出，再结合原声复核重点
  起音；账本只用于定位，不代替试听和osu!试打。

## 表达取舍静态审计

- 2026-11-13新增`scripts/summarize_expression_tradeoffs.py`及
  [协议](expression-tradeoff-audit-protocol.md)、[结果](expression-tradeoff-audit-results.md)。
  只读冻结的《樱花草》10K分析、默认选择和执行计划JSON；不运行选择器、模型或音频
  处理，不读取Reference，不改默认行为。三份输入SHA在协议和机器报告中固定。
- 现有1,496个选中对象中，导出的382个LN分布在381个段落/声部/原始起点组。
  另有1,886条被省略候选满足当前`min_ln=240ms`表达阶段的持续音资格条件；
  按省略理由为`group_capacity_or_density` 1,370、`duplicate_pitch` 316、
  `represented_by_chord` 200。折叠同段落、同声部、同原始起点的模型类别后为
  1,357组，其中1,308组当前没有同起点的已选LN。它们不是可直接加回的音符或质量提升。
- 声部组数：other 1,253组、bass 76组、piano 28组。ignored报告位于
  `test-data/expression-tradeoff-audit-v1-20261113/report.json`，SHA-256
  `c6fa144d3d03e320d6f227b6fa0844d8e6b26d99388126a242a50206b5858732`；
  逐组区间只标出相对当前同声部LN的未覆盖分析时钟范围，不是可加时长或音乐价值排名。
- 验证：`tests/test_expression_tradeoffs.py`为4 passed，全量
  `.\.venv\Scripts\python.exe -m pytest -q`为2,322 passed、89 skipped；
  `compileall -q manialab scripts tests`、`py_compile`及`git diff --check`通过。
  未做人工试听、osu!试打、Timing或音乐验收。下一步应把结构队列作为日后人工复核
  的定位工具；不得依审计排序直接调声部选择或LN默认值。

## dev.16 版本发布

- 当前`main`为`43a2f0a3c15cb837dcab4e93ced2da693a476832`，最新已发布
  `v0.1.0-alpha.6-dev.15` tag为`ead17d5e07f2936025272463dc1a5d56678e4923`。
  tag之后的主线变更为PR #83发布记录和PR #84表达取舍静态审计；没有应用运行逻辑、
  默认采音或模型变化。
- 准备dev.16只同步`manialab.__version__`、README当前下载入口、文档索引和发布说明，
  以使新tag/Windows包源码快照准确对齐合并后的main。源码中的新审计CLI不进入默认
  应用路径。Demucs checkpoint单独再分发授权仍未核实，发布说明继续披露该风险。
- 本地验证：`.\.venv\Scripts\python.exe -m pytest -q`为2,322 passed、89 skipped；
  `compileall -q manialab scripts tests`、版本导入`0.1.0-alpha.6-dev.16`及
  `git diff --check`通过。PR #85的Windows验证通过并合入`main`，merge commit为
  `42e5493ad17456d6d8b37c334162e226203126a9`；annotated tag
  `v0.1.0-alpha.6-dev.16`精确指向该提交。
- Windows tag CI run `43363747645`于2026-11-13成功，耗时6分46秒：2,411项测试中
  2,322 passed、89 skipped、零失败/错误；模型校验、worker/Windows便携包构建、
  clean PATH与Unicode数据路径离线EXE、便携分析/导出/取消/重启、worker关停及
  验证报告上传均通过。CI报告明确标记没有音乐或Timing验收，测试夹具为合成音频。
- 工件留在ignored目录`build/portable/verified-dev16-43363747645/`。Windows ZIP为
  698,958,003字节，SHA-256
  `e8c2f10eacc4b2dec0e0600f525c05b914b9a3fc91f15ede9dd6696d1bc22dfc`；本地4,230项
  全量CRC通过且无重复路径。包内73项源码清单均与tag文件内容一致；本机
  `manialab/planner.py`为CRLF而CI清单记录LF，按既有换行等价校验复核通过。
- GitHub Pre-release已发布：
  [ManiaLab alpha.6 dev.16](https://github.com/zhyzx35607/ManiaLab/releases/tag/v0.1.0-alpha.6-dev.16)。
  ZIP、SHA256SUMS和验证JSON三个远端资产的大小及digest与本地逐一相符：
  ZIP 698,958,003字节、SHA-256
  `e8c2f10eacc4b2dec0e0600f525c05b914b9a3fc91f15ede9dd6696d1bc22dfc`；
  SHA256SUMS 226字节、SHA-256
  `f9ee09ba685de12514f553f5db2e0ed5a828fc49d3a6b841f6fa1ee5deffc59a`；
  验证JSON 1,992字节、SHA-256
  `262be03d47a408c1a31ac71f85a81043f05d39138effc1a4a6f3bd0408b344d5`。
  固定模型随包；Demucs checkpoint单独再分发授权仍未核实。
- 本版没有音乐/Timing质量验收，也没有改变默认采音行为。人工听审、候选标签复核
  与osu!试打仍未完成；用户暂时没有耳机，本阶段不把静态检查或合成音频替代为试听结论。
  后续在不重复模型实验、不改写未验证默认行为的前提下，继续可离线验证的结构/工具工作，
  并保留人工音乐验收为开放目标。

## dev.16 后续主线与 checkpoint 许可范围复核

- 2026-11-13续接核对：工作区从 `main` 创建本阶段文档分支前干净，
  `main`/`origin/main` 均为 `94c442ec13cd734b9ae205eee662f18ef05909b3`。
  最新 Windows Pre-release 仍是 dev.16，tag 固定在 `42e5493ad17456d6d8b37c334162e226203126a9`；
  PR #87 已将 LN 互动特征批量编码集成到当前主线，PR #88 补记合并状态。
  #87 只优化研究评估运行时间，不接入默认生成，也不构成键型或音乐质量改善。
- GitHub 记录的 #87 验证为序列定向测试112 passed、4 skipped，全量
  `pytest` 为2,340 passed、89 skipped，`compileall`、定向 `py_compile`、
  `git diff --check`通过；两项 Windows `verify`通过。没有重跑模型对照、
  读取 test split、音频或本地研究产物。
- checkpoint许可复核只读取模型清单、torchaudio 2.9.1随包源码及上游许可讨论，
  没有读取权重内容或修改包内文件。四声部 `hdemucs_high_trained.pt` 的官方
  torchaudio bundle说明训练数据包含MUSDB-HQ训练/测试集及150首Meta内部歌曲，
  并指向原HDemucs训练仓库；该bundle文档/源码没有声明此checkpoint的单独许可。
  Demucs issue #327 中维护者 `adefossez` 说明当时的预训练权重不受代码MIT覆盖、
  仅供科学用途，但答复未点名torchaudio文件。2024年另一位贡献者在同一讨论中表示
  MUSDB训练的模型应仅用于研究；这与torchaudio披露的训练来源相关，但仍不是精确到
  `hdemucs_high_trained.pt` 的授权。该讨论中维护者的答复早于六声部v4 checkpoint
  `htdemucs_6s.th`，后续贡献者评论也没有列出该文件；两者都是重要风险证据，
  不是这两个具体文件的单独授权文本。
  截至本次核对，两个随包checkpoint的单独再分发授权都未核实；这不是侵权结论，
  也不能从Demucs MIT或torchaudio BSD-2-Clause推导模型权重许可。
- 复核命令：`gh pr view 87 --json body,commits,files,reviewDecision,statusCheckRollup`、
  `gh pr checks 87`、`gh release list --limit 5`、
  `Get-NetTCPConnection -State Listen -LocalPort 8765,8766`。
  当前 `127.0.0.1:8765` 和 `:8766` 仍是原有只读审核页HTTP服务，分别服务
  `test-data/expression-audio-cohort-v3-stem-review-v1` 与
  `test-data/sakura-chord-evidence-audio-review-v1`；未改标签或导出。
  本次没有发现神经推理任务，也没有试听；用户暂时无耳机，音乐与Timing验收仍开放。
- 本阶段只更新第三方声明和本进度文档，不改应用、模型、音频、审核标签或默认行为。
  dev.17按既定开发测试版目标沿用dev.16中相同的固定模型文件，继续明确披露
  checkpoint授权未核实；这不代表取得了授权，也不扩展模型范围或用途。
  商业化或新增/更换模型权重的公开分发，仍须先核实权利人许可或调整打包方案。

## dev.17 版本发布

- 2026-11-13从干净的 `main` `67c7fb761bdede3a4d09dd1f3a89c76631cae2e9`
  准备dev.17。版本准备PR #90两项Windows `verify`通过并合并，merge commit、
  annotated tag `v0.1.0-alpha.6-dev.17` 与发布时 `main` 均精确为
  `3a5f0be09a27613d220e5fa91590863d5d7a2db0`。
- dev.16 tag之后的功能性差异为PR #87序列先验LN互动特征批量编码及分组验证优化；
  它是研究评估工具，不接入默认制谱，不重跑已完成的14组对照，不表示模型或音乐质量
  提升。PR #86/#88/#89为发布、合并状态和第三方许可范围文档记录。
- 只读检查dev.16 ZIP模型文件清单，确认包内含
  `beat-this-final0.ckpt`（81,058,141字节）、
  `hdemucs_high_trained.pt`（334,697,255字节）和
  `htdemucs_6s.th`（54,996,327字节）；`scripts/build_windows.py`将完整`assets`
  目录打入包并逐项校验模型SHA。dev.17继续沿用此范围，许可风险如实列入发布说明，
  不声称授权已解决。
- 当前版本准备仅变更`manialab.__version__`、README下载入口、文档索引及发布说明；
  不改应用逻辑、默认行为、模型权重、音频或研究产物。人工听审、Timing与osu!试打
  仍未完成，不能由CI代替。
- 本地验证：`.\.venv\Scripts\python.exe -m pytest -q`为2,340 passed、89 skipped
  （71.39秒）；`.\.venv\Scripts\python.exe -m compileall -q manialab scripts tests`、
  dev.17版本断言及`git diff --check`通过。未运行模型推理或读取音频/参考谱。
- Windows tag CI run `43373546222` 于2026-11-13成功，耗时7分14秒。JUnit记录
  2,429项测试、2,340 passed、89 skipped、零失败/错误；tag/source版本检查、
  模型校验下载、隔离音频环境、worker与Windows ZIP构建、clean PATH及Unicode路径
  离线EXE、分析/导出、取消/重启、worker关停、验证报告和工件上传均通过。
  报告说明合成音频夹具不构成音乐或Timing验收。
- 工件下载并独立复核于ignored目录
  `build/portable/verified-dev17-43373546222/`。正式Windows ZIP为698,968,541字节，
  SHA-256 `082c33977d6b2c8bc7d950f4240ff52bc879ade2cea1da0a9a275651b670d97b`；
  4,232项全量CRC通过、无重复路径。三个模型SHA分别与 `assets/deep-models.json`
  一致；包内EXE SHA为
  `55c7b2a65dfead84710e0455117e5d6cf3e0d9d44e32afd6921c8d3940357ede`，
  release manifest、源码版本及CI验证JSON相符。
- GitHub Pre-release已发布：
  [ManiaLab alpha.6 dev.17](https://github.com/zhyzx35607/ManiaLab/releases/tag/v0.1.0-alpha.6-dev.17)。
  三个资产名称、大小和digest均与本地复核一致：ZIP 698,968,541字节，
  SHA-256 `082c33977d6b2c8bc7d950f4240ff52bc879ade2cea1da0a9a275651b670d97b`；
  SHA256SUMS 226字节，SHA-256
  `4c97bc3a66105a58cd0bc4b8b99a8fd69b7b47da07eddf4a875c12cbfbf3cd8b`；
  验证JSON 1,992字节，SHA-256
  `d9f8a148cb5fe59d72d2820d2ccff44a1343cbc0d5d07722eb15496ab53a42c8`。
- 此版继续沿用dev.16的固定模型范围，checkpoint单独再分发授权未核实；这不是已获
  授权或商业用途声明。PR #87仅为研究评估加速，不表示准确率提升。人工听审、
  Timing和osu!试打仍未完成，用户暂时没有耳机。

## 分段表达偏好与 CPU 并发修复

- 2026-11-13续接后直接核对当前 `main`、进程和用户反馈：源码干净并与
  `origin/main` 同步在 `c574dc7`，dev.17是最新已发布预览；没有活动神经推理。
  用户明确要求由项目自动音频分析继续推进，不把逐首人工听审设为前置条件。
- 修复 `propose_expression` / `expression_evidence` 对段落局部设置的漏读：自动声部方案
  现在按所在段落的 `ln_ratio` 和声部权重选择表达，实际音符及人声乐句候选也按局部
  `min_ln` 和权重过滤。此前这些逻辑部分读取全曲值，会令段落建议与生成输出不一致。
  增加不同段落长短倾向、最短LN门槛、本地静音和人声乐句门槛的回归测试。
- 基础 Basic Pitch ONNX 会话的算子线程从3改为项目规定的2，跨算子线程保持1；新增
  配置测试。不改检测候选、默认全曲偏好、音尾推断或模型权重，也不重放已完成的模型
  研究和《樱花草》全量候选实验。
- 验证：`.\.venv\Scripts\python.exe -m pytest -q` 为 **2,344 passed、89 skipped**
  （60.46秒）；`compileall -q manialab scripts tests`、全部静态JS `node --check`
  和 `git diff --check` 通过。测试包括音频与生成逻辑的既有合成夹具，没有运行研究训练、
  读取本地音乐Reference或改写忽略的逐曲产物。这里验证的是段落设置生效与运行线程上限，
  不是音乐质量验收。
- PR #92 的两项Windows `verify`均通过（run `43431529253`、`43431591372`），已于
  2026-11-13合并；merge commit和`main`为
  `2264e327ccdc206ab5571efb59556857a770f0d2`。

## 局部方案生成 API 回归

- 2026-11-13在PR #92合并后追加端到端回归：通过`/api/projects/<id>/generate`和实际
  `.osu`导出运行固定的合成12秒分析，包含跨段落不同的LN倾向及最短长度。导出中第一段
  为短键、第二段为LN且结束时间正确，证明局部设置经过项目推荐计划、自动声部方案、
  选择、定轨并到达最终谱面，不只是单测函数返回值。
- 验证：`tests/test_server.py`为15 passed；专项生成用例为1 passed；`compileall`、
  全部静态JS `node --check`和`git diff --check`通过。该回归不运行神经推理，不读取真实
  音乐或Reference，只检查真实应用路径中的候选格式和表达配置。
- PR #92与本端到端回归都不证明现实音乐中的候选正确、Timing可靠或谱面可玩；下一步继续
  检查固定音频分析经过完整生成路径时的声部和段落表达，保留用户试听作为独立验收，
  但不要求逐首听审才能继续开发。

## 冻结分析推荐生成诊断

- 2026-11-13使用已有`scripts/evaluate_recommendations.py`，将冻结的《樱花草》分析
  经真实Flask项目API依次运行`recommendations`、`apply-recommendation`、`generate`、
  `.osu`导出和解析回读。输入为283.84秒、6,053个候选、13个结构段落；分析文件SHA-256
  `26a0250568b2ce149022c4aa1549af2d9dd49b045372959a6b26b5342ba5313a`，分析内检测拍线
  标记为`stable=false`。
- 命令：
  `.\.venv\Scripts\python.exe scripts\evaluate_recommendations.py test-data\sakura-tsumugi-expression-onset-v3\analysis.json --output test-data\sakura-recommendation-generation-v1 --keys 10 --bpm 74 --offset 178 --shift 22.5`
  使用诊断时钟74 BPM、chart offset 178ms、audio shift 22.5ms、4分拍内细分；脚本开启
  吸附并显式确认该时钟。没有读取真实音频、Reference或MAESTRO保留测试，没有重新运行
  音频采音/分离/转录或研究训练。生成使用应用既有轨道先验（SHA-256
  `a7b60de4c9d8d4d86c9c16cc4b79001607fc6c069bfb6a752f743de1847fb12c`）与编排配置
  （SHA-256 `f5ec85ffdd85afc6712577e263f8540a795631d9c513b74d05e13fc84160c29c`）。
- 三种10K推荐均生成成功；API校验有效、导出再解析与保存音符一致、原始候选证据未变。
  程序统计如下，手部压力和审查提醒是内部结构指标，不是质量分：

  | 推荐 | 物件数 | LN比例 | 分段物件/秒范围 | 手部压力峰值 | 结构提醒 |
  | --- | ---: | ---: | ---: | ---: | --- |
  | melody | 1,050 | 63.1% | 1.55–4.17 | 6.076 | 无 |
  | hybrid | 2,088 | 39.1% | 1.69–9.60 | 10.970 | 无 |
  | layers | 2,965 | 44.5% | 2.40–13.33 | 15.721 | `held_hand_load` 3项、短连打37处 |

- 这组数据下`melody`是较保守的自动候选：密度和内部手部压力最低、没有结构提醒；
  `layers`最密且出现负载提醒。它们只用于程序侧比较，不证明哪一版更符合音乐，也没有
  改变应用默认值。所有版本报告`off_grid=0`仅因为开启了吸附，不能作为Timing通过：
  原始候选中5,085个起点被移动，1,338个候选原始起点距网格超过30ms；三个输出分别仍
  有90、167、343个被选起点移动超过30ms。诊断时钟的24个12秒窗口中有5个可疑。
- 输出保留在ignored目录`test-data/sakura-recommendation-generation-v1/`。报告SHA-256
  `10a8eddd5e3df4a3ae6455a591d8a17e0ef505a8963d64db48d719ac379fd1f0`；三个诊断`.osu`
  SHA-256依次为`10K-melody.osu`
  `6c5e91f7391b07a7baa23e65d0ca2167b218dbe05f2abc9ac619d76717dde2f6`、
  `10K-hybrid.osu` `a3c4d998d3248f72245a6fa5818fc49758678a7aca3bbb2d67903072f37cf01d`、
  `10K-layers.osu` `89bcfe34e5a3563725b22f665f9e7b77e07ccb4c4ce0cfa548523911f39972b9`。
  运行使用analysis-only占位音频且未导出音频包，不能直接作为可播放谱面交付。
- 下一步从冻结的`timing_attacks`继续做自动拍线稳定性诊断，重点区分半速候选与偏移、
  分段覆盖问题；在Timing未有可靠证据前，不把吸附后的整齐谱面当作验收结果，也不要求
  用户逐首试听才能推进代码和自动评估。

## dev.18版本准备

- PR #94已于2026-11-13合并到`main`，merge commit为
  `562e32be4f9a3fcdf12a830be17b8e29845fcf5a`；两项Windows `verify`均通过。
  最新已发布版仍为dev.17，tag早于PR #92/#93及本次进度记录。
- dev.18准备只同步`manialab.__version__`、README下载入口、文档索引和发布说明，
  使便携包快照覆盖分段表达修复、Basic Pitch线程限制和生成API回归；不增加模型、
  改默认生成偏好或打包本地用户数据。冻结分析诊断仅作为程序侧证据，不宣称音乐提升。
- Windows包继续使用既有固定模型范围；`THIRD_PARTY_NOTICES.md`记录的checkpoint
  单独再分发授权风险仍未核实。
- 本地全量`pytest`为2,345 passed、89 skipped；`compileall -q manialab scripts tests`、
  全部静态JS `node --check`、dev.18版本导入断言和`git diff --check`均通过。
- 版本准备PR、tag CI、包体复核及GitHub Pre-release待完成。
## 2026-11-16 项目目录与 Git 清理

- `docs/README.md` 和 `scripts/README.md` 已有目录/入口索引，因此没有迁移
  `docs/`、`scripts/` 或实验文件；历史命令、复现协议和测试导入仍依赖这些路径。
- 本机开始整理时有72个本地分支：68个分支已包含于`origin/main`，4个未合并。
  经逐个确认提交可从`origin/main`到达后，清除了67个已合入的本地旧分支；保留
  `main`、当前dev.18 PR分支、3个其他未合并分支和本次独立清理分支。不删除远端分支。
- 清除18个可再生的`__pycache__/`和`.pytest_cache/`目录。检查发现根目录
  `dist/ManiaLab/`不是空缓存，而是dev.4便携构建快照，含钢琴和吉他ONNX、乐器worker
  与发布manifest，因此保留。`.venv/`、`.venv-audio/`、`build/`、`release/`、
  `test-data/`、本地模型目录和`ManiaLab.spec`也均保留。
- GitHub 于本日核对时，最新预发布为dev.17，dev.18版本准备PR #95仍打开且Windows
  `verify`通过。该状态只记录核对时点，不表示dev.18已打tag或发布。

## 钢琴专用模型与发布包范围审计

- 只读对比本地保留ZIP中央目录及发布manifest，未解压或读取模型权重。alpha.6
  dev.2、dev.3、dev.4包含`piano-regression.onnx`；dev.4 ZIP为1,354,151,023字节，
  piano manifest非空。alpha.5起该文件未稳定进入包。
- dev.7 ZIP为698,916,155字节，dev.17 ZIP为698,968,541字节；两者均保留
  `manialab/piano.py`、Piano detector前端配置和ONNX Runtime，但都没有
  `piano-regression.onnx`，发布manifest的`piano_model`为`null`。dev.17相对dev.7
  仅增52,386字节，不能据此说近期每版都持续删减钢琴功能；更准确地说，CI包一直缺少
  专用权重，因此依赖该模型的钢琴专用采音不可用。
- 差异来源是打包输入范围不固定：`assets/deep/`和`assets/instruments/`被Git忽略；
  `scripts/fetch_audio_models.py`只获取`assets/deep-models.json`中的固定三种音频模型，
  Windows workflow没有获取钢琴/吉他模型或构建乐器worker；`scripts/build_windows.py`
  则在本地可选模型存在时才验证并打入包。dev.4还有2,667项`assets/instruments/`
  内容、吉他模型和乐器worker，dev.7/dev.17没有这些可选组件。dev.4到dev.17体积差为
  655,182,482字节，主要可见差异包括instrument子树压缩后约351.5 MiB、压缩后的钢琴
  模型139,908,297字节和吉他模型88,306,215字节；这不是全部差额的完整分摊。
- 因此当前首先是“源码保留功能入口、CI便携包没有可选权重”的发布可复现性问题，
  不是最新源码彻底删除钢琴分析。若后续决定恢复该能力，应单独确定模型下载/校验、
  许可和便携验收策略；本阶段不下载、生成或打包模型，也不改变应用功能或默认行为。

## 《樱花草》Timing 半速/倍速静态诊断

- 2026-11-16 续接后确认工作树原为干净 `main`，HEAD 与 `origin/main` 同为
  `ba3c512`；没有发现 ManiaLab 推理或 pytest 任务。既有 `127.0.0.1:8765`、
  `:8766` 音频审核页服务保持运行。冻结输入
  `test-data/sakura-tsumugi-expression-onset-v3/analysis.json` SHA-256 与记录一致：
  `26a0250568b2ce149022c4aa1549af2d9dd49b045372959a6b26b5342ba5313a`。
- 新增 `scripts/diagnose_sakura_timing_ambiguity.py`：读取该 JSON 已存的
  `tempo.comparisons` 与 `timing_attacks`，不调用 `estimate_timing`；对 74、148、
  296 BPM 固定网格计算拍点、半拍、细分及 12 秒局部支持，比较折返相位。
  脚本校验输入 SHA，输出目录存在时拒绝覆盖。新增协议、结果文档和测试；不改
  `manialab/timing.py`、Timing 默认算法或稳定性门槛。
- 诊断显示应用原选 148 BPM/204ms，得分 0.7100；296 BPM 为 0.7047，74 BPM
  为 0.6397，三者均有重叠的网格支持。鼓轨 148 BPM 有 4/24 个可疑窗口，集中
  于 0–36 秒及 192–204 秒；相位误差范围约 -14.25 至 +4.22ms。Offset 跨来源在
  同候选拍速折返后相差约 1–2ms，未显示明显内部全局相位冲突，但绝对物理 Offset
  未验证。半拍/拍点覆盖成对相近，不能仅靠起音定强拍或听感 BPM；`stable` 仍为
  false，生成确认门槛不变。
- 协议、结果页分别为
  [诊断协议](sakura-timing-ambiguity-protocol.md)和
  [诊断结果](sakura-timing-ambiguity-results.md)。机器报告保留在新的 ignored
  目录 `test-data/sakura-timing-ambiguity-v1-20261116/report.json`，SHA-256 为
  `b56d3d8a3dffe5936a98a4b64f0a0d40dc4b911d576d12af6aa9ee326568122e`。报告固定输入
  SHA，记录脚本 SHA `660fcb3b487d5292d2301d96af49d1c561cb14635e14dc3502cb56594c7b4341`
  和 `manialab/timing.py` SHA
  `6e8bf724db50aafd70af4fee9260d4bdee5ac6917c0063fb29fb6c854cb40e2b`。
  定向测试 `tests/test_diagnose_sakura_timing_ambiguity.py` 为3 passed；
  全量 `pytest -q` 为2,348 passed、89 skipped（75.56秒），
  `compileall -q manialab scripts tests`、`git diff --check`通过。提交`3d9f001`，
  PR #97 的两项 Windows `verify` run `43795303419`、
  `43795336438`均通过。未读取音频、Reference 或保留测试；未运行模型、重新估计
  Timing 或生成谱面。

## 恢复钢琴模型的 Windows 包复现

- 2026-11-16 从干净 `main` `2663084` 新建分支
  `fix/restore-piano-model-package-20261116`。仅检查既有进程和产物：没有活动的
  ManiaLab 推理或 pytest；`127.0.0.1:8765`、`:8766` 仍是原有静态音频审核页服务。
  `assets/deep/piano-regression.onnx` 与 `assets/piano-model.json` 记录一致，大小
  154,214,195 字节、SHA-256
  `7488435a431ba2863de16c56be24f3626dc75e38cc22f3103e193a677494fafa`。
- 新增固定来源的 `scripts/fetch_piano_model.py`：下载并校验 Zenodo 原始 checkpoint、
  固定上游源码归档及许可证证据，使用锁定的 PyTorch/ONNX/NumPy 依赖在 CPU 上从全零
  合成张量导出图；输出目录存在或身份不符时拒绝覆盖。以忽略目录中既有、经哈希核验的
  权重/源码隔离复建，结果与本地 ONNX 字节级一致；未读取歌曲音频或谱面，未运行音乐模型
  推理。checkpoint SHA-256 为
  `c3fa9730725bf4a762f1c14bc80cd5986eacda01b026f5a4a2525cd607876141`，源码提交为
  `0226e74cbc805660e34bbd6a8fed2083890ebb88`，模型许可记录为 CC-BY-4.0。
- Windows 便携构建现在将此钢琴 ONNX 作为必需资源，并核对模型、来源元数据和随包许可；
  workflow 在便携构建时从固定来源复建模型。实际便携 EXE 验证增加 ONNX Runtime CPU
  零输入烟测，要求输入/七路输出契约、输出形状及有限值正确；发布报告核对烟测与包内
  manifest 身份一致。该烟测只验证打包模型可以在 CPU 上加载和运行，不是音频或音乐质量
  验收；模型仍为显式选择、默认关闭。
- 第三方声明已明确钢琴模型随 Windows 包分发但默认关闭。依赖锁增加固定导出依赖。
  本地全量测试为2,360 passed、89 skipped；`compileall`、全部静态 JS `node --check`
  和 `git diff --check`通过。Windows 干净 runner 的来源下载、模型复建、EXE/ZIP、
  离线便携检查及发布验证报告尚待 workflow_dispatch 验证。PR #98 已创建；第一次
  `workflow_dispatch` run `43800741956` 的 attempt 1、2 都在既有
  `Fetch verified audio models` 步骤获取 `Beat This! final0` 时遇到上游 HTTP 500，
  钢琴模型复建步骤尚未运行。
- 为解除该既有下载阻塞，增加了固定 revision 的校验后备用镜像。镜像仓库固定提交
  `f864b9916e54ab70a6dfae0f7fbe8e37cd7fa68a`；本机新建 ignored 验证目录后下载的文件为
  81,058,141字节，SHA-256 与现有 `assets/deep-models.json` 官方权重身份一致
  （`8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331`）。获取脚本仅在
  原地址失败时尝试该镜像，并仍执行相同字节数及 SHA-256 校验，因此不改变打包模型内容。
  将随钢琴模型恢复阶段一并提交并重新触发完整 Windows 验证。
- PR #95（dev.18 版本准备）仍为 open，当前 GitHub 状态为 `DIRTY/CONFLICTING`；最新
  预发布仍为 dev.17。后续先验证并合并本阶段 PR，再把 #95 对齐当前主线后另行验证发布。
  此阶段不重跑已完成的模型实验；钢琴音频质量、Timing 和真实音乐验收仍未完成。
- 后续核对 PR #98：workflow_dispatch run `43801902532` 已通过全量测试、固定音频模型
  获取和隔离依赖安装，但生成的钢琴 ONNX 在 SHA 校验时失败；按校验顺序，它的大小已
  符合 `154,214,195` 字节，实际哈希没有记录。PR 检查 run `43801833398`虽通过测试，
  但便携构建步骤因不是手动/发布事件而全部跳过；本阶段仍没有 CI ZIP 或真实 EXE 烟测
  证据。不得因此放宽固定模型身份。本次诊断改动让错误显示期望/实际大小与 SHA，并在
  导出校验失败时生成小型 ONNX 节点/初始化器哈希摘要；workflow 仅上传该摘要（7天），
  不上传模型权重。下一步重新运行完整 Windows workflow_dispatch，根据摘要定位字节差异，
  然后再决定确定性导出修复。

## 钢琴 ONNX 跨机器确定性修复

- 继续分析 run `43804348688` 的导出诊断：旧本地基准 ONNX 与 CI 图均为154,214,195字节、
  566个节点和222个 initializer，但CI中107个 initializer payload 与基准不同，差异集中
  在Conv/BatchNorm融合权重。按名称比较会被不同导出路径误导，因此另按数据类型、shape
  和payload SHA多重集核查。
- 首轮高精度 Decimal 融合试验没有匹配基准（111/222个payload不同），已弃用其算法；
  原试验文件留在ignored目录，不覆盖。以固定上游checkpoint直接对照后，改用锁定NumPy
  的明确 float32 运算顺序：`float32 gamma / sqrt(float32 variance + epsilon)`，之后以
  float32计算卷积权重和偏置。该公式的56组Conv/BatchNorm权重与偏置都逐位匹配本机旧图。
- 两次独立的新目录导出均生成154,223,955字节、SHA-256
  `31c5d8c87fd8cccaf98b85d8c97821c587c52c2495800ba3c41925912f06ad0f`。新旧图的222个
  initializer `(dtype, shape, payload SHA-256)`多重集相同，566个节点的顺序、名称、算子
  与属性一致；值名称/连接的序列化不同，因此文件身份独立固定，不放宽SHA校验。
- ONNX Runtime CPU（算子线程2、跨算子线程1）对旧图和新图分别运行全零输入及固定种子的
  合成输入；七路输出形状均为四路`[1,1001,88]`和三路`[1,1001,1]`、所有值有限，逐元素
  bitwise equal。该项只证明导出等价与运行契约，不是钢琴音频或音乐质量验收。
- `assets/piano-model.json` 与 `manialab/piano.py` 当前固定身份更新为上述新 SHA/大小。
  原本机ignored ONNX仍保留旧SHA `7488435a...`，未覆盖；历史研究与dev.2验证文档中的旧
  SHA继续保留，以准确描述它们实际使用的文件。诊断artifact名增加`github.run_attempt`，
  避免同一run重跑时重名。
- 本地验证：`tests/test_piano_packaging.py`为13 passed；全量`pytest -q`为
  **2,363 passed、89 skipped**（59.95秒）；`compileall -q manialab scripts tests`、
  全部静态JS `node --check`、JSON解析及`git diff --check`通过。未读取用户音频、参考谱或
  MAESTRO保留测试，未做音乐质量验收。
- 新身份由`fetch_piano_model.py`严格校验接受（`already_verified`）；ignored原模型仍保持
  旧SHA `7488435a...`，本次没有覆盖本地模型，也没有把任何 ONNX/音频/研究产物加入Git。
- 提交`616363e`推送至PR #98。Windows `workflow_dispatch` run `43808668376`于
  2026-11-16成功，耗时11分14秒：全量测试、固定音频模型下载、隔离依赖、钢琴ONNX复建、
  音频worker、Windows ZIP、clean PATH + Unicode路径实际EXE离线验证、关停检查、报告和
  工件上传全部通过。报告记录2,452项测试（89 skipped、零failures/errors），钢琴模型
  CPU smoke输出契约通过；便携夹具为合成音频，音乐/Timing验收标记为false。对应 PR
  `verify` run `43808625349`也对同一提交成功；push run `43808620246`因同组
  `workflow_dispatch`触发并发取消而结束为cancelled，不是测试失败。
- 手动run便携包下载到ignored目录`build/portable/verified-piano-43808668376/`并独立复核：
  ZIP 888,431,751字节，SHA-256
  `11a75d9273f3575acad77c11171100038cb627c9798b0636a6fe5e48e5055029`；4,280项CRC全过、
  无重复路径。包内钢琴ONNX为154,223,955字节、SHA-256
  `31c5d8c87fd8cccaf98b85d8c97821c587c52c2495800ba3c41925912f06ad0f`，与随包清单及验证
  报告一致；EXE SHA-256 `5f07de6ebb095c73ca5409c7fece0d49235df6a9a34d5c8210e624fe6497044d`
  与验证报告一致。validation JSON把源码提交记为`616363e`、版本记为dev.17，符合当前
  PR尚未包含dev.18版本准备的状态；这只是验证工件，不是Release。
- 模型重建、ZIP与实际EXE/关停验证已满足合并门槛；合并PR #98后，dev.18 PR #95须重新
  对齐新`main`并另行完成版本准备验证。钢琴真实音频质量和音乐/Timing验收仍未完成。

## dev.18 PR 重对齐

- PR #98 已于2026-11-16合并到`main`，merge commit为
  `7280819529ff139e111ede02d834d4756b729028`。手动便携验证来源提交`616363e`，完整
  模型重建、ZIP、EXE、关停检查及独立包体复核详见上节；这不代表音乐质量验收。
- 为继续PR #95，在本地`codex/prepare-dev18-20261113`分支合入最新`main`，仅
  `docs/PROGRESS.md`发生内容冲突；保留两边阶段记录，并更新 README 和 dev.18 发布说明，
  把PR #98恢复的钢琴模型能力及默认关闭/验收限制准确纳入版本范围。
- PR #95 已完成 Windows 验证并合并；其 main merge commit 为
  `dbc4c04ac1dda528b6ac637ff432f79fe154bbd6`。后续精确 tag、tag CI 与包体验证结果见
  下方“dev.18 tag CI 与便携包复核”。
- 对齐后的本地全量`pytest -q`为2,363 passed、89 skipped（55.24秒）；
  `compileall -q manialab scripts tests`、全部静态JS `node --check`、dev.18版本与新钢琴
  模型身份断言、`git diff --cached --check`均通过。PR head Windows 检查、tag CI、最终
  便携包独立复核和 Pre-release 均已完成；真实音乐、Timing 与人工试听验收仍开放。

## dev.18 tag CI 与便携包复核

- PR #95 已于2026-11-16合并，merge commit 为
  `dbc4c04ac1dda528b6ac637ff432f79fe154bbd6`。PR head 的 Windows `verify` run
  `43887395734`与 push run `43887382182` attempt 2 均通过；后者全量测试用时1分31秒。
- annotated tag `v0.1.0-alpha.6-dev.18` 精确指向上述 `main` 提交。Windows tag CI
  run `43901369558`于2026-11-17完整通过，耗时11分04秒：tag/source 版本检查、2,452项
  测试（89 skipped、零 failures/errors）、固定音频模型下载、钢琴 ONNX 确定性复建、
  worker 与 Windows ZIP 构建、clean PATH 和 Unicode 路径离线 EXE、worker 关停、报告及
  工件上传全部成功。报告的合成音频夹具只验证软件与便携运行，不是音乐或 Timing 验收。
- run `43901369558`的便携产物下载并独立复核于 ignored 目录
  `build/portable/verified-dev18-43901369558/`。ZIP 888,433,245字节，SHA-256
  `d148f42f0b4b057bb44caab35a14343aca88acb200c5168f3d3105e90d5350cc`；4,281个条目 CRC
  全过且无重复路径。包内 EXE SHA-256 为
  `99e91298bf4efe396f7d076adc53a1c2f0e0a0c1fd2be51c0f5628d3ff36be27`；钢琴模型为
  154,223,955字节，SHA-256 `31c5d8c87fd8cccaf98b85d8c97821c587c52c2495800ba3c41925912f06ad0f`。
  报告的源码提交、版本、模型身份均与 tag 和包内 manifest 一致；SHA256SUMS 两项均匹配。
- GitHub Pre-release 已发布：[ManiaLab alpha.6 dev.18](https://github.com/zhyzx35607/ManiaLab/releases/tag/v0.1.0-alpha.6-dev.18)，
  附上述 Windows ZIP、SHA256SUMS 与机器验证 JSON。远端资产 digest 与本地完全一致：ZIP
  `d148f42f0b4b057bb44caab35a14343aca88acb200c5168f3d3105e90d5350cc`、SHA256SUMS
  `7794b51a328021a31cb48b55771aadeefa1603e30f5cbade70923de467e61496`、validation JSON
  `4f606b869465ddce4349cd81b5e59787edd944de49f9e44ff6d568f889cea05e`。原本机 ignored 模型
  与研究产物未覆盖或提交。
- 当前 release 说明和进度只记录构建/运行验证；音乐质量、Timing、人工试听与 osu! 试打
  仍未验收，钢琴模型默认关闭，Demucs checkpoint 单独再分发授权仍未核实。


## 2026-11-17 第一阶段：应用分层与数据可靠性

- 从审查基线 `d2e6f0d` 开始，确认工作区干净；只有既有 8765/8766 听审静态服务运行，没有模型任务。未读取用户音频或 MAESTRO 保留测试。
- 制谱编排移入 `manialab/generation.py`，HTTP 路由只校验请求并提交任务；新增 `preferences.py` 明确原音频时钟和制谱时钟。音符新增可选 `audio_start`，原分析不改写，旧数据继续可读。
- 修复提前量、零点裁剪和量化跨段后误用声部/表达偏好的问题。音符策略按原音频起点查询；占轨事件按换算后的制谱区间查询；跨段音符量化合排时容量取较紧值。覆盖 0、22.5、正负 50ms、无结构旧分析、两种定轨器及 API 实际导出。
- 导入拒绝跨输入及大小写重名覆盖和文件/目录冲突；嵌套原谱导出将补传音频放入原相对引用位置。声轨全请求预校验，失败/取消只清理本次未提交文件及本次新建解码集合；已成功集合保留。异常使用持久日志，并记录项目/任务标识。
- README 聚焦当前使用入口，历史实验归档 `docs/research-history.md`；新增 `docs/architecture.md`，修正快速开始的旧分段描述与构建缺失的钢琴准备步骤。研究脚本及协议原路径不变。
- 首轮全量 `python -m pytest -q`：2,395 passed、89 skipped（63.74秒）。之后补全跨上传目录冲突与合排容量检查，相关组合回归122 passed；`compileall`、静态 JS 语法与 `git diff --check`通过。最终全量为 **2,398 passed、89 skipped（66.54秒）**；以 PR head CI 为合并依据。
- 本阶段是软件可靠性修复，不声称提升真实音频采音准确率或完成音乐验收。下一阶段：组件身份缓存、构建能力清单、前端候选状态和可复用浏览器回归。


## 2026-11-17 第二阶段：组件契约与浏览器回归

- 第一阶段 PR #101 已合并，main merge commit `9784525101c33743c77d94b43649a8da4c37d8dd`；head `1f1da33` 的两项 Windows verify 均成功。
- 后续独立复核发现更窄的容量边界：前段 LN 量化到下一段开始时，原段偏好可能覆盖下一段更低的 max_held。本阶段将实际持续占轨约束独立按制谱时间计算，加入原5990ms→6000ms边界、两条LN而下一段上限1的回归。
- 钢琴状态验证大小/SHA，区分 missing/mismatched/verified/读取错误；文件签名变化或30秒过期重新校验，推理前强制校验。模型来源选择支持当前完整SHA命名的版本文件，历史模型原路径不变；构建规范为包内固定文件名。
- 本机从已验证的 `build/piano-model-deterministic-20261116-c-work/piano-regression.onnx` 复制到 ignored `assets/deep/piano-regression-31c5d8c87fd8cccaf98b85d8c97821c587c52c2495800ba3c41925912f06ad0f.onnx`，API状态与构建校验均通过。原模型仍为SHA `7488435a431ba2863de16c56be24f3626dc75e38cc22f3103e193a677494fafa`，未覆盖；没有重新导出或进行真实音频推理。
- 新增 `scripts/package_assets.py`：standard/extended 明确构建范围，源码资源按清单暂存；额外历史模型不进入包。extended必须具有已验证吉他与Tsumugi组件，不能因本地目录碰巧存在就改变标准包。manifest、随包帮助、发布验证共享能力声明。
- 修复目标键数改变后的旧候选导出/编辑状态；预览继续使用候选自己的键数。普通 PR 增加锁定 Playwright 合成回归，覆盖音频生成→换键数→编辑→重开→导出，以及原谱严格保时；不需要私人语料、Edge或模型下载。
- Mapperatorinator 的 OMP/MKL 统一2线程，实际运行器显式设置 PyTorch 算子2、跨算子1。主环境无torch的进程生命周期夹具显式提供线程契约替身；真实模型运行仍要求原独立环境。
- 本地专项68 passed；两条真实Edge浏览器回归通过（14.8秒），原有静态服务不受影响。一次全量捕获生命周期夹具没有torch的问题，修正夹具后11项进程/原模型测试通过；最终全量 **2,405 passed、89 skipped（67.48秒）**。完整Windows构建结果待追加。
- 文档补充 `docs/development.md`，当前乐器能力表对齐dev.18，历史实验继续保留。以上为工程验证，不代表采音或音乐质量通过验收；本阶段不创建新的Release标签。
- `a330647` 的完整 Windows run `44004762861` 已成功：Python 2,405 passed、89 skipped（59.29秒），Chromium 两项通过（8.7秒），固定钢琴复建、worker/EXE/ZIP、离线 clean PATH、Unicode 数据路径及关停均通过。CI 记录 ZIP 888,441,706 字节、SHA `49136f93b69ee89447a980c1c5329cbabd9b8c35b4eb517f09abba101f98e617`。小型报告与完整日志下载至 ignored `build/structure-20261117-validation/ci-44004762861/`；未下载整包作本机独立复核。该记录仅对应这个源码提交。
- 合并前独立复核发现 worker 子树仍使用 `copytree`，可夹入额外文件；本地目录未发现私有文件，但旧校验只覆盖 EXE，不能证明依赖树未改变。补充两个 builder 的完整 `worker_files` 清单，暂存前后与包后验证文件集合、大小、SHA，并拒绝链接；实际复制只遍历已核验清单。Tsumugi 运行验证前后也核对依赖树，报告绑定完整清单，不能沿用另一组 DLL 的旧报告。历史清单缺失时要求重建，不修改旧产物或给它补写身份。
- 新增回归覆盖 worker 额外私有文件、复制期间新增文件、依赖内容变化、缺失文件、不安全路径和旧运行报告复用；相关构建专项32 passed。此补充尚需新的完整 Windows 验证；前一个成功 run 不作为新提交已完成打包验证的证据。
- 补充后的本地全量 `.venv/Scripts/python.exe -m pytest -q` 为 **2,416 passed、89 skipped（59.46秒）**；`compileall -q manialab scripts tests` 与 `git diff --check`通过。两次独立代码复核均已结束；当前没有模型队列，仅保留原有8765/8766静态服务。
- 最终代码 `6ff49743e440551368e8538eaa1f4153fd06d1bc` 的完整 Windows run `44006822579` 于2026-11-17成功，普通PR run `44006831028`同样成功。完整run记录2,416 passed、89 skipped（70.20秒），Chromium两项通过（9.6秒）；固定钢琴复建、2,360个worker运行文件清单核验、标准包EXE/ZIP、离线clean PATH、Unicode路径、6–10K导出、失败取消和关闭停止worker全部通过。
- CI ZIP为888,687,430字节、SHA `b56110625d473ade02f124f684d7c5abba8727670ea65bc378f3d2b3216c39a6`，EXE SHA `250d721f5848ec7353523c6f754a08b74e9351dcf70b25d056ff2dbe7710a7a9`。小型报告、JUnit与完整日志已保存到 ignored `build/structure-20261117-validation/ci-44006822579/`；未下载整包作本机独立复核。后续提交仅补文档与验证结果，不更改已验证的应用/构建代码。本轮没有创建Release，已发布dev.18仍是先前tag产物。
- 当前进一步研究入口：冻结多曲起音的Timing偏差/歧义、按段落声部的省略原因、LN音尾/占轨回归；见[检查与优化记录](project-review-20261117.md)及[架构入口](architecture.md)。扩展包新清单契约、真实音乐质量、Timing与最终osu!试打仍未验收，不重跑已完成神经模型队列。

## 2026-11-17 第三阶段：冻结多曲回归与容量边界

- 从 PR #102 合并后的 `85b1d6a1d30b7aa063cc2eb714706023774a4c84` 新建
  `codex/frozen-generation-regression-20261117`。沿用10个已知开发片段的 revision 8
  分析，未读取其音频或参考谱，没有运行模型；默认26例和分段40例在改生产代码前保存。
- 新增 `scripts/replay_frozen_generation.py`：核对计划、输入、基线和源码SHA，串行调用
  真实生成与导出回读，保存声部/段落省略、LN、Timing门槛、输出容量及逐ID差异。
  旧目录拒绝覆盖，所有逐音产物保留在ignored `test-data/`。
- 10个定向用例首先失败，证明两类不同问题：结构化路径只在本组起音检查自身可行性，
  因未来低持条边界或其他声部候选时刻而整组省略；无结构路径只在起音约束LN，漏掉
  后续段落的低持条上限。现在结构化路径先冻结实际可选音集合，再使用最终求解器同样的
  起音/段落检查点判断是否按起音拆组；无结构路径把段落起止加入LN约束。
- 不改变纯边界密度语义，不裁短保留LN来绕过占轨；旧路径保持已选起音。补充原音频段
  量化合排、永久裁去的和弦层不能收紧其他声部、边界放键和正负提前量回归。
- 本地全量 `.venv/Scripts/python.exe -m pytest -q` 为 **2,440 passed、89 skipped
  （58.06秒）**；Edge两项浏览器流程通过（14.0秒）；`compileall`和`git diff --check`
  通过。独立复核157项相关测试通过，并用500组合成条件核对组内可行性与完整事件检查一致。
- 两次对照均完成，默认26例及三推荐30例的ID、起止、轨道和证据全部不变。10例压力配置
  中4例变化：新增23个、删除15个ID，共享音起止和证据不变，432个共享音重新定轨。
  压力取舍同时包含人声/鼓点减少，完整披露于[结果](frozen-generation-regression-results.md)，
  不以增加LN或求解目标值代表听感改善。全部66例结构、网格、输出容量和导出回读通过。
- 产物：`test-data/frozen-generation-fixed-20261117/`（报告SHA
  `88e24c5b64a7daf6074acfedd3fcb34726f8ceb05ab49f350efccadb821948f7`）及
  `test-data/frozen-generation-sections-fixed-20261117/`（报告SHA
  `67a77fe2d354ae5be800613706e947b49f2cf5a2f8ef8ebe649082bb6682525a`）。
  运行命令见协议：两个固定计划分别使用 `--baseline` 指向同名baseline目录，线程上限2。
- 自动Timing门槛仍仅3/10片段接受；诊断显式确认时钟不等于自动成功。全部选中LN均缺
  历史音尾来源字段，明确记作unknown；分段累计大起音移动为5,078→5,080，不声称采音改善。
  PR #103 head `cc605035a14eea283f1555e6598bbba716eda576` 的Windows
  push run `44012919492`、PR run `44012999021` 均通过Python与Chromium检查，
  于2026-11-17合并为 `9c80fc347283377389ca2b651e64a1e32d12d0bd`。
  本阶段未重新构建便携包或发布新Release。继续审查Timing确认与诊断状态。

## 2026-11-17 第四阶段：Timing诊断与波形时钟

- 从PR #103合并后的main创建 `codex/timing-diagnostics-clock-20261117`。
  已复现：5个原属12–24秒窗口的起音因22.5ms提前量重新分窗，原可疑窗口消失；
  再次检查/拟合/生成丢失原鼓轨与混音补查来源；1个起音返回零检查段却没有证据不足状态。
- 后端新增共享 `analysis_timing_diagnostics`：原音频起点固定窗口归属，当前制谱时间
  计算残差；重算质量，仅继承来源标签；返回不足窗口、可评估状态和完整时钟身份。
  API检查、重新拟合和生成报告使用同一入口，自动Timing接受条件不变。
- 冻结10首已有片段各检查0、22.5、50、-50ms，40例全部满足窗口归属不随提前量改变。
  对照PR #103的22.5ms报告，8首边界计数得到修正，可疑窗口数均不变；未读音频或参考，
  未运行模型。产物 `test-data/timing-diagnostic-clock-20261117/report.json`，SHA
  `c9f2352c11567fe52777796204c76d860fe8ad9a7a9b874f2a731392f7888cc6`。
- 前端统一波形原始/制谱时钟转换，候选叠加采用候选自己的提前量，切换显示保持同一物理位置；
  BPM/细分/半速/倍速编辑保留未舍入Offset锚点。完整时钟与请求序号共同使旧诊断失效，
  原分析结论明确标注原始参数，证据不足单独显示；旧候选可读并提示重查。
- 复核另复现手填93.5ms被后端取整94ms后前端静默丢弃响应，以及参考偏好载入恢复137 BPM
  草稿后误显示120 BPM旧诊断。检查/拟合前按后端规则显式取整并提示；恢复草稿后核对诊断身份。
  正负93.5/94.5ms、延迟响应、旧项目、保存重开、波形起止叠加与对齐均有浏览器回归。
- 默认26例再次运行真实生成与导出回读，ID、起止、轨道及证据字段全部不变；报告位于
  `test-data/frozen-generation-timing-fixed-20261117/report.json`，SHA
  `e93c35b05963308fc38a93c7f46c86f88934ab6b2ff9c59f158b91bf43296880`。
  命令与范围见[Timing诊断结果](timing-diagnostic-clock-results.md)。
- 本地Python全量 **2,454 passed、89 skipped（65.87秒）**；真实Edge浏览器全组
  **12 passed（28.0秒）**，运行 `npm run test:ui`，环境
  `MANIALAB_TEST_BROWSER=msedge`、`MANIALAB_TEST_PORT=8978`。静态JS语法、编译与差异检查通过。
  未重新构建便携包，未改变自动Timing接受门槛；下一阶段复用完整歌曲已有候选检查6–10K声部与LN。
- PR #104 head `1bdf9543fb751ee40412012d1529e85107a5a603` 的Windows push
  run `44015311356`、PR run `44015375676` 均通过，于2026-11-17合并为
  `9bf9f5347e728b2a9b3c2dbda1d803b4e6015e42`。没有创建新Release。

## 2026-11-17 第五阶段：LN来源快照与完整歌曲

- 从PR #104合并后的main创建`codex/ln-evidence-review-20261117`。完整歌曲冻结计划
  `test-data/frozen-fullsong-plan-20261117/plan.json` SHA
  `749dc371b73cbf52406fa52e6ef58a89f0cb5d57bf89e034eecc643d6ea334ee`：
  revision 13《樱花草》既有6,053候选，默认/历史无人声设置各6–10K；未读音频、Reference或保留测试，未跑模型。
- 原始10例生成、导出回读和容量全部通过，基线报告位于
  `test-data/frozen-fullsong-baseline-20261117/report.json`，SHA
  `2670de5f1b491b149b10bcab4ca2de881499b4ea1b936495646a737e5754039d`。
  人工74 BPM/178ms/22.5ms时钟仅用于诊断，原自动Timing仍不接受。
- 默认10K有434条LN，其中74条尾移超过30ms；原界面只列起音移动。另复现量化塌缩乐句
  仍声明代表内部候选、手动新增被称为检测、表达尾被称为物理检测尾的语义问题。
  新`evidence.py`为生成候选保存检测与表达的原音频快照，编辑继承，局部锁定物件使用旧快照。
  旧候选来源不足明确未知，不按重用ID借用新分析。
- 乐句塌缩时清除最终乐句标志，内部省略理由不再宣称由不存在的LN表达；没有自动添加音符。
  前端检查器和谱面检查独立到`static/evidence-review.js`，起音/尾端分别提供前12项定位，
  使用候选自身时钟和对应声轨；人工添加、人工修改、未知来源和模型预测明确区分。
- 完整10例对照已通过，选中ID、起止、轨道及既有证据字段全部不变；全部15,032条输出
  的原检测快照与冻结分析逐字段一致，表达时间的提前量换算与既有`evidence_end`相符。
  对照报告`test-data/frozen-fullsong-evidence-20261117/report.json` SHA
  `b824bce54286d46259fff1b178062fe1748ddf4dceaaf067e5cae2c1014318b0`。
- 快照实现首轮全量Python **2,459 passed、89 skipped（66.99秒）**；真实Edge全组
  **15 passed（35.2秒）**，新增3项证据展示/定位流程均先复现失败再修复通过。
  JS语法、`compileall`、`git diff --check`通过。具体字段、命令及范围见
  [LN来源与全曲复核](ln-evidence-review-results.md)。准备完整Windows便携构建验证。
- 独立复核后补局部重写保护：同ID重分析跨重写边界时拒绝与锁定旧物件合并，避免重复ID和
  错绑快照；仅为锁定ID保留原人工编辑记录；新快照字段存在但缺ID时明确未知，不降级借用
  本轮整首分析的新决策。前两项已由真实生成/编辑失败回归复现，专项44 passed；补充后全量
  Python **2,461 passed、89 skipped（56.72秒）**。上述整谱对照保留原源码身份，没有覆盖重跑。
- 旧格式局部候选也补充区域外来源保护：真实API生成`[8978,9978)`ms区域后，模拟缺少
  整个快照字段的历史项目；下界新生成音可使用自身决策，上界锁定音及其他区域外音明确未知。
  修复前复现错误借用2310ms新音尾，修复后定向通过；最终真实Edge全组
  **16 passed（36.5秒）**，使用`MANIALAB_TEST_BROWSER=msedge`、`MANIALAB_TEST_PORT=8978`
  执行`npm run test:ui`。Python编译、JS语法、差异检查通过。
- 提交`b7bfb090f44d0c7ce84ca14e15581a9c909c35aa`已创建PR #105。普通Windows
  run `44018594628`通过：Python2,461 passed、89 skipped（72.85秒），Chromium
  16 passed（33.1秒）。完整便携run `44018546073`成功：Python2,461 passed、89 skipped
  （67.28秒），Chromium16 passed（31.0秒）；固定钢琴复建、2,360个worker文件、标准ZIP、
  实际EXE离线clean PATH/Unicode路径、6–10K导出、失败取消、重开和关停检查通过。
- CI记录ZIP为888,717,778字节、SHA
  `7e82f3781e591b0c67076a2986a1b1f9dc767f301f1e488e161b9a8040f3ca3c`，EXE SHA
  `042e123c478f924f4e7e753f175a0bd6f61fc23ffd43290381163e10641e294e`。小型报告与完整日志
  已下载至ignored `build/structure-20261117-validation/ci-44018546073/`并复核；没有下载
  整包作本机独立检查。PR #105已合并为`5d0ecb392d8e1d498706f5a56682b561b793c1b8`。
  本包对应`b7bfb09`源码，版本字符串仍为dev.18；不是原发布tag的资产，没有创建新Release。

## 2026-11-17 第六阶段：复核定位与Timing草稿恢复

- 基于PR #105提交创建`codex/review-navigation-clock-20261117`。LN事件按钮把制谱
  时间直接传入波形细节，实际Edge复现原音频显示偏22.5ms、原谱候选切换音频草稿后偏50ms。
  现在用候选自身时钟转换波形坐标，播放跳转与试听继续使用实际导出时间；两个真实浏览器
  回归先失败后通过，并核对真实试听请求、波形采样窗口与光标。
- 保存声部方案、段落边界、采用本地建议三个入口均复现：恢复137 BPM/50ms草稿后，仍显示
  120 BPM候选的旧诊断。共享`captureTimingDraft`/`restoreTimingDraft`保留字段、显式确认
  和未舍入Offset锚点，失效不匹配诊断及波形标记；参考偏好载入也使用同一入口。
- 独立复核补出编排建议派生状态：保存方案后，临时载入120 BPM发出的建议不能用于137 BPM
  草稿，实际采用被后端拒绝。恢复后重新请求建议并使旧请求失效，两个失败场景修复通过。
  相同时钟保留诊断、恢复后直接采用建议、提前量再次改变仍保持原Offset锚点均有回归。
- 定向LN两项通过（9.9秒），草稿恢复三项通过（12.2秒）；最终真实Edge全组
  **21 passed（51.6秒）**，JS语法和`git diff --check`通过。
  没有修改生产Python、运行模型或重放历史音频实验。API编排计划与清除计划的草稿保持路径
  尚待下一小阶段核对；音乐质量和最终试打未验收。
- PR #106 head `4f5f4956644a15ec28623ae9840407e8dd3abf09`的Windows push
  run `44020084238`及PR run `44020109376`均通过，已合并为
  `f393f49408080cad59bd0f5b3d17c334ee8b4fca`。push记录Python2,461 passed、89 skipped
  （57.69秒），Chromium21 passed（38.6秒）；日志保存在ignored验证目录。
  本阶段没有重建便携包，前述完整便携证据只对应PR #105提交。

## 2026-11-17 第七阶段：API计划与清除计划的草稿保护

- 基于PR #106提交创建`codex/plan-draft-state-20261117`。实际浏览器复现两个操作把
  137 BPM/50ms/8等分草稿覆盖为旧候选参数；现在成功API计划与清除计划共用草稿恢复入口，
  保留显式确认、对照Offset、采音引擎和未舍入锚点，清除期间锁定交互。
- 新增5项真实浏览器流程，修改前2 failed/3 passed，修复后5 passed（18.1秒）。
  成功只改变计划与偏好；失败/取消保留完整旧项目，生成、重开、导入仍采用各自时钟。
  隔离测试服务拒绝外部planner请求，用确定性替身测试真实HTTP、任务队列、取消和保存。
- 完整Edge **26 passed（1.1分钟）**；JS语法、fixture编译与差异检查通过。
  没有修改生产Python、请求外部API、运行神经模型或重复历史音乐实验。
- 下一项已复现：局部重写把本轮整首提案的省音记录标成最终候选记录，导致实际锁定保留的
  钢琴音仍显示已省略。两音合成证据已保存于ignored验证目录；后续区分提案与最终输出范围，
  不据此宣称采音/音乐质量已完成。
- PR #107 head `cd319dd1bb8088808de5e6188d390c2167582d11`的Windows push
  run `44021666071`、PR run `44021695389`均成功，合并为
  `52ab4104cca8363595e016e108f4ac16c1c55cfb`。push记录Python2,461 passed、89 skipped
  （46.17秒），Chromium26 passed（43.3秒）。没有重建便携包或发布新版本。

## 2026-11-17 第八阶段：局部重写报告范围

- 基于PR #107创建`codex/regional-selection-audit-20261117`。真实合成复现：钢琴1000–1800ms
  和贝斯3500–4200ms先生成整谱，再重写`[3000,5000)`并静音钢琴；实际输出仍有两音，
  原报告却写选中1音、钢琴已省略。原因是整首预选发生在局部过滤和锁定物件合并之前。
- 新`selection_report.py`将局部候选报告分为实际输出与整首提案。顶层按最终物件记录保留、
  锁定、本次重写数量、声部、LN比例与制谱起止；旧完整提案保存在`proposal`并显式标明范围。
  不给锁定旧音补写本轮分析来源或声学时间；来源继续使用候选快照。整谱生成路径不变。
- 保留顶层结构/表达快照供导出书签使用。人工新增、重分析同名ID、原谱重定轨、保存回读、
  JSON/OSU导出及后续手动编辑已有专项；修改前3项失败，修复后相关组合29项、API/结构组合
  49项通过，新增保存导出4项通过。完整Python为 **2,465 passed、89 skipped（61.89秒）**。
- 前端实际保留/锁定/重写数量从候选物件与范围计算；整首省音提案折叠展示，旧项目明确标注
  范围不确定，不能把锁定保留音当作当前候选省音。新旧局部与整谱3项、既有来源4项合计
  **7 passed（22.2秒）**。独立复核再现重分析复用ID后新提案误标已保留，现明确提示同ID的
  旧物件锁定、来源分别核对；真实3000ms新提案与1000ms旧锁定音并存，跳转只定位新提案。
- 新增同ID回归后定向 **8 passed（22.3秒）**；全组曾暴露新测试读取项目早于载入完成，
  改为等待准确项目标题后最终完整Edge **30 passed（1.2分钟）**。JS语法、fixture编译、
  `compileall`与`git diff --check`通过。没有修改选音、定轨或运行新的音频推理。
- PR #108首轮push检查通过，但PR检查在既有来源测试重开后读取尚未绘制的hitbox失败
  （29 passed、1 failed）。测试改为等待项目实际载入并轮询点击框，不加固定休眠、不改生产；
  两次独立Edge来源组均4 passed（15.2秒、14.3秒），已提交补充等待CI复核。
- 最终head `f53e2d411eb0a75b88587e40c9446df087bdd0c7`的Windows push
  `44025453538`、PR `44025459570`均通过，PR #108合并为
  `94534a74fff71caa0b25abdfeb82e15709f86bc4`。push记录Python2,465 passed、89 skipped
  （65.53秒），Chromium30 passed（59.5秒）；完整日志已保存ignored验证目录。

## 2026-11-17 第九阶段：局部重写的乐句依赖边界

- 真实生成复现：先保留1000ms、1500ms两个短键，再启用以1000ms为起点的乐句，
  只重写`[1250,2500)`。整首提案用新乐句覆盖1500ms起音，但范围外锁定仍是旧短键，
  导致1500ms起音静默消失。修改前证据在ignored
  `build/structure-20261117-validation/regional-phrase-before.json`。
- 新`regional.py`只在内部被省略起音属于重写范围、其代表乐句起点在范围外时拒绝，
  提示扩大范围包含乐句起点和内部起音或整谱生成。边界按真实提前量、裁剪、量化后的
  制谱起点判断。普通跨界持续音及没有内部覆盖关系的历史路径不受影响。
- 不凭旧候选的phrase标志、support ID或本轮提案签名放行：手动缩短和重分析都可能
  保留这些字段。即使旧长音看似相同，当前也要求一起重写相关乐句，以免推断不存在的来源身份。
  全谱的表达、选择、定轨不变，不自动补音，也不把容量落选改为成功。
- 四项真实行为回归先失败后通过；含API失败保留完整项目、扩大范围后生成及OSU回读的
  根专项8项通过。独立补充容量落选、非整数拍格、编辑、缺快照、内部ID重分析、量化塌缩和
  历史路径8项，合计 **16 passed（2.02秒）**；全量Python **2,481 passed、89 skipped
  （57.25秒）**，运行 `.venv/Scripts/python.exe -m pytest -q`。
- 复用4个已有开发片段各6K及原键数，8例均复现旧行为删除目标起音，新保护均拒绝不完整
  替代范围；扩大范围后均生成成功。独立复核2,662个区域外锁定物件逐字段不变，OSU回读
  起止/轨道一致。6例提案乐句因容量落选、2例实际选中；扩大范围不保证恢复音乐内容。
  每组只检查最早可复现的旧保留内部起音，采用显式确认Timing的诊断，未跑模型或读音频。
  产物 `test-data/regional-phrase-scope-20261117/`，报告SHA
  `f44aa3c2da373711f3a956a7d1406764a2c786b90abe49372bcc7c641c28dd05`。
  本机命令 `.venv/Scripts/python.exe build/structure-20261117-validation/check_regional_phrase_scope.py`，
  输出目录不覆盖；输入/旧候选哈希与既有冻结记录一致。范围及限制见
  [局部乐句边界结果](regional-phrase-boundary-results.md)。
- 独立复核再现反方向：重写`[1000,1500)`生成新乐句却保留区外旧1500ms短键；旧音手动
  移到2250ms时，即便原分析起音在范围内也会重复。新增保护仅检查已选中乐句明确替代的
  内部ID是否仍被旧候选锁定。容量落选、量化塌缩、旧内部音已删除/已移入、普通持续音允许。
  修改前6 failed、5 passed，最终三文件 **28 passed（2.80秒）**；真实API两种方向失败
  均完整保留项目，扩大范围生成及导出通过。最终全量Python **2,493 passed、89 skipped
  （60.65秒）**；JUnit保存在ignored `build/structure-20261117-validation/regional-phrase-final-junit.xml`。
- 最终源码重新核对8正向、2反向冻结场景全部通过。8例扩大范围后的整份候选除创建时间/耗时
  外与上一版产物一致，2,662个锁定物件不变；两例反向旧行为明确保留新乐句及旧内部音，
  新保护拒绝。脚本 `build/structure-20261117-validation/check_regional_phrase_scope_final.py`，
  运行 `.venv/Scripts/python.exe` 加该路径；输出 `test-data/regional-phrase-scope-final-20261117/`，
  报告SHA `91215012b54cc4e52710a9c499c7fe8d9cf47bb98aef288ce2f34b80e8ebb3a9`。
  没有重跑神经模型、重建便携包或发布版本；音乐质量仍未验收。
- PR #109 head `0e25e5b126fd9cd533d6e14b714f85f031d1cb12`的Windows push
  `44044780933`、PR `44044861229`均成功，合并为
  `3dc07058a60335d565a586f8ce69161cd9f09019`。push记录Python2,493 passed、89 skipped
  （58.79秒）、Chromium30 passed（1.0分钟），完整日志已保存在ignored验证目录。

## 2026-11-17 第十阶段：集成便携验证

- 针对九阶段合并提交`3dc07058a60335d565a586f8ce69161cd9f09019`执行
  `gh workflow run windows.yml --ref main`。前两次run `44049902518`、`44056957927`取消；
  最终run `44057633497`于2026-11-17成功。CI实际head为
  `0e25e5b126fd9cd533d6e14b714f85f031d1cb12`，其Git tree
  `4974a83b759a5e7237eac3877b8c656aa538bb2f`与合并提交一致。
- 完整standard构建、固定钢琴模型、实际EXE离线/Unicode路径、失败取消与恢复、6–10K导出、
  worker关停及产物归档全部通过。报告确认83个源码文件、2,360个worker文件；JUnit共2,582项，
  2,493 passed、89 skipped、0 failures。没有下载大型ZIP或创建Release。
- 只下载小型报告与日志到ignored目录
  `build/structure-20261117-validation/ci-44057633497/`。便携报告SHA
  `e2c0d7124e6bd11eba5c01267e3ae8b2d12669e0249316082ba40d549b4b192e`；JUnit SHA
  `46e839c519be8b82c038adfdc703f1b56778c547747a99e4863f39f9de7685c4`；关停报告SHA
  `bb5a6c3edd2e1b8c8a4aabd147b5724c5a735c4b60f3ea185529473989ae513c`。

## 2026-11-17 第十一阶段：声部取舍证据复核

- 只读核验发现[表达取舍协议](expression-tradeoff-audit-protocol.md)中的冻结
  `selection.json` SHA 少写一个`5`，已更正。当前冻结文件、审计脚本常量和历史报告
  `input_sha256`均为`24f23cc1d59e7452e5ede424f10054d285b0d2ecdd26b33095e651716f0b0b44`。
- 66例结果中，26例默认配置的音符集合未变化；仅4例显式`max_held=5/1`压力配置发生
  取舍，共新增23条、删除15条，删除理由均为容量或密度。声部权重均为1，段落没有
  `main_voice`，且`salience_estimated=false`。这些静态证据可确认容量约束造成跨声部取舍，
  不能证明具体取舍是音乐错误；没有据此改选音权重、重跑模型或重读音频。
- 真实音频采音、重要声部判断、LN音乐性和osu!试打仍是开放验收，不以静态审计或构建通过代替。
- PR #110的PR run `44060991936`与push run `44060934067`最终均通过并合并；push run曾取消后重跑，期间取消状态在PR检查中显示失败，促成本阶段避免依赖取消去重。

## 2026-11-17 第十二阶段：Windows验证去重

- 同一提交因push与pull_request同时触发`.github/workflows/windows.yml`，重复运行Python和浏览器测试。取消其中一条后，GitHub将取消状态显示为失败检查，必须重跑才能合并。
- `verify` job现仅对PR、`main`、tag、手动触发和提交信息带`[portable]`的push执行；普通功能分支push跳过job，避免重复计算且返回成功跳过状态。没有移除`[portable]`便携构建入口。
- `docs/development.md`已同步说明触发规则。提交`30d01ce9021c5763dd0bbab66ce1348ad77a3a7c`的push run `44064113519`为`skipped`，PR run `44064156960`的Python与浏览器测试通过，验证普通push不再留下失败检查。没有重新触发`[portable]`便携构建；其完整验证已有run `44057633497`和PR #105 push run `44018546073`作为证据。未改音乐算法或模型输入。
- PR #111已合并为`cca4a21f831c3aa5b2cad989b6c287b7dbe50f01`。合并后复核确认run `44064879215`为`skipped`、PR run `44064888429`与main run `44065394291`均成功。
- 对`Projects.save`及应用写入串行边界做只读复核：同目录临时文件加`os.replace`保留当前单进程原子替换语义，服务写路由由`RLock`保护；没有复现缺陷，不据测试盲区改写保存实现。定向`tests/test_editing.py tests/test_import_reliability.py`为28 passed。未来多进程共享数据目录仍无锁与专门回归，且原子替换不等于断电级`fsync`持久性；这些是边界说明，不是当前已确认故障。

## 2026-11-17 第十三阶段：未知任务的 API 状态

- 只读复核发现`GET /api/jobs/<jid>`与`POST /api/jobs/<jid>/cancel`对未知ID直接索引任务字典，抛出的`KeyError`被全局处理器误映射为400。新增两个API回归后先复现2项失败，响应均为`400`。
- 两个路由现在在任务锁内显式查找；任务不存在返回`404 {"error":"任务不存在"}`，已知任务行为不变。
- `.venv/Scripts/python.exe -m pytest -q tests/test_server.py`为17 passed；全量`.venv/Scripts/python.exe -m pytest -q`为2,495 passed、89 skipped（60.82秒）。`compileall -q manialab scripts tests`、`node --check`全部静态JavaScript文件及`git diff --check`通过。等待PR Windows CI；未运行模型、读取音频或改变研究产物。

## 2026-11-17 第十四阶段：最近项目切换请求竞态

- 只读前端审查发现快速连续选择最近项目时，旧GET响应无条件调用`loadState`，可能覆盖较晚选择。新增可控延迟的Edge回归：先请求`UI timing clock`并挂起响应，再加载`UI timing wave`，最后放回旧响应；修复前页面被旧项目覆盖，真实复现通过。
- `loadState`及最近项目请求共用单调请求序号；每次选择发出新序号，所有成功项目载入都会使旧请求失效。迟到响应不能覆盖最新项目，也不影响现有载入路径。
- 单项Edge回归通过；完整`npm run test:ui`使用`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=8978`，31 passed（1.3分钟）。静态JavaScript与新测试文件`node --check`、`git diff --check`通过；Windows PR CI尚未运行。未改服务端、模型或研究输入。

## 2026-11-17 第十五阶段：播放路由模块边界

- `server.py`继续承载项目与任务 API，但播放、独立声部读取、试听渲染和波形窗口处理集中在嵌套路由中。将这四个路由搬到`manialab/playback_routes.py`，由`create_app`显式传入`projects`、`current_audio`和`requested_tempo`。
- 保留同一Flask app、URL、视图函数名、鉴权和错误处理；未把依赖更广的谱面/ZIP导出一并迁移。架构导航同步更新。
- 定向`tests/test_server.py tests/test_audition.py`为20 passed；全量Python为2,495 passed、89 skipped（64.41秒），`compileall -q manialab`通过；完整Edge `npm run test:ui`为31 passed（1.2分钟）。PR run `44071878909`与合并后main run `44073069913`成功，重复push run `44071832433`跳过。未运行模型或改变音频数据。

## 2026-11-17 第十六阶段：任务生命周期服务边界

- 将单线程`ThreadPoolExecutor`、共享`RLock`、任务表和后台执行闭包从`server.py`移入无Flask依赖的`manialab/task_service.py`。路由仍负责请求校验与构造业务操作；同步项目写路由和任务提交共享同一把锁，单项目占用与全局单任务限制保持不变。
- `ProjectTasks`集中查询状态、取消、进度、错误日志、提交及清理时机。只在操作未取消且成功后保存快照；失败/取消保留旧状态并以`cleanup(False)`清理本次未提交资源。保留`app.extensions['executor']`和`['jobs']`兼容既有调用与测试，并增加`task_service`入口。
- 新增`tests/test_task_service.py`覆盖成功持久化、失败不落盘、取消不落盘、清理标记和并发任务阻止。服务端、导入清理与吉他任务组合定向 **46 passed（12.29秒）**；全量`.venv/Scripts/python.exe -m pytest -q`为 **2,498 passed、89 skipped（67.69秒）**。`compileall`与`git diff --check`通过。
- 本机Edge完整`npm run test:ui`初次为30 passed、1项“段落草稿恢复”勾选状态断言失败；该用例单项重跑通过，随后完整复跑为 **31 passed（1.2分钟）**。默认Chromium尝试因本机未安装Playwright对应headless二进制，使用`MANIALAB_TEST_BROWSER=msedge`完成浏览器验证。Windows PR CI待运行。
- 检查时本地8765与8766端口仍由既有听审材料HTTP服务占用，本阶段未停止或改写这些进程；没有运行模型、读取音频或修改研究脚本/产物。

## 2026-11-17 第十七阶段：项目路由模块边界

- 将`/api/import`、项目读取、谱面切换和候选切换从`server.py`移入`manialab/project_routes.py`。通过`create_app`注册到同一个Flask app，保留URL、HTTP方法、endpoint名称、应用级鉴权与错误处理，以及任务服务共享锁和写入前项目空闲检查。
- 增加回归验证导入后切谱同步音频引用并清理声部/分析/预览缓存、旧项目`input_timing`读取回填、未知谱面拒绝，以及非法候选索引不改变项目状态。
- 定向`.venv/Scripts/python.exe -m pytest -q tests/test_project_routes.py`为7 passed；全量`.venv/Scripts/python.exe -m pytest -q`为2,505 passed、89 skipped（65.62秒）。`.venv/Scripts/python.exe -m compileall -q manialab scripts tests`与`git diff --check`通过。完整Edge `npm run test:ui`为31 passed（1.3分钟）；PR Windows run `44079580314`与合并后main run `44080283011`成功。CI未下载音频模型或构建便携包；没有读取用户音频或修改研究脚本/产物。

## 2026-11-17 第十八阶段：导出路由模块边界

- 将`/api/projects/<pid>/export/<kind>`从`server.py`提取到`manialab/export_routes.py`，由应用组合层显式传入项目仓库、资源目录和当前音频解析器；保留`export` endpoint、JSON/OSU/OSZ格式和应用级鉴权、异常映射及响应头处理。
- 保留音频候选`timing_revision`拦截、固定Timing一致性校验、原谱引用、6–10K导出参数、偏移书签、嵌套谱面目录和缺失音频补包行为。只读对照未发现迁移兼容差异。
- 定向`tests/test_server.py tests/test_import_reliability.py tests/test_editing.py`为45 passed；全量`.venv/Scripts/python.exe -m pytest -q`为2,505 passed、89 skipped（62.42秒）；`.venv/Scripts/python.exe -m compileall -q manialab scripts tests`与`git diff --check`通过。完整Edge `npm run test:ui`为31 passed（1.2分钟）。PR Windows run `44082425764`与合并后main run `44082995836`成功，均未运行音频模型或便携构建；未读取用户音频或修改研究脚本/产物。

## 2026-11-17 第十九阶段：Timing诊断路由边界

- 将`POST /api/projects/<pid>/timing-refit`和`timing-check`迁入`manialab/timing_routes.py`。模块注入项目仓库与原有`requested_tempo`闭包；BPM/Offset校验和共享时钟策略继续由应用组合层维护。
- 保留endpoint与HTTP方法、稀疏攻击证据拒绝、音频时钟攻击点转换、Offset重拟合及诊断报告合并；只读对照未发现行为差异。
- 定向`tests/test_timing_diagnostic_clock.py tests/test_server.py`为31 passed；全量`.venv/Scripts/python.exe -m pytest -q`为2,505 passed、89 skipped（62.36秒）；`.venv/Scripts/python.exe -m compileall -q manialab scripts tests`和`git diff --check`通过。完整Edge `npm run test:ui`为31 passed（1.2分钟）。PR Windows run `44085269448`与合并后main run `44085905987`成功；未运行模型、读取用户音频或修改研究脚本/产物。

## 2026-11-17 第二十阶段：编排、段落与表达路由边界

- 将`/plan`、`/clear-plan`、`/recommendations`、`/structure`、`/expression`及`/apply-recommendation`迁入`manialab/arrangement_routes.py`。保持 endpoint 名、路径、HTTP 方法、异步任务提交、共享锁、项目忙碌检查、证据签名校验和计划/表达状态联动。
- planner通过应用组合层显式注入`app.extensions["planner"]`，浏览器fixture沿用该注入点隔离外部请求；段落检测继续使用原音频解析器与偏移转换闭包。
- 全量`.venv/Scripts/python.exe -m pytest -q`为2,505 passed、89 skipped（84.66秒）；完整Edge `npm run test:ui`为31 passed（1.3分钟）；`.venv/Scripts/python.exe -m compileall -q manialab scripts tests`和`git diff --check`通过。PR #119 Windows run `44088189406`、合并后main run `44088861668`成功；未运行模型、读取用户音频或修改研究脚本/产物。

## 2026-11-17 第二十一阶段：采音与导入声轨路由边界

- 将`POST /api/projects/<pid>/analyze`和`provided-stems`迁入`manialab/analysis_routes.py`。项目仓库、任务服务、当前音频解析、任务提交、分离器及各检测器由应用组合层显式注入；分析函数与声轨处理工厂集中在`app.extensions['analysis_services']`，测试可替换单个依赖而不触发真实模型。
- 保持现有endpoint、请求校验、专用模型/声部兼容检查、共享任务锁、异步提交身份、旧结果保留，以及导入失败/取消时新上传与解码文件的清理行为。
- 定向API与可靠性验证为66 passed；全量`.venv/Scripts/python.exe -m pytest -q`为2,505 passed、89 skipped（87.83秒）；完整Edge `npm run test:ui`为31 passed（1.3分钟）；`.venv/Scripts/python.exe -m compileall -q manialab scripts tests`与`git diff --check`通过。PR #120 Windows run `44091644703`成功并合并为`754530c76da46a57bfdbe8b6e62050b501412ec9`；合并后main run `44092230242`成功，API/领域测试及合成浏览器通过，模型与便携构建步骤跳过。未运行模型、读取用户音频或修改研究脚本/产物。

## 2026-11-17 第二十二阶段：Mapperatorinator路由边界

- 将`POST /api/projects/<pid>/engine-install`和`engine-generate`迁入`manialab/upstream_routes.py`。应用组合层注入共享`MapperEngine`、项目仓库、异步提交器、当前音频解析器和Timing闭包；路由模块承接原模型输入、输出谱面修正与remap报告。
- 保持endpoint及视图函数名、请求校验、模型任务生命周期、Timing确认门槛、原始键型、固定时钟对齐和候选状态提交语义。
- 定向`tests/test_server.py tests/test_upstream.py`为26 passed；全量`.venv/Scripts/python.exe -m pytest -q`为2,505 passed、89 skipped（87.72秒）；完整Edge `npm run test:ui`为31 passed（1.3分钟）；`.venv/Scripts/python.exe -m compileall -q manialab scripts tests`与`git diff --check`通过。独立对照未发现路由语义差异；PR #121 Windows run `44093748990`和合并后main run `44094348023`均成功，合并提交`34e3ece8322a1283250532c80ca228ae183c5ddb`。CI的模型与便携构建步骤按触发条件跳过。
- 未运行模型、读取用户音频或修改研究脚本/产物。

## 2026-11-17 第二十三阶段：候选生成与编辑路由边界

- 将`POST /api/projects/<pid>/generate`和`edit`迁入`manialab/candidate_routes.py`。应用组合层注入现有参数规范器、Timing解析闭包及异步任务提交器；纯候选生成继续委托`generation.py`。
- 保留候选版本冲突检查、编辑基准时间戳、旧候选继承、落线门槛、编辑历史、报告刷新和最近8个候选状态更新；参考谱风格导入保持独立。
- 定向候选、编辑、生成时钟、表达、局部生成与证据测试为139 passed；全量`.venv/Scripts/python.exe -m pytest -q`为2,505 passed、89 skipped（84.02秒）；完整Edge `npm run test:ui`为31 passed（1.3分钟）；`.venv/Scripts/python.exe -m compileall -q manialab scripts tests`和`git diff --check`通过。PR #122 run `44096021565`与合并后main run `44096681056`均成功，合并提交`3e1059bed55bcb874751c360ce9f7540cfb79d81`；模型与便携构建步骤按触发条件跳过。
- 本阶段未运行模型、读取用户音频或修改研究脚本/产物。

## 2026-11-17 第二十四阶段：项目参考风格导入边界

- 将`POST /api/projects/<pid>/reference-style`并入`manialab/project_routes.py`，与谱面导入、选择和项目偏好操作同组；应用组合层注入参数规范器及风格分析器。
- 保持10MB `.osu` 限制、UTF-8 校验、共享任务锁、明确键数偏好优先、风格元数据保存，以及不覆盖音频分析/既有候选的行为。
- 定向`.venv/Scripts/python.exe -m pytest -q tests/test_project_routes.py tests/test_server.py::test_reference_style_preserves_audio_candidates_and_target_keys tests/test_style.py`为10 passed；全量Python为2,505 passed、89 skipped（84.05秒）；完整Edge `npm run test:ui`为31 passed（1.3分钟）；`compileall`和`git diff --check`通过。PR run `44098248603`和合并后 main run `44098766111`通过；普通 push run `44098175931`成功跳过。
- 未运行模型、读取用户音频或修改研究脚本/产物。

## 2026-11-17 第二十五阶段：共享数值求解边界

- 将 HiGHS 共享锁、MILP 单线程选项及 warning 过滤从`arrange.py`移至`manialab/solver.py`，并为`gesture_flow.py`的线性规划增加同锁求解入口。表达与研究模块不再从定轨模块引用私有求解器状态或实现；保留`expression._solve_binary`和`gesture_flow.linprog`的测试替换点。
- 保持调用方 MILP 时间/间隙参数、强制`threads=1`与`parallel=False`、LP 的`highs-ds`方法/超时，以及两种 SciPy warning 类别和原窄消息过滤。MILP 与 LP 共用同一个非重入锁，不在调用侧嵌套加锁。
- 新增 fake solver 参数与 warning 回归，以及 MILP/LP 跨线程互斥验证。定向求解器、表达、手势流、锚点和布局测试为47 passed；全量`.venv/Scripts/python.exe -m pytest -q`为2,509 passed、89 skipped（87.03秒）；完整Edge `npm run test:ui`为31 passed（1.4分钟）；`compileall`与`git diff --check`通过。PR #124 run `44102406451`与合并后 main run `44103378364`通过，合并提交`e37b3833a2a4a0f390777f7ee4823161091371a7`；模型与便携构建步骤按条件跳过。未运行音频模型、读取用户音频或改动研究脚本/研究产物；音乐质量与 osu! 试打仍待独立验收。

## 2026-11-17 第二十六阶段：序列窗口编码边界

- 将`SequenceRow`、节拍位置转换、单窗与批窗特征编码从`sequence_prior.py`迁入`sequence_windows.py`；模型 JSON 校验、`SequencePrior`和镜像评分仍由`sequence_prior.py`负责。
- `sequence_prior.py`继续重导出原有窗口类、时钟和编码入口，训练/研究脚本无需改路径；`layout.py`改为分别从新窗口模块和模型模块导入。
- 新增兼容导出的逐项身份回归，并验证窗口模块不承担模型加载/评分。定向`.venv/Scripts/python.exe -m pytest -q tests/test_sequence_prior_runtime.py tests/test_sequence_prior_remap.py tests/test_layout.py tests/test_sequence_windows_boundary.py`为59 passed（8.75秒）；全量`.venv/Scripts/python.exe -m pytest -q`为2,511 passed、89 skipped（85.52秒）；完整Edge `npm run test:ui`在`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=8978`下为31 passed（1.3分钟）；`compileall`与`git diff --check`通过。PR #125 run `44106980785`与合并后 main run `44107895343`成功，合并提交`dc1de5e67badccf058a85113f16fc94817ffdc52`。
- 未读取用户音频/参考谱、触碰保留的 MAESTRO 测试或运行模型实验；音乐质量与 osu! 试打仍待独立验收。

## 2026-11-17 第二十七阶段：序列特征 schema 边界

- 将序列特征字段名、固定维度、LN 交互与乐句上下文配置、schema 修订号及版本映射从编码实现移入`manialab/sequence_schema.py`。`pattern_sequence.py`与运行窗口模块继续重导出同一对象，既有研究脚本导入路径保持不变。
- 增加单一 schema 所有者和兼容重导出的身份测试；全量`.venv/Scripts/python.exe -m pytest -q`为2,512 passed、89 skipped（62.72秒），Edge `npm run test:ui`在`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=8978`下为31 passed（1.2分钟），`.venv/Scripts/python.exe -m compileall -q manialab scripts tests`与`git diff --check`通过。PR #126 Windows run `44108562881`与合并后 main run `44110037927`均成功，合并提交`76b760a53e93298b0be80f2ac87820a5338dda48`。
- 不更改特征值、模型文件、缓存协议或训练脚本；未运行模型实验、读取用户音频/参考谱或访问保留的 MAESTRO 测试。音乐质量与 osu! 试打仍待独立验收。

## 2026-11-17 第二十八阶段：Timing 拟合原语边界

- 将 BPM 常见拍速归一化、相位估计、网格拟合/评分、固定 BPM 偏移重拟合、精确拍速建议及模型拍间隔支持移入`manialab/timing_fitting.py`。`timing.py`保留公共时钟、诊断、自动估速和候选对齐，并继续重导出旧拟合入口。
- 定向 Timing/诊断/生成时钟/边界测试为72 passed；全量`.venv/Scripts/python.exe -m pytest -q`为2,513 passed、89 skipped（60.75秒）；Edge `npm run test:ui`在`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=8978`下为31 passed（1.2分钟）；`compileall`与`git diff --check`通过。PR #127 Windows run `44111017307`与合并后 main run `44112344487`均成功，合并提交`c7af19cad28391a7f9e737be3d56dba949851e06`。
- 只移动纯拟合实现，未更改算法、音频输入或模型实验；未读取用户音频/参考谱或访问保留的 MAESTRO 测试。音乐质量与 osu! 试打仍待独立验收。

## 2026-11-17 第二十九阶段：音频证据轻量边界

- 将共享采样参数移入`manialab/audio_config.py`；将声部末尾采样对齐、局部能量归属、拍速证据比较与跨声部串音拒绝移入不导入 ONNX Runtime 的`manialab/audio_evidence.py`。`audio.py`原路径重导出常量和函数，`analyze_deep`仍按原模块全局查找函数，测试与外部脚本 monkeypatch 入口不变。
- 两个只执行证据核验的研究脚本直接依赖轻量模块。定向音频、导入声轨、模型调用替身和兼容边界测试为62 passed（6.67秒）；Edge `npm run test:ui`在`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=8978`下为31 passed（1.2分钟）；`compileall`与`git diff --check`通过。PR #128 Windows run `44113265599`成功并合并为`4d250f27004eaadd87b346b3dd177a5234107c3d`；合并后main run `44114098052`成功。未执行模型下载、模型构建或便携包构建。
- 未调用真实模型、读取用户音频/参考谱或访问保留的 MAESTRO 测试；不更改音频算法或研究产物。音乐质量与 osu! 试打仍待独立验收。

## 2026-11-17 第三十阶段：音频事件分析边界

- 将`pitched_notes`、`rhythmic_evidence`、`fuse_attacks`、`refine_releases`移入不导入 ONNX Runtime 的`manialab/audio_analysis.py`。`audio.py`继续重导出相同函数对象，分析编排仍通过原模块全局入口调用，保留现有导入和 monkeypatch 行为；解码、模型分窗推理、缓存与组合流程仍在`audio.py`。
- 用 AST 对照合并前`main`与新模块，四个函数实现完全一致；独立子进程导入测试确认`audio_analysis`不会加载`onnxruntime`。音频事件、Timing、声部集成和替身回归共120 passed（9.90秒）；`.venv/Scripts/python.exe -m compileall -q manialab tests`及`git diff --check`通过。未运行真实模型测试或浏览器测试。PR #129 Windows run `44115907561`成功并合并为`a1f573c7187bf94a929db441f95af8d89c54ac8f`；合并后main run `44116712054`成功。
- 下一步检查音频编排剩余职责及调用边界，选取可单独验证的纯数据/缓存边界；音乐质量与 osu! 试打仍需独立验收。

## 2026-11-17 第三十一阶段：音频 IO 轻量边界

- 将`run_hidden`与FFmpeg单声道PCM解码移入`manialab/audio_io.py`。`audio.py`保留相同进程辅助对象和旧签名`decode`包装，并把旧`audio.run_hidden`作为显式runner传入，既有导入及替换点可继续工作。
- `provided_stems.py`、`separation.py`、`upstream_routes.py`和`arrangement_routes.py`改为直接依赖轻量音频IO/配置/证据模块。独立子进程验证这些模块导入后`manialab.audio`与`onnxruntime`都未载入。
- `.venv/Scripts/python.exe -m pytest -q tests/test_audio_io_boundary.py tests/test_audio_analysis_boundary.py tests/test_audio_evidence_boundary.py tests/test_provided_stems.py tests/test_stems.py tests/test_upstream.py tests/test_server.py tests/test_import_reliability.py`为59 passed（15.43秒）；`.venv/Scripts/python.exe -m pytest -q tests/test_separation.py`为2 passed（0.72秒）。`.venv/Scripts/python.exe -m compileall -q manialab tests`与`git diff --check`通过。未运行真实模型、用户音频或浏览器测试；PR CI待运行。
- 阶段 30 PR #129的Windows run `44115907561`和合并后main run `44116712054`均成功，合并提交为`a1f573c7187bf94a929db441f95af8d89c54ac8f`；本阶段 PR #130 Windows run `44117906701`与合并后main run `44118573172`均成功，合并提交`36826779232a165c3b7f8f07ee09aebd9cb52d63`。下一步检查剩余音频分析编排和缓存职责，先做只读边界审查，再决定是否需要迁移；长期音乐质量与 osu! 试打仍独立验收。

## 2026-11-17 第三十二阶段：表达选音边界

- 将`resolve_phrase_decisions`、和弦层级选项、候选去重、容量预检、鼓点回退与`select_expression`移入`manialab/expression_selection.py`。`expression.py`继续负责声部方案生成、校验和证据转换，并重导出原选择辅助函数。
- `expression.select_expression`保留原签名，通过包装层将`expression._solve_binary`转交新模块，两阶段求解均继续使用既有可替换求解器入口。AST 对照确认六个辅助函数实现不变，旧选择函数签名一致。
- 定向表达、solver、LN、Timing 生成和声部容量测试共364 passed（6.94秒）；全量`.venv/Scripts/python.exe -m pytest -q`为2,522 passed、89 skipped（63.52秒）；完整Edge `npm run test:ui`为31 passed（1.2分钟）。`.venv/Scripts/python.exe -m compileall -q manialab tests`与`git diff --check`通过。独立导入测试确认选择模块不反向导入`expression.py`或`audio.py`。未运行真实模型或用户音频。
- PR #131 Windows run `44121345391`通过并合并为`6f814dcbdab9ede914f6a86b5d84147e96a1d901`；合并后 main run `44122130127`成功。后续架构审查确认`sequence_prior.py`已是模型加载/评分门面，窗口编码已在阶段 26 拆出；下一步转为按版本职责整理`pattern_sequence.py`的基础、LN交互和长上下文编码，保留旧入口并冻结特征定义。长期音乐质量与 osu! 试打仍独立验收。

## 2026-11-18 第三十三阶段：序列特征编码边界

- 将版本化特征实现拆到`sequence_features_common.py`、`sequence_features_base.py`、`sequence_features_interactions.py`、`sequence_features_phrase_context.py`和`sequence_features_mirror.py`。`pattern_sequence.py`保留 schema、常量、编码器及原辅助函数的兼容重导出；模型输入字段、顺序和计算逻辑不作更改。
- AST 对照确认迁移的12个函数定义与原实现一致。新增边界测试验证旧导入名仍指向同一函数/schema 对象，并确保编码实现模块不反向导入兼容门面。
- 序列特征、schema、窗口、训练编码、运行时、布局与评估定向测试为135 passed、4 skipped（8.67秒）；全量`.venv/Scripts/python.exe -m pytest -q`为2,524 passed、89 skipped（65.48秒）；完整Edge `npm run test:ui`为31 passed（1.1分钟）。`.venv/Scripts/python.exe -m compileall -q manialab tests`和`git diff --check`通过。未运行研究训练、模型、用户音频或参考谱。
- PR #132 Windows run `44124525798`与合并后 main run `44125358706`均成功，合并提交`86ed7e88468197f3eb458a60c016eab19739a709`；Python/API与合成浏览器检查通过，模型和便携构建步骤按触发条件跳过。长期音乐质量与 osu! 试打仍独立验收。后续继续只读审查剩余领域模块，依据实际函数职责和调用/测试边界选择下一阶段。

## 2026-11-18 第三十四阶段：本地编排计划边界

- 将`EVIDENCE_THRESHOLD`、`plan_evidence`、`evidence_signature`、`validate_plan`和`adjusted_segments`迁入`manialab/recommendation_plan.py`。`recommendations.py`继续重导出同一对象；`generation.py`与`arrangement_routes.py`直接依赖计划模块，减少加载建议生成实现的耦合。
- 新增兼容身份和独立导入边界测试。定向`.venv/Scripts/python.exe -m pytest -q tests/test_recommendation_plan_boundary.py tests/test_recommendations.py tests/test_server.py::test_generation_applies_local_recommendation_expression_to_export`为24 passed；`.venv/Scripts/python.exe -m compileall -q manialab tests`与`git diff --check`通过。
- 本地全量测试期间并行运行浏览器测试导致资源耗尽：Python套件报告2,522 passed、89 skipped、4 failures，其中1项合成音频测试内存分配失败、3项导入边界测试无法创建子进程；浏览器套件因Node堆内存耗尽报告8个worker失败、23项未运行。失败发生时系统可用内存约1.1 GiB，且进程快照显示非本项目`eye-kg-llm-ui` Vite进程占用异常高；保留该用户进程及本项目两个资料服务，不能将此轮全量结果记作通过。
- PR #133 的Windows run `44128674193`和进度更新后的 run `44129552127`均通过，合并后 main run `44130292999`也通过：全量Python/API为2,526 passed、89 skipped，Edge UI为31 passed；音频模型与便携构建步骤按条件跳过。干净runner结果覆盖本地资源不足造成的全量验证缺口；没有运行研究训练或模型实验。
- 未读取用户音频/参考谱或保留的 MAESTRO 测试，未改动研究脚本或研究产物。结构整理不代表音乐质量或模型效果提升；osu! 试打仍独立验收。

## 2026-11-18 第三十五阶段：LN 批量特征边界

- 将`_sequence_windows_ln_interactions`从`sequence_windows.py`迁入`manialab/sequence_window_interactions.py`，新模块只依赖 NumPy 与`sequence_schema.py`；窗口模块仍导出同一函数对象，`sequence_prior.py`旧兼容入口不变。
- 更新窗口边界测试，检查旧导出对象身份，并验证独立导入新模块不加载`sequence_windows.py`或`sequence_prior.py`。本机资源不足期间，定向 pytest 在收集阶段未能运行；`git diff --check`通过。
- PR #134 Windows run `44133239396`和进度更新后 run `44134458472`，以及合并后 main run `44135099313`均通过：全量Python/API为2,527 passed、89 skipped，Edge UI为31 passed；模型及便携构建步骤按条件跳过。该干净runner结果完成本地资源不足时未能执行的运行时与训练编码兼容验证。
- 未运行研究训练、模型或用户音频；未访问保留的 MAESTRO 测试，未改动研究脚本或研究产物。结构迁移不代表序列模型或音乐质量提升；osu! 试打仍独立验收。

## 2026-11-18 第三十六阶段：定轨配置边界

- 将`DEFAULTS`、`BOUNDS`、`INTEGER`和`parameters`从`arrange.py`迁入只依赖标准库`math`的`manialab/arrange_config.py`。`arrange.py`继续重导出同一对象；`planner.py`、`recommendations.py`与`project_routes.py`改为直接依赖轻量配置模块。
- 新增兼容对象身份、默认设置副本及独立导入测试；子进程验证配置模块不加载`arrange`、NumPy或SciPy，三个直接消费者不加载`arrange`或SciPy。定向`.venv/Scripts/python.exe -m pytest -q tests/test_arrange_config_boundary.py tests/test_core.py tests/test_selection.py tests/test_selection_capacity_boundaries.py tests/test_recommendations.py tests/test_project_routes.py`为83 passed（14.35秒）；`.venv/Scripts/python.exe -m compileall -q manialab tests`与`git diff --check`通过。
- PR #135 Windows run `44137877217`与合并后 main run `44138712410`均通过：全量Python/API为2,531 passed、89 skipped，Edge UI为31 passed；模型与便携构建步骤按条件跳过。
- 本阶段未运行研究训练、模型、用户音频或参考谱，未访问保留的 MAESTRO 测试，也未修改研究脚本或研究产物。结构迁移不代表音乐质量或模型效果提升；osu! 试打仍独立验收。

## 2026-11-18 第三十七阶段：候选选音与 LN 目标边界

- 将`select_notes`和`choose_holds`的选音、密度/和弦限制与LN目标求解迁入`manialab/candidate_selection.py`；`generation.py`直接依赖新模块，候选起音筛选与后续`arrange.py`定轨成为显式分层。
- `arrange.py`保留原有公开函数签名的兼容包装，并继续让旧`_solve_binary`与`choose_holds`替换点生效；研究脚本与测试无需批量改导入路径。新模块不反向导入`arrange.py`。
- 选音/LN、容量、来源边界、生成时钟与冻结回放定向测试复验为143 passed（13.43秒）；`compileall`与`git diff --check`通过。PR #136 run `44217101175`和最终提交 run `44217665635`均通过：Python/API为2,533 passed、89 skipped，Edge UI为31 passed；合并提交`3f3879fb91d3072a2e321ede9ae538414f4e2282`的 main run `44218152214`通过。
- 未运行研究训练、模型、用户音频或参考谱，未访问保留的 MAESTRO 测试，也未修改研究脚本或研究产物。重组代码不代表音乐质量或模型效果提升；osu! 试打仍独立验收。

## 2026-11-18 第三十八阶段：候选边界重构后的冻结回放

- 按原协议、数值线程上限2串行重放默认26例与分段40例，并分别对照既有`frozen-generation-fixed-20261117`和`frozen-generation-sections-fixed-20261117`。计划SHA与协议一致：`0696d759598b57ceaaefe3eb52ca3e78f0d443cee60c0816c4cda2d733fad593`、`bdcc88e9d1e9d56679f9f69b116a16f82a53099318663bf661420d4fef3c5a8b`。
- 66/66例均通过结构、网格、输出容量及导出回读检查，容量违规为0；与既有fixed基线逐案无音符、证据或轨道差异。自动Timing门槛为9/26和12/40，与此前记录相同。这确认候选模块边界迁移保持确定性行为，不代表Timing、采音或音乐质量提升。
- 使用`OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`、`OPENBLAS_NUM_THREADS=2`运行`replay_frozen_generation.py`。输出分别位于ignored的`test-data/frozen-generation-architecture-20261118/`和`test-data/frozen-generation-sections-architecture-20261118/`，报告SHA为`3da98cd55a8a6aa0002ac0d19f2ac7bda55c47edc77ab32ea71545f22420d2e8`和`ec44562aac67e4db261949bee4b02524f3e9c14d57c4e30d41abfd4ca30b7beb`。默认、分段命令分别使用`plan.json`/`section-plan.json`及各自同名的`fixed-20261117`基线目录。
- 未读取音频、参考谱或保留的 MAESTRO 测试，未运行模型。结果仍来自已用于开发的10首片段；独立泛化集、多曲来源完整LN/乐句证据、真实声音主次与可闻性以及 osu! 试打尚未完成。

## 2026-11-18 第三十九阶段：非空稀疏窗口限制自动 Timing

- 只读复核默认与分段冻结报告：现有自动Timing门槛分别接受9/26、12/40；所有拒绝项均有5个已评估窗口、0个不足窗口。因此这些结果不能证明新发现的稀疏窗口缺口造成了既有拒绝。
- 新增合成复现：前后完整窗口各有24个对齐起音，中间完整12秒窗口有4个偏拍起音；旧逻辑仍标记稳定并允许未确认量化。共享窗口计数用于估速门槛和分析诊断；完整且非空、少于5次起音的窗口要求手动确认，并在结果中列出`sparse_windows`。完全静音窗口及不足12秒的尾段不因此拒绝；显式确认仍可继续。
- 定向`.venv/Scripts/python.exe -m pytest -q tests/test_timing.py tests/test_timing_diagnostic_clock.py tests/test_generation_clock.py`为73 passed；设置`OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`、`OPENBLAS_NUM_THREADS=2`后全量`.venv/Scripts/python.exe -m pytest -q`为2,535 passed、89 skipped（71.33秒）。`.venv/Scripts/python.exe -m compileall -q manialab scripts tests`和`git diff --check`通过。
- 没有重跑冻结生成：冻结计划保存的是已算出的Timing，不会重新执行估速；新合成回归直接覆盖了缺陷边界。未运行研究脚本、模型或用户音频，未读取参考谱或保留的 MAESTRO 测试。PR #137 Windows run `44225629152`通过，合并后 main run `44226100575`成功；长期Timing与音乐质量仍需独立泛化数据、试听和 osu! 试打验收。

## 2026-11-18 第四十阶段：通用声部激活缓存边界

- 将分离声部 Basic Pitch 激活缓存的版本、NPZ 读写及音频/分离身份校验迁入`manialab/audio_cache.py`；`audio.py`继续负责推理与分析编排，并通过原路径使用缓存。缓存版本3、文件名、字段及压缩格式不变；钢琴、吉他和 Tsumugi 专用缓存保持独立。
- 新增缓存格式往返、音频与分离身份失配、旧版本/缺字段/损坏/缺失缓存，以及新模块不加载 ONNX 推理运行时的测试。独立只读审查未发现行为兼容问题；既有`test_stems.py`与音尾来源测试覆盖`analyze_deep`缓存重建、复用和回读路径。
- 定向`.venv/Scripts/python.exe -m pytest -q tests/test_audio_cache.py tests/test_audio_cache_boundary.py tests/test_stems.py tests/test_provided_stems.py tests/test_audio_release_provenance.py tests/test_separation.py`为53 passed（4.40秒）；`.venv/Scripts/python.exe -m compileall -q manialab/audio.py manialab/audio_cache.py tests/test_audio_cache.py tests/test_audio_cache_boundary.py`和`git diff --check`通过。未运行全量本地测试。PR #139 最终提交 Windows run `44233665325`通过：Python/API为2,543 passed、89 skipped，Edge UI为31 passed；合并提交`acbfcdd682a6e350a20a3e60aaad187923938147`及 main run `44234039374`均成功；模型及便携构建步骤按触发条件跳过。
- 未运行研究训练、音频模型或用户音频，未读取参考谱或保留的 MAESTRO 测试，也未改动研究脚本与研究产物。该结构调整不代表模型或音乐质量提升；独立泛化、试听与 osu! 试打仍待验收。

## 2026-11-18 第四十一阶段：Mapperatorinator 事件适配边界

- 将 note-like 记录到 Mapperatorinator `Event` 序列的转换迁入`manialab/mapper_event_adapter.py`；`domain.upstream_events`保留同签名兼容入口并惰性委托，使领域模块导入不再加载第三方事件类型。
- 保持事件时间/轨道排序、同刻 LN 尾优先顺序及每个事件的 TIME_SHIFT/MANIA_COLUMN 展开方式不变。新增逐事件顺序与24-token数量回归，以及领域/适配器双向导入边界测试。
- 定向`.venv/Scripts/python.exe -m pytest -q tests/test_mapper_event_adapter.py tests/test_core.py tests/test_generation_clock.py tests/test_upstream.py`为60 passed（12.02秒）；`.venv/Scripts/python.exe -m compileall -q manialab/domain.py manialab/mapper_event_adapter.py tests/test_mapper_event_adapter.py`和`git diff --check`通过。未运行全量本地测试。PR #140 Windows run `44236677859`通过：Python/API为2,548 passed、89 skipped，Edge UI为31 passed；模型及便携构建步骤按触发条件跳过。
- 不运行模型、研究训练或用户数据，不访问参考谱及 MAESTRO 保留测试。该依赖边界调整不代表模型或音乐质量提升。

## 2026-11-18 第四十二阶段：半速 / 倍速歧义门槛

- 自动 Timing 检出同一首歌存在网格拟合合格的半速 / 倍速 BPM 时，默认要求用户试听确认；有拍点模型时，只有最佳 BPM 自身支持度至少`.55`且比每个八度竞争项至少高`.20`，才据此解除歧义。第三种 BPM 获得较高模型支持、或模型对竞争项支持相近时，不再误标稳定。
- 所有未消歧的八度竞争 BPM 均进入`tempo.candidates`供前端选择，并按 BPM 去重；返回诊断保留竞争项及拍点支持差。无歧义时延用原有稳定与量化路径。
- 合成回归覆盖等强和强弱起音、快慢两个方向、轻微抖动、模型明确消歧、模型支持接近、无关 BPM 更获模型支持、多个八度竞争项以及未确认量化拦截。定向`.venv/Scripts/python.exe -m pytest -q tests/test_timing.py tests/test_timing_diagnostic_clock.py tests/test_generation_clock.py`为81 passed；数值线程限制为2时全量`.venv/Scripts/python.exe -m pytest -q`为2,556 passed、89 skipped（76.72秒）；Edge `MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=8978 npm run test:ui`为31 passed（约1.2分钟）；`.venv/Scripts/python.exe -m compileall -q manialab scripts tests`和`git diff --check`通过。
- 本阶段只验证合成 Timing 门槛与现有接口兼容；未运行音频模型、研究训练或用户音频，未读取参考谱及 MAESTRO 保留测试，也未改研究脚本或研究产物。尚未用小数精确候选的独立拟合指标参与八度竞争筛选；该边界、独立泛化、真实试听和 osu! 试打仍待后续验证。结果不代表真实音乐质量或模型效果提升。

## 2026-11-18 第四十三阶段：精确 BPM 竞争项保留独立拟合

- 将精确 BPM 候选自身的覆盖率、误差与拍点支持用于半速 / 倍速竞争筛选；未消歧的精确竞争项继续进入可选 Timing 候选。精确候选之间按`.001 BPM`去重，保留同一整数 BPM 舍入附近、但全曲漂移不同的拟合；普通 BPM 候选沿用`.15 BPM`容差。
- 新增合成回归覆盖`120.37 BPM`的`60.18 BPM`精确竞争项，以及相差`.12 BPM`的近邻精确候选和半速竞争列表，确认各候选保留自己的拟合质量与选择入口。
- 定向`.venv/Scripts/python.exe -m pytest -q tests/test_timing.py`为51 passed；最终全量（数值线程上限2）`.venv/Scripts/python.exe -m pytest -q`为2,558 passed、89 skipped（71.14秒）。`.venv/Scripts/python.exe -m compileall -q manialab scripts tests`及`git diff --check`通过。PR #142 Windows run `44248668628`通过并合入`main`为`f35b22f6c50ddfe9a1aa0926ab2121087dce6451`：Python/API为2,558 passed、89 skipped，Edge合成UI为31 passed；模型下载及便携构建步骤按条件跳过。
- 未运行模型、研究训练或用户音频，未读取参考谱、下载语料及 MAESTRO 保留测试，未改研究脚本或逐音产物。合成 Timing 通过不代表独立音乐泛化、真实试听或 osu! 试打验收；这些长期目标仍待完成。
- 后续继续检查 Timing 候选的 API/UI 往返与人工确认边界，再转入独立泛化和试听验收，不重复既有模型实验。

## 2026-11-18 第四十四阶段：冻结 Timing 复算与候选往返回归

- 新增`scripts/reestimate_frozen_timing.py`，按固定计划 SHA `0696d759598b57ceaaefe3eb52ca3e78f0d443cee60c0816c4cda2d733fad593`及逐份分析 SHA 校验输入；将26个配置按10份唯一分析去重，重用已保存的`timing_attacks`和拍点模型`beats_ms`，不读音频、不运行模型、不覆盖已有输出目录。汇总报告保存在 ignored `test-data/frozen-timing-reestimate-20261118/report.json`，不输出逐音内容。
- 10/10份分析的 BPM 与 stable 结论均和冻结分析一致，仍只有3/10为 stable。`4709745`细分从3改为4，覆盖率`0.4987 → 0.6324`、中位误差`30 → 18.75 ms`、相对冻结计划 Offset 为`-1 ms`；`5647839`细分从3改为4，覆盖率`0.3864 → 0.5260`、中位误差`35.25 → 24.31 ms`、计划 Offset 差为`0 ms`。两者仍为`stable=false`；历史代码复算结果相同，未发现近期 PR 引入回归。该复算来自既有开发片段，不构成独立泛化或音乐验收。
- 新增合成浏览器往返：选择`120.37 BPM`精确候选（原音频 Offset `115.25 ms`），把制谱提前量从`22.5 ms`改为`37.5 ms`后，谱面 Offset 从`93 ms`调整为`78 ms`并清除旧确认；未确认生成被后端拒绝且不保存候选，勾选确认后生成、保存并重载，BPM、Offset、提前量和确认状态均保持。
- 完整验证：数值线程上限2时`.venv/Scripts/python.exe -m pytest -q`为`2,562 passed, 89 skipped`（70.20秒）；`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=8981 npm run test:ui`为`32 passed`（约1.2分钟），其中 Timing 文件11项全过；`.venv/Scripts/python.exe -m compileall -q manialab scripts tests`、`node --check tests/browser/timing.spec.cjs`及`git diff --check`通过。没有运行模型、训练、用户音频或参考谱，没有访问 MAESTRO 保留测试。
- PR #143 Windows validation run `44260840665`成功；合并提交`271c1b1148d08e18fac2d1656c2c786c6b5519d3`的 main run `44261260834`也成功。当前源码树与`origin/main`一致。
- 本机 D 盘清理：删除的 dev.3/dev.4 发布 ZIP 大小与 SHA-256 均和仍在线的 GitHub Release 资产一致；清除时删除112个 Python `__pycache__` 目录及`.pytest_cache`。本地 dev.14 ZIP 对应 Release 不存在，故保留；dev.17/dev.18 包、全部模型、研究数据和验证报告也保留。D 盘可用空间从清理前`50.30 GiB`升至清理后`52.74 GiB`，测试运行后复测为`52.70 GiB`；仅清理本地可再下载副本与可重建缓存，没有改动 Git 跟踪数据。
- 后续转入独立歌曲泛化、真实试听和 osu! 试打；现有 Timing 结果仍不能替代音乐质量验收。

## 2026-11-18 第四十五阶段：钢琴起音候选到 6K–10K 谱面对照

- 复用 Nova《樱花草》已有 Kong、Transkun 与起音保留输出，不重跑模型；同一音频 SHA、非钢琴候选 SHA、表达计划、手动时钟和生成参数保持不变。钢琴两侧只比较起音/音高，统一强度为0.5；不使用 Transkun velocity 或 offset，关闭钢琴LN，不读取 Reference。
- Kong 有1,557个钢琴候选，Transkun官方1,275个，起音保留版1,277个。每种键数最终选中的钢琴物件为173对169，鼓为301对303，总物件为1,614对1,612；461条 LN 均来自 `other`。起音保留版新增的2个候选没有改变任一键数的最终选音。
- 6K、7K、8K、9K、10K 各输出Kong/Transkun官方/起音保留三张谱，共15张。每种键数Transkun相对Kong有1,553个相同事件、59个新增、61个移除；布局代价分别变化`-4.444`、`-2.739`、`-12.998`、`+2.421`、`-10.386`。布局代价是启发式，方向并不一致，不能据此认定键型更好；起音最大移动仍为101ms，超过30ms的移动计数仅从2,908降至2,905。结构、固定网格和导出回读通过只确认制谱正确性，不代表采音或音乐验收。
- 定向`.venv\Scripts\python.exe -m pytest -q tests/test_piano_transcription_chart_ab.py`为2 passed（1.68秒）；数值线程上限2时全量`.venv\Scripts\python.exe -m pytest -q`为2,568 passed、89 skipped（66.74秒）；`.venv\Scripts\python.exe -m compileall -q manialab scripts tests`和`git diff --check`通过。研究复现入口为`scripts/check_piano_transcription_chart_ab.py`；本地产物在ignored `test-data/audio-transcription-chart-ab-20261118/`，汇总SHA-256为`C05D1AADAB50CB5070E9F0B1EC5786FD40921A52BBBA69B73615065C32830B3C`。只提交脚本、测试和去逐音数据后的文档。
- 既有采音研究仍是混合结论：11段历史开发实录起音F1从`0.953677`升至`0.986837`，但91份控制的漏采从8升至25，另有20个Kong能找到而Transkun漏掉的实录起音。此次对照没有直接听审、试打或读取Reference，故不证明《樱花草》钢琴采音正确，也不证明6–10K键型改善；当前证据不支持全局替换默认采音。
- 下一步先对可追溯的新增/移除片段做盲化原声听审，再由玩家比较相同范围的6K–10K键型；结论记录完成前不将实验候选设为默认。长期歌曲泛化和真实音乐质量目标仍未完成。
- PR #145 Windows validation run `44283528751`通过；合并提交`f20626f543508200a829bccb74c9e8199bf02b59`的 main run `44284232486`通过。Python/API与合成浏览器流程成功，模型下载和便携构建按触发条件跳过。

## 2026-11-18 第四十六阶段：采音与 6K–10K 键型效果审计

- 复核《樱花草》Kong/Transkun 对照：Transkun 候选少约18%，每种键数钢琴最终物件少4个；1,553个相同事件、59个新增、61个移除。布局启发式在6K/7K/8K/9K/10K分别变化`-4.444/-2.739/-12.998/+2.421/-10.386`，不支持一致改善；起音最大移动仍101ms，超过30ms的移动仅`2,908→2,905`。结合既有实录 F1 提升但控制漏采`8→25`及20个新增实录漏采，仍不切换默认采音。
- 扩充`scripts/summarize_frozen_generation_matrix.py`：逐案读取并校验冻结`variant.json`哈希，保留每列/声部列负载，并对同一分析、同一配置下的不同键数比较候选ID、起止时间及轨道列。新增6项矩阵摘要回归，覆盖键列负载、跨键变化和非法轨道。
- 默认矩阵的10份分析在6K和10K都选出3,147个相同候选、531条LN；6K到10K有2,737/3,147个事件换列（86.97%），起止时间、选中ID无变化。6K到7K/8K/9K各只有2份分析，换列比例为68.86%/73.41%/84.07%，不能据此判定手感改善。分段配置的10K `melody` 与`layers`分别为469/298与1,388/904个事件/LN，显示参数敏感性而非质量高低。默认1,363条、分段7,986条LN的音尾来源全部为`unknown`；自动Timing仍仅3/10通过。
- 完整数值、限制与下一步见[生成矩阵、键型与采音效果审计](generation-effect-audit-2026-11-18.md)。新汇总仅读取既有冻结计划、回放和变体；未读音频、Reference或保留测试，未运行模型/推理，也未改生产生成与采音默认行为。ignored 汇总报告分别为`test-data/frozen-generation-matrix-key-layout-default-20261118/report.json`（SHA-256 `AC275C7F3F95FF444B1353E29A00BD02522AB91EF66441AA80ECAE071379490B`）和`test-data/frozen-generation-matrix-key-layout-sections-20261118/report.json`（SHA-256 `116166969042C7F1BAF1D2381C8A336100F97F9128D014A340A62745E3DFC93C`）。
- 验证：`.venv/Scripts/python.exe -m pytest -q tests/test_frozen_generation_matrix.py tests/test_piano_transcription_chart_ab.py`为8 passed；`.venv/Scripts/python.exe -m compileall -q scripts tests`和`git diff --check`通过。PR #147 Windows run `44290064124`通过，合并提交`f947c2862e2d2cc40c14c7e3b6b1b4a8db303000`的 main run `44290737467`也通过；Python/API 与 Edge UI 检查成功，模型下载及便携构建按条件跳过。
- 同日生成并校验本地盲测包 `test-data/audio-transcription-chart-ab-playtest-20261118/Sakura-Piano-AB-6-10K.osz`：10 张 Mania 谱（6K–10K 各两张）、共用同一音频、ZIP CRC 通过、无模型名称泄漏，包 SHA-256 `f399fbd88c7efec888254c045852f30f28d3dff747b38fc9b571c336ee89a49e`；解盲表保存在包外的 ignored 文件。此项只确认包完整和盲化，不代表已在 osu! 加载或完成试听/试打。
- 该阶段只完善效果证据和人工验收入口，不是采音或键型提升。真实声音盲听、6K–10K玩家试打和独立歌曲泛化仍未完成；当前证据仍不支持替换默认采音。

## 2026-11-18 第四十七阶段：全曲 LN 音尾分段定位

- 新增只读 `scripts/summarize_frozen_generation_ln_segments.py`，校验冻结计划 SHA、10 个 case ID 和每个 `variant.json` SHA，再按冻结结构段落/声部统计 detected、expressed 与最终 release reason；分别报告原音频时钟的模型/表达尾移和谱面时钟的量化尾移。段落边界采用左闭右开；缺失起音、快照或尾端明确保留，不补猜。回归覆盖两个时钟分离、恰好30ms阈值、段落边界、缺失来源、段落外短键、重复 ID、哈希不符和输出目录保护。
- 复核版输出位于 ignored `test-data/frozen-generation-ln-segments-reviewed-20261118/report.json`，SHA-256 `14ECE116D7592A675A8DFB364920B80A08F429E48A8FCA5F2AA5BE249843EF16`。10 个冻结配置共 4,043 条 LN，全部可归入段落、均有快照和可测尾端；`expressed.end - detected.end` 的大于30ms计数为0，`note.end - evidence_end` 的大于30ms计数为698。默认6K–10K各有70/72/75/73/74条，无人声配置为64/65/69/68/68条；较集中的 `other` 声部段落为187.501–215.474s和97.501–110.069s。跨配置会复用事件，这些是定位数量而非独立样本。
- 汇总结果说明大尾移发生在量化阶段，不是表达阶段移动了模型预测尾端；不证明模型尾端等于真实松键，也不判断格线长短更好或更差。报告仅用于安排原声复听，未读取音频、Reference或 MAESTRO 保留测试，也未运行模型/用户数据。
- 本地验证：`.venv\\Scripts\\python.exe -m pytest -q tests/test_frozen_generation_ln_segments.py`为7 passed；`compileall`与`git diff --check`通过。PR #149 Windows run `44310862594`通过，进度更新后 run `44311633143`通过（Python/API与合成浏览器流程成功；模型与便携构建按条件跳过）。PR #149 已于2026-11-18合并，merge commit `d991217fd260e9bf8ec66c05ad5294beb0ba6492`；合并后 main run `44312826642`通过，Python/API与合成浏览器流程成功，模型与便携构建按条件跳过。

## 2026-11-18 第四十八阶段：钢琴采音差异盲听包

- 修正`scripts/build_piano_onset_blind_review.py`：先按 A/B 报告中的`audio_reference`定位源音频，再核对报告记录的音频 SHA-256；此前哈希核对发生在路径赋值之前，脚本会直接报未定义变量。
- 新增真实路径/哈希匹配、错误哈希拒绝及生成包 CRC/盲化回归。定向`.venv\Scripts\python.exe -m pytest -q tests/test_piano_onset_blind_review.py`为7 passed；`.venv\Scripts\python.exe -m compileall -q scripts/build_piano_onset_blind_review.py tests/test_piano_onset_blind_review.py`和`git diff --check`通过。
- 使用已缓存《樱花草》10K输出、无模型推理生成ignored评审包`test-data/piano-onset-blind-review-20261118/piano-onset-review.zip`：84段差异（Transkun独有40、Kong独有44），每段原起音前1秒/后1.5秒，22050Hz，无逐段归一化。解盲映射留在包外；ZIP SHA-256为`b5339477e8932d563fb87a3c162c27003295eeabd5de5677089e41e6e3c8d298`，manifest SHA-256为`e8b524ff86f6535b8854f5982bdd0d3b9585b40e34871d288e6a17d2f0c6a376`。独立复核确认84个WAV时长/采样率、ZIP CRC、盲化标签和包外解盲项均正确。
- 此包还没有人工评分，不产生采音改善结论；其验证只覆盖10K已选钢琴物件的新增/移除可闻性。此前报告中 Transkun 开发录音F1较高但控制样本漏采增加，仍不支持换默认。键型更无玩家试打反馈；现有6K–10K `.osz`能评整体谱面偏好，不能隔离纯定轨。
- 验收下一步：用户或玩家完成盲听CSV和同键数osu! A/B试打记录后，先锁定评分表再解盲汇总；如需证明纯定轨手感，另做固定音符/Timing/LN端点、只改变轨道的盲测谱包。PR与CI状态待记录。

## 2026-11-18 第四十九阶段：编辑后的段落书签

- 通过真实 `/edit` 与 `/export/osu` 接口复现：在带段落书签的音频候选上仅修改一条轨道，原来的 `1480,2980` 书签全部消失。手工编辑与局部生成现共用候选生成时保存的段落和表达记录。手工编辑重新计算实际音符报告，旧候选缺少段落快照时书签留空。
- `.\.venv\Scripts\python.exe -m pytest -q tests/test_edited_candidate_bookmarks.py tests/test_editing.py tests/test_regional_selection_report.py tests/test_evidence_snapshot.py` 为 23 passed。回归通过 HTTP 连续换轨、删音、重开项目并回读 `.osu` 和 `.osz`。输入为临时目录中的合成谱面。
- 接下来验证同一谱包各难度的保存与恢复、原谱变速和拍号的试听。人工音乐评审仍待记录。
- PR #151 合并后的 `main` run `44322594836` 成功。PR #152 run `44324693260` 和合并后 `main` run `44325499743` 均成功，合并提交为 `5902e947`。

## 2026-11-18 第五十阶段：同一谱包的多难度编辑与原谱试听

- 难度切换按原谱保存候选、手工修改、活动版本和分析。解码、声部与推理缓存按音频分开，同一音频可复用派生文件，既有项目保留原目录。生成 Mapperatorinator 谱之后也能从难度列表返回原音频项目。
- 严格重定轨模式的滚动节拍器和试听 WAV 读取各段原谱红线与拍号，换速交界只有一次点击。首次导入可从原音频直接解码试听窗口，带缓存时沿用已有 WAV。已回归验证两条路径从非整数毫秒起点截取至音频末尾的样本一致。
- 源码验证真实输入 `test-data/audio-transcription-chart-ab-playtest-20261118/Sakura-Piano-AB-6-10K.osz`。运行 `scripts/verify_chart_workspaces.py <源包> --sha256 f399fbd88c7efec888254c045852f30f28d3dff747b38fc9b571c336ee89a49e --out test-data/acceptance-chart-workspaces-20261118`。10/10 张谱各保存两份候选，关闭应用服务后逆序重开，16,130 个物件、4,610 条 LN 的起止及除轨道外的原始物件字段全部一致。两个导出格式的谱面内容相同，各自音频 SHA 匹配。该数字按十张难度累加。报告 SHA-256 `77a8b7f3888967edc63666868b097a883095f5b273ffa4a6a4b2470632eca6f5`。
- 本地项目切换、编辑与服务路由定向测试 48 项通过。`tests/test_imported_audition.py tests/test_audition.py tests/test_audio_io_boundary.py` 11 项通过。Edge 已覆盖难度保存与打开以及 3/4 转 5/4 的试听 WAV 与实时网格逐拍核对。人工原声听审与 osu! 玩家手感评分待记录。
- PR #153 首次运行的轻量依赖回归指出 `project_routes` 在导入时加载 SciPy。`chart_workspace` 现仅在首次解析新谱时导入段落标记解析器。
- PR #153 最终 Windows run `44329416293` 通过，合并提交 `80c1b71b`。

## 2026-11-18 第五十一阶段：可填写的盲听评分与同音符定轨试打

- 钢琴盲听网页和 `review-template.csv` 共用七列，`audio_start_ms` 保存整数毫秒，并保留音频相对路径。Edge 实际填写含双引号、逗号、中文和换行的备注，下载后通过 CSV 解析核对七列。定向 Python 8 项通过（包括定轨包回归），Edge 实际下载 1 项通过。
- 用 `scripts/build_piano_onset_blind_review.py --input test-data/audio-transcription-chart-ab-20261118 --output test-data/piano-onset-blind-review-csv-20261118` 生成新的本地评审包。旧包和新包的 84 份 WAV、评分模板和解盲表逐字节一致，两包 CRC 通过。新版 `piano-onset-review.zip` SHA-256 为 `842de24e7e415cbf1d1ad7eb3aa8a26a75e4b3f03dfc2bbaa550c4844051f55c`。
- 另用 `scripts/build_layout_playtest.py <第四十六阶段盲测包> --sha256 <该包 SHA> --chart Source-A-10K.osu --out test-data/layout-only-playtest-20261118` 构建定轨试打。所有难度共用一种选音、相同原声、1,612 个物件、461 条 LN 和相同变速。两侧使用产品内建的两种定轨模式、同一随机种子且关闭键型先验。每个键数检查导出后的全部非轨道物件字段、Timing 和占轨。6K 至 10K 分别有 572、739、702、855、664 个可玩物件的位置不同。
- 玩家入口 `test-data/layout-only-playtest-20261118/participant/` 含 `Layout-AB-6-10K.osz`、`scores.csv` 和填写说明。谱包 SHA-256 `eb2369cc1fd954fae10f01aa902d39a6bae64bb40fd130c721bc08621cd14735`。需要玩家先填写实际试打结果再解盲。当前人工盲听和手感偏好均待验收。

## 2026-11-18 第五十二阶段：两首外部音乐的采音到出谱验证

- 分离启动器显式传入 `--threads 2`，覆盖四声部、六声部和只补节拍三条路径。worker 的直接调用默认值也改为 2；算子间线程为 1。`tests/test_separation.py` 两项使用真实 FFmpeg 转码并检查子进程参数。
- 在同一新目录先后运行 `scripts/verify_independent_audio.py prepare test-data/independent-audio-20261118` 与 `scripts/verify_independent_audio.py verify test-data/independent-audio-20261118`。公式、输入和判据先于推理记录在 `protocol.json`（SHA-256 `5656b2215a86d813400e2234b8a07b2799f667e79d318cffb6ca7ff90e4d7bfd`）。上游 HQ OGG、原始署名与转换后的各 24 秒 WAV 均保留在该 ignored 目录。
- Setuniman《Sweet Waltz》和 Admiral Bob feat. Snowflake《Choice》的鼓贝斯选轨分别在 31.36 和 20.32 秒完成四声部实算，自动接受 150/68 BPM。各键数分别为 120 个物件/90 条 LN、100 个物件/18 条 LN。共 10 份 OSZ 保留预期音频，1,640 个起点和放键端点全部落线，回读键位与端点一致。总报告 SHA-256 `84a6ac045372df545dda0817df67461ab3fecd9d90f435799670effcb0c264b9`。
- 《Sweet Waltz》的模型下拍呈三拍周期，首次音频模式导出固定为四拍；《Choice》10K 有 8 个物件归为 `other`，含 1 条 LN。[验收记录](acceptance-20261118.md)给出来源和复听位置。两段均已备好试听与 6K–10K 谱包。当前未收集真人评分。
- PR #154 合并后 `main` run `44331863473` 成功。PR #155 Windows run `44332141244` 通过，合并提交 `29880324` 的 `main` run `44333019827` 通过。

## 2026-11-18 第五十三阶段：把有证据的小节强拍带到试听与导出

- 音频模式在稳定时钟上核对整段拍点、下拍覆盖和红线相位。至少有三个完整且一致的小节，才采用检测到的每小节拍数。草稿时钟、候选快照、手工编辑、局部重写、两种导出、实时节拍器、试听 WAV 与 Mapperatorinator 的固定红线上下文共用此规则。旧候选沿用四拍，原谱重定轨采用原谱的变速与拍号。
- 复用第五十二阶段《Sweet Waltz》的分析，在独立项目重新生成 6K–10K。五份 `.osu` 各只有红线的一个字节从 `4` 变为 `3`。五份合计 600 个物件、450 条 LN、1,050 个端点，原音频逐字节相同。对导出时钟和偏移 100ms 的对照分别渲染八秒 WAV，独立解码逐拍核对每份的 20 次点击。回放脚本为本地 `test-data/check-cached-waltz-meter-20261118.py`，输出在 `test-data/cached-waltz-meter-20261118/`，报告 SHA-256 为 `b02bd5a41850b3529805f06640e6919fd5b9e4c7cad36bd604b105dd686e5fb7`。
- 定向 Python 97 项和 Edge 13 项通过。设置 `OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`、`OPENBLAS_NUM_THREADS=2` 后运行 `.venv/Scripts/python.exe -m pytest -q` 为 2,620 passed、89 skipped（104.35 秒）。`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=8984 npm run test:ui` 为 36 passed。`.venv/Scripts/python.exe -m compileall -q manialab scripts tests` 与 `git diff --check` 通过。
- 下一步用 osu! 官方解码器独立读回谱面，并验证 Windows 便携可执行文件。采音、声部分段和 6K–10K 手感的真人评分入口见[验收记录](acceptance-20261118.md)。

## 2026-11-19 第五十四阶段：官方 Mania ruleset 独立读回谱包

- 新增 `scripts/verify_official_decoder/`：固定 `ppy.osu.Game.Rulesets.Mania` NuGet `2026.1026.0`（对应上游 commit `b94ee4d835d65683d027ae8f47a3e61fa4899ad4`），使用官方 `LegacyBeatmapDecoder` 和 Mania converter 解析 `.osz`；独立比较器核对规则、键数、物件类型/列/起止、原声引用、红线拍号及滚速变化。
- 官方解码实际读取 17 个 `.osz` 中的 26 张难度：17,824 个物件、5,602 条 LN、23,426 个起放端点、27 条红线与 3 次滚速变化均与 ManiaLab 读取器一致。定轨 A/B 的 5 对音符起止与原声一致而轨道不同；《Sweet Waltz》5 对新旧谱物件一致、拍号分别为三拍和四拍。合成控制还覆盖拍号、速度变化与负向输入拒绝。官方报告 SHA-256 `4204c7c75cea95294fe3df5ab822cd46c26017f811f6a2a21e767afde6df793d`；对照摘要 SHA-256 `a060c8026e303f27a15a0adfe8fd5023ecc2443714074cc4656ec8ff69df080d`。本地完整谱包/报告/SDK 均在忽略的 `test-data/official-decoder-20261118/`，不提交。
- 新增比较器负向测试，分别篡改键数、列摘要、起放端点摘要、红线拍号、滚速及音频引用并确认拒绝；另覆盖偏移校正的正负半毫秒舍入边界。`tests/test_audio_meter.py tests/test_official_decoder_compare.py tests/test_upstream.py` 为 42 passed。最终全量 `.venv/Scripts/python.exe -m pytest -q` 为 2,629 passed、89 skipped（69.25 秒）；`compileall`、JS 语法检查与 `git diff --check` 通过。复用同一 17 包输入进行第二次官方解码，26 张谱的所有汇总关系再次一致；复跑小结位于 ignored 的 `test-data/official-decoder-20261118/validation/replay-20261119/summary.json`，SHA-256 `422c4daf668a178148ee69927bdfd617bb59f10b83a8ab29fc1bf3f9885d21ee`。
- PR #156 Windows run `44344455889`通过，合并提交 `64d2855bba174de07781206cfec091ec12042381`。合并后手动 `workflow_dispatch` run `44345356104` 全绿：Python/API 为 2,629 passed、89 skipped（JUnit 2,718 项，0 failures/errors），Edge 36 passed；固定音频 worker 与完整 Windows archive 构建成功，便携 EXE 的离线/干净 PATH/Unicode 数据路径测试及退出 worker 清理测试通过。实际 workflow 总用时 10 分 30 秒。`standard` profile 包含四声部、六声部与钢琴能力，不含 guitar/instruments。EXE SHA-256 `59c8c41f775365de318d3d99093f258c36821ac1fa187fb47e88579c4a083e7c`；构建的 `ManiaLab-0.1.0-alpha.6-dev.18-windows-x64.zip` 为 888,813,966 字节、SHA-256 `f64d0562dd76c700c8d4be6f69df2075176b323684601e32d81cf1cd9ed175e8`。验证摘要位于 ignored 的 `test-data/portable-verification-20261119/`，便携报告 SHA-256 `d44086db197b763f70bb19f0691b53d8944a995bbfd1aaf73af1e7ceb8167fb`、退出报告 SHA-256 `6564844645e63e2dcf4352ba939ba6ec769fdf91c09e7787caafca799185e95d`。ZIP 是 Actions 临时构建产物，并未发布成新 Release；Release 页当前仍指向 dev.18 的旧包。仅有 Actions `setup-node` Node 20 迁移至 Node 24 的弃用提示。
- 固定依赖版本及复现步骤见 [官方解码对照](../scripts/verify_official_decoder/README.md)。工程与独立音乐链路已有可复核验收，但这不代表采音、声部清晰度或玩家体验已验收；钢琴盲听与 6–10K/LN 玩家试打评分仍待收集，长期歌曲泛化验收继续开放。

## 2026-11-19 第五十五阶段：合并后完整便携构建验收

- 在官方 Mania 解码及三拍强拍改动合入 `main` 后，对源码提交 `64d2855bba174de07781206cfec091ec12042381` 手动触发 Windows Release workflow。锁定环境的完整 pytest JUnit 为 2,718 项、0 failures/errors、89 skipped；Edge UI 为 36 passed。钢琴 ONNX 确定性构建、音频 worker 和 standard Windows archive 均完成。
- 从实际 EXE 启动两次，干净 PATH 下完成离线分析及 6K–10K 五档生成导出（均零 off-grid），验证推荐选择、过期 Timing 拒绝、取消/失败恢复，以及 Unicode 音频名 `音频测试.wav` 的项目关闭重开。manifest 报告 114 个源文件、2,360 个 worker 文件、standard profile；能力为 audio/four/six/piano，guitar/instruments 未启用。EXE SHA-256 `59c8c41f775365de318d3d99093f258c36821ac1fa187fb47e88579c4a083e7c`。
- 独立关停报告确认关闭应用时 worker 已退出。ZIP 为 888,813,966 字节、SHA-256 `f64d0562dd76c700c8d4be6f69df2075176b323684601e32d81cf1cd9ed175e8`。验证资产保留在 ignored `test-data/portable-verification-20261119/`；只下载了小型验证报告，未下载近 889MB ZIP。该 zip 为临时 Actions artifact，未经版本号/发布流程发布；当前公开 Release 仍是 v0.1.0-alpha.6-dev.18。因此后续源码已验，用户直接下载的公开版尚未包含本阶段和更早的 main 更新。首页和[工程验收记录](acceptance-20261118.md)已注明此边界。
- 完成状态严格限定为工程和机器验证：官方读回、Windows 实际 EXE 运行以及两首独立音乐到 6K–10K 导出均有可复现结果。人工钢琴盲听表及 6K–10K/LN 定轨试打表为空；两首独立歌曲听审尚待真人完成。真实音乐质量、采音和手感目标不宣布通过，不因本阶段 CI 变绿而结项。

## 2026-11-19 第五十六阶段：Core 便携配置与来源完整性

- 公开 Windows 构建切换到 `core` profile：只带 Basic Pitch、应用谱面配置和基础运行时，排除 Demucs/Beat This!/钢琴/吉他/Tsumugi 权重、音频 worker、uv 与 Mapperatorinator 安装器。混音采音、已有谱面重定轨、导入声轨和基础 6K–10K 生成保持可用；没有把授权未核实的 Demucs checkpoint 放进新公开构建。`standard`/`extended` 本地构建所需的先验、模型元数据、分离资源与 Mapperatorinator 安装文件继续按 capability 暂存。
- 补齐 `assets/model-manifest.json`：构建清单、便携 EXE 验证和最终 ZIP 验证均核对 Basic Pitch `v0.4.0` 固定来源 URL 与 SHA-256 `2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec`。发布报告新增 Core 模型身份和按 profile 判定的关停证据；Core 单独确认运行时无音频 worker、应用可退出，非 core 保留原 worker 退出检查。
- 最终实际 Core Windows 构建 ID `codex-core-20261119-b`，产物留在 ignored `build/portable/codex-core-20261119-b/`。EXE SHA-256 `488d85c591c0624c385be9bee2447843c355101b98143558b955e474acb01363`；ZIP 112,643,004 字节，SHA-256 `5d6e845d4772c977db3f93f9a0d63ffdbe73343cf7d3409ee1da8a575ef7d139`。发布收集器逐项验证源码快照、EXE/manifest/ZIP 一致性、Basic Pitch 字节身份、JUnit、portable 报告和 shutdown 报告；ZIP 1,753 个条目 CRC 全通过、无重复路径，仅含三个 Core 资源，未发现其它 ONNX/权重、worker、uv 或安装器文件。
- 实际 EXE 合成运行记录位于 ignored `test-data/portable-core-20261119-b/report.json`，SHA-256 `0422c85fd8ccca9d33323aad46f3b9a75ebe961051b556aa3c3b99147e1b3689`：基础分析得到5个候选；6K–10K各生成5个物件/5条 LN、0 off-grid；导入合成 `other` 声轨分析导出通过；deep/deep6 缺失组件返回预期错误；Unicode 文件名、项目重启保存和损坏音频恢复检查通过。关停报告 `test-data/shutdown-core-20261119-b/report.json` SHA-256 `c192bf8da5e21b4a254457ec7f13703ae57fb3dfc1ae5e4a992ece0953f2ec71`，确认未发现 worker、应用正常退出。JUnit `test-data/core-validation-20261119-a/junit-final.xml` 为2,735项、2,646 passed、89 skipped、0 failures/errors，SHA-256 `4ab15eb302bff04d50d8ca929f237f27bfcbfe81d26d5730313b28f3d600c8a6`；Playwright UI 36 passed。最终再次运行 `compileall`、全部 `static/*.js` 的 `node --check` 与 `git diff --check` 均通过。
- 发布证据摘要位于 ignored `build/portable/codex-core-20261119-b/release/ManiaLab-0.1.0-alpha.6-dev.18-windows-x64-core-validation.json`，摘要 SHA-256 `3d4fbb5c88d89f36f2c37f274076a143f8a9ad2ed043d63087c309b510735eaa`，source commit 为 `af276a9ab44cc838f84f551329b830ef766ead68`；摘要所引用的118个源码快照、ZIP、EXE 和验证报告均逐项校验。PR #158 Windows run `44366735851`成功：Python/API 为2,646 passed、89 skipped，Edge 为36 passed；Core EXE/ZIP、合成流程、worker-free shutdown 与发布摘要收集通过。PR文档同步 run `44368149916`也通过。PR于2026-11-19合并为`6200891e88c53bf03b0a4c317e437c004278056d`；main run `44368922837`成功，Python/API为2,646 passed、89 skipped，Edge为36 passed，Core Windows构建及实际EXE验收通过。main Actions artifact中的ZIP为111,713,727字节、SHA-256 `58900b58213513dec9d782b3989e2c1634c4c6a9d1f391632c0fa97e2f2fd97c`；该包未发布为新Release，当前公开版仍是dev.18。机器/工程验证不代表音乐质量或玩家体验已验收，真人钢琴盲听、定轨试打和两首独立歌曲听审继续待办。

## 2026-11-19 第五十七阶段：真人评审草稿恢复与解盲前锁定

- 钢琴盲听网页新增按音频与每段片段哈希隔离的本地草稿自动保存、重新打开恢复与清除；导出继续保留原七列 CSV。`scripts/refresh_piano_blind_review.py` 可从已有冻结模板与84段渲染片段生成新页面，拒绝不安全片段路径，不重解码原音、不运行模型、不访问或复制解盲映射。
- 新页面和 ZIP 生成在 ignored `test-data/piano-onset-blind-review-csv-20261119/`，保留84段短片和原模板哈希 `98ec483e0c8864876be2652b13b128dd253aff7fdb80b887fe35eef177b6764a`。ZIP SHA-256 `779100eee1578db14b728b58db965c1cf28c14aee508e2aab4f3558a7532f5d9`，含盲化参与者文件、不含解盲表。原有2026-11-18评审包未覆盖，包外解盲表没有读取。
- `scripts/build_layout_playtest.py` 今后同时产生可填写的 `scores.csv` 与不可变空白基线 `scores-template.csv`；现有ignored评审包的基线副本也已创建，源评分 CSV 保持不变。新增 `scripts/check_blind_review_scores.py`：严格解析两种评分格式、匹配钢琴片段身份或定轨谱包哈希，允许未试打键数和反转 A/B 次序，拒绝半填无效行；锁定时以新目录保存评分原始字节和只含哈希/计数/时间的回执，整个流程不读取解盲文件。
- 新增两首独立歌曲的结构化复听空白表，区分起音/Timing、声部、段落、LN和手感意见；真实评审仍由人完成，不计算自动音质分。
- 对现有两张 CSV 只做字段/计数检查，未输出评分内容。钢琴盲听为84行、0开始、0完整、`lockable=false`；定轨为6K–10K五行、0开始、0完整、未覆盖键数0、`lockable=false`。因此没有创建锁定回执，也未解盲；两首独立歌曲仍未取得真人听审记录。
- 独立源码审阅发现刷新时没有确认旧参与者目录与清单记录的盲评 ZIP 一致，短片损坏可能沿用旧片段编号。现先验证参与者 ZIP 的 SHA-256、CRC、完整成员和每个文件的字节一致性，再创建新输出；合成测试覆盖短片与模板变更，并确认失败时不留下输出目录。没有读取解盲表。
- 最终验证：`.venv\Scripts\python.exe -m pytest -q` 为2,666 passed、89 skipped（67.41秒）；定向评审用例28 passed；`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=8997 npm run test:ui` 为36 passed（1.4分钟）。`compileall`、Playwright测试文件`node --check`及`git diff --check`通过。浏览器用例覆盖刷新生成、草稿持久化/清除及含逗号、引号、中文和换行的 CSV 下载；锁定器测试使用合成表，不接触本地评分、解盲表或音频。
- PR #160 Windows run `44382403042`通过（Python/API 2,666 passed、89 skipped；Edge 36 passed），于2026-11-19合并为`1fa101247c5de3f5c03d6691b0dee1329d11b987`。合并后 main run `44383586303`通过相同 Python/API 与 Edge 检查；便携 EXE步骤按触发条件跳过。
- 工程改进只提高评审回收的可恢复性和身份/完整性，不构成采音、Timing、声部清晰度、LN或玩家体验改善。下一步仍需收集完整钢琴盲听、实际游玩键数的定轨评分和两首歌曲复听，再先锁定表格、后由真人决定是否解盲和如何修改产品。

## 2026-11-19 第五十八阶段：盲评刷新事务发布

- `refresh_piano_blind_review.py` 在输出父目录的唯一临时目录内完成页面、CSV、短片、ZIP 和 manifest；全部成功后才将目录重命名为正式输出。若复制或写盘失败，临时目录自动清理，正式目标不会留下半成品；仍拒绝覆盖已有目标。
- 输出本身为符号链接时在路径规范化前拒绝，避免悬空链接把产物重定向到外部目录；覆盖发布前目标被并发创建的冲突场景，确认保留已有文件并清理 staging。
- 新增磁盘失败、悬空符号链接与发布前目标冲突合成测试；定向刷新/锁定/钢琴页面测试为29 passed、1 skipped（当前 Windows 环境不允许创建符号链接）。设置数值线程上限2运行全量`.venv\Scripts\python.exe -m pytest -q`为2,668 passed、90 skipped（67.05秒）；`compileall`及`git diff --check`通过。该阶段没有读取 ignored 音频、盲评评分或解盲数据。
- PR #161 Windows run `44386569096`及合并后 main run `44387300930`均通过 Python/API 与合成浏览器检查；本次便携 EXE 步骤按 workflow 触发条件跳过。合并提交为`e11663b932c23f17303aef50941b7ef69ada275e`。
- 下一步继续处理 `ProjectTasks.jobs` 对终态任务和结果无界保留的问题；真人钢琴盲听、6K–10K/LN定轨试打和两首独立歌曲复听仍缺人工结果，整体音乐验收保持开放。

## 2026-11-19 第五十九阶段：限制终态任务历史

- `ProjectTasks` 默认最多保留最近64条完成、失败或取消的任务记录；提交新任务时才清除最旧的终态记录，不触碰 queued/running 任务。近期结果字段继续可查询，已清除 ID 沿用未知任务的404行为。
- 新增保留策略回归测试：用2条上限验证旧终态记录和取消入口失效、近期结果仍可读、运行任务在裁剪期间保留，最终记录数不超过上限。
- 定向`.venv\Scripts\python.exe -m pytest -q tests/test_task_service.py tests/test_server.py`为22 passed；全量设置数值线程上限2运行`.venv\Scripts\python.exe -m pytest -q`为2,669 passed、90 skipped（65.23秒）；`compileall`和`git diff --check`通过。
- PR #162 Windows run `44388833902`及合并后 main run `44389967517`均通过 Python/API 与合成浏览器检查；便携 EXE 步骤按 workflow 触发条件跳过。合并提交为`ae0ee05e51e4046c27c3e4f3163dacf8917e803a`。该工程内存治理不代表音质或玩家体验提升。

## 2026-11-19 第六十阶段：整次导入共享展开预算

- 导入总写入预算与 HTTP 请求体上限共用512 MiB常量；普通音频/`.osu`、所有 `.osz` 成员的展开大小均计入同一预算。ZIP 成员数限制也跨整次导入累计；超额归档在写入前拒绝，既有失败清理会删除本次新项目并保留此前项目。
- 先用 Python 标准库 ZIP 结束目录读取器读取声明条目数，在构造完整 `ZipFile` 中央目录前拒绝超过限制的归档；保留完整解析后的条目数防线，并先检查条目数再遍历大小。
- 新增合成 multipart 用例，分别覆盖多归档展开字节合计超限、跨归档条目数超限、普通音频加归档超总写入预算；另验证超条目归档在创建完整 `ZipFile` 前被拒绝。
- 首轮定向`.venv\Scripts\python.exe -m pytest -q tests/test_import_reliability.py tests/test_core.py`为49 passed；加入中央目录预检后同一命令为50 passed。全量设置数值线程上限2运行`.venv\Scripts\python.exe -m pytest -q`为2,673 passed、90 skipped（65.50秒）；`compileall`和`git diff --check`通过。新增测试确认超条目 ZIP 在构造完整 `ZipFile` 前被拒绝，原始输入游标保持不变。
- PR #163 已合并为`d705c9c65e40cf08beaa8c732e079773e555c9d4`。PR Windows run `44393454396`成功；合并后 main run `44394113412`成功，Python/API为2,674 passed、89 skipped，Edge UI为36 passed；便携构建按触发条件跳过。导入资源上限属于工程健壮性改进，不构成音乐质量验收。真人盲听、6K–10K/LN试打和两首独立歌曲复听继续待完成。

## 2026-11-19 第六十一阶段：任务取消与进程关停

- `ProjectTasks` 增加关闭状态：应用开始关停后拒绝新任务、请求取消排队及运行任务，并等待清理完成；线程启动失败时取消遗留 Future、关闭不可用的 executor 并执行未提交任务的清理。提交响应返回锁内任务快照，避免后台完成时与 HTTP JSON 序列化共享可变字典。终态任务取消请求保持兼容的成功响应，但不再改写终态记录。
- 启动器退出时依次关闭 Waitress socket/request dispatcher 与任务服务；即使服务运行或关闭阶段抛错，后续清理仍会执行。Waitress dispatcher 按其接口等待最多5秒，任务服务随后进行协作取消和等待。
- 新增可取消 FFmpeg 子进程执行器，在约200ms轮询周期内检查取消，并终止、回收子进程；接入普通/深度采音、导入声轨、分离预转码、结构检测及可选上游模型解码。结构特征与边界扫描也检查取消；SciPy 的单次 STFT 等原生向量化调用无法中途打断，取消在当前调用返回后生效。
- 新增线程启动失败、`BaseException` 任务恢复、关停取消/清理、稳定响应快照、Waitress dispatcher 清理、FFmpeg 终止回收、结构任务 API 取消以及结构扫描取消检查。定向跨模块验证为127 passed；全量设置数值线程上限2运行`.venv\Scripts\python.exe -m pytest -q`为2,681 passed、90 skipped（67.63秒）；`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=9003 npm run test:ui`为36 passed（1.4分钟）。`.venv\Scripts\python.exe -m compileall -q manialab scripts tests launcher.py`与`git diff --check`通过。
- PR #164 Windows run `44403326416`通过；合并后 main run `44405337764`成功，Python/API为2,682 passed、89 skipped，Edge UI为36 passed，便携构建按触发条件跳过。合并提交为`6d779209561b6b767b5f6886d75e9713bf371227`。没有创建新 Release。
- 本阶段没有运行模型、研究或音乐质量实验，也没有读取 ignored 音频、评审评分、解盲表或模型产物。
- 本阶段只改善任务可取消性、异常恢复与应用关停，不代表采音或谱面音乐性改善。真人钢琴盲听、实际键数的6K–10K/LN试打及两首独立歌曲听审仍未完成；公开 Release 仍是 dev.18。

## 2026-11-19 第六十二阶段：结构分析频谱分块

- 将段落结构检测的整曲 STFT 改为每块最多512帧，按原始全曲帧网格切片，并在线累加到500ms频带统计；不改变频带、窗口、缩放或平滑规则。频谱与功率中间数组在块间释放，取消检查可在块与频带计算边界响应。
- 新增与原整曲 SciPy STFT 参考公式的合成数值等价测试（强制11帧分块），并验证检测在首个频谱块后响应取消。定向`tests/test_structure_expression.py`为34 passed。
- 合成180秒单声道`float32`输入的`tracemalloc`峰值比较：原整曲参考为90.85 MiB，分块实现为12.16 MiB，约7.5倍降低；这是合成内存基准，不代表真实音乐质量或所有机器上的绝对峰值。
- 设置 OpenMP / OpenBLAS / MKL 线程上限2运行`.venv\Scripts\python.exe -m pytest -q`为2,682 passed、90 skipped（66.01秒）；`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=9004 npm run test:ui`为36 passed（1.4分钟）。`.venv\Scripts\python.exe -m compileall -q manialab tests`与`git diff --check`通过。
- PR #166 Windows run `44410686981`通过；进度文档更新后的 run `44411505246`再次通过，Python/API为2,683 passed、89 skipped，Edge UI为36 passed；便携构建按触发条件跳过。PR 已合并为`e684551abd5071d75f33febefe76e0bd7b182135`，合并后 main run `44412123852`成功，Python/API与 Edge UI 计数相同。
- 本阶段仅改变结构特征的内存占用和取消粒度，需以旧实现数值对照守住输出；没有运行模型、研究或音乐质量实验，也未读取 ignored 音频、评分、解盲表或模型产物。人工听审及玩家试打仍未完成，公开 Release 仍为 dev.18。

## 2026-11-19 第六十三阶段：音频身份哈希流式化

- 新增共享`sha256_file`，以1 MiB块计算源音频 SHA-256，避免分析缓存和声部分离缓存校验将完整导入音频读入内存。分析与分离路径都将现有取消回调传入哈希过程；每块读取前检查取消，抛出`InterruptedError`并及时关闭文件。
- 合成测试校验跨多个数据块的摘要与标准 SHA-256 一致、相关路径不调用`Path.read_bytes()`、哈希中途取消后最多读取首个1 MiB块，以及分析缓存仍正确复用、分离路径不进行整文件读取。
- 定向`.venv\Scripts\python.exe -m pytest -q tests/test_audio_cache.py tests/test_audio.py tests/test_separation.py`为20 passed。全量设置 OpenMP / OpenBLAS / MKL 线程上限2运行`.venv\Scripts\python.exe -m pytest -q`为2,685 passed、90 skipped（65.28秒）；`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=9005 npm run test:ui`为36 passed（1.4分钟）。`.venv\Scripts\python.exe -m compileall -q manialab scripts tests launcher.py`及`git diff --check`通过。
- 独立只读复核未发现明显哈希正确性回归；建议正向测试覆盖多块拼接，已将输入扩至约3 MiB并重新运行定向与全量测试。没有运行模型、研究或真实音频；未读取 ignored 音频、评审评分、解盲表、模型权重或研究产物。
- PR #167 Windows run `44415178836`及本文件同步后的 run `44416022976`均通过，Python/API为2,686 passed、89 skipped，Edge UI为36 passed；便携构建按触发条件跳过。
- PR #167 已合并为`76c3dc00b6466682f4bd3bbbacd5e50535b3a370`；合并后 main run `44417472152`成功，Python/API为2,686 passed、89 skipped，Edge UI为36 passed；便携构建按触发条件跳过。
- 本阶段只降低大音频身份校验的内存占用并增加哈希取消响应，不改变音频推理、缓存版本或音乐输出，不代表真实音乐质量提升。公开 Pre-release 仍为 dev.18；钢琴盲听、6K–10K/LN 试打与两首独立歌曲真人听审仍待完成。

## 2026-11-19 第六十四阶段：限制 ZIP 中央目录内存

- 在读取完整 ZIP 中央目录前，使用现有 EOCD/ZIP64 元数据预检新增的32 MiB中央目录字节上限；与已有成员数量上限独立生效，避免条目计数正常但中央目录声明异常大时，`ZipFile`先将整段目录读入内存。
- 新增合成归档用例：保持归档条目数正常，直接改写合成 ZIP 的 EOCD 中央目录大小字段至超限值，不 mock 结束记录解析器；确认在构造`ZipFile`前拒绝，上传流回到初始位置且不留下文件。定向`.venv\Scripts\python.exe -m pytest -q tests/test_import_reliability.py tests/test_core.py`为51 passed。
- 全量设置 OpenMP / OpenBLAS / MKL 线程上限2运行`.venv\Scripts\python.exe -m pytest -q`为2,686 passed、90 skipped（65.91秒）；`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=9006 npm run test:ui`为36 passed（1.4分钟）。`.venv\Scripts\python.exe -m compileall -q manialab scripts tests launcher.py`、全部`static/*.js`的`node --check`及`git diff --check`通过。
- PR #168 首轮 Windows run `44418933013`通过，Python/API为2,687 passed、89 skipped，Edge UI为36 passed；直接改写合成 EOCD 的测试增强后，Windows run `44420946296`再次通过相同计数，便携构建按触发条件跳过。独立只读复核未发现解析或兼容性问题；实现依赖 CPython `zipfile` 私有接口，Python 主次版本升级时需重新复核。
- 本阶段只增加 ZIP 元数据资源限制，不更改正常谱包解析/解压结果；未读取用户上传谱包或音频，未运行模型、研究或音乐质量实验。当前公开 Pre-release 仍为 dev.18；真人钢琴盲听、6K–10K/LN试打与两首独立歌曲听审仍待完成。

## 2026-11-19 第六十五阶段：迁移 GitHub Actions Node 运行时

- CI 将`actions/setup-node`固定到官方`v7.1.0`提交`949feb2413d6458794dcd2491c4babbbce0c15c1`，采用 Node 24 action runtime；项目安装的 Node 版本仍为22，npm cache配置不变。当前`package.json`无`packageManager`字段，workflow也未配置`registry-url`或依赖`NODE_AUTH_TOKEN`。
- PR Windows run `44424536788`成功，Python/API为2,687 passed、89 skipped（70.17秒），Edge UI为36 passed（1.2分钟），Core EXE步骤按触发条件跳过。日志确认 action 安装 Node `v22.23.5`，未发现原有“Node.js 20 actions are deprecated”告警。
- 仅修改CI action版本与本进度文档；不改变应用运行逻辑、研究脚本或音乐处理结果。PR进度同步与合并后 main run 待记录；当前公开 Pre-release 仍为 dev.18。真人钢琴盲听、6K–10K/LN试打与两首独立歌曲听审未完成。

## 2026-11-19 第六十六阶段：原模型后处理取消与发布回滚

- Mapperatorinator 返回后，在读取/解析及报告计算边界检查协作取消；所有报告计算完成后才发布新谱和工作区快照。取消或失败会删除本次新谱，并恢复旧快照；原来没有快照时移除新建快照及空目录。快照备份使用同目录原子改名，避免把大型 JSON 快照读入内存。
- `archive_active_chart` 无论写入或替换成功与否都会清理 `.tmp` 文件。故障注入测试模拟只写入部分 JSON 后抛出磁盘错误，确认不遗留临时或目标快照。发布取消测试覆盖旧快照存在和不存在两种路径。
- 定向`.venv\Scripts\python.exe -m pytest -q tests/test_server.py -k "original_model_cancel or archive_active_chart"`为4 passed、18 deselected。全量设置 OpenMP / OpenBLAS / MKL 线程上限2运行`.venv\Scripts\python.exe -m pytest -q`为2,690 passed、90 skipped（87.92秒）；`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=9017 npm run test:ui`为36 passed（1.4分钟）。`.venv\Scripts\python.exe -m compileall -q manialab scripts tests launcher.py`、全部`static/*.js`的`node --check`及`git diff --check`通过。
- PR #170 Windows runs `44430965427`、`44431613071`和最终文档复验`44432358696`均通过，Python/API为2,691 passed、89 skipped，Edge UI为36 passed；Core EXE步骤按触发条件跳过。PR合并提交为`3cf043aa7c027a905aaa1aa88ca551e84c9557d5`；合并后main run `44432819183`成功，Python/API为2,691 passed、89 skipped，Edge UI为36 passed。本阶段没有运行模型、研究或音乐质量实验，也没有读取 ignored 音频、谱包、盲评评分、解盲表、权重或研究产物。
- 本阶段只改进取消后的文件事务完整性，不代表采音或谱面音乐性改善。公开 Pre-release 仍为 dev.18；真人钢琴盲听、6K–10K/LN试打与两首独立歌曲听审仍待完成。后续工程审查重点为进程关停是否需要统一总截止时间。

## 2026-11-19 第六十七阶段：统一任务与子进程关停期限

- 启动器在关停入口创建单调时钟绝对 deadline（15秒）；先关闭任务提交入口并广播取消，再关闭 Waitress 服务和 dispatcher，最后只等待同一个 deadline 的剩余时间。任务服务的关闭锁获取、运行任务和清理等待均受该时间约束。已通过最终取消检查并进入原子项目保存的任务允许完成提交，关闭结果仍反映是否在 deadline 内等到它结束。
- 任务内部使用 callable `CancellationContext`，普通 `cancelled()` 调用保持兼容；deadline 与任务 JSON 布尔 `cancel` 分离。FFmpeg 回收、Mapperatorinator/uv Windows 子进程树终止、声部分离和专用采音 worker 都复用该 deadline，终止与 kill 阶段不会重新获得完整预算。
- 若 OS 未能在期限内确认子进程退出，清理会明确抛出 `ChildProcessError`，保留原取消异常链、标记任务失败并使关停结果返回失败；不会把“Python 任务线程结束”误报为“子进程已退出”。Windows Mapperatorinator 清理即使根进程已先退出，也会再次枚举并终止子进程；无法打开、终止或确认子进程退出都会写入共享失败状态。启动器将任务服务未能在期限内停止提升为关停异常，原服务器异常保留为异常上下文。合成测试覆盖锁竞争、共享预算耗尽、FFmpeg 无法回收、信号失败、根进程先退出及任务服务失败结果。
- 验证：关停/进程边界定向测试76 passed；全量设置 OpenMP / OpenBLAS / MKL 线程上限2运行 `.venv\Scripts\python.exe -m pytest -q` 为2,704 passed、90 skipped（82.74秒）；`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=9022 npm run test:ui` 为36 passed（1.6分钟）。`compileall`、全部`static/*.js`的`node --check`及`git diff --check`通过。
- 关停期限只约束协作式清理和等待，不是任意 native 算子或卡死 Python 线程的硬退出 SLA；`ThreadPoolExecutor`中的非守护线程仍可能阻止解释器退出。未运行模型、研究或音乐质量实验，也未读取 ignored 音频、谱包、盲评评分、解盲表、权重或研究产物。
- PR #172 Windows run `44443163914`通过，合并提交为`a1cb5375bd54f7998aa0ae395f0aa0064dabfee1`；合并后 main run `44443998819`成功，Python/API为2,705 passed、89 skipped，Edge UI为36 passed。Core便携EXE构建、离线运行、Unicode数据路径与进程关停检查均通过。
- 本阶段只约束协作式清理和等待，不代表音乐质量改善；真人钢琴盲听、6K–10K/LN玩家试打和两首独立歌曲听审仍未完成，公开 Pre-release 仍为dev.18。

## 2026-11-19 第六十八阶段：OSZ 导出使用磁盘临时文件

- `/api/projects/<pid>/export/osz`不再把完整压缩包累积在`BytesIO`；改用磁盘`TemporaryFile`承接归档，交由Flask响应关闭回调释放。压缩构建或响应创建失败时立即关闭临时文件。归档成员、路径、内容与压缩格式保持不变。
- 新增合成项目测试验证临时文件仍开放至响应消费/关闭，下载归档CRC、谱面和音频内容正确；注入归档构建异常时确认临时文件立即关闭。未访问用户音频或谱包。
- 定向`.venv\Scripts\python.exe -m pytest -q tests/test_export_resources.py tests/test_server.py tests/test_project_routes.py tests/test_import_reliability.py tests/test_audio_meter.py tests/test_edited_candidate_bookmarks.py`为86 passed；异常注入强化后`tests/test_export_resources.py`为2 passed。最终全量设置OpenMP / OpenBLAS / MKL线程上限2运行`.venv\Scripts\python.exe -m pytest -q`为2,706 passed、90 skipped（70.95秒）。`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=9024 npm run test:ui`为36 passed（1.6分钟）；`compileall`、全部`static/*.js`的`node --check`及`git diff --check`通过。
- PR #173 Windows run `44446658555`通过，Python/API为2,707 passed、89 skipped，Edge UI为36 passed；Core便携EXE构建、离线运行、Unicode数据路径和进程关停均通过。PR合并提交为`aac6bfee354b1fad3cf4cbf5e308ecbd7915d18c`；合并后main run `44447294349`成功，测试计数相同。
- 本阶段只降低大型OSZ导出的内存驻留量，不改变谱面音乐结果；未运行模型、研究或音乐质量实验，未读取ignored音频、谱包、评审评分、解盲表、权重或研究产物。音乐质量验收仍需真人盲听、玩家试打及独立歌曲复听。

## 2026-11-19 第六十九阶段：项目状态保存临时文件清理

- `Projects.save()`在JSON写入或原子替换成功/失败后统一清理同目录`state.tmp`；正常保存仍使用`os.replace`发布完整`state.json`，失败时不覆盖之前有效的状态文件。
- 两条合成故障注入先在旧实现上复现：部分写入抛错与`os.replace`抛错都会遗留`state.tmp`；修复后均确认旧状态字节不变且临时文件不存在。
- 定向`.venv\Scripts\python.exe -m pytest -q tests/test_storage.py tests/test_task_service.py tests/test_server.py -k "not test_real_http_planner_mock"`为36 passed、1 deselected。全量设置OpenMP / OpenBLAS / MKL线程上限2运行`.venv\Scripts\python.exe -m pytest -q`为2,708 passed、90 skipped（89.42秒）；`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=9025 npm run test:ui`为36 passed（1.5分钟）。`compileall`与`git diff --check`通过。
- PR #174 Windows run `44448768663`通过，Python/API为2,709 passed、89 skipped，Edge UI为36 passed；Core便携EXE构建、离线运行、Unicode路径和关停检查通过。合并提交为`d5d1cc70201bbb6556a849587cc52d7261e82b37`；合并后 main run `44449536131`成功，验证结果相同。
- 本阶段只清理状态写入失败后的临时文件，不更改状态格式或音乐处理；未运行模型、研究或音乐质量实验，也未读取ignored音频、谱包、评审评分、解盲表、权重或研究产物。真人音乐验收仍未完成。

## 2026-11-19 第七十阶段：拒绝无有效谱面时的多音频歧义导入

- 导入项目时继续统计可用音频；当没有可解析的原生 mania 谱面且存在多个音频文件时返回明确的 400 错误，由既有失败清理删除本次新项目。有效谱面选择流程和单音频回退行为保持不变。
- 新增合成 API 测试覆盖：无谱面/无效 `.osu` 搭配多个音频时拒绝并清理新项目、保留此前项目；无效 `.osu` 搭配单音频时仍成功且没有虚构谱面。
- 定向`.venv\Scripts\python.exe -m pytest -q tests/test_import_reliability.py`为26 passed。全量设置 OpenMP / OpenBLAS / MKL 线程上限2运行`.venv\Scripts\python.exe -m pytest -q`为2,711 passed、90 skipped（73.20秒）；`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=9026 npm run test:ui`为36 passed（1.5分钟）。`compileall`、全部`static/*.js`的`node --check`及`git diff --check`通过。
- PR #175 Windows run `44452212429`通过，Python/API为2,712 passed、89 skipped，Edge UI为36 passed；便携构建按触发条件跳过。合并提交为`32564c4ae08382dc29e802cd8704bb0689d1febf`；合并后 main run `44452857369`成功，Python/API为2,712 passed、89 skipped，Edge UI为36 passed，便携构建同样按触发条件跳过。此前 PR #174 main run `44449536131`已通过Core便携EXE、离线、Unicode路径和关停检查。
- 本阶段只让歧义输入显式失败，不代表音频或谱面音乐质量改善。没有运行模型、研究或音乐质量实验，也没有读取ignored音频、谱包、评审评分、解盲表、权重或研究产物。公开 Pre-release 仍为dev.18；真人钢琴盲听、6K–10K/LN玩家试打及两首独立歌曲听审仍未完成。

## 2026-11-19 第七十一阶段：提示谱面缺失的配套音频

- 当已导入难度声明的 `AudioFilename` 未找到时，播放器下方持续提示该难度没有配套音频，并说明恢复方式是把对应 `.osu` 与音频重新导入。不会自动沿用前一难度的音轨；现有缺失源 API 400 行为不变。当前没有在既有项目内直接补配文件的操作，恢复方案会创建一个新项目。
- 新增 Edge 合成浏览器用例：先导入有配套音频的难度，再切换到引用缺失音频的难度；确认提示出现且播放器 `src` 被清除。身份保护由 `tests/test_project_routes.py` 覆盖。
- 本地验证：`.venv\Scripts\python.exe -m pytest -q` 为2,711 passed、90 skipped（70.97秒）；新增浏览器用例定向运行1 passed，完整`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=9028 npm run test:ui`为37 passed（1.5分钟）。`compileall`、全部`static/*.js`的`node --check`及`git diff --check`通过。
- PR #177 Windows runs `44457046075`与文档复验`44457739696`均通过；合并后 main run `44458199401`成功，Python/API为2,712 passed、89 skipped，Edge UI为37 passed；便携构建按触发条件跳过。合并提交为`b69acf7f1db7a3ae3aafde4ca3e744c64c466be9`。
- 本阶段闭合上一轮识别的 P2 界面恢复提示缺口；不是数据错配修复，未新增音频绑定 API，不改变导入、存储或导出合同。未运行模型或研究，也未读取 ignored 音频、谱包、评审、权重或研究产物。公开 Pre-release 仍为dev.18；真人音乐验收与目标 profile/许可门槛仍未完成。

## 2026-11-19 第七十二阶段：补齐候选删除的浏览器生命周期覆盖

- 已有领域/API 测试覆盖删除校验与持久化，但此前没有浏览器端到端用例。新增合成 Edge 流程：从画布选择音符并删除，核对删除日志与新候选，切回父版本后再选择删除版本，重开项目后确认两版仍可切换且删除结果保留。
- 定向浏览器用例1 passed；完整`MANIALAB_TEST_BROWSER=msedge MANIALAB_TEST_PORT=9032 npm run test:ui`为38 passed（4.2分钟）。测试只使用`tests/browser/fixture_server.py`创建的合成项目，没有模型或用户媒体。
- 关闭问题账本 COV-01。此阶段只补浏览器工作流覆盖，没有修改产品逻辑，也不代表音符删除以外的阶段1–3跨路径审查已经完成。
- PR #179 Windows run `44461711637`与合并后 main run `44462358541`均成功；Python/API为2,712 passed、89 skipped，Edge UI为38 passed，便携构建按触发条件跳过。PR于2026-11-19合并为`ef4e788d59c696b48af241ed40652549923449c6`。
- 本阶段没有发现音符删除产品行为缺陷；只是补齐生命周期覆盖。阶段1–3跨路径审查仍在进行。

## 2026-11-19 第七十三阶段：修复嵌套谱面的项目内父目录音频引用

- osu! 谱面[格式文档](<https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format)>)规定 `AudioFilename` 相对当前谱面文件夹。此前导入 `charts/diff.osu` 引用 `../song.wav` 时，路径在 OSZ 根内仍被 `safe_path` 因含 `..` 拒绝，导致整个导入返回400并清理新项目；OSZ 导出检查相同引用也会被拒绝。
- 新增 `resolve_project_reference`：先基于谱面目录解析相对资源，再校验规范化路径仍位于项目 `files` 根内；拒绝盘符、绝对路径、UNC、系统保留名及越界符号链接。归档成员仍使用原有严格 `safe_path`，没有放宽 ZIP 提取边界。
- 合成 API 回归覆盖 POSIX 与 Windows 分隔符的 `../song.wav` 导入、音频接口读取、候选生成与 OSZ 往返（仅包含一份根目录原音频），以及 `../../`、Unix绝对路径、盘符和UNC引用的拒绝与新项目清理。先行测试复现旧行为：合法根内父目录引用返回400。
- 定向`.venv\Scripts\python.exe -m pytest -q tests/test_import_reliability.py tests/test_project_routes.py tests/test_export_resources.py tests/test_core.py tests/test_server.py tests/test_official_decoder_compare.py`为102 passed。`.venv\Scripts\python.exe -m compileall -q manialab scripts tests launcher.py`、全部`static/*.js`的`node --check`及`git diff --check`通过。未运行模型、研究或读取用户媒体/盲评数据。
- 本地未运行全量 pytest：部分测试会读取 `.gitignore` 下的研究与音频目录；完整回归交由无这些本地文件的 PR Windows runner。PR 与 main CI 待补。
- 此阶段修复的是文件引用兼容性及导出完整性，不代表音质或音乐验收改善。阶段1–3其余跨路径审查、非Core profile许可处置及真人音乐门槛继续开放。
- PR #180 Windows run `44465032776`与合并后main run `44466336310`均成功；Python/API为2,719 passed、89 skipped，Edge UI为38 passed，便携步骤按条件跳过。PR于2026-11-19合并为`6888912147e5deed81545cb2376ef2c5066cb65c`。

## 2026-11-19 第七十四阶段：清理分离 worker 启动失败的临时音频

- 声部分离预转码成功后，若 worker 的 `subprocess.Popen` 因 `OSError` 无法启动，旧实现会遗留 `separation-input.wav`。新增合成故障注入测试，先在旧实现上确认文件残留，再确认修复后删除。
- 启动或日志打开抛出 `OSError` 且尚无进程对象时清理 WAV；进程启动后仍只在 `stop_process` 确认退出时删除，不能确认停止时保留可能仍被读取的输入。独立只读复核通过。
- 定向`.venv\Scripts\python.exe -m pytest -q tests/test_separation.py tests/test_audio_cache.py`为12 passed；`compileall -q manialab tests`和`git diff --check`通过。未运行模型、研究或真实音频。
- PR #181及合并后main run `44468079196`、`44468606711`均通过；main Python/API为2,720 passed、89 skipped，Edge UI为38 passed，Core便携步骤按触发条件跳过。此修复只闭合一条资源失败路径，不代表阶段3集成审查或音乐质量验收完成。

## 2026-11-19 第七十五阶段：分离算法修订纳入缓存身份

- 复现缓存清单只校验音频哈希、模型哈希、输入处理版本和stem文件是否存在，没有校验分离算法版本；下游`separation_identity`也未包含算法修订。因此算法实现变化但输入和模型不变时，旧stem及其activation可能被复用。
- 为四/六声部profile增加独立`algorithm_revision`，由父进程传递至worker；缓存命中、worker返回清单和下游分离身份都检查/记录该字段。缺少该字段的历史清单会重算，原有`revision`用途不变。
- 将`scripts/separate_six.py`加入便携音频worker源码fingerprint，并新增合成测试保证六声部脚本变化会令旧worker manifest校验失败，避免以后只改六声部实现时被漏检。
- 定向`.venv\Scripts\python.exe -m pytest -q tests/test_separation.py tests/test_audio_cache.py tests/test_build_provenance.py`为19 passed；`compileall -q manialab scripts tests`、全部`static/*.js`的`node --check`及`git diff --check`通过。没有运行全量测试、真实音频或模型。
- 独立只读审查确认缓存修订链路一致，未发现阻断问题；源码fingerprint遗漏由本阶段一并补齐。PR #182 run `44471599763`和合并后main run `44472118121`均成功，Python/API `2725 passed, 89 skipped`、Edge `38 passed`，Core便携步骤按条件跳过；合并提交为`58c48b9966da2b8a798fe5baa32d0fdc932bfaf7`。
- 本阶段仅保证分离结果身份与缓存一致性，不代表分离音质或音乐验收改善。公开Pre-release仍为dev.18；人工盲听、6K–10K/LN试打、两首独立歌曲复听及Demucs权重再分发许可仍未闭环。

## 2026-11-19 第七十六阶段：Basic Pitch 模型身份进入 activation 缓存

- 合成回归先在旧实现上复现三条路径：整曲`activations.npz`只校验格式版本与音频哈希；分离声部缓存只校验音频与分离身份；ONNX session全局单例不跟踪模型身份。同一输入切换模型A/B时，旧缓存仍被复用，直接推理也不会重建session。
- 增加Basic Pitch模型内容身份与独立`BASIC_PITCH_INFERENCE_REVISION`。整曲缓存格式升级到3，分离声部格式升级到4；旧格式自动miss。整曲与每声部缓存均校验音频、分离（适用时）、模型身份和推理修订。
- ONNX session按模型身份缓存；分析开始时绑定本地session引用，避免并发调用替换全局session后改变当前推理所用模型。模型文件缺失的纯合成/cache-only路径使用稳定路径身份；实际文件使用SHA-256。
- 定向合成测试为`tests/test_audio_cache.py tests/test_audio.py -k "not real_model" tests/test_stems.py tests/test_audio_release_provenance.py tests/test_provided_stems.py tests/test_instrument_integration.py tests/test_guitar_integration.py`，结果76 passed、1 deselected；真实模型测试明确排除。复核时修正session回归对真实`_session_model_identity`字段的重置，并将整曲cache identity变量改名为`model_fingerprint`。复跑`compileall`、全部`static/*.js`的`node --check`和`git diff --check`通过；未运行全量测试、未读取模型权重或用户音频。
- 独立只读复审未发现阻断问题。PR #183 run `44476736699`与合并后main run `44477712665`均成功，Python/API `2728 passed, 89 skipped`、Edge `38 passed`，便携步骤按触发条件跳过；合并提交为`3427f179e17068f4b4581c70fa37d96583e5816e`。本阶段只保证模型和推理身份变更后重算，不代表Basic Pitch音准提升或音乐验收完成。公开Pre-release仍为dev.18。

## 2026-11-19 第七十七阶段：导入项目目录创建失败时清理孤儿目录

- 阶段1只读审计发现 `Projects.create()` 创建 `<uuid>/files` 时，父目录可能已创建而子目录操作失败；调用发生在导入路由的异常清理范围之外，留下没有 `state.json`、不会出现在项目列表中的目录。
- 新增合成 API 故障注入：模拟 `<uuid>` 创建成功、`files` 创建抛出 `OSError`。旧实现返回500后留下UUID目录；`Projects.create()` 现在在第二步失败时仅移除本次新建目录并重抛原异常。
- 定向`.venv\Scripts\python.exe -m pytest -q tests/test_server.py::test_import_removes_project_when_creation_fails_after_parent_mkdir tests/test_server.py::test_failed_import_removes_partial_project tests/test_import_reliability.py tests/test_storage.py`为37 passed；`compileall -q manialab/storage.py tests/test_server.py`和`git diff --check`通过。未读取媒体或运行模型。
- 本阶段只修复项目创建的失败清理，不改变导入成功行为或项目格式。阶段1–3其余工程审查以及真人音乐、profile许可验收继续开放。
- PR #184 Windows run `44479294481`与合并后 main run `44480110018`均成功；main Python/API为2,729 passed、89 skipped，Edge UI为38 passed，Core便携步骤按触发条件跳过。合并提交为`d5d1e374e9d315c9d39662da796e14008ee193c7`。故障注入确认创建失败后无孤儿目录，ENG-09关闭。

## 2026-11-19 第七十八阶段：计划 API 取消和 HTTP 传输边界

- `/api/projects/<pid>/plan` 现在把任务取消上下文传给 planner；外部请求期间每50毫秒检查取消状态并关闭连接，返回前再次确认取消，避免旧计划被提交。真实本机 HTTP 阻塞 fixture 复现了原行为；修复后任务在2秒内进入`cancelled`、项目状态不变，唯一 worker 可继续处理下一次规划任务。
- planner 使用 asyncio 流式 HTTP 传输，保留 HTTPS 证书校验、本地 HTTP、环境代理、代理 Basic 认证、禁止重定向；请求协程期限为90秒，响应体最多128 KiB、响应头最多64 KiB。支持临时1xx响应和chunked响应；拒绝混淆长度、超限响应、非法端口及 endpoint 控制字符。HTTPS代理 CONNECT 使用显式`host:port` authority。
- 定向验证：`.venv\Scripts\python.exe -m pytest -q tests/test_server.py -k "real_http_planner or planner_rejects_control or planner_rejects_invalid or cancelled_http_planner"`为13 passed；`.venv\Scripts\python.exe -m pytest -q tests/test_task_service.py tests/test_core.py::test_plan_bounded_and_no_times`为14 passed。PowerShell命令`$env:MANIALAB_TEST_BROWSER='msedge'; $env:MANIALAB_TEST_PORT='9045'; .\node_modules\.bin\playwright.cmd test --grep='API plan' --workers=1`为3 passed；`compileall -q manialab scripts tests launcher.py`、12个`static/*.js`的`node --check`和`git diff --check`通过。PR runs `44493364596`、文档复验`44495044806`与main run `44496133010`均通过，Python/API为`2741 passed, 89 skipped`、Edge为`38 passed`；Core便携构建步骤按触发条件跳过。合并提交为`25460346bf87e847b68ae1aad00300b22d4590a3`。
- 代理覆盖包括实际本机 HTTP 代理 Basic 认证和模拟 HTTPS 代理的 CONNECT/TLS 升级调用。后者没有执行真实证书验证或 TLS-over-TLS 握手；这是当前测试缺口。
- 取消边界：在 Python 3.12 中，asyncio DNS `getaddrinfo`使用默认线程池；取消网络协程不能中断已运行的系统解析线程。`asyncio.run()`结束时默认执行器可能额外等待最多5分钟，因此90秒不是端到端硬超时，DNS挂起时任务 worker 的释放也没有2秒保证。相关依据见[asyncio DNS 文档](https://docs.python.org/3.12/library/asyncio-eventloop.html#dns)和[asyncio.run 文档](https://docs.python.org/3.12/library/asyncio-runner.html#asyncio.run)。ENG-10 保持开放，见验收账本。
- 本阶段没有运行模型、研究或音乐质量实验，也没有读取 ignored 音频、谱包、评分、解盲表、权重或研究产物。PR与main CI均已通过；公开 dev.18 发布包不包含本次主线改动，真人音乐验收与权重许可仍未闭环。

## 2026-11-19 第七十九阶段：补齐发布失败与声轨导入清理覆盖

- 阶段1跨路径只读审计没有确认产品行为缺陷，但发现两处失败路径的集成断言不足：原模型候选发布、工作区归档后若最终项目状态保存失败；已提交声轨后再导入长度不匹配文件。
- 为原模型最终保存失败增加两种工作区状态的故障注入：原先无快照与已有快照。两种情形均验证任务报错、旧项目状态保持、刚发布的谱面被删除、临时/回滚文件不残留；已有快照的原始字节恢复，无快照时新工作区目录移除。
- 强化声轨长度不匹配的 API 回归，验证失败请求不改变旧项目和先前成功的上传/解码目录集合；本次新增上传目录与解码缓存目录会清除。
- 定向`.venv\Scripts\python.exe -m pytest -q tests/test_server.py::test_original_model_save_failure_rolls_back_files_and_workspace tests/test_provided_stems.py::test_provided_upload_keeps_existing_analysis_and_old_set_on_failure`为3 passed。Windows PR runs `44498461836`、最新 head 文档复验`44499686610`成功；合并后 main run `44500510492`成功，Python/API `2743 passed, 89 skipped`、Edge `38 passed`，Core便携与模型步骤按触发条件跳过。
- 本阶段仅增加合成失败注入覆盖，不代表声部或音乐质量改善；未运行模型、研究或音乐验收，也未读取ignored音频、谱包、评分、解盲表、权重或研究产物。合并后 main run `44500510492`通过后，阶段1生命周期跨路径工程审查收口；阶段2–3及外部验收仍开放。

## 2026-11-19 第八十阶段：统一手动 Offset 的半毫秒舍入

- 阶段2 Timing/时钟审查确认自动投影与手动输入使用不同规则是有意边界：自动 Offset 推导保留 JavaScript `Math.round`，与服务端缺省 Offset 的`floor(value + .5)`一致；用户显式提交的小数 Offset 使用 ties-to-even，与`chartOffset()`及服务端显式 Offset 的 Python `round()`一致。
- 复现到两条显式输入路径遗漏该规则：粘贴参考 BPM/Offset 和波形「对齐到拍线」仍调用`Math.round`。合成输入`-1.5 ms`时旧实现写入`-1 ms`，而检查/生成归一化得到`-2 ms`；现已改用`chartOffset()`。自动候选、自动估计与提前量投影保持原行为。
- `tests/browser/timing.spec.cjs`新增回归并明确锁定自动与手动两种规则；浏览器 Timing 全组`12 passed`，Python `tests/test_audio_meter.py tests/test_timing_diagnostic_clock.py`为`40 passed`；两处 JS `node --check`、fixture `compileall`和`git diff --check`通过。没有运行模型或真实音频。
- PR #187 已合并为`5b51d049f608772413251da28f0d2e1948e72a97`；PR run `44507507235`及main run `44508222088`成功，main Python/API `2743 passed, 89 skipped`、浏览器`39 passed`，Core便携步骤按触发条件跳过。该结果只覆盖工程契约，不证明音乐 Timing 已验收。

## 2026-11-19 第八十一阶段：候选声轨试听与波形来源一致性

- 核对实际Git、进程与产物后，从上一暂停点的未提交实现续接；没有运行研究脚本或重新执行模型实验。候选保存`source_separation`身份和清单副本；模型生成的试听WAV存入`stems-review/<identity>/`，独立于可被替换的分离缓存。导入声轨继续使用原有不可变集合。播放前核对快照清单与WAV内容哈希，缺失、篡改或历史来源未知时显式报错。
- 试听、原声轨播放与波形按候选/当前采音上下文绑定来源；切换视图、声部、项目时丢弃旧的异步试听/波形响应。跨标签切换后，携带旧来源身份或旧音频路径的请求被拒绝。混音只切换视图时保留播放位置。局部重写不允许把新声轨来源与旧候选的锁定部分混合。
- 续接审查补上跨视图点击「试听谱面起止」/「试听原始采音」时自动采用目标上下文的修复与浏览器回归，避免按钮模式与来源身份冲突。Python定向回归初次为72 passed、1 failed，原因是拍号集成夹具缺少新增契约要求的音频/模型身份；已补齐合成身份，保留原有拍号断言。最终本地Python/API全量为`2749 passed, 90 skipped, 1 deselected`（167.64秒，显式排除真实模型单测），Edge完整组`44 passed`（2.3分钟）。
- 本地验证命令：`$env:MANIALAB_TEST_BROWSER='msedge'; $env:MANIALAB_TEST_PORT='8978'; npm run test:ui`；Python使用`.venv\Scripts\python.exe -m pytest -q -k 'not real_model'`，并限制OMP/MKL/OPENBLAS线程为2。`compileall -q manialab scripts tests launcher.py`、全部静态JS与新增浏览器测试`node --check`、`git diff --check`通过。UI临时产物位于ignored `tests/artifacts/browser/`，`.last-run.json`为passed且无失败项；测试结束后未发现ManiaLab测试进程。
- PR #188已合并为`0e1b7d687fbf503339c0b7175fca57ad9324d5d8`。PR Windows run `44525372079`及合并后main run `44526302353`均成功：Python/API `2751 passed, 89 skipped`、浏览器`44 passed`，Core便携与模型步骤按触发条件跳过。本地保存的完整CI日志为ignored `build/source-identity-93d3460/pr-188-windows.log`和`main-188-windows.log`；本阶段ENG-12关闭。
- 本阶段保证来源可追溯与故障可见，不声明采音准确率、音乐Timing、声部听感或LN可玩性改善。未读取用户音频、参考谱、保留测试曲、权重或逐音研究产物；真人音乐与发布许可门槛继续开放。
- 下一步：继续核对段落/LN来源证据经过手动编辑、局部重写、候选切换及JSON/OSU/OSZ导出后的对应关系，仅对可复现差异做最小修复。阶段2整体尚未收口；ENG-10的DNS取消与真实HTTPS代理边界、非Core profile/许可和MUS-01人工回收继续开放。此次没有创建新Release。

## 2026-11-19 第八十二阶段：并行跨路径审查与来源/取消集成回归

- 用户明确允许3–4个子agent同时持续工作，覆盖旧的最大1个限制；当前3名分别负责段落/LN来源证据、候选编辑与导出、取消和资源生命周期，由主agent整合。每名只操作分配文件，不再分派子agent，神经模型与研究脚本不运行。PR #189合并提交`8243feb`的main run `44527791593`成功，Python/API `2751 passed, 89 skipped`、浏览器`44 passed`；完整日志与阶段81回执保存于ignored `build/source-identity-93d3460/`。
- 新增`tests/test_candidate_source_export.py`：整谱/局部两种父候选使用不同的合成WAV与复用ID，连续修改起尾、增加人工LN、删除tap后，核对JSON保留候选快照与当前分析的区别、OSU/OSZ端点/轨道及音乐字节、旧声轨字节，再切回父候选并重开应用。人工新增物件没有采音证据。此测试从人工seed父候选开始，不冒称覆盖模型生成。
- 新增`tests/test_task_lifecycle_flow.py`：排队取消不执行解码、解码/清单完成后取消、最终状态保存失败三条合成故障路径，均验证旧项目/声轨字节及目录集合保持、本次新资源清理，随后worker能处理新任务。晚失败使用Event屏障固定时序，未依赖概率竞争。
- 主agent复核与LN agent独立只读复审未发现新增测试的阻断问题。主agent定向`.venv\Scripts\python.exe -m pytest -q tests/test_candidate_source_export.py tests/test_task_lifecycle_flow.py tests/test_evidence_snapshot.py tests/test_editing.py tests/test_edited_candidate_bookmarks.py tests/test_export_resources.py tests/test_task_service.py tests/test_provided_stems.py tests/test_recommendations.py`为66 passed；新增文件`compileall`和`git diff --check`通过。LN两批现有回归分别121 passed和165 passed；remap保时/原始LN音效/资源引用/重开定向11 passed，未发现产品缺陷。
- 本阶段仅增加5项非平凡集成回归，未修改产品逻辑。PR/main Windows验证待补；不重复本地全量/UI或模型实验。另一个确定性故障注入确认：操作被取消后`cleanup(False)`抛`PermissionError`时，任务仍显示cancelled、资源残留，但`shutdown()`返回True且清理失败列表为空；实际文件锁下的unlink/rollback适用。ENG-13已单列，下一阶段修复与本阶段测试分开提交。
- 后续继续保持3名子agent推进独立检查点。阶段2–3及真人音乐、非Core profile/权重许可尚未整体验收，不以新增测试数量宣称音乐改善。
- PR #190已合并为`2e22f1e08952d267f970a33cd4d393ce3b96f6ab`；PR Windows run `44530422421`及main run `44531157525`均成功：Python/API `2756 passed, 89 skipped`、浏览器`44 passed`，Core便携与模型构建步骤按条件跳过。完整CI日志保存在ignored `build/cross-workflow-62e24c0/`；COV-02/03关闭。

## 2026-11-19 第八十三阶段：资源清理失败的任务与关停可见性

- 旧实现合成复现：取消或保存失败后的`cleanup(False)`抛`PermissionError`只写日志，任务仍可能cancelled、文件残留但`shutdown()`误报True。现在清理异常登记到取消上下文并向关停传播，job标记`cleanup_failed=true`；未提交时进入error并移除完整恢复保证，已提交时保留done/result，明确提示已保存但资源清理失败。文件清理失败不占住worker，后续正常任务不会继承旧job的失败状态；已有失败记录时关停不报完全成功。
- 真实Windows无窗口合成进程/临时文件核查又确认提供声轨回滚使用`rmtree(ignore_errors=True)`会吞掉实际文件锁。声轨cleanup现在分别尝试本次上传目录与新解码目录，已不存在的路径正常忽略，其他删除失败汇总为`ExceptionGroup`交给任务服务；旧成功集合不删除。底层unlink故障注入覆盖上传/解码/两处同时失败×取消/最终保存失败共6例。
- 浏览器`runJob`在保留已保存结果及成功摘要后，追加清理失败消息并保持可见提示。新增`tests/browser/task-cleanup.spec.cjs`先在旧界面复现普通成功文案覆盖警告，修复后单项1 passed，完整Edge组45 passed；测试使用真实重定轨与OSU导出，仅对job响应注入展示标记，与API真实故障注入互补。后续普通成功能清除旧警告。
- 新增`tests/test_task_cleanup_errors.py`覆盖取消、最终保存失败、已提交成功、executor提交失败四个回调异常入口；与声轨6例、既有任务/取消/服务API/声轨回归合跑71 passed（20.60秒）。命令为`.venv\Scripts\python.exe -m pytest -q tests/test_task_cleanup_errors.py tests/test_provided_cleanup_errors.py tests/test_task_lifecycle_flow.py tests/test_task_service.py tests/test_cancellation.py tests/test_server.py tests/test_provided_stems.py`；OMP/MKL/OPENBLAS线程为2。Edge命令`MANIALAB_TEST_BROWSER=msedge`、端口8978后`npm run test:ui`，完整45 passed（1.6分钟）；相关`compileall`、两处JS语法与`git diff --check`通过。
- 主agent独立复核声轨删除差异与新回归；候选/导出agent独立复核任务服务设计；生命周期agent独立复核UI及展示测试，均未发现阻断项。PR/main Windows验证待补。UI临时产物位于ignored `tests/artifacts/browser/`，本阶段不运行模型或读取用户媒体、权重和研究产物。
- 剩余边界明确开放：ENG-14无可用参考书签/Kiai导致旧结构被清空，修复已在独立文件排队；ENG-15确认子进程未回收后仍可开始下一任务，可能破坏串行进程要求，需专用门禁，不能因本阶段允许文件清理失败后复用worker就宣称此项解决；ENG-16固定BPM重拟合把123.456789四舍五入成123.45679，真实Edge丢弃诊断，另行修复。本阶段不代表阶段2–3、音乐或发布许可整体验收完成。
- PR #191已合并为`0f89440ab8563fd57b1348da605fc901252e7d96`；PR Windows run `44532957056`成功，Python/API `2766 passed, 89 skipped`、浏览器`45 passed`，Core便携与模型步骤按条件跳过。main run `44533920894`实际失败：Python/API通过，浏览器44 passed/1 failed，旧证据回看未成功选中音符；完整/失败日志保存在ignored `build/task-cleanup-00080e0/main-191-windows.log`及`main-191-failed.log`。选择helper修补和后续验证随第八十四阶段记录，不把失败main改写为成功。

## 2026-11-19 第八十四阶段：拒绝不可用的参考结构边界

- 合成API先复现全部书签越界时仍接受空结果、把旧结构改为整曲单段并清空plan/手动expression；Kiai虽已有空结果拒绝，先过滤浮点时间再round仍允许±0.1ms回落到首尾端点，触发相同状态变化。两轮旧实现分别出现2项和4项真实失败。
- bookmarks/Kiai现在先按既定Python规则得到整数音频时间，再过滤最终音频范围；全部不可用时拒绝任务并保留完整旧状态。合法正负提前量仍只平移毫秒边界，不按BPM缩放。用户显式手动`action=edit,boundaries=[]`仍可清成整曲单段，此修复只拒绝不可用参考。
- 新增`tests/test_structure_clock_audit.py`14项API回归，覆盖越界/舍入端点、合法bookmarks/Kiai×±50ms、非法手动边界，核对旧项目/候选不变及合法操作正确失效当前plan/expression。`.venv\Scripts\python.exe -m pytest -q tests/test_structure_clock_audit.py tests/test_structure_expression.py tests/test_generation_clock.py tests/test_edited_candidate_bookmarks.py tests/test_timing_diagnostic_clock.py`为83 passed（4.61秒，OMP/MKL/OPENBLAS=2）；相关`compileall`、`git diff --check`通过。主agent只读复核未发现阻断项。
- 未新增正式测试文件的临时合成核验确认最终状态保存失败保留旧项目，以及两难度参考标记/结构往返独立恢复；不把这些核验冒称为完整媒体或音乐验收。PR/main Windows验证待补；本阶段不改Timing拟合算法、Offset舍入、音符或音频，不读取用户素材/模型和研究产物。
- 下一阶段分别整合ENG-16的固定BPM精度与mode迟到响应失效、ENG-15的进程专用门禁，以及非Core验证请求/打包来源fingerprint缺口；仍由3名子agent持续推进不同检查点。
- PR #192首轮后另补主线浏览器验证等待：`selectNote`原先只等hitbox存在并单次点击，失败日志中检查器仍为未选择提示；现按当前hitbox重试点击并等待实际选择的ID，不降低原始/未知来源断言。失败单项Edge重复10次通过，完整evidence组待汇总，新head CI仍需通过。此为验证交互同步，不能代替另已发现的native媒体加载/播放状态产品问题。
- 完整evidence组复验10 passed（29.5秒）；新head PR run `44535877376`成功，Python/API `2780 passed, 89 skipped`、浏览器`45 passed`，Core便携与模型步骤按条件跳过。PR #192已合并为`f3b7707c8700dcc58e4294b154394b731f9031d6`；main run `44536768911`待最终核验。PR日志位于ignored `build/reference-boundary-68af4d7/pr-192-windows.log`。
- 合并后main run `44536768911`实际成功：Python/API `2780 passed, 89 skipped`、浏览器`45 passed`，Core便携与模型步骤按条件跳过。ENG-13/14关闭；主线完整日志为ignored `build/reference-boundary-68af4d7/main-192-windows.log`，不覆写此前失败记录。

## 2026-11-19 第八十五阶段：Timing重拟合保留固定时钟身份

- 真实API与Edge复现固定BPM123.456789返回123.45679，界面保留原输入且严格时钟比较令诊断为null。数值拟合实际使用原请求BPM，损失只在`_score`显示字段；路由现将已验证请求BPM覆盖回拟合结果后统一用于响应/诊断，不改拟合算法或Offset舍入。
- 另一条真实API屏障复现：audio模式请求refit后切remap，迟到响应仍改已禁用Offset并显示audio诊断；audio→remap→audio往返也会误接收。mode变化现在递增Timing请求身份并清除旧诊断/定位；新audio请求仍可正常检查。
- 新增`tests/test_timing_refit_precision.py`5例与`tests/browser/timing-refit-precision.spec.cjs`7例，覆盖精度BPM、正负提前量、负Offset ties-to-even、BPM10/1000和Offset±600000、模式切换与往返。浏览器旧响应先由真实API计算，再以明确屏障释放并等待原handler结束，未mock修复后的成功数据。
- 主agent复跑`.venv\Scripts\python.exe -m pytest -q tests/test_timing_refit_precision.py tests/test_timing_diagnostic_clock.py`为19 passed（5.01秒）；Edge端口8978定向`timing-refit-precision.spec.cjs timing.spec.cjs task-cleanup.spec.cjs`合计20 passed（35.7秒），同时确认已合入的清理失败警告保持。相关`compileall`、两处JS语法与`git diff --check`通过。生命周期agent独立只读复审未发现阻断项。
- 完整因果脚本/JSON/截图在ignored `build/timing-refit-precision-20261119-ln/`，不含用户资产；该阶段不读模型或真实研究音频。PR/main验证待补。native媒体快速切换归零/加载中显式暂停被旧回调恢复的问题已另列ENG-20，不由Timing mode修复代签；后续阶段单独整合。
- PR #193已合并为`575a4a5e83aa19d282a34f627a3488be744b0cab`；PR run `44537412322`成功，Python/API `2785 passed, 89 skipped`、浏览器`52 passed`，Core便携与模型步骤按条件跳过。main run `44538269327`待核验；PR日志位于ignored `build/timing-identity-788f5f9/pr-193-windows.log`。
- 合并后main run `44538269327`成功，同为Python/API `2785 passed, 89 skipped`、浏览器`52 passed`，Core便携与模型步骤按条件跳过；ENG-16关闭，完整日志位于ignored `build/timing-identity-788f5f9/main-193-windows.log`。

## 2026-11-19 第八十六阶段：已报告子进程回收失败后的任务门禁

- 真实无窗口Python sleep子进程在terminate故障注入后仍活着，旧服务允许第二任务运行；根进程已退、后代回收失败的Mapper路径也缺少门禁。`CancellationContext`现将进程回收失败与普通文件清理失败分开，专用标记向父上下文传播且保持粘性；`ensure_idle()`持锁在注册新job/分配上传目录前拒绝继续计算，提示先确认相关进程退出后重启。
- 文件cleanup失败继续保持阶段83的worker可复用契约，不触发进程门禁。旧callable仅有普通报告接口时保留诊断fallback；生产ProjectTasks使用专用context。即使旧根后来退出也不自动解除标记，因为没有完整后代身份/句柄证据，不能凭PID或根poll声称全部回收。
- 新增`tests/test_process_cleanup_gate.py`7项：真实子进程失败、成功回收、普通文件失败兼容、跨context粘性、旧callable、Windows根退后代失败、API预分配门禁。旧实现先为6 failed/1 passed，修复后相关组55 passed（8.24秒）；真实child启动另外改用`sys._base_executable`避开Windows venv redirector测试歧义，修改后单项1 passed，精确PID核验及finally kill/wait确认无遗留。相关`compileall`、`git diff --check`通过；候选/发布agent独立只读复审未发现阻断项。
- 已运行命令为`.venv\Scripts\python.exe -m pytest -q tests/test_process_cleanup_gate.py tests/test_task_cleanup_errors.py tests/test_task_service.py tests/test_cancellation.py tests/test_audio_io_boundary.py tests/test_provided_cleanup_errors.py tests/test_task_lifecycle_flow.py tests/test_upstream.py -k 'not real_model'`，OMP/MKL/OPENBLAS=2。base直接child小修后仅复跑该真实进程用例，1 passed（3.66秒），不重复整组。
- 此阶段只门禁已经上报的进程回收失败，不能证明通用`stop_process`自动回收完整Windows进程树。另已实测venv redirector根退出而payload仍活、旧helper返回True且无标记，列ENG-21；原型在ignored `build/task-process-tree-draft-20261119-01/`继续验证，不把设计当实现。复现/设计保留于ignored `build/task-child-quarantine-20261119-01/`。
- PR/main验证待补；没有运行神经推理、真实用户素材或研究实验。神经推理仍串行，工具并发只用于互不冲突的工程检查。
- PR #194已合并为`aba2d44b80219c1737247239859e528b02fcf7d1`；PR run `44538891542`成功：Python/API `2792 passed, 89 skipped`、浏览器`52 passed`，Core便携与模型步骤按条件跳过。main run `44539773445`待核验；日志在ignored `build/process-gate-05e6d18/pr-194-windows.log`。
- PR #194合并后main run `44539773445`成功，同为Python/API `2792 passed, 89 skipped`、浏览器`52 passed`；ENG-15关闭，日志在ignored `build/process-gate-05e6d18/main-194-windows.log`。

## 2026-11-20 第九十二阶段：局部生成事务与发布证据身份链

- 阶段2只读审查未发现已复现的生产错配；发现真实regional generation、连续编辑、JSON/OSU/OSZ及重开后来源身份验证目前分散在测试中。现有跨路径测试曾手工给父候选添加region字段，没有走真实regional生成入口。
- 在`tests/test_candidate_source_export.py`新增合成API集成：通过真实异步`POST /generate`生成父候选和regional child，核对`source_evidence`、selection决策及分离身份；随后编辑区域音符，验证JSON/OSU/OSZ导出、切回父候选和regional child，以及重开后的输出与来源身份。
- `tests/test_task_lifecycle_flow.py`新增regional child最终状态保存失败故障注入：任务返回error且不泄露result，原父候选和来源快照不变、worker恢复空闲；恢复保存后重试成功并提交child。
- `scripts/verify_portable.py`为每次运行生成`verification_run_id`，仅绑定本次调用新增的吉他/乐器专项行；`scripts/collect_release_validation.py`按profile capability校验组件身份、专项检查完整性及其与本次EXE/manifest/run ID的一致性，拒绝旧运行行拼接。
- 复审发现core/standard报告可夹带未绑定的extended专项检查行并进入通过摘要；collector现拒绝profile capability之外的吉他、取消及乐器专项行，六种core/standard组合均有负例测试。
- 本轮聚焦命令`.venv\Scripts\python.exe -m pytest -q --tb=short tests/test_candidate_source_export.py tests/test_task_lifecycle_flow.py tests/test_release_validation.py tests/test_verify_portable_core.py tests/test_process_cleanup_gate.py tests/test_upstream.py tests/test_regional_selection_report.py -k 'not real_model'`为`122 passed`（73.17秒）；`.venv\Scripts\python.exe -m compileall -q manialab scripts`及`git diff --check`通过。所有音频均为合成输入，未运行模型、EXE或研究脚本，也未读取用户素材。PR/main尚未验证；本阶段是覆盖与发布门禁增强，不是音乐质量结论。
- 后续阶段92补充：候选新增持久`generation_origin`，连续编辑继承，旧候选沿唯一可用父链回填；缺失/歧义/循环/非有限时间键或父链裁剪后无法确认时显示`unknown`。Edge覆盖多代编辑、父子切换、刷新和裁剪；新增手动逐段表达与regional rewrite组合链，确认锁定音符、expression/structure快照、编辑重开和JSON导出一致。任务测试另覆盖状态保存成功后cleanup阻塞期间收到HTTP cancel，最终以`done/result`收尾并保留已提交项目状态、cleanup标记及worker复用。
- collector现将所有profile的`licenses/runtime/`与`runtime-inventory.json`逐文件路径/大小/SHA绑定release manifest，并核验ZIP根目录及`_internal`副本；extended另绑定吉他/Tsumugi组件及专项许可。ZIP路径门禁拒绝大小写、尾随点/空格规范化、文件/目录节点冲突、Windows非法字符及保留设备名。源码/Windows workflow公开支持范围明确为`core`。
- 最终聚焦命令`.venv\Scripts\python.exe -m pytest -q --tb=short tests/test_candidate_source_export.py tests/test_task_lifecycle_flow.py tests/test_release_validation.py tests/test_verify_portable_core.py tests/test_process_cleanup_gate.py tests/test_upstream.py tests/test_regional_selection_report.py tests/test_generation_origin_ambiguity.py -k 'not real_model'`为`174 passed`；`npm run test:ui -- tests/browser/regional-audit.spec.cjs`为`5 passed`；`compileall`、全部`static/*.js`的`node --check`及staged/unstaged `git diff --check`通过。均为合成输入，未运行模型、EXE或研究脚本，也未读取用户素材。PR #201原run `44677307367`于`05:47:56 UTC`结束为`cancelled`，step 6未完成且日志不可用；本地增强待推送后的新CI验证。本阶段是覆盖和证据门禁增强，不是音乐质量结论。

## 2026-11-19 第八十七阶段：便携验证请求与构建来源闭包

- 真实合成API复现`verify_portable.py`非Core六声部检查仍使用缺少source_identity的声轨/波形请求，guitar/piano四条均400；Core不走此循环。现提取小helper，按当前采音`context=evidence`及身份请求，即使选中候选A、当前分析B，也核对B的实际声轨字节与波形峰值；身份缺失时在发送请求前拒绝。
- 受控源树复现实际打包/运行依赖的vendored `event.py`、`template.osu`变化不进入旧指纹。source_snapshot现在覆盖vendor Python包闭包、模板、构建recipe，以及启用原模型安装器时的源码ZIP/requirements；排除后复制的uv二进制，Core不增加未包含组件要求。该修复只约束以后新构建，不为历史dev.18包补签源闭包。
- 新增`tests/test_portable_capability_audit.py`9项；已运行`.venv\Scripts\python.exe -m pytest -q --tb=short tests/test_portable_capability_audit.py tests/test_verify_portable_core.py tests/test_build_provenance.py`为15 passed（2.04秒，PYTHONUTF8=1、OMP/MKL/OPENBLAS=2）。相关`compileall`、`git diff --check`通过，主agent只读复核未发现阻断项。测试只执行提取的纯source_snapshot函数与真实Flask合成API，不启动builder入口。
- 本阶段没有构建、下载、运行EXE/模型或读取用户素材。真实nonCore包的离线/Unicode/专用模型/关停仍未验证；当时collector也尚未强制extended专用fixture覆盖。后续阶段92补上对应源码门禁，ENG-05/06仍开放，历史dev.18包不因此补验。
- PR #195已合并为`ed16855bfb0b911dab5d97874f72719d79a2d31e`；PR run `44540158809`成功，Python/API `2801 passed, 89 skipped`、浏览器`52 passed`，Core便携与模型步骤按条件跳过。main run `44541152663`待核验；日志在ignored `build/portable-source-d7ddc19/pr-195-windows.log`。

## 2026-11-19 第八十八阶段：发布运行报告绑定清单摘要

- 合成复现确认：同一EXE字节配上不同source/static内容和新`release-manifest.json`/ZIP时，旧portable与shutdown报告原先仍可被collector接受；仅EXE SHA无法证明来源声明一致。portable唯一`release_manifest`记录、shutdown身份和collector均绑定实际manifest原始字节`release_manifest_sha256`，且collector另要求ZIP内manifest字节与本地已验证manifest完全相同；旧缺字段、摘要错配、重复身份和仅改ZIP内空白均拒绝。
- 新增同EXE改source/static后旧报告拒绝、仅旧portable摘要更新而shutdown仍拒绝、跨Core/standard身份缺失/重复/版本错误及ZIP manifest字节变化等回归。定向命令为`.venv\Scripts\python.exe -m pytest -q --tb=short tests/test_release_evidence_identity.py tests/test_verify_shutdown.py tests/test_release_validation.py`，最终`47 passed`；相关compileall、`git diff --check`通过。未执行EXE；合成字节不代表实际便携运行。
- portable/shutdown/collector绑定实际manifest原始字节，collector同时核对ZIP内manifest字节。历史dev.18及旧报告不自动补签。PR #196 merge commit `256ecf8`，PR run `44544668731`与main run `44546117834`均成功。ENG-19源码契约已闭合；真实Core集成另记第九十一阶段。
- 手动CLI JUnit与源码身份残余仍由COV-04记录；托管CI同job没有发现跨run混用。该边界不扩大成本阶段音乐质量结论。

## 2026-11-19 第八十九阶段：Windows worker 进程树回收

- 真实venv redirector复现旧stop_process只等待根PID返回成功、payload仍存活；Windows worker现以CREATE_SUSPENDED创建，先加入non-breakaway Job再恢复初始线程，覆盖分离与专用乐器worker。停止时按统一取消deadline终止Job、等待根和可验证成员句柄，并把成员枚举、终止、等待、active count或句柄关闭异常传播到既有进程清理门禁；PID在打开句柄后还会以IsProcessInJob确认，避免误杀Job外复用PID。
- 启动故障注入额外复现Popen Handle已关闭后基础析构仍调用WaitForSingleObject，触发unraisable WinError 6；子类析构现检查closed标志。warning-as-error tests/test_worker_process_tree.py为20 passed；完整生命周期/进程组合为62 passed（OMP/MKL/OPENBLAS=2），compileall与git diff --check通过。没有运行模型、EXE或真实音频。
- PR #197 merge commit `75179ae`；最终PR run `44547867366`及main run `44549577592`成功。外来PID、venv与根已退出子进程、启动/回收故障、closed-handle析构、句柄关闭与共享deadline均有覆盖；warning-as-error worker测试20 passed，本地生命周期组合62 passed。成员PID快照和TerminateJobObject间仍可能出现没有逐个预持有句柄的新成员；Job终止及active count是补充验证，不宣称所有竞态下内核同步回收。
- ENG-21源码契约已闭合；Core合成运行不证明任意worker竞态，也不等于音乐质量改善。

## 2026-11-19 第九十阶段：native媒体切换与提示音等待状态

- 快速切换声部、视图、候选或项目时，native媒体source请求递增身份并清理旧metadata listener；暂停播放保留目标位置，加载中显式pause不会被旧回调恢复，播放拒绝显示为错误。`waiting`事件清理已排提示音，只有媒体达到`HAVE_FUTURE_DATA`后再安排后续提示音。
- Playwright helper先检查`readyState`再等`loadedmetadata`，修复Edge上事件早于listener注册而超时的测试竞态。定向Chromium/Edge各10 passed；PR #198 run `44549463878`、main run `44552057267`成功，main全组Python/API `2857 passed, 89 skipped`、Edge `62 passed`。合并提交`bee69ca`。本地回归覆盖真实HTMLMediaElement缓冲/暂停/seek，不是打包EXE浏览器端到端。
- ENG-20源码/浏览器契约已闭合；打包Core静态服务器里的媒体播放仍未验证。此阶段不证明音乐可听性、Timing或玩家体验改善。

## 2026-11-19 第九十一阶段：冻结main上的Core构建与身份链验证

- 前置核对：main `bee69cad232a01206051ba59f422961c17e94259`与origin/main相同、工作区干净；`licenses/FFmpeg-GPLv3.txt`存在；`licenses/FFmpeg-build-info.txt`构建后无tracked diff；没有ManiaLab/pytest/研究进程。唯一产物ID为`core-evidence-20261119-bee69cad232a-20261119T232220475-32408`，位于ignored的`build/core-evidence-runs/`、`build/portable/`、`test-data/core-evidence/`，没有覆盖历史目录或创建Release。
- 指定计划执行的新鲜pytest为`2856 passed, 90 skipped`（JUnit tests=2946、failures/errors=0），Edge完整组`62 passed`；Core PyInstaller build成功。新EXE的portable verifier通过：打包Basic Pitch v0.4.0 synthetic analysis、6/7/8/9/10键生成导出、提供合成other stem分析导出、缺失deep engine错误、重启保持、Unicode音频路径和坏音频后服务继续；Core关停验证通过且退出前/后均无worker。Collector验证fresh JUnit、source snapshot、EXE/manifest/ZIP身份、CRC、Core精确payload与Basic Pitch身份。
- Core ZIP为`ManiaLab-0.1.0-alpha.6-dev.18-windows-x64-core.zip`，SHA-256 `83cb0453d1ad4e502cc619cb82949f6f285fa70c80980ab6d9f675b3fb35700c`；EXE SHA-256 `d275ee2c4eb893d8ac350feb9da8867713c9c6826ddc92a95f8b929e9ebf07dc`；raw release-manifest SHA-256 `719ad2ba71d5aa436aed879101dc6ae255598b15ba418d54f4b458e033012469`。portable/shutdown与实际文件、collector一致。
- 原计划脚本执行到所有build、portable、shutdown和collector步骤后，在最终PowerShell 5.1断言因JSON根数组被包装为嵌套`Object[]`而返回非零，误报`EXE identity chain differs`；实际EXE SHA在manifest/portable/shutdown中完全一致。ignored计划脚本已显式枚举portable报告数组；在Windows PowerShell 5.1下用同一run目录里的JUnit、browser JSON、portable/shutdown报告、collector、manifest和EXE重新执行最终断言，逐项通过，没有再次build或启动EXE。保留首次exit code=1及后续final reconciliation；详细审计保存在该run目录内。
- 本阶段只证明Core profile合成/打包检查，不证明standard/extended包、非Core声部分离/专用模型/权重许可、Tsumugi parity、真实歌曲质量、Timing听感、真人盲听、6K–10K/LN实玩或打包服务器里的浏览器播放。没有读取或修改ignored音频、谱包、权重及逐音研究产物，也没有创建Release。
