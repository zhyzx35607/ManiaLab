# 架构与开发入口

ManiaLab 分为应用、音乐领域、运行组件和研究工具四层。新增功能先放进对应模块，再由路由组合；历史研究文件保持原路径，避免破坏协议、测试导入和输入哈希。

| 层 | 入口 | 职责 |
| --- | --- | --- |
| 启动与 HTTP | `launcher.py`、`manialab/server.py`、`manialab/task_service.py`、`manialab/project_routes.py`、`manialab/analysis_routes.py`、`manialab/arrangement_routes.py`、`manialab/candidate_routes.py`、`manialab/upstream_routes.py`、`manialab/export_routes.py`、`manialab/timing_routes.py`、`manialab/playback_routes.py` | 本机访问、任务生命周期、项目读写/谱面偏好导入、采音与导入声轨、编排建议/段落/表达、候选生成与编辑、Mapperatorinator集成、Timing检查/重拟合、JSON/OSU/OSZ导出、播放、试听、波形响应 |
| 制谱服务 | `manialab/generation.py` | 组合证据、表达、量化、选音、定轨和报告；不依赖 Flask 或磁盘 |
| 音乐领域 | `domain.py`、`timing.py`、`preferences.py`、`expression.py`、`arrange.py`、`layout.py`、`solver.py` | 音符与谱面、时钟、分段策略、占轨约束、轨道搜索与共享数值求解 |
| 音频组件 | `audio.py`、`separation.py`、`piano.py`、`guitar.py`、`instrument_detector.py` | 分离、采音、模型身份、独立缓存和 CPU 推理 |
| 项目与文件 | `storage.py`、`provided_stems.py` | 原子保存、安全路径、上传冲突检查、声轨来源与失败清理 |
| 浏览器 | `static/` | 当前候选、待应用参数、试听与编辑；不重新解释模型能力 |
| 构建与研究 | `scripts/`、`docs/` | 固定构建、运行验证、冻结协议与研究结果，见[脚本索引](../scripts/README.md) |

`analysis_routes.py`集中采音和导入声轨接口。应用组合层显式提供项目任务、音频路径、分离器和检测器；可替换分析函数保存在`app.extensions['analysis_services']`，以便测试隔离模型与文件处理依赖。

`solver.py`统一封装进程级 HiGHS 调度锁、MILP 单线程选项与 SciPy 警告过滤。定轨、表达和研究规划直接依赖该基础模块，不再借用`arrange.py`作为求解运行时入口。

`upstream_routes.py`集中Mapperatorinator安装与生成接口；应用组合层传入共享`MapperEngine`、项目仓库、异步任务提交及当前项目Timing闭包。原模型生成保留输入键型、经确认的固定时钟对齐和完整remap报告。

`candidate_routes.py`集中普通候选生成与编辑。生成算法仍由无HTTP依赖的`generation.py`提供；编辑在原候选上派生新版本，保留版本历史、Timing/落线报告和声部证据。

项目谱面导入与参考风格偏好导入共用`project_routes.py`。参考风格路由只更新项目级风格/设置，不覆盖音频分析或候选；参数规范器和风格分析器由组合层注入。

## 两种时钟

原始音频时钟用于检测证据、结构段落和音乐表达。制谱时钟用于实际音符起止、拍线、占轨、试听提示音及导出。

`chart_clock_notes` 保存 `audio_start`，然后应用提前量；量化只改制谱起止。`evidence_start/evidence_end` 继续表示量化前的制谱时间，因此现有量化偏差报告含义不变。旧项目没有 `audio_start` 时可读取，重新从分析生成后获得该字段；不会改写旧分析。

`ChartPreferences` 同时持有原音频区间和换算后的制谱区间。按音符查询声部权重、LN 资格时使用原起点；按时间查询持续占轨时使用制谱区间。量化把不同段落的音符合到一排时，声部策略各自保留，公共密度、和弦和占轨容量取较紧值，布局软偏好取原起点最早的音符。原谱严格重定轨不进行音频时钟转换。

Timing诊断的12秒窗口固定属于原音频，窗口残差按当前制谱起音与拍线计算。
`analysis_timing_diagnostics` 是检查、Offset重新拟合和生成报告的共享入口；返回完整时钟身份、
可评估/证据不足状态以及已有的鼓轨/混音补查来源，不继承旧窗口的质量结论。
诊断可评估不等于自动Timing可接受，手动确认和原稳定性门槛保持独立。

`evidence.py`保存每个选中物件的原检测和表达快照，二者都是原音频时间；
`evidence_end`仍是量化前的表达尾在制谱时钟的位置，不能直接称作物理检测尾。
编辑继承快照，局部重写的锁定物件保留旧来源；旧候选没有快照时不能按ID借用后来分析。
前端的谱面检查与来源说明独立在`static/evidence-review.js`。

仅修改声部/段落/参考偏好或采用本地建议时，共享`captureTimingDraft`/`restoreTimingDraft`
保留未生成的拍线草稿、显式确认和未舍入Offset锚点。恢复后核对诊断身份，并重算候选拍线
列表和本地建议；不能把响应临时载入的旧候选BPM用于当前草稿的诊断或建议。

局部生成的`selection`由`selection_report.py`整理为`regional_output`，统计实际合并结果；
完整选择器报告放在`selection.proposal`并标作`full_song_proposal`。其省略理由和求解器
分数描述整首预选，不能解释锁定旧物件。实际输出决策只含制谱起止，不冒认原始检测时间。

`regional.py`在局部合并前校验乐句替代范围：若区间内起音由区间外的新乐句代表，
要求扩大范围或整谱重写。两端按实际提前量和量化后的制谱起点计算；旧锁定物件可能是
短键、编辑后的LN或另一轮分析，不能仅凭同ID、phrase标志推断它承担本次覆盖。
反向也检查实际锁定集合：区内新乐句已选中，但明确被其代表的内部ID仍锁在区外时拒绝，
防止乐句与旧内部音重复；旧音手动移位后以保存物件的起点判断。

## 项目提交与失败

`ProjectTasks` 在 `task_service.py` 持有单工作线程、任务表和应用级 `RLock`。后台操作由路由提供，服务负责状态推进、取消、日志和提交边界；同步项目写路由也使用同一把锁并检查项目是否正运行任务。操作只修改加载的项目快照，成功且未取消才原子保存。失败和取消保留上一版本；清理回调按是否已提交决定是否移除新上传文件和解码目录。`app.extensions['executor']` 与 `['jobs']` 保留为服务对象的兼容入口。

上传路径采用大小写不敏感的冲突检查；同一谱包先完整校验，再写入。嵌套谱面导出保留原始相对引用，必要时将补传音频写入该引用位置。

## 验证与下一步

运行 `python -m pytest -q` 验证领域逻辑与 API；音乐质量仍由冻结输入比较、原声核查和最终试打共同评估。合成测试不能证明新模型更准。

组件身份、构建范围、候选与目标键数状态，以及合成浏览器回归已经接入。运行命令和 standard/extended 的边界见[开发与验证](development.md)。构建资源按明确清单暂存；worker 的每个运行文件记录大小与 SHA，打包前后核对，不能把本地额外文件夹入发布包。

后续音乐研究使用既有冻结输入，分别观察 Timing 偏差、声部省略与 LN 音尾退步。Timing 的当前诊断见[拍速歧义结果](sakura-timing-ambiguity-results.md)，声部省略见[表达取舍审计](expression-tradeoff-audit-results.md)，LN 历史对照见[固定候选重放](expression-current-replay.md)。比较时先确认输入身份和脚本对当前代码的适用性；不得把不同冻结版本的数量直接当作改善，也不要重跑已经完成的模型队列。逐首人工反馈不作为自动开发的前置条件，最终听感与 osu! 试打仍单独验收。
