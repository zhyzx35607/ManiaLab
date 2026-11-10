# 续接进度

更新：2026-11-11，Asia/Shanghai。本文件用于中断或额度耗尽后恢复工作；
先核对实际进程、Git 和产物，不能把这里记录的运行中状态当作已经完成。

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
- 自动 Timing、真实混音的漏采/串音、声部连续性和 LN 音乐性尚未最终验收。
  不宣称已超越 Mapperatorinator。质量标准见[目标](music-quality-target.md)。

## 本阶段新增

- 修复同占用、同强度、同表达含义候选的去重顺序偏置，沿用原对齐评分择优，
  同分按ID；不同尾长、乐句代表和不同表达依据不作等价替换。25项专项及
  相关检查共114 passed。固定《樱花草》6–10K重放、定轨、导出回读通过：
  各键数160处候选代表变化，11对实际保留，全部other；物件/LN总数不变。
  6K/7K/9K分别额外2/1/2处对象取舍，8K/10K对象内容不变；不作音乐收益
  结论。详见[重复采音选择](expression-duplicate-selection.md)，新比较入口
  `scripts/check_expression_duplicates.py`，产物`test-data/expression-duplicates-v1`。
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
及MuScriptor全部36输入、评分，实际exit 0。当前没有排队的神经推理。
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
- 下一次提交更新本文件并单独开 PR、检查、合并。已有源码与小阶段记录
  保存在 GitHub；本地 ignored 音频/权重仍需本机保留，GitHub源码不包含它们。
