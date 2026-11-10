# 续接进度

更新：2026-11-10，Asia/Shanghai。本文件用于中断或额度耗尽后恢复工作；
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
- 自动 Timing、真实混音的漏采/串音、声部连续性和 LN 音乐性尚未最终验收。
  不宣称已超越 Mapperatorinator。质量标准见[目标](music-quality-target.md)。

## 本阶段新增

- Harmonica 第三方复现的固定对照入口：全部35份已有 URMP 输入及40秒静音。
  安全载入真实权重已通过，116个存储张量逐值核对、679444参数、6个解码
  控制通过。没有本机声学成绩；不是论文作者权重。未查到源码许可，未打包。
  见[协议](harmonica-reproduction-protocol.md)。
- MuScriptor 公开 GGUF small 已下载并核 SHA，逆名称/布局转换为 safetensors；
  正向转换与回读逐值相同。原 Python 模型严格载入122个状态张量通过，
  102441984参数。15个纯符号事件控制和6个计分控制通过；保留完整token、
  人为封口、错误区间与原始类别。本阶段尚无真实采音分数。
  见[权重准备](muscriptor-gguf-preparation.md)和
  [比较协议](muscriptor-gguf-transcription-protocol.md)。
- 以上均为研究工具，未改变应用默认，没有新训练或新 Release。

## 正在执行的固定队列

当前任务的 shell session `69438` 等待 Mega53 分离进程退出，然后依次运行
以下三项。任何步骤失败即停止。session/PID 仅用于本次会话定位，重开后先
确认命令行和启动时间，不能凭旧PID启动、结束或重复任务。

1. 分离任务本身：session `42354`，观察到的 Python PID `15668`。
   `scripts/check_mega53_long_context.py --output test-data/mega53-long-context-v1`
   使用作者20秒窗和拼接规则，共22份PCM、44次前向。日志：
   `test-data/mega53-source-review-v1/long-context-run.log`。
2. 下游采音：应用 `.venv` 运行
   `scripts/check_mega53_transcription.py --separation test-data/mega53-long-context-v1 --output test-data/mega53-long-transcription-v1`。
   日志：`test-data/mega53-source-review-v1/long-transcription-run.log`。
3. Harmonica：音频 `.venv-audio` 运行
   `scripts/check_harmonica_transcription.py --output test-data/harmonica-urmp-v1`。
   日志：`test-data/harmonica-source-review-v1/inference-run.log`。
4. MuScriptor：应用 `.venv` 运行
   `scripts/check_muscriptor_urmp.py --weights test-data/muscriptor-recovered-small-v1/muscriptor-small-recovered-f16.safetensors --output test-data/muscriptor-urmp-v1`。
   日志：`test-data/muscriptor-cpp-review-v1/urmp-run.log`。

根 `report.json` 未出现不等于没有运行；先看日志、进程和中间回执。失败时保留
已有证据，排查后使用新输出目录或有身份核对的恢复流程，不能直接覆盖。
两新模型全部推理结束后才读取逐音参考。尚未开始的报告不能用来发布成绩。

唯一活动子代理 `/root/piano_stream_exports` 负责静态审核和完成后的独立复算，
不运行神经模型。独审目录：`test-data/mega53-long-independent-v1`、
`test-data/mega53-long-transcription-independent-v1`、
`test-data/muscriptor-independent-v1`。Harmonica独审待补。

## 已知结果与下一步

- Mega53旧6秒短窗的波形改善没有转成整体采音改善，不能直接接默认；
  长窗是对同一来源的补充验证。见[旧结果](mega53-transcription-results.md)。
- 当前《樱花草》固定候选10K选择为1495物件/373LN，是定轨前重放，
  不是新谱面或试听验收。见[当前规则重放](expression-current-replay.md)。
- 等队列完成，独立核对各乐器的起音/完整区间、目标缺席误报与人工封口。
  有收益的候选再用真实 Nova 声轨验证，保留失败项；不只凭波形或平均分接入。
- 下一次提交更新本文件并单独开 PR、检查、合并。已有源码与小阶段记录
  保存在 GitHub；本地 ignored 音频/权重仍需本机保留，GitHub源码不包含它们。
