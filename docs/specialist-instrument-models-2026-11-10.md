# 新增的乐器专用模型与接入决定

继续查阅作者源码、公开权重及解码实现后，新增以下具体候选。此前Tsumugi、
YourMT3、ROSVOT等已有比较，不重新包装为本轮新发现。

| 项目 | 实际用途 | 当前处理 |
|---|---|---|
| STRAdi offline | 独奏小提琴起音、音尾、活动与F0 | 已下载源码及94MB权重，实际完成12条小提琴对照；起音92.85%、完整区间83.15%，低于当前Tsumugi，见[结果](stradi-violin-results.md)。 |
| xavriley/hf_midi_transcription | 贝斯、萨克斯、吉他的专用CRNN转录 | FiloBass 20K／FiloSax 25K已完成固定17＋2条CPU对照；起音／区间F1分别88.43%／56.37%、44.68%／28.72%，低于对应Tsumugi，见[完整结果](hf-instrument-specialists-results.md)。两份吉他权重已下载，[完整60例对照](hf-guitar-specialists-protocol.md)正在串行执行，尚未完成；不替换默认。 |
| Banquet / query-bandit | 用示例音频查询，从混音再提取目标乐器波形 | 两份作者权重、推荐权重共同增益诊断共96次前向全部完成。两权重的8组混音整窗SI-SDR均未胜过原混音；另一权重仅小号中央窗局部改善，见[实测](banquet-separation-results.md)。未接入采音前处理；不是逐音转录。 |
| PESTO | 单F0音高轨迹及发声活动辅助证据 | 已在全部35个弦管乐输入实际运行；直接筛选音符导致明显退步，见[完整结果](pesto-instrument-support-results.md)，未接默认。 |

STRAdi来源为`https://github.com/seingreen/STRAdi`，commit
`6b9612056b2a31775ef833dc38337ed6040b1f8c`。offline权重来自作者v1.0.0
Release，SHA256 `2b344f91fe5d49499faf553b982ab5944ee6b9be46de07879ca727fa54f64123`。
代码Apache2；README仅明确source code，权重未另列条款，因此只本地比较。
模型采用整片峰值归一化，解码含6秒、输入末端和同音重弹封口；不能直接把
CSV尾当LN。输入、真实长度、原始四个输出头和版本均由新入口保存。

HF MIDI来源为`https://github.com/xavriley/hf_midi_transcription`，commit
`96f6797881e9497cbfc8f8e5deccea9c1f2f7adc`，依赖fork
`https://github.com/xavriley/piano_transcription_inference` commit
`7568dc7f78b625e40cf9776e2806d164006610e3`。权重仓库为
`https://huggingface.co/xavriley/midi-transcription-models`，revision
`b7bec65a2b860aca72856b0feef58b5df407b777`，公开无gate、模型卡MIT。

| 权重 | 字节 | SHA256 |
|---|---:|---|
| filobass_20000_iterations.pth | 98,882,013 | cbc792a1b6b6002a73ee0a433c0fe5e097ee728034fe2ea70b24bd06b5026e84 |
| filosax_25k.pth | 99,341,469 | 448cf2c8ea6d4b77f7435f5b9a496211ea60300c5c17fa9c754da764f75f3a6a |
| guitar-gaps.pth | 99,178,877 | 65483e7c0e340a90415b15b520687587698c8c728f5fa470a205f13ee45c6513 |
| guitar-fl.pth | 98,916,957 | 50d93dba89bdd3401849bc735614478e83d9f46d21fa3f71d8aca5acc0a52028 |

不能直接照搬其包装层：`forward`原样返回输入；自定义阈值没有传给底层；
底层使用宽松权重载入，输出按采样数而不是帧数裁切。仓库中的76字节
`model.safetensors`只是dummy参数，真正的网络在上述pth中。应按固定单个
文件取权重，复用底层模型并保存原始激活。当前代码README和pyproject标MIT，
但两个源码仓库未见独立LICENSE文件，归属记录保留此差异。

萨克斯论文`https://arxiv.org/html/2405.16687v1`说明FiloSax训练27.5K步，
公开文件名却是25K，不能直接套论文分数；FiloBass论文
`https://arxiv.org/html/2311.02023v1`主要介绍数据集，未确认该20K权重的
准确训练划分。本轮已按[固定协议](hf-instrument-specialists-protocol.md)完成全部17个贝斯和2个萨克斯输入，安全严格载入后保存原始窗口及物理时钟输出。推理累计347.001秒，复用三条兼容性结果后新增275.142秒；汇总阶段旧URMP缺少`available`字段的问题通过核验原产物后离线重算恢复，没有重跑网络。萨克斯104个漏采的起音头在正确音高附近均未过默认阈值，但对应区间内frame仍有活动，不能归纳为F0全面失败。

Banquet来源为`https://github.com/kwatcharasupat/query-bandit`，commit
`79ed5bb75e5c3a40cd319d9d990cee913fc65c26`，作者权重Zenodo记录13694558。
代码MIT，权重元数据CC BY-NC-SA4.0。10秒query短了循环、长了截断。
已严格载入完整网络，包括checkpoint中的PaSST编码器；未修改应用依赖。
固定使用不重叠query与混音片段，并保留目标独奏、目标缺席和静音控制。
推荐权重`ev-pre-aug.ckpt`的实测及统一增益诊断均不支持直接接到采音前面。
作者仓库issue6的普通用户反馈提供另一权重线索，但没有作者确认，已按
[相同输入协议](banquet-alternate-protocol.md)完成32例验证，不覆盖原结果。
另一权重的整窗8/8仍低于原混音；中央窗只有小号高1.064dB，不能据此选入默认。
目前研究是单6秒窗口，未采用作者CLI重叠拼接；人工混音、低采样率单声道
与查询选择等限制详列于协议，不能将本机结果推广为全部应用场景无效。

全部静态记录和官方响应身份在ignored
`test-data/instrument-model-extra-search-2026-11-10/notes.md`。
原始音频、模型、逐音预测均留在ignored目录，公开仓库只保存适配与方法记录。
