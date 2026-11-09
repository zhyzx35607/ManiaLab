# 钢琴起音与持续状态双头适配

2026-11-10，在本轮真实双头前向、预检和训练之前固定。本实验从原权重开始，检验起音和持续状态一起更新能否减少错误而不造成逐音退步。它不修改旧起音头实验的协议或选轮结果，也不是 LN 标签训练或产品验收。

## 固定资料与模型

仅访问已缓存的 6 train（449、1046、518、1176、926、1222）和 4 validation（1219、628、1134、752）。每曲原输入前180秒、35个10秒窗、5秒步进；只评分 validation 前170秒起音，保留完整音尾。selection SHA256 为 `34947943dcf11daf1b68939d005e05fd7331fe40413e575e70140a88248eb1aa`，缓存 report SHA256 为 `29a84fe641efbda504cd815dfea777de81ae8452fb6fc3fad1ebb87e36ad0a8f`。原缓存只读；新标签、mask、结果、代码快照独立保存。

Kong 上游提交 `0226e74cbc805660e34bbd6a8fed2083890ebb88`，原 checkpoint SHA256 `c3fa9730725bf4a762f1c14bc80cd5986eacda01b026f5a4a2525cd607876141`。复制原 `reg_onset_fc` 和 `frame_fc`，各为 Linear(512,88)，合计90288参数；原网络及所有 buffer 保持 eval/frozen。不要从旧起音 epoch1 继续。

复用每窗原 H、F0、offset 和其他固定输出；每个 batch 都以当前 onset 重新执行完整1001帧的双向 frame GRU，不能复用旧 Z/frame 或裁中间500帧再推理。保留上游 `onset.detach()`：onset BCE 只更新 onset FC，frame BCE 只更新 frame FC；两个头的前向数值仍耦合。GRU 在 no_grad 内，frame FC 在其外。

## 标签与训练配置

起音沿用原物理时钟三角目标 `max(0,1-|t-onset|/0.05)`。新增 frame 目标取同 pitch MIDI 踏板区间的半开并集：`onset <= t < sustain_end`。key release、踏板 sustain、可听衰减和谱面 LN 是四种不同概念。此标签不声称逐 bit 复现上游 round 后 inclusive 的 frame 标签。

先沿用原唯一归属，再要求整数全局帧 `5 <= 500*window_index+j < 17995`。每曲17990帧，每头每 epoch 共9498720个 pitch-frame 单元；完整1001帧仍用于前向。全局50ms边缘保护排除输入外起音的三角目标渗入，不删除正常内部重叠窗的上下文。跨过局部窗尾或全局180秒的持续音，在可观察输入内继续为正，不在截断点制造放手。

固定 seed42，float32，Torch 2.9.1 CPU、2 intra/1 inter线程，AdamW lr1e-5、weight_decay1e-4，batch4完整窗，两个头联合 clip norm1，完整2个epoch。每轮由独立 seed42 NumPy generator 产生覆盖全部210窗的排列并保存。loss 是两个头各自在 mask 内的无类别权重 BCEWithLogits 均值之和。不增广、不调阈值、不根据结果追加轮次。

## 训练前必须通过

1. 保存本协议、所有执行代码及缓存/模型身份 hash，派生 frame 目标和 mask 单独落盘并校验。显式允许清单只含上述十份录音，test 不读取。
2. 用原两个 FC 重放全部350窗，七输出逐样本一致；官方 deframe、本地拼接和原缓存整曲输出及解码全部相同。不事后放宽浮点容差。
3. 临时副本执行真实训练 batch 的独立梯度检查：onset loss 仅更新 onset FC，frame loss 仅更新 frame FC；联合一步后两个头均变化，原网络无梯度、参数/buffer不变。正式训练重新从原头和 seed42 开始。
4. 核验全局唯一归属、半开区间、同 pitch 重弹、重叠并集和右删失。恰好90288可训练参数，每头每 epoch9498720监督格。

## 固定开发门槛与选择

解码固定 onset0.3、offset0.3、frame0.1、pedal offset0.2。按现有同 pitch 一对一最大匹配报告两个成套口径：50ms起音加 `max(50ms,20%参考时长)` 音尾；40ms起音加80ms音尾。key、sustain分别评分。预测尾超界不从分母移除，参考不可观察尾单列。

epoch0始终可保留；epoch1/2候选必须同时满足：

- 两种口径、每曲均无新增漏采、额外预测不增加、紧邻真实重弹双起音覆盖不减少。固定 epoch0 选定的参考起音集合，候选的 necessary 起音覆盖必须包含它，不能只比较数量。
- 两种口径、每曲原本完整匹配的每个 sustain 参考仍完整匹配。固定 epoch0 已选完整匹配集合 B，要求候选完整区间的 necessary 覆盖包含 B。possible/necessary 考虑所有最大基数配对，不以次级时间误差挑有利身份；歧义完整保留。这是保守拒绝门槛。
- 对前后在所有最大起音匹配中均具有唯一固定配对的同一参考，若 sustain 尾绝对误差增加严格超过80ms则拒绝。使用未舍入时间；记录所有可比较尾误差及歧义。该值只属于本轮未来实验，不是通用音乐阈值。
- 全部门槛通过后，合并50ms起音F1或sustain完整区间F1至少一项严格提高。合格候选按起音F1、sustain区间F1依次降序，再按 epoch 升序选择；无合格候选就保留epoch0。loss下降、概率变化本身不算进步。

1219/ref373（MIDI62、起音51.683593750秒）的旧0.85秒拖尾是已知开发回归例，必须遵守相同逐音门槛。逐音结束来源、区间失得和波形/频谱另作诊断，不能由平均分替代。

## 解释与后续

这四份 validation 已用于方案设计，只能叫开发门槛；公共数据也可能参与基础模型预训练。原解码600帧/6秒上限仍存在，本实验没有修改它；可观察参考中 train 有6条、validation有49条 sustain 大于6秒。两个线性头不能保证恢复底层特征丢失的音，也不保证解决混音或分离残留。

保留test继续不读。只有合格且无已知阻塞的候选才能按 `piano-acoustic-candidate-validation-protocol.md` 完成91控制、11段SMD、完整Nova《樱花草》回归，再读取四份test。研究结果不自动替换应用；波形/频谱检查不记为直接试听，不凭一次训练发布新成品。
