# 逐音力度与突出声音：固定对照协议

本轮不把模型velocity接入候选strength，不用它删音或自动指定主旋律。
先检验它是否能区分同一声轨里不同音符的强弱，再考虑段落编排。

固定使用Tsumugi源码`eba1c7d05e4d6a53cc4e8916d742d9623a3fdc38`，
模型revision `c6ee8a22626cd7185031ee71ce7cea27ec862781`中的
`best_velocity_model.pth`，54996249字节，SHA256
`df8b0877c8b71cb5c3c224d69d386d67520417ecbc03728f02a2951abc4d2ae1`。
按checkpoint配置建立原模型，以`weights_only=True`读取并严格加载全部权重。
CPU两计算线程、一跨算子线程；8秒不重叠窗，保留原PCM幅度、音符时间和音高。
同一首窗内已有的音、未知尾也保留查询，但不把查询变成新的起音或放手证据。

作者模型需要精确GM program；现有转录只提供宽类别，`program=-1`。
作者forward会把负program截成0，即钢琴，不能当作支持未知音色。因此实验
显式枚举预测宽类别内全部GM program，每个都只是假设，不是重新识别的音色。
特殊非GM类采用作者明确规定的代表音色，并标出假设。模型类别错分仍会影响结果。

固定测试在观察模型输出之前确定：

1. 现有smoke清单全部四个完整片段（bass、guitar、other、vocals），复用原始
   音符查询；同一PCM分别乘以−12、−6、0、+6dB，并增加全零音频对照。
   不裁幅、不重新归一化，不跑转录来改变查询集合。统计逐音随增益单调性、
   同类别不同program的范围、静音输出，不能把非零velocity称为检测到了声音。
2. 固定四组双音音高(60,67)、(48,69)、(55,72)、(62,73)，每组分别低音强、
   高音强、等强；原子声音为明确记录的谐波合成信号。强弱幅比4:1，保留每个
   分量的独立RMS、同一混音内两个音的查询。program 0和40分别作为假设。
   检验交换强弱后排序是否翻转；纯合成控制不能替代真实乐器验收。
3. 保存未取整期望velocity和127类logits，保留全部program/增益情况，不择优。
   推理前后核验输入文件与PCM身份，所有结果留在新的ignored目录。

模型不是逐音存在概率；静音也有softmax输出。作者README说明伪教师是波形
能量rank，合成SoundFont训练使用固定增益，因此MIDI力度、录音响度、听觉显著性
和主旋律身份不能等同。通过上述控制也只获得继续评估的资格，不证明适合直接
控制音符取舍或LN长度。若失败，记录反例并保留现有应用行为。

来源：作者仓库`https://github.com/anime-song/tsumugi`的
`instrument_agnostic_amt/velocity/{README.md,modeling/model.py,cli/infer_velocity.py}`；
权重`https://huggingface.co/anime-song/instrument_agnostic_amt`。
