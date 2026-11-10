# Banquet 乐器查询分离的固定小对照

先记录输入与处理，再运行模型。目的：检查“其他”乐器的细分波形是否有可用方向；
这不是逐音转录，也不声称已经改善Nova声轨或完成自动乐器规划。

作者源码 `https://github.com/kwatcharasupat/query-bandit` 固定
`79ed5bb75e5c3a40cd319d9d990cee913fc65c26`。推荐权重 `ev-pre-aug.ckpt`
来自作者Zenodo13694558，645470187字节，SHA256
`657295888781e62ef50593002720d2edb3858b9e5bbfabf0c54f715a0da4b9e2`；下载后
另与作者MD5校验相等。代码MIT，权重CC BY-NC-SA4.0；本轮仅本地研究，不打包。

复用作者`PasstFiLMConditionedBandit`和`bandit-query.yml`推理配置。checkpoint
包含PaSST查询网络；构造时只关闭重复权重下载，随后对全部参数严格载入并逐
tensor比对。使用安全张量读取，不恢复训练优化器、不下载其他模型。已有CPU
Torch环境加隔离依赖，不修改主应用环境。两计算线程、一跨算子线程、串行。

固定URMP比较扩展manifest中按原顺序每种乐器首次出现的作品，得到8组：
12小提琴、12大提琴、13中提琴、25萨克斯、31小号、31圆号、31长号、38低音提琴。
一曲有多个同类乐器时合并为该类目标，不假设模型可以区分两个小提琴演奏者。
不根据结果改选曲子。

- 每曲所有独立原声轨除以声轨数再求和，构成已知混音；不单独归一化。
- 查询为目标类原声轨0–10秒；测试为20–26秒，样本时间不重叠，但乐句可能重复。查询是已知独立声轨，
  属于理想条件，不等于软件会自动从混音找到示例。
- 输入来自既有16kHz、mono、int16版本，以`/32768`转浮点，经作者使用的
  torchaudio重采样到44.1kHz，然后复制为双声道。高频与空间信息不会因此恢复。
- 对每组串行运行混音、仅目标、去掉目标、静音四种输入，共32次。查询固定。
- 使用作者网络一次6秒前向；分别计整窗和中央2–4秒的波形分数。未复现作者
  CLI的6秒窗／0.5秒hop及反射补齐，不能把单窗结果当作全曲CLI成绩。
- 保留输入浮点波形与每个输出、输入哈希、全部源文件哈希、实际依赖及耗时。
  不读转录标注，不吸拍，不按输出质量删例。

有目标时比较输出与已知目标的SNR、去均值SI-SDR、增益和MAE，同时记录混音
基线；去掉目标和静音时只报告残留RMS、峰值等，零参考的SNR/SI-SDR不定义。
“仅目标”检查提取过程有没有损伤本来已分好的乐器。完整8组才汇总。

```powershell
.venv/Scripts/python.exe scripts/check_banquet_separation.py --prepare-only
.venv-audio/Scripts/python.exe scripts/check_banquet_separation.py --output test-data/banquet-separation-v1
```

该实验不检验自动查询发现、真实Nova串音、原版立体声混音、逐音音尾或最终LN。
波形分源有益也需要后续同一音频的采音对照，不能直接据分离分数替换生成流程。

## 基线完成后的唯一电平诊断

完整8组基线完成后观察到多个目标被严重削弱，才决定新增本段诊断。作者训练
按目标dBFS优先抽片（>-36，失败放宽到>-48），而原始测试信号偏安静；作者
BYOQ入口没有归一化，不能声称这是漏掉的上游步骤。

预先固定全8组、每组原来的四种输入，唯一增益为
`g=.95/max(abs(original_mixture).max(),abs(original_query).max())`。
不读参考音符、不用独立目标、缺席输入或输出质量决定g；同组四种输入与查询
全用同一个g。网络输出除以g，还原原始量纲后使用原分数。保存放大前后RMS、
峰值、原始网络输出及还原结果，全部32次完成后再比较，不能逐例取较好值。

这是事后提出、统一执行的一次联合电平敏感性诊断，无法区分query和mixture
各自的作用；峰值归一也不保证进入训练RMS范围。不会据它自动修改应用默认。

```powershell
.venv-audio/Scripts/python.exe scripts/check_banquet_input_level.py --source test-data/banquet-separation-v1 --output test-data/banquet-input-level-v1
```
