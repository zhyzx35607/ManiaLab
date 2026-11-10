# Mega 53 作者长窗拼接复核

2026-11-10，在新声学推理前固定。旧6秒单窗结果全部保留；本轮只检验较长
上下文和作者拼接方法是否改变同一批弦管乐来源的表现。没有据结果挑乐器、
改起音阈值或过滤预测标签，不改应用默认。

仍为8类、5首作品，混音、仅目标、目标缺席、全零四种控制。作者源码、权重、
输出头和CPU线程设置与[上一实验](mega53-separation-protocol.md)相同。
从同一原16kHz单声道声轨取物理10–30秒，按原声轨数量的共同增益相加，
升采样44.1kHz并复制双声道。所有输入至少有完整30秒。

调用固定作者源码的 `utils/model_utils.py::demix`，采用配置中的882000点
（20秒）窗口、2倍重叠、10秒步长、2秒淡入淡出、batch1。20秒输入等于
两倍border，不触发两端反射扩展；第二窗只有10秒，依原条件补零到20秒。
因此每份独立PCM执行两次20秒网络前向，不能称整曲推理。关闭仅CUDA生效
的autocast，运行保持CPU float32，不做输入归一化。

相同浮点PCM可以共享前向，只同时请求事先固定的独立输出头；每种用途保留
对应头和输出，不能相加重叠类别。原函数不改算法，用包装器验证网络输出
有限值，并在原 `nan_to_num` 之前断言拼接结果有限。另以恒等及反相信号
验证6秒、20秒、22秒输入的边界与反射拼接；这些是适配检查，不是采音分数。

每份输出保留完整20秒及物理20–26秒截取，后者继续与旧实验完全相同的
6秒原输入比较。较长音源升采样在6秒截取边缘可能不同，逐项记录最大差和
变动样本数，要求中央物理22–24秒输入逐样点完全一致。不能把长上下文
说成网络输入完全不变，也不能用SI-SDR代替起音和音尾验证。

分离完成后再用原 `check_mega53_transcription.py` 对相同6秒原输入和新6秒
分离输出做完整采音，沿用Tsumugi other_v1_5、中央2秒、50ms起音容差、
精确半音及既有音尾容差。原脚本在全部推理完成后才读取标签。所有缺席控制、
仅目标控制、两个声部完整覆盖的并集比较必须保留。最终结论同时报告旧6秒
与新长窗，不能只报更好的部分。

下面是原始运行命令，需要已有上述本地冻结资产和隔离研究环境。脚本拒绝
覆盖已有输出；复跑时给分离及转录各自指定新目录，并保持路径对应。

```powershell
.venv/Scripts/python.exe -B scripts/check_mega53_long_context.py --prepare-only
.venv-audio/Scripts/python.exe -B scripts/check_mega53_long_context.py --output test-data/mega53-long-context-v1
.venv/Scripts/python.exe -B scripts/check_mega53_transcription.py --separation test-data/mega53-long-context-v1 --output test-data/mega53-long-transcription-v1
```

作者工具的附加依赖ml-collections1.1.0与absl-py2.3.1装在ignored研究目录，
不修改应用依赖。没有使用用户音源、上传音频、重新训练或评估定轨/LN音乐性。
