# 26类鼓起音模型的独立声轨检查

2026-11-09，推理前记录。来源为作者项目`pier-maker92/ADT_STR`，代码固定`77dbef225e7029478ebfb916f20c7a00274f0f12`；官方权重仓库`Pierfrancesco/adt-str`固定`a33c5c6b191a4ca1e0f6dc22140947485eb36ce8`，使用文档默认setting-tau-0.8，不按《樱花草》评分挑选三个变体。来源许可证CC-BY-SA-4.0，代码与权重留在本地研究目录，未打包分发。

只输入已核验的作者Nova鼓轨。运行原模型、原贪心sample和原tokenizer，严格加载全部safetensors权重。模型使用24kHz、2.56秒分块、10ms起音token；原decode为每个起音固定补0.1秒音长，故仅评价鼓起音和类别，绝不能当作真实音尾或LN。原始token另存，审查不完整事件、非法类别与越块时间；若解码出现错配，先保留原样并报告。

遵循官方推理预处理：PCM幅度、单声道、torchaudio重采样，末块补零，不额外峰值归一化。批大小2以适应本机CPU，max_length512与配置相同。记录PyTorch/TorchAudio/Transformers版本，与项目要求的差异明示。

全曲先生成，再读Reference评分。40ms容差，固定22.5ms制谱提前量，原始物理时间另外保留；不读作者Timing、书签、列号做推理。kick={35,36}，严格snare={38,40}、扩展snare另加side-stick37及clap39，hi-hat={42,43,44}均为项目自定义音色编号，不套用标准GM编号。第4列以hi-hat为主但含其他鼓，附加hi-hat+tom及全部非kick/snare的诊断分数，不用其中最高者冒称hi-hat准确率。

与全曲参考起音及三类作者列比较，并报告各类别数量、重复事件、漏检与新增事件。Reference是作者取舍而非完整鼓标注，额外预测可能是未选音。此曲是已使用的开发基准；结果不能证明跨歌曲采音或整体成品质量。
