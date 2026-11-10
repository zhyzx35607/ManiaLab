# 贝斯与其他伴奏专用采音

dev.3完整测试包包含Tsumugi的贝斯和其他伴奏模型及独立CPU运行程序，
不需要用户安装Python、Torch或Docker。它是默认关闭的可选项。

1. 导入原曲，选择四／六声部分离；已有Nova声轨也可直接导入。
2. 勾选“贝斯与其他伴奏专用采音（测试）”。只处理bass和other；钢琴、鼓、
   吉他和人声继续各自原有路径。
3. 默认只把起音交给编排。若要尝试这些声部的持续音，再勾选
   “允许模型预测音尾参与长条”。没有可追溯模型尾的事件仍只能做短键。
4. 分析后检查拍线、段落和音色分类。模型分类是预测，不等于独立分离出的乐器。
   可在逐段表达中分别设置持续音、连续起音、起音或不采。
5. 生成候选后用声轨／起止提示试听，再导出 `.osz` 试打。

开关允许模型音尾参与长条，不意味着每个长区间都必须变成LN。段落方案、
层数、密度和定轨容量还会决定是否保留。同一曲中改变开关会保存独立采音
缓存；同设置再次分析可复用经过身份和结果校验的缓存。

原浮点采样、声轨身份、模型与代码哈希都进入缓存键。取消或失败不会替换
已有分析。贝斯和other的模型事件权重为1，不能当作100%置信度；当前不对
这两类候选套用旧Basic Pitch的瞬态校准、尾修整或基于置信度的串音过滤。
两者仍经过原有局部声源能量检查，分离残留或误采仍可能存在。

起音也检查原始窗口来源：曲中没有起音预测的持续区间不再直接成为新按键。
原输入开头已经活动的区间单独保留并标记，避免删除真正的开头音；不会把
这个例外用于曲中窗口。原预测和排除原因均保留在证据目录。完整损益见
[起音来源复核](tsumugi-onset-provenance.md)。

同一固定短录音的两模型、两音尾模式，已通过便携EXE与研究输出逐音／逐窗
一致性检查，并验证冷运行与缓存复用。完整音乐评价另见[17份贝斯](bass-model-results.md)、
[35份弦管乐](tsumugi-urmp-results.md)、[原Nova全曲18份导出](tsumugi-fullsong-results.md)。
其中存在局部退步、量化大移动和主次方案不足，不视为完整音乐验收。

## 开发和重新打包

作者源码固定 `eba1c7d05e4d6a53cc4e8916d742d9623a3fdc38`，
模型库 `anime-song/instrument_agnostic_amt` 固定
`c6ee8a22626cd7185031ee71ce7cea27ec862781`。只准备
`best_model_bass_v2.pth`、`best_model_other_v1_5.pth`。
本仓库不提交模型二进制。来源、MIT归属和哈希见
`manialab/instrument_bundle.py`、`manialab/tsumugi-source.json`。

源码清单保存当前Windows CRLF检出的原始字节哈希；重新取得作者代码时
使用仓库本地 `core.autocrlf=true`，核验失败时先检查换行，不能跳过校验。
准备后的便携程序完全不依赖Git。

使用独立Python3.12.10 CPU环境，依赖版本见
`requirements-tsumugi-research.txt`。若依赖在单独目录，开发入口支持
`MANIALAB_INSTRUMENT_PYTHON` 和 `MANIALAB_INSTRUMENT_DEPS`。

```powershell
# 作者代码及两权重已经下载到对应目录后：
.venv/Scripts/python.exe scripts/prepare_instrument_bundle.py --source <作者源码目录>
.venv-audio/Scripts/python.exe scripts/build_instrument_worker.py --dependencies <依赖目录>
.venv/Scripts/python.exe scripts/verify_instrument_worker.py --fixtures <固定真实音频对照目录> --output <新验证目录>
.venv/Scripts/python.exe scripts/build_windows.py
```

验证所用原音频仅在本地研究目录，不随软件或源码发布。验证程序会真正运行
两个模型的两种音尾模式并与已冻结研究证据逐项比较，全部通过才写入
`assets/instruments/verification.json` 和完成标志。主程序打包会拒绝源码、
模型、EXE或验证证据不匹配的组件。缺少可选资产时通用构建仍可生成基础包；
此类包会将专用入口禁用，不是包含新组件的完整dev.3测试包。
