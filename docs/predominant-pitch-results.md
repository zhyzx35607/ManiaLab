# 突出旋律算法的实际运行与限制

从作者来源取得的Essentia.js和libfmp均已在本机Windows CPU跑过真实计算。
它们提供独立的音高轮廓证据，但当前结果不足以直接控制主次、删除伴奏或
产生LN放手。可复用WASM入口为`scripts/research_predominant_pitch.cjs`，
仅用于研究；没有给用户发行包新增Node、Docker或WSL依赖。

## Essentia.js / Melodia

原生Essentia官方安装文档说明Windows Python绑定未支持；此次PyPI发行清单
也未找到Windows wheel。因此使用作者的Essentia.js 0.1.3 npm包，gitHead
`f46c91c08bdf263d5f3d575ab8fb0f9b81695acf`，包SHA256
`161c53527a45b5de6ca21199084f7d92d6a2a5551cbd031e23ae8e461686bfdf`。
本机Node22.13.0，WASM自报Essentia2.1-beta6-dev。该包构建脚本没有锁住原生
Essentia的唯一commit；不能把今天下载的原生源码称为这个二进制的精确源码。

研究入口在加载前核验两份UMD文件SHA256及原PCM身份，接收44100Hz、mono、
小端float32。显式先做EqualLoudness，保留原幅度、不另行归一化。固定frame2048、
hop128，保存全部pitch与pitchConfidence。结果没有完整复音、乐器身份或音符
起止，也不是校准过的置信概率。代码与完整参数保留在入口中。

四个双正弦控制从0–4秒持续，在2秒交换.12/.03幅度。事先固定的稳定区间
为0.2–1.8和2.2–3.8秒：五度和四八度的第一段未检出；八度交换后仍跟踪
较弱的440Hz；半音两段均跟踪输入幅度较强者。另对原输出报告完整两半区间，
没有挑掉边缘帧。这些控制与velocity的四谐波信号不同，不合并成绩。
等响过滤会改变不同频率的相对能量，“幅度更大”也不直接等于感知主旋律。

原算法面对4秒全零PCM抛出`trying to calculate mean of empty array`。
新入口对精确全零PCM明确返回`exact_zero_pcm_skipped`、`algorithm_called=false`，
不伪造“算法返回了空轮廓”。其他失败仍记录为`algorithm_failed`并返回失败码。
没有给近静音音频调新的主观阈值。

已用20秒既有URMP小提琴片段真实运行。新入口的6892帧pitch/confidence与独立
原调用逐值完全一致，此次约8.61秒。它验证运行和接入一致性，不是主旋律准确率。
输出最后一帧中心在20.00109秒，属于右侧补帧；原数组保留并另计补帧数，
后续关联音符只能使用中心时间小于输入时长的帧，不能据补帧拉长LN。
拒绝覆盖已有输出、输入SHA不符、非有限PCM及非法字节长度的检查均通过。

运行时没有纳入发行包。作者包标AGPL-3.0，原生源码头标AGPL-3.0-or-later，
且另有第三方组件归属；这些文件不能被标成ManiaLab的MIT代码。本次只在
本地使用固定原包，没有替换为未经核验的第三方二进制。

作者来源：
`https://github.com/MTG/essentia.js`、`https://essentia.upf.edu/installing.html`、
`https://essentia.upf.edu/reference/std_PredominantPitchMelodia.html`、
`https://essentia.upf.edu/licensing_information.html`。

## libfmp

作者仓库`https://github.com/meinardmueller/libfmp`，固定源码
`ed057ba7f884e0fb267809707b2657ca034359a6`。谐波显著度和动态规划的原始
计算函数在现有隔离依赖中实际运行，保留运算，只移除可选jit装饰器并绕开
包级绘图import；未安装完整包。该源码与许可为MIT，已保存版权文本。

22050Hz的一秒440Hz控制得到173帧、中位440Hz；全零PCM却同样得到173帧、
中位55Hz，虽然显著度矩阵全零。无约束DP没有无声状态，不能把轨迹当新采出的
音符。无JIT路径各耗约7.39、5.03秒，不作为整曲便携性能承诺。
目前仅是可进一步改进的音高显著度基础，未取代默认采音，也未测试真实音乐。

## 记录

- 下载、原来源清单、控制、许可文本、独立复核：
  `test-data/section-salience-model-survey-2026-11-10/`。
- 可复用入口验证：`test-data/predominant-pitch-adapter-v1/verification.json`，SHA256
  `7f045509a0e597c3878226546c6c50f4270a55cccaea433b59ac6bad7c1e12ee`。
- 对照的真实片段原报告SHA256：
  `294998b177b15750d3330957bad75eafaa0d9449f85423634215effd273d5c4a`。
