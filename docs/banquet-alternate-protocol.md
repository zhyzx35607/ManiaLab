# Banquet 作者另一权重的固定对照

这是在推荐权重和电平诊断之后提出的新候选实验，不覆盖原结果，也不按每例选择
最好的模型。触发线索是作者仓库issue6：2025-05-19一位用户报告推荐权重
`ev-pre-aug.ckpt`在管弦乐上几乎无输出，称`ev-pre-fz-bal.ckpt`可工作。
截至本次获取，评论为0，没有作者确认。该反馈只提供待验证方向，不证明权重
损坏、问题原因或本机能复现改善。

来源 `https://github.com/kwatcharasupat/query-bandit/issues/6`。完整API响应及
评论响应保存在本地`test-data/banquet-model/issue6*.json`。

另一权重仍取自作者同一Zenodo记录13694558，文件`ev-pre-fz-bal.ckpt`，
551785899字节，作者MD5为`94691fb1c477ac811b5ab57ed5edcfe5`。下载后检查大小
和MD5，固定SHA256为
`527ba0b44f0e58abd805e246ebb3dac21d80d8741b75da4c4f4cf6c0853bc7f8`，
运行入口再次强制核验。源码、模型结构、PaSST、依赖版本和每个6秒前向
均与原实验相同；完整严格载入且逐tensor比对，不读取优化器或额外预训练路径。
运行仍CPU两计算线程、一跨算子线程，前一组神经推理退出后才开始。

直接复用原8组`inputs.npz`，不重新挑选、重采样或归一化。query仍是0–10秒，
测试20–26秒；每组混音、仅目标、目标缺席、静音四种输入，共32次，原始电平。
不采用上一轮peak增益。不按曲子换模型、不改query，也不读取转录标注。

输入文件、原始结果和原runner/adapter均先核验SHA。新入口只在本进程临时将
原loader的checkpoint路径与身份验证结果设为已核验的另一权重；原loader文件
与原实验文件保持不变。报告保留新权重、variant入口、原adapter及所有输出SHA。
文件名中的`fz`不被当作已验证的推理结构或因果解释，实际结构以严格载入为准。

使用同一整窗和中央窗SNR、去均值SI-SDR、gain及残留指标；32次全部完成后
再给比较，不以一个示例宣布有用。原[基线协议](banquet-separation-protocol.md)
中的mono低采样率、理想独立query、人工混音、单窗非OLA和非盲测限制全部保留。
这里仍没有逐音转录或LN音乐验收。

```powershell
.venv/Scripts/python.exe scripts/check_banquet_alternate.py --source test-data/banquet-separation-v1 --prepare-only
.venv-audio/Scripts/python.exe scripts/check_banquet_alternate.py --source test-data/banquet-separation-v1 --output test-data/banquet-alternate-v1
```
