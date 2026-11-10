# MuScriptor 公开 GGUF 的研究权重准备

这一步把固定的公开 small GGUF 还原成原作者 Python 代码可读取的 safetensors 和配置文件，不需要 Windows C++ 工具链。它保留 GGUF 内已有的 FP16/FP32 数值，不能恢复原始 FP32 精度。张量检查与神经网络实际载入、推理、采音验收是不同步骤。

## 固定来源

- 转换器：`DamRsn/muscriptor.cpp`，commit `8589219cc3b584bce970f95aec2ec4a04e53b280`。
- GGUF：`DamRsn/muscriptor-gguf`，revision `13a851294c5b5701966e850c7eba326f9da97e8d`，文件 `v1/muscriptor-small-f16.gguf`。
- 完整文件 209425152 bytes，SHA256 `925f55af65a20ebc4f8b45ceaf095a12b72493d436cb112623cd0041a1af23d4`。
- GGUF 来源公开且未 gated；原始 safetensors 没有下载。内嵌的原始 checkpoint hash/revision 只是保存的来源信息，不代表本次独立核验了原始文件。
- 转换器代码 MIT；权重 CC BY-NC 4.0。输出保留权重 LICENSE、NOTICE、原作者和转换作者署名及转换器版权声明。

本地研究使用 ignored `test-data/muscriptor-cpp-review-v1` 中已经下载并固定的资产。脚本本身不联网或安装依赖；它也不随公开仓库提交模型文件。

## 转换与验证

`scripts/prepare_muscriptor_gguf.py` 在读取数据前验证完整 GGUF、原转换器各依赖源码和许可文本的固定 SHA256。small 的配置是 `dim=768, num_heads=12, num_layers=14, card=1393`，必须包含全部 122 个张量。

名称映射来自作者未经改动的 `_GLOBAL_MAP` 和 `_LAYER_MAP`。只有 mel filterbank 需要逆转置；注意 GGUFReader 的 `tensor.data` 已经是 NumPy shape，其他权重不能再转置。QKV 没有重新排列或拼接。旧键 `emb.0.*`、`linears.0.*` 保留给原作者的兼容加载器，映射记录另列当前单流模型的键名。

每个张量都检查名称集合、shape、dtype 和 finite；缺失、额外、错形状、错精度、非有限值会失败。逆转换后调用原作者 `_translate` 正向再转换，逐张量要求原 GGUF 数据 SHA256 完全相同。安全格式写出后重新读取，验证全部张量和 safetensors 元数据。五个小型损坏控制检查上述拒绝路径，不运行模型。

```powershell
.venv/Scripts/python.exe -B scripts/prepare_muscriptor_gguf.py --prepare-only
.venv-audio/Scripts/python.exe -B scripts/prepare_muscriptor_gguf.py --output test-data/muscriptor-recovered-small-v1
```

第二条命令拒绝覆盖已有目录，包括部分失败产物。复跑使用新的输出目录。隔离数据依赖固定为 `gguf==0.19.0`、`safetensors==0.5.3`，放在研究资产的 `deps/`；输入资产中的原转换器模块必须与脚本固定的哈希一致。数据转换使用 `.venv-audio` 已有的 NumPy、PyYAML 等基础依赖。普通 `.venv` 可以执行 preflight，但本机缺少 GGUF 导入所需的 PyYAML，不能直接执行转换。

输出包括 safetensors、`config.json`、全部 GGUF 元数据、逐张量映射和校验值、往返与安全格式回读结果、损坏控制、许可文本和总报告。脚本显式检查没有导入 Torch 或 ONNX Runtime。只有最后产生 `report.json` 且 `completed=true` 才表示本步骤完成。

本机的 `test-data/muscriptor-recovered-small-v1` 已完成上述 122 张量的逐值往返与安全格式回读检查，五项损坏输入均被拒绝。恢复后的 safetensors SHA256 是 `2111582d9f3df37619dfbc8b4f10071b4621f143bfa3f664f0efe7b293a208e8`。此处的成功只指数据格式和数值保持，不包含下一节的模型载入或推理。

## 后续边界

当前原作者 clone 为 `8a1ef88672cc23c26619adaff37ec48d82c0b906`；C++ 工具自身引用的原作者版本是 `7f213afecf23bd6a1b8672aa223690ee9807cefb`。二者应分别记录，不能当成同一实现。

后续可用原作者 `TranscriptionModel.load_model(local_path, device='cpu', dtype='float32')`，旁边保留配置文件，执行真实 strict load 和加载后数值核查。CPU FP32 计算使用的是 GGUF 已量化数值，不会获得原始 FP32 权重精度。本准备脚本没有执行这个步骤，也没有生成音符。

实际采音研究还需要保存完整 token、分块边界、原始 note start/end 与音尾来源，再与冻结输入的参考比较。不得将原作者的重弹截尾、缺 tie 边界封口或末尾 10ms 回退解释为独立的 LN 放手预测。数据转换成功不能说明采音质量或 LN 已达标。
