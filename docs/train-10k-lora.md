# 10K/LN LoRA 微调

## 4050 一键开训

1. 把 ManiaLab 项目目录和 `workspace_exp/lora_10k_kit/dataset/` 一起复制到 RTX 4050 电脑。若项目副本没有 `engines/torosu_cli/torosu-cli.exe`，请一并复制该文件，或确保目标机可用 Rust `cargo`。
2. 在项目根目录打开 PowerShell。
3. 执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\training\run_train_4050.ps1
```

启动器建立独立 Python 3.10 环境，安装 CUDA 12.8 版 PyTorch 与锁定的 Mapperatorinator 依赖，然后默认使用 `v32-mini` 开训。Python、模型、HF、Torch、pip、临时文件与训练输出都放在 `workspace_exp/lora_10k_kit/runtime/` 和 `workspace_exp/lora_10k_kit/training/`；该套件不依赖固定盘符。

默认配置为 micro-batch 1、梯度累积 16、bf16、gradient checkpointing、LoRA `r=16 / alpha=32`；若目标环境支持 bitsandbytes，则使用 8-bit AdamW，否则使用标准 AdamW。训练样本按真实 LN 桶加权：10K LN `4x`、Hybrid `2x`、Rice `0.25x`、辅助 LN `1x`，真实 `hold_note_ratio` 与 `difficulty` 不改写。每轮记录验证 NLL、hold 起点/放手准确率，并对最多两首验证集 10K 歌自由生成、调用 Torosu `analyze` 统计 LN 偏差、`peak_held`、minijack、pill 与外侧轨均衡度。

## 常用参数

```powershell
# 从最近的 checkpoint 续训
powershell -ExecutionPolicy Bypass -File .\training\run_train_4050.ps1 -Resume

# 改用完整版 V32
powershell -ExecutionPolicy Bypass -File .\training\run_train_4050.ps1 -Model v32

# 覆盖 Torosu CLI 位置
powershell -ExecutionPolicy Bypass -File .\training\run_train_4050.ps1 -TorosuCli C:\tools\torosu-cli.exe
```

LoRA 目录包含 `adapter_config.json`、`adapter_model.safetensors` 和 `lora-manifest.json`，可在 ManiaLab V32 生成设置中作为 `lora_path` 使用。具体最新目录以 `training-report.json` 的 `adapter_dir` 为准；训练断点、epoch 评测与训练报告都保存在被忽略的实验目录中。

## 本机 CPU 冒烟

CPU 仅用于短验证，不运行完整训练：

```powershell
python -m training.train_10k_lora --device cpu --num-workers 0 --max-steps 1 --epochs 1 --gradient-accumulation-steps 1 --checkpoint-every-steps 1 --validation-batches 1 --validation-samples 0
python -m training.train_10k_lora --device cpu --num-workers 0 --max-steps 2 --epochs 2 --gradient-accumulation-steps 1 --checkpoint-every-steps 1 --validation-batches 1 --validation-samples 1 --generation-window-seconds 20 --resume-latest
```

两条命令在第二步前冷启动模型并从第 1 步 checkpoint 恢复；脚本会对真实验证歌曲的 20 秒片段自由生成，全长生成留给支持 CUDA 的训练机。运行时间、峰值内存、实际设备和每轮指标记录在 `training-report.json` 与 `validation-metrics.jsonl`。
