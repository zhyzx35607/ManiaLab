# 6–10K LoRA 训练操作卡

## 人工审核先行

三层语料位于 `workspace_exp/lora_6_10k_kit/`，按歌曲、音频哈希和谱包分组后做 85%/15% Train/Val 划分。Tier 1 为本机 `Creator=zhyzx35607` 的谱，Tier 2 为 Sayobot 中文谱，Tier 3 为 Ranked/Loved 泛化谱。

先打开 `workspace_exp/lora_6_10k_kit/zhyzx35607_review.md` 和 `.csv`，逐行在 `ReviewDecision` 填 `KEEP` 或 `DROP`。自动 `REVIEW_INVERSE` 只是提示，必须人工判定；空白决定不会进入正式语料。重新构造正式集：

```powershell
$kit = "D:\project\ManiaLab\workspace_exp\lora_6_10k_kit"
$env:TMP = "$kit\runtime\tmp"
$env:TEMP = $env:TMP
$env:TMPDIR = $env:TMP
$env:HF_HOME = "$kit\runtime\cache\huggingface"
$env:TORCH_HOME = "$kit\runtime\cache\torch"
$env:PIP_CACHE_DIR = "$kit\runtime\cache\pip"
python -m training.prepare_10k_dataset `
  --local-songs-dir "C:\Users\31270\AppData\Local\osu!\Songs" `
  --review-csv "$kit\zhyzx35607_review.csv"
```

正式训练器只接受 `review_status=human_reviewed`。未审核清单生成的 `dataset_provisional/` 仅允许最多两步 CPU smoke，不得用于长训。语料、缓存、断点、试打包都留在被忽略的 `workspace_exp/lora_6_10k_kit/`。

## 本机 CPU

`training/run_train_cpu_overnight.ps1` 复用已有 Python runtime，不安装工具；缺依赖、找不到阶段 #06 Mapperatorinator source / Torosu CLI、C 盘低于 5 GiB 或 D 盘低于 4 GiB 时会在训练前停止。HF、Torch、pip、Cargo 和临时缓存均固定在 D 盘。

默认模型缓存位于 kit 的 `runtime/cache/huggingface/manialab/`；如已有经校验的 V32 权重，可在启动前设置 `MANIALAB_V32_MODEL_CACHE` 指向该 D 盘目录，runner 会复用而不重复下载。

审核前只验证 provisional 全链路时：

```powershell
powershell -ExecutionPolicy Bypass -File .\training\run_train_cpu_overnight.ps1 -Smoke
```

该命令顺次执行冷启动 1 步和 `--resume-latest` 续至 2 步，限定 CPU、单 worker、20 秒验证片段。真正长训必须先逐行完成人工审核并重建 `dataset/`：
需要验证另一个以 `dataset_provisional` 开头的临时数据目录时，可通过 `-SmokeDatasetName dataset_provisional_<name>` 指定；该参数不影响正式训练路径。

```powershell
powershell -ExecutionPolicy Bypass -File .\training\run_train_cpu_overnight.ps1
powershell -ExecutionPolicy Bypass -File .\training\run_train_cpu_overnight.ps1 -Resume
```

默认 `v32-mini`、梯度累积 8、每 25 步 checkpoint、最多保留 8 个断点；Ctrl+C 在完整优化器更新边界原子保存后退出。用 `-Steps 100` 可设置绝对 optimizer-step 上限。首次训练如缺少固定版本模型权重，会将其下载到 D 盘的 kit 模型缓存。

每轮验证固定抽取 10K LN 与 Hybrid/Rice 歌曲生成阶段 `.osz`，输出到 `workspace_exp/lora_6_10k_kit/progress_osz/`；`training/validation-metrics.jsonl`、`training/metrics-curve.csv` 和 `training/training-report.json` 记录 NLL、LN 比例偏差、peak_held、minijack、pill、外侧小拇指轨均衡、耗时与进程峰值 RSS。断点包含 epoch 批次游标、消费的 Beatmap ID、采样分布和数据集 manifest 哈希；换数据集不会续接旧断点。

## RTX 4050

在已完成人工审核的数据副本上运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\training\run_train_4050.ps1
powershell -ExecutionPolicy Bypass -File .\training\run_train_4050.ps1 -Resume
```

启动器保留 CUDA/bf16、micro-batch 1、梯度累积 16 与 8-bit AdamW 可用时的加速路径，并在安装依赖前拒绝未审核语料。Tier 权重为 `1:2:3 = 4:2:1`，风格桶权重为 LN `1.2`、Hybrid `1.0`、Rice `0.7`；两维权重用于确定性 beatmap 重采样，报告也记录实际消费的层级、键数和风格分布。
