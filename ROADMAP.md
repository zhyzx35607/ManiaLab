# ManiaLab 2.0 路线图（超级缝合怪架构）

目标：在纯 CPU 环境下，打造真正“导出进 osu! 打起来像人类好谱”的 6–10K（核心聚焦 10K / 10K LN）自动生谱引擎。

## 三层架构分工
1. **听觉与分轨层（保留并精简）**
   - 开源多声部分离（HTDemucs 四/六声部，兼容外部 Nova 等分轨导入）+ Beat This! / 鼓轨 Timing + 双时钟（22.5ms 可调基准）与对音试听。
2. **写谱大脑层（双轨实打 PK，胜者留任）**
   - **A 选手**：分轨重混输入（可按需压低人声/突出主乐器）+ `Mapperatorinator v32-mini`（快速解码）生成高乐感底谱 + 分轨能量包络校验 LN 尾部。
   - **B 选手**：抛弃 Basic Pitch + MILP 硬凑；利用 Sayobot 6–10K Ranked/Loved 谱面与配对音频，训练纯 CPU 秒级推理的「节拍网格多声部→节奏/和弦/LN」轻量模型。
3. **双手与键型层（真·Torosu 核心）**
   - 直接引入 `D:\project\Torosu` 的键型转换与模式识别核心（Wide Hands 10K 独立双手、motif/trill/chordjack 保持、125ms 防 minijack、无 pill、智能扩缩和弦、真实难度与键型分析），作为所有生谱与转键的统一排键引擎。

## 阶段计划
- **Phase 0（工单 #01）**：大换血与彻底止血——归档旧研究、清 60+ 残留 worktree 与 40+ GB 构建废料、删死代码与废脚本、落地新版 `AGENTS.md`。
- **Phase 1（工单 #02）**：接入真·Torosu 键型引擎 + 一键多版本 `.osz` 实打对比导出工具。
- **Phase 2A（工单 #03）**：A 选手实测——`Mapperatorinator v32-mini` 纯 CPU 测速 + 分轨重混控声部 + Torosu 10K 端到端出谱。
- **Phase 2B（工单 #04）**：B 选手实测——精选 6–10K Ranked 音频配对数据管线 + 轻量节拍网格写谱模型原型 + 与 A 选手在同批真实歌曲上实打 PK。
- **Phase 3（待实打 PK 后定稿）**：收敛胜出主脑、精修 LN/段落起伏、大刀阔斧精简 UI 并补齐谱面可视化预览。
