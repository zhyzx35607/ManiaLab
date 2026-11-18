# osu!mania 官方解码对照

这个独立工具使用 ppy 发布的 `ppy.osu.Game.Rulesets.Mania` 包。固定版本 `2026.1026.0`
对应 ppy/osu 提交 `b94ee4d835d65683d027ae8f47a3e61fa4899ad4`。输入为 ManiaLab
导出的原生 mania `.osz`，音符起止使用整数毫秒。工具记录官方解码器与规则集的程序集版本。

在仓库根目录的 PowerShell 中运行。需要 .NET 10 SDK 和 Python 3.11 以上。
将 `$package` 指向本地谱包。一次可以向解码器提供多份谱包。两份 JSON 均需使用新文件名。

```powershell
$project = 'scripts/verify_official_decoder/OfficialDecoder.csproj'
$work = 'test-data/official-decoder-local'
$package = 'path/to/chart.osz'
dotnet build $project --configuration Release --artifacts-path "$work/build"
dotnet "$work/build/bin/OfficialDecoder/release/OfficialDecoder.dll" "$work/official.json" $package
python scripts/verify_official_decoder/compare.py "$work/official.json" "$work/comparison.json"
```

`Program.cs` 直接从 ZIP 读取各难度，调用 `LegacyBeatmapDecoder` 和 `ManiaBeatmapConverter`。
`compare.py` 逐谱验证键数、短键与长条数量、每个物件的列和起止、原声文件引用、红线和滚速。
报告使用 SHA-256 保存按起点、终点和列排序的物件摘要。玩家听感与手感评分见项目验收记录。
