using System.Globalization;
using System.IO.Compression;
using System.Reflection;
using System.Security.Cryptography;
using Encoding = System.Text.Encoding;
using System.Text.Json;
using osu.Game.Beatmaps;
using osu.Game.Beatmaps.Formats;
using osu.Game.Beatmaps.Legacy;
using osu.Game.IO;
using osu.Game.Rulesets;
using osu.Game.Rulesets.Mania;
using osu.Game.Rulesets.Mania.Beatmaps;
using osu.Game.Rulesets.Mania.Objects;

if (args.Length < 2)
    throw new ArgumentException("Usage: OfficialDecoder output.json chart.osz [chart.osz ...]");

CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
var ruleset = new ManiaRuleset();
using var rulesets = new AssemblyRulesetStore();
Decoder.RegisterDependencies(rulesets);
var results = new List<object>();
foreach (string archivePath in args.Skip(1))
{
    using var bundle = ZipFile.OpenRead(archivePath);
    var charts = bundle.Entries.Where(entry => entry.FullName.EndsWith(".osu", StringComparison.OrdinalIgnoreCase))
                               .OrderBy(entry => entry.FullName, StringComparer.Ordinal).ToArray();
    if (charts.Length == 0)
        throw new InvalidDataException($"No .osu difficulty in {archivePath}");
    var difficulties = new List<object>();
    foreach (var entry in charts)
    {
        using var stream = entry.Open();
        using var reader = new LineBufferedReader(stream);
        var decoder = Decoder.GetDecoder<Beatmap>(reader);
        if (decoder is not LegacyBeatmapDecoder)
            throw new InvalidDataException($"Expected a legacy beatmap in {entry.FullName}");
        var raw = decoder.Decode(reader);
        var converter = new ManiaBeatmapConverter(raw, ruleset);
        if (raw.BeatmapInfo.Ruleset.OnlineID != 3 || !converter.IsForCurrentRuleset || !converter.CanConvert())
            throw new InvalidDataException($"Expected a native mania difficulty in {entry.FullName}");
        var map = (ManiaBeatmap)converter.Convert();
        var objects = map.HitObjects.Select(note => (
            start: Milliseconds(note.StartTime),
            end: Milliseconds(note is HoldNote hold ? hold.EndTime : note.StartTime),
            lane: note.Column,
            type: note is HoldNote ? "ln" : note is Note ? "tap" : throw new InvalidDataException("Unexpected mania object")
        )).OrderBy(note => note.start).ThenBy(note => note.end).ThenBy(note => note.lane).ToArray();
        if (objects.Any(note => note.lane < 0 || note.lane >= map.TotalColumns || note.end < note.start))
            throw new InvalidDataException($"Invalid lane or times in {entry.FullName}");
        var legacy = (LegacyControlPointInfo)map.ControlPointInfo;
        difficulties.Add(new {
            entry = entry.FullName, mode = raw.BeatmapInfo.Ruleset.ShortName, keys = map.TotalColumns,
            version = raw.BeatmapInfo.DifficultyName, audio = raw.BeatmapInfo.Metadata.AudioFile,
            objects = objects.Length, holds = objects.Count(note => note.type == "ln"),
            rows_sha256 = Digest(objects.Select(n => $"{n.start},{n.end},{n.lane},{n.type}\n")),
            endpoints_sha256 = Digest(objects.Select(n => $"{n.start},{n.end},{n.type}\n")),
            timing = legacy.TimingPoints.Select(p => new { time = p.Time, period = p.BeatLength, meter = p.TimeSignature.Numerator }),
            velocities = legacy.DifficultyPoints.Select(p => new { time = p.Time, value = p.SliderVelocity })
        });
    }
    using var input = File.OpenRead(archivePath);
    results.Add(new { path = Path.GetFullPath(archivePath), sha256 = Convert.ToHexStringLower(SHA256.HashData(input)), difficulties });
}
var output = new {
    game = AssemblyVersion(typeof(LegacyBeatmapDecoder).Assembly),
    ruleset = AssemblyVersion(typeof(ManiaRuleset).Assembly),
    archives = results
};
using var report = new FileStream(args[0], FileMode.CreateNew);
JsonSerializer.Serialize(report, output, new JsonSerializerOptions { WriteIndented = true });
Console.WriteLine($"{results.Count} packages decoded with {ruleset.Description}.");

static int Milliseconds(double value) => value >= 0 && value == Math.Truncate(value)
    ? checked((int)value) : throw new InvalidDataException($"Expected integer chart milliseconds, got {value}");

static string Digest(IEnumerable<string> rows)
{
    using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
    foreach (string row in rows)
        hash.AppendData(Encoding.ASCII.GetBytes(row));
    return Convert.ToHexStringLower(hash.GetHashAndReset());
}

static string AssemblyVersion(Assembly assembly) =>
    assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()!.InformationalVersion;
