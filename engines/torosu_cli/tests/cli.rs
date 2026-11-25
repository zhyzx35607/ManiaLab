use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

const MAP_7K: &str = "osu file format v14

[General]
AudioFilename: audio.ogg
Mode: 3

[Metadata]
Title: Native CLI fixture
Artist: Test
Creator: Test
Version: Original
BeatmapID: 42
BeatmapSetID: 24

[Difficulty]
HPDrainRate:5
CircleSize:7
OverallDifficulty:7
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
0,500,4,2,0,70,1,0

[HitObjects]
36,192,0,1,8,0:0:0:0:
109,192,250,128,0,1000:0:0:0:0:
183,192,500,1,0,0:0:0:0:
256,192,750,1,0,0:0:0:0:
329,192,1000,1,0,0:0:0:0:
402,192,1250,1,0,0:0:0:0:
475,192,1500,1,0,0:0:0:0:
";

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let index = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("torosu-cli-test-{}-{index}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_torosu-cli"))
        .args(args)
        .output()
        .unwrap()
}

fn map_for_keys(keys: usize) -> String {
    let columns = [0usize, 1, 2, 3, 4, 5, 6];
    let objects = [
        (0, 1, 8, "0:0:0:0:"),
        (250, 128, 0, "1000:0:0:0:0:"),
        (500, 1, 0, "0:0:0:0:"),
        (750, 1, 0, "0:0:0:0:"),
        (1000, 1, 0, "0:0:0:0:"),
        (1250, 1, 0, "0:0:0:0:"),
        (1500, 1, 0, "0:0:0:0:"),
    ];
    let hit_objects = columns
        .into_iter()
        .zip(objects)
        .map(|(lane, (time, kind, hitsound, tail))| {
            let lane = lane.min(keys - 1);
            let x = ((lane as f64 + 0.5) * 512.0 / keys as f64).floor() as usize;
            format!("{x},192,{time},{kind},{hitsound},{tail}")
        })
        .collect::<Vec<_>>()
        .join("\n");
    MAP_7K
        .replace("CircleSize:7", &format!("CircleSize:{keys}"))
        .replace(
            "36,192,0,1,8,0:0:0:0:\n109,192,250,128,0,1000:0:0:0:0:\n183,192,500,1,0,0:0:0:0:\n256,192,750,1,0,0:0:0:0:\n329,192,1000,1,0,0:0:0:0:\n402,192,1250,1,0,0:0:0:0:\n475,192,1500,1,0,0:0:0:0:",
            &hit_objects,
        )
}

#[test]
fn strict_convert_preserves_raw_object_order_metadata_and_timing() {
    let temp = TempDir::new();
    let input = temp.path("source.osu");
    let output = temp.path("nested/converted.osu");
    fs::write(&input, MAP_7K).unwrap();

    let result = run(&[
        "convert",
        "--input",
        input.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--keys",
        "10",
        "--density",
        "none",
        "--strict",
        "--keep-jacks",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let converted = fs::read_to_string(output).unwrap();
    let parsed = rosu_map::from_str::<rosu_map::Beatmap>(&converted).unwrap();
    let source = rosu_map::from_str::<rosu_map::Beatmap>(MAP_7K).unwrap();
    assert_eq!(parsed.circle_size, 10.0);
    assert_eq!(parsed.hit_objects.len(), source.hit_objects.len());
    assert_eq!(
        parsed.control_points.timing_points,
        source.control_points.timing_points
    );
    for (before, after) in source.hit_objects.iter().zip(&parsed.hit_objects) {
        assert_eq!(before.start_time, after.start_time);
        assert_eq!(
            matches!(
                before.kind,
                rosu_map::section::hit_objects::HitObjectKind::Hold(_)
            ),
            matches!(
                after.kind,
                rosu_map::section::hit_objects::HitObjectKind::Hold(_)
            )
        );
        assert_eq!(before.samples, after.samples);
    }
    let in_hit_objects = converted
        .split("[HitObjects]")
        .nth(1)
        .expect("converted chart has HitObjects");
    for row in in_hit_objects.lines().filter(|row| !row.trim().is_empty()) {
        row.split(',')
            .next()
            .expect("hit object has an x coordinate")
            .parse::<u16>()
            .expect("osu! x coordinates remain integer text for the Python bridge");
    }
    assert!(converted.contains("Title: Native CLI fixture"));
    assert!(converted.contains("Version: Original"));
    assert!(converted.contains("BeatmapSetID: 24"));
}

#[test]
fn strict_convert_covers_every_source_and_target_key_count() {
    use rosu_map::section::hit_objects::HitObjectKind;

    let temp = TempDir::new();
    for source_keys in 6..=10 {
        for target_keys in 6..=10 {
            let input = temp.path(&format!("source-{source_keys}.osu"));
            let output = temp.path(&format!("out-{source_keys}-{target_keys}.osu"));
            let source_text = map_for_keys(source_keys);
            fs::write(&input, &source_text).unwrap();
            let result = run(&[
                "convert",
                "--input",
                input.to_str().unwrap(),
                "--output",
                output.to_str().unwrap(),
                "--keys",
                &target_keys.to_string(),
                "--density",
                "none",
                "--strict",
            ]);
            assert!(
                result.status.success(),
                "{source_keys}K -> {target_keys}K: {}",
                String::from_utf8_lossy(&result.stderr)
            );
            let map = rosu_map::from_str::<rosu_map::Beatmap>(&fs::read_to_string(output).unwrap())
                .unwrap();
            let source = rosu_map::from_str::<rosu_map::Beatmap>(&source_text).unwrap();
            assert_eq!(map.circle_size, target_keys as f32);
            assert_eq!(map.hit_objects.len(), 7);
            assert_eq!(
                map.hit_objects
                    .iter()
                    .filter(|object| matches!(
                        object.kind,
                        rosu_map::section::hit_objects::HitObjectKind::Hold(_)
                    ))
                    .count(),
                1
            );
            assert!(source
                .hit_objects
                .iter()
                .zip(&map.hit_objects)
                .all(|(a, b)| {
                    a.start_time == b.start_time
                        && match (&a.kind, &b.kind) {
                            (HitObjectKind::Circle(_), HitObjectKind::Circle(_)) => true,
                            (HitObjectKind::Hold(a), HitObjectKind::Hold(b)) => {
                                a.duration == b.duration
                            }
                            _ => false,
                        }
                }));
        }
    }
}

#[test]
fn non_strict_convert_keeps_native_density_expansion_semantics() {
    let temp = TempDir::new();
    let input = temp.path("source.osu");
    let output = temp.path("expanded.osu");
    fs::write(&input, MAP_7K).unwrap();
    let result = run(&[
        "convert",
        "--input",
        input.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--keys",
        "10",
        "--density",
        "full",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let map =
        rosu_map::from_str::<rosu_map::Beatmap>(&fs::read_to_string(output).unwrap()).unwrap();
    assert_eq!(map.circle_size, 10.0);
    assert!(map.hit_objects.len() > 7);
}

#[test]
fn strict_density_expansion_preserves_raw_source_objects() {
    let temp = TempDir::new();
    let input = temp.path("source.osu");
    let output = temp.path("expanded.osu");
    fs::write(&input, MAP_7K).unwrap();

    let result = run(&[
        "convert",
        "--input",
        input.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--keys",
        "10",
        "--density",
        "full",
        "--strict",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let source = fs::read_to_string(input).unwrap();
    let converted = fs::read_to_string(output).unwrap();
    let source_rows = hit_object_rows(&source);
    let converted_rows = hit_object_rows(&converted);
    assert!(converted_rows.len() > source_rows.len());
    for source in source_rows {
        let fields = source.split_once(',').unwrap().1;
        assert!(
            converted_rows
                .iter()
                .any(|row| row.split_once(',').unwrap().1 == fields),
            "source object fields changed or were dropped: {source}"
        );
    }
    assert!(converted.contains("BeatmapSetID: 24"));
    assert_eq!(
        rosu_map::from_str::<rosu_map::Beatmap>(&converted)
            .unwrap()
            .circle_size,
        10.0
    );
}

#[test]
fn strict_density_expansion_handles_a_source_without_final_newline() {
    let temp = TempDir::new();
    let input = temp.path("source.osu");
    let output = temp.path("expanded.osu");
    let source = MAP_7K.trim_end().to_owned();
    fs::write(&input, &source).unwrap();

    let result = run(&[
        "convert",
        "--input",
        input.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--keys",
        "10",
        "--density",
        "full",
        "--strict",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let converted = fs::read_to_string(output).unwrap();
    assert!(hit_object_rows(&converted).len() > hit_object_rows(&source).len());
    rosu_map::from_str::<rosu_map::Beatmap>(&converted).unwrap();
}

fn hit_object_rows(osu: &str) -> Vec<&str> {
    let mut in_hit_objects = false;
    let mut rows = Vec::new();
    for line in osu.lines() {
        if line.starts_with('[') && line.ends_with(']') {
            in_hit_objects = line == "[HitObjects]";
            continue;
        }
        if in_hit_objects && !line.trim().is_empty() && !line.trim_start().starts_with("//") {
            rows.push(line);
        }
    }
    rows
}

#[test]
fn analyze_reports_pattern_shares_modifiers_and_hand_load() {
    let temp = TempDir::new();
    let input = temp.path("source.osu");
    fs::write(&input, MAP_7K).unwrap();
    let result = run(&["analyze", "--input", input.to_str().unwrap(), "--json"]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["keys"], 7);
    assert_eq!(value["objects"], 7);
    assert_eq!(value["ln"], 1);
    assert!(value["patterns"]
        .as_array()
        .unwrap()
        .iter()
        .all(|pattern| { pattern["ratio"].is_number() && pattern["percentage"].is_number() }));
    assert!(value["minijack_count"].is_number());
    assert!(value["pill_count"].is_number());
    assert!(value["hand_balance"].as_f64().unwrap() >= 0.0);
}

#[test]
fn ln_shape_keep_is_byte_identical_and_fln_keeps_object_times() {
    let temp = TempDir::new();
    let input = temp.path("source.osu");
    let keep = temp.path("keep.osu");
    let fln = temp.path("fln.osu");
    fs::write(&input, MAP_7K).unwrap();

    let kept = run(&[
        "ln-shape",
        "--input",
        input.to_str().unwrap(),
        "--output",
        keep.to_str().unwrap(),
        "--mode",
        "keep",
    ]);
    assert!(kept.status.success());
    assert_eq!(fs::read(&input).unwrap(), fs::read(keep).unwrap());

    let reshaped = run(&[
        "ln-shape",
        "--input",
        input.to_str().unwrap(),
        "--output",
        fln.to_str().unwrap(),
        "--mode",
        "fln",
    ]);
    assert!(
        reshaped.status.success(),
        "{}",
        String::from_utf8_lossy(&reshaped.stderr)
    );
    let source = rosu_map::from_str::<rosu_map::Beatmap>(MAP_7K).unwrap();
    let result =
        rosu_map::from_str::<rosu_map::Beatmap>(&fs::read_to_string(fln).unwrap()).unwrap();
    assert_eq!(source.hit_objects.len(), result.hit_objects.len());
    assert!(result.hit_objects.iter().any(|object| {
        matches!(
            object.kind,
            rosu_map::section::hit_objects::HitObjectKind::Hold(_)
        )
    }));
    assert!(source
        .hit_objects
        .iter()
        .zip(&result.hit_objects)
        .all(|(left, right)| left.start_time == right.start_time));
}
