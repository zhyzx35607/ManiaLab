use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
    process,
};

use anyhow::{anyhow, bail, Context, Result};
use rosu_map::{
    section::{general::GameMode, hit_objects::HitObjectKind},
    Beatmap,
};
use serde_json::{json, Value};
use torosu_cli::commands::trainer::{
    analysis::{layout, patterns},
    mania,
    model::{
        Density, FlnConverterParams, KeyConverterParams, LengthUnit, NoteLength, ShortLnParams,
    },
    tasks::{fln_converter, key_converter, short_ln},
};

const BOOLEAN_FLAGS: &[&str] = &["--json", "--strict", "--keep-jacks", "--no-keep-jacks"];

fn main() {
    if let Err(error) = run(env::args().skip(1).collect()) {
        eprintln!("torosu-cli: {error:#}");
        process::exit(2);
    }
}

fn run(args: Vec<String>) -> Result<()> {
    let Some((command, rest)) = args.split_first() else {
        bail!("expected convert, analyze, or ln-shape; use --help for usage");
    };
    if command == "--help" || command == "-h" {
        print_help();
        return Ok(());
    }

    let options = parse_options(rest)?;
    match command.as_str() {
        "convert" => convert(&options),
        "analyze" => analyze(&options),
        "ln-shape" => shape_lns(&options),
        other => bail!("unknown command: {other}"),
    }
}

fn parse_options(args: &[String]) -> Result<HashMap<String, Option<String>>> {
    let mut options = HashMap::new();
    let mut index = 0;
    while index < args.len() {
        let flag = &args[index];
        if !flag.starts_with("--") {
            bail!("unexpected argument: {flag}");
        }
        if options.contains_key(flag) {
            bail!("duplicate option: {flag}");
        }
        let value = if BOOLEAN_FLAGS.contains(&flag.as_str()) {
            None
        } else {
            index += 1;
            Some(
                args.get(index)
                    .context(format!("missing value for {flag}"))?
                    .clone(),
            )
        };
        options.insert(flag.clone(), value);
        index += 1;
    }
    if options.contains_key("--keep-jacks") && options.contains_key("--no-keep-jacks") {
        bail!("--keep-jacks and --no-keep-jacks cannot be used together");
    }
    Ok(options)
}

fn required<'a>(options: &'a HashMap<String, Option<String>>, name: &str) -> Result<&'a str> {
    options
        .get(name)
        .and_then(Option::as_deref)
        .with_context(|| format!("missing required option {name}"))
}

fn optional<'a>(options: &'a HashMap<String, Option<String>>, name: &str) -> Option<&'a str> {
    options.get(name).and_then(Option::as_deref)
}

fn ensure_only(options: &HashMap<String, Option<String>>, allowed: &[&str]) -> Result<()> {
    for name in options.keys() {
        if !allowed.contains(&name.as_str()) {
            bail!("unknown option: {name}");
        }
    }
    Ok(())
}

fn parse_keys(value: &str) -> Result<usize> {
    let keys = value
        .parse::<usize>()
        .context("--keys must be an integer")?;
    if !(6..=10).contains(&keys) {
        bail!("--keys must be between 6 and 10");
    }
    Ok(keys)
}

fn parse_density(value: &str) -> Result<Density> {
    match value {
        "none" => Ok(Density::None),
        "dense" => Ok(Density::Dense),
        "more-dense" => Ok(Density::MoreDense),
        "full" => Ok(Density::Full),
        _ => bail!("--density must be none, dense, more-dense, or full"),
    }
}

fn read_beatmap(path: &Path) -> Result<(String, Beatmap)> {
    let text =
        fs::read_to_string(path).with_context(|| format!("could not read {}", path.display()))?;
    let parse_text = text.trim_start_matches('\u{feff}');
    let beatmap = rosu_map::from_str::<Beatmap>(parse_text)
        .map_err(|error| anyhow!("invalid .osu file: {error}"))?;
    if beatmap.mode != GameMode::Mania {
        bail!("only native osu!mania maps (Mode: 3) are supported");
    }
    Ok((text, beatmap))
}

fn validate_playstyle(keys: usize, playstyle: Option<&str>) -> Result<()> {
    if let Some(name) = playstyle {
        if !layout::layouts(keys)
            .iter()
            .any(|candidate| candidate.name == name)
        {
            bail!("unsupported {keys}K playstyle: {name}");
        }
    }
    Ok(())
}

fn convert(options: &HashMap<String, Option<String>>) -> Result<()> {
    ensure_only(
        options,
        &[
            "--input",
            "--output",
            "--keys",
            "--density",
            "--playstyle",
            "--keep-jacks",
            "--no-keep-jacks",
            "--strict",
        ],
    )?;
    let input_path = PathBuf::from(required(options, "--input")?);
    let output_path = PathBuf::from(required(options, "--output")?);
    let keys = parse_keys(required(options, "--keys")?)?;
    let density = parse_density(optional(options, "--density").unwrap_or("none"))?;
    let playstyle = optional(options, "--playstyle");
    validate_playstyle(keys, playstyle)?;
    let strict = options.contains_key("--strict");
    let keep_jacks = !options.contains_key("--no-keep-jacks");
    let (source_text, source_map) = read_beatmap(&input_path)?;
    let mut beatmap = rosu_map::from_str::<Beatmap>(source_text.trim_start_matches('\u{feff}'))
        .map_err(|error| anyhow!("invalid .osu file: {error}"))?;
    let params = KeyConverterParams {
        target_keys: keys as u8,
        density,
        keep_jacks,
        playstyle: playstyle.map(str::to_owned),
    };
    let extras = key_converter::apply_with(&mut beatmap, &params, None)
        .context("Torosu key conversion failed")?;

    let output_text = if strict {
        let encoded = beatmap
            .encode_to_string()
            .context("could not serialize converted map")?;
        let rows = hit_object_rows(&encoded);
        if rows.len() != extras.len() {
            bail!("Torosu object identity metadata does not match its output");
        }
        let extra_lines = rows
            .into_iter()
            .zip(&extras)
            .filter_map(|(line, is_extra)| (*is_extra).then_some(line))
            .collect::<Vec<_>>();
        let result = strict_output(
            &source_text,
            &source_map,
            &beatmap,
            keys,
            &extras,
            &extra_lines,
        )?;
        let (_, checked) = read_text_beatmap(&result)?;
        validate_lane_layout(&checked, keys)?;
        result
    } else {
        torosu_cli::osu_fixups::sanitize(
            beatmap
                .encode_to_string()
                .context("could not serialize converted map")?,
            None,
        )
    };
    if let Some(parent) = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("could not create {}", parent.display()))?;
    }
    fs::write(&output_path, output_text)
        .with_context(|| format!("could not write {}", output_path.display()))?;
    Ok(())
}

fn hit_object_rows(osu: &str) -> Vec<&str> {
    let mut in_hit_objects = false;
    let mut rows = Vec::new();
    for line in osu.lines() {
        if line.starts_with('[') && line.ends_with(']') {
            in_hit_objects = line.eq_ignore_ascii_case("[HitObjects]");
            continue;
        }
        if in_hit_objects && !line.trim().is_empty() && !line.trim_start().starts_with("//") {
            rows.push(line);
        }
    }
    rows
}

fn read_text_beatmap(text: &str) -> Result<(String, Beatmap)> {
    let beatmap = rosu_map::from_str::<Beatmap>(text.trim_start_matches('\u{feff}'))
        .map_err(|error| anyhow!("invalid generated .osu file: {error}"))?;
    if beatmap.mode != GameMode::Mania {
        bail!("generated map is not osu!mania");
    }
    Ok((text.to_owned(), beatmap))
}

fn strict_output(
    source_text: &str,
    source: &Beatmap,
    output: &Beatmap,
    keys: usize,
    extras: &[bool],
    extra_lines: &[&str],
) -> Result<String> {
    if extras.len() != output.hit_objects.len() {
        bail!("Torosu object identity metadata does not match its output");
    }
    let original_objects: Vec<_> = output
        .hit_objects
        .iter()
        .zip(extras)
        .filter_map(|(object, is_extra)| (!is_extra).then_some(object))
        .collect();
    if source.hit_objects.len() != original_objects.len() {
        bail!(
            "strict conversion dropped or duplicated source objects ({} -> {})",
            source.hit_objects.len(),
            original_objects.len()
        );
    }
    let mut source_order: Vec<usize> = (0..source.hit_objects.len()).collect();
    source_order.sort_by(|&a, &b| {
        source.hit_objects[a]
            .start_time
            .total_cmp(&source.hit_objects[b].start_time)
    });
    let mut x_by_source = vec![None; source.hit_objects.len()];
    for (&source_index, after) in source_order.iter().zip(original_objects) {
        let before = &source.hit_objects[source_index];
        if (before.start_time - after.start_time).abs() > 0.01 {
            bail!("strict conversion changed object ordering or start times");
        }
        let before_note = mania::note(before);
        let after_note = mania::note(after);
        match (before_note, after_note) {
            (Some((_, before_end)), Some((after_x, after_end))) => {
                if (before_end - after_end).abs() > 0.01
                    || matches!(before.kind, HitObjectKind::Hold(_))
                        != matches!(after.kind, HitObjectKind::Hold(_))
                {
                    bail!("strict conversion changed a note end time or LN type");
                }
                x_by_source[source_index] = Some(after_x);
            }
            (None, None) => {}
            _ => bail!("strict conversion changed an object type"),
        }
    }

    patch_raw_osu(source_text, &x_by_source, keys, extra_lines)
}

fn patch_raw_osu(
    source: &str,
    x_by_object: &[Option<f32>],
    keys: usize,
    extra_lines: &[&str],
) -> Result<String> {
    let mut output = String::with_capacity(source.len() + 32);
    let mut section = "";
    let mut object_index = 0usize;
    let mut circle_size_written = false;
    let mut hit_objects_seen = false;
    let mut extras_written = false;
    let line_ending = if source.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    };
    for raw_line in source.split_inclusive('\n') {
        let content = raw_line.trim_end_matches(['\r', '\n']);
        let newline = &raw_line[content.len()..];
        if content.starts_with('[') && content.ends_with(']') {
            if section.eq_ignore_ascii_case("HitObjects") && !extras_written {
                if !extra_lines.is_empty() && !output.ends_with('\n') {
                    output.push_str(line_ending);
                }
                for line in extra_lines {
                    output.push_str(line);
                    output.push_str(line_ending);
                }
                extras_written = true;
            }
            section = &content[1..content.len() - 1];
            hit_objects_seen |= section.eq_ignore_ascii_case("HitObjects");
            output.push_str(raw_line);
            continue;
        }
        if section.eq_ignore_ascii_case("Difficulty")
            && content
                .split_once(':')
                .is_some_and(|(name, _)| name.trim().eq_ignore_ascii_case("CircleSize"))
        {
            output.push_str(&format!("CircleSize:{keys}{newline}"));
            circle_size_written = true;
            continue;
        }
        if section.eq_ignore_ascii_case("HitObjects")
            && !content.trim().is_empty()
            && !content.trim_start().starts_with("//")
        {
            let x = x_by_object
                .get(object_index)
                .context("raw object count mismatch")?;
            if let Some(x) = x {
                let mut parts = content.split(',').map(str::to_owned).collect::<Vec<_>>();
                if parts.len() < 5 {
                    bail!("invalid HitObjects row at index {object_index}");
                }
                parts[0] = x.to_string();
                output.push_str(&parts.join(","));
                output.push_str(newline);
            } else {
                output.push_str(raw_line);
            }
            object_index += 1;
            continue;
        }
        output.push_str(raw_line);
    }
    if section.eq_ignore_ascii_case("HitObjects") && !extras_written {
        if !extra_lines.is_empty() && !output.ends_with('\n') {
            output.push_str(line_ending);
        }
        for line in extra_lines {
            output.push_str(line);
            output.push_str(line_ending);
        }
        extras_written = true;
    }
    if !circle_size_written {
        bail!("strict conversion requires a CircleSize field in [Difficulty]");
    }
    if !hit_objects_seen {
        bail!("strict conversion requires a [HitObjects] section");
    }
    if !extras_written && !extra_lines.is_empty() {
        bail!("strict conversion could not append native density notes");
    }
    if object_index != x_by_object.len() {
        bail!(
            "raw object count mismatch ({} chart rows, {} objects)",
            object_index,
            x_by_object.len()
        );
    }
    Ok(output)
}

fn validate_lane_layout(beatmap: &Beatmap, keys: usize) -> Result<()> {
    let mut last_start: Vec<Option<f64>> = vec![None; keys];
    let mut last_end: Vec<Option<f64>> = vec![None; keys];
    for object in &beatmap.hit_objects {
        let Some((x, end)) = mania::note(object) else {
            continue;
        };
        if !x.is_finite() || !(0.0..512.0).contains(&x) || !end.is_finite() {
            bail!("invalid note coordinate or LN end time");
        }
        let lane = mania::column(x, keys);
        if last_start[lane].is_some_and(|start| (start - object.start_time).abs() < 0.01) {
            bail!(
                "two notes share lane {} at {} ms",
                lane + 1,
                object.start_time
            );
        }
        if last_end[lane].is_some_and(|end| end > object.start_time + 0.01) {
            bail!(
                "overlapping notes on lane {} at {} ms",
                lane + 1,
                object.start_time
            );
        }
        last_start[lane] = Some(object.start_time);
        last_end[lane] = Some(end);
    }
    Ok(())
}

fn analyze(options: &HashMap<String, Option<String>>) -> Result<()> {
    ensure_only(options, &["--input", "--json", "--playstyle"])?;
    if !options.contains_key("--json") {
        bail!("analyze requires --json");
    }
    let input_path = PathBuf::from(required(options, "--input")?);
    let playstyle = optional(options, "--playstyle");
    let (_, beatmap) = read_beatmap(&input_path)?;
    let keys = mania::key_count(&beatmap);
    validate_playstyle(keys, playstyle)?;
    let target_layout = layout::layout(keys, playstyle);
    let analysis = patterns::analyze(&beatmap, &target_layout);
    let rows = patterns::rows(&beatmap);
    let pill_count = rows
        .iter()
        .filter(|row| {
            target_layout.groups().iter().any(|group| {
                let held = row.mask & group.mask;
                (0..keys.saturating_sub(1))
                    .any(|lane| held & (1 << lane) != 0 && held & (1 << (lane + 1)) != 0)
            })
        })
        .count();
    let mut hand_load = [0.0f64; 2];
    let mut long_notes = 0usize;
    for object in &beatmap.hit_objects {
        let Some((x, end)) = mania::note(object) else {
            continue;
        };
        let lane = mania::column(x, keys);
        match target_layout.fingers[lane].hand {
            layout::Hand::Left => hand_load[0] += 1.0,
            layout::Hand::Right => hand_load[1] += 1.0,
            layout::Hand::Either => {
                hand_load[0] += 0.5;
                hand_load[1] += 0.5;
            }
        }
        long_notes += (end > object.start_time) as usize;
    }
    let minijack_count = analysis.modifiers.get("minijack").copied().unwrap_or(0);
    let hand_balance = if hand_load[0].max(hand_load[1]) == 0.0 {
        1.0
    } else {
        hand_load[0].min(hand_load[1]) / hand_load[0].max(hand_load[1])
    };
    let mut value = serde_json::to_value(analysis)?;
    let object = value
        .as_object_mut()
        .context("analysis did not serialize as an object")?;
    object.insert(
        "objects".to_owned(),
        json!(object.get("notes").and_then(Value::as_u64).unwrap_or(0)),
    );
    object.insert("ln".to_owned(), json!(long_notes));
    object.insert("long_notes".to_owned(), json!(long_notes));
    object.insert("minijack".to_owned(), json!(minijack_count));
    object.insert("minijack_count".to_owned(), json!(minijack_count));
    object.insert("pill".to_owned(), json!(pill_count));
    object.insert("pill_count".to_owned(), json!(pill_count));
    object.insert("left_notes".to_owned(), json!(hand_load[0]));
    object.insert("right_notes".to_owned(), json!(hand_load[1]));
    object.insert("hand_balance".to_owned(), json!(hand_balance));
    if let Some(patterns) = object.get_mut("patterns").and_then(Value::as_array_mut) {
        for entry in patterns {
            if let Some(row) = entry.as_object_mut() {
                if let Some(pattern) = row
                    .get("pattern")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                {
                    let share = row.get("share").and_then(Value::as_f64).unwrap_or(0.0);
                    row.insert("name".to_owned(), json!(pattern));
                    row.insert("ratio".to_owned(), json!(share));
                    row.insert("percentage".to_owned(), json!(share * 100.0));
                }
            }
        }
    }
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

fn shape_lns(options: &HashMap<String, Option<String>>) -> Result<()> {
    ensure_only(options, &["--input", "--output", "--mode"])?;
    let input_path = PathBuf::from(required(options, "--input")?);
    let output_path = PathBuf::from(required(options, "--output")?);
    let mode = required(options, "--mode")?;
    let (source_text, mut beatmap) = read_beatmap(&input_path)?;
    match mode {
        "keep" => {
            if let Some(parent) = output_path
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
            {
                fs::create_dir_all(parent)
                    .with_context(|| format!("could not create {}", parent.display()))?;
            }
            fs::write(&output_path, source_text)
                .with_context(|| format!("could not write {}", output_path.display()))?;
            return Ok(());
        }
        "fln" => {
            let zero = NoteLength {
                unit: LengthUnit::Ms,
                value: 0.0,
            };
            let params = FlnConverterParams {
                gap: NoteLength {
                    unit: LengthUnit::Beat,
                    value: 8.0,
                },
                min_length: zero,
                rework_existing_ln: true,
            };
            fln_converter::apply(&mut beatmap, &params).context("Torosu FLN conversion failed")?;
        }
        "short_ln" => {
            let params = ShortLnParams {
                min_length: NoteLength {
                    unit: LengthUnit::Beat,
                    value: 16.0,
                },
                max_length: NoteLength {
                    unit: LengthUnit::Beat,
                    value: 2.0,
                },
            };
            short_ln::apply(&mut beatmap, &params).context("Torosu short-LN conversion failed")?;
        }
        _ => bail!("--mode must be fln, short_ln, or keep"),
    }
    let output = torosu_cli::osu_fixups::sanitize(
        beatmap
            .encode_to_string()
            .context("could not serialize LN-shaped map")?,
        None,
    );
    if let Some(parent) = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("could not create {}", parent.display()))?;
    }
    fs::write(&output_path, output)
        .with_context(|| format!("could not write {}", output_path.display()))?;
    Ok(())
}

fn print_help() {
    println!(
        "torosu-cli\n\
         Usage:\n\
           torosu-cli convert --input <in.osu> --output <out.osu> --keys <6..10> [--playstyle <name>] [--density <none|dense|more-dense|full>] [--keep-jacks|--no-keep-jacks] [--strict]\n\
           torosu-cli analyze --input <in.osu> --json [--playstyle <name>]\n\
           torosu-cli ln-shape --input <in.osu> --output <out.osu> --mode <fln|short_ln|keep>"
    );
}
