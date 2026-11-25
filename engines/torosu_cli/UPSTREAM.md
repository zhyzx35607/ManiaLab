# Upstream Source

The copied Torosu algorithm sources in this crate are from
`D:\project\Torosu` at commit `83e7df6d71383975f247e2807ab30ee5a95788b5`.
That checkout was read-only during this work.

Copied production modules:

- `src-tauri/src/commands/trainer/tasks/key_converter.rs`
- `src-tauri/src/commands/trainer/tasks/fln_converter.rs`
- `src-tauri/src/commands/trainer/tasks/short_ln.rs`
- `src-tauri/src/commands/trainer/analysis/layout.rs`
- `src-tauri/src/commands/trainer/analysis/patterns.rs`
- `src-tauri/src/commands/trainer/analysis/zones.rs`
- `src-tauri/src/commands/trainer/{mania,model,timing}.rs`
- `src-tauri/src/commands/trainer/osu_fixups.rs`

`analysis/corpus.rs` is an offline Songs-folder research test module and is
not part of the production dependency closure. `accents.rs` and audio decoding
are also omitted: this CLI does not accept audio or expose scratch layouts;
the key converter is invoked through its upstream no-audio path.

The upstream MIT license is included as `LICENSE`. The key converter's
audio-dependent upstream test module is opt-in behind the
`upstream-tests` Cargo feature. Its production mapping algorithm is unchanged;
the standalone adapter exposes the converter's existing per-object density
marker so it can preserve source rows while appending native density extras.
