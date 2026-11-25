# Torosu CLI

This crate is a standalone, headless extraction of the Mania key-conversion
and pattern-analysis core documented in `UPSTREAM.md`. It has no Tauri or GUI
dependency.

## Build on the ManiaLab Windows workstation

Keep Rust downloads and build scratch on D:. The GNU toolchain uses Rust's
self-contained linker, so Visual Studio Build Tools are not needed.

```powershell
$env:CARGO_HOME = 'D:\.cargo'
$env:RUSTUP_HOME = 'D:\.rustup'
$env:CARGO_TARGET_DIR = 'D:\rust-build\mania-torosu\target'
$env:TMP = 'D:\rust-build\mania-torosu\tmp'
$env:TEMP = $env:TMP
$env:TMPDIR = $env:TMP
New-Item -ItemType Directory -Force $env:TMP | Out-Null
& D:\.cargo\bin\rustup.exe run stable-x86_64-pc-windows-gnu cargo build `
  --release --locked --manifest-path engines/torosu_cli/Cargo.toml `
  --target x86_64-pc-windows-gnu
$env:MANIALAB_TOROSU_CLI = 'D:\rust-build\mania-torosu\target\x86_64-pc-windows-gnu\release\torosu-cli.exe'
```

Run the Rust regression suite with the same environment and target:

```powershell
& D:\.cargo\bin\rustup.exe run stable-x86_64-pc-windows-gnu cargo test `
  --locked --manifest-path engines/torosu_cli/Cargo.toml `
  --target x86_64-pc-windows-gnu
```

The Python bridge also searches `CARGO_TARGET_DIR` and the crate's normal
`target` directory. It never falls back to the old layout engine when the
native executable is missing or fails. ManiaLab invokes the strict mode,
which keeps all original object fields byte-for-byte except each note's x
coordinate, and can append Torosu density extras without re-serializing the
source rows.
