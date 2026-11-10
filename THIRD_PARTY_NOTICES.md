# Third-party notices

ManiaLab's own code is MIT licensed. Bundled third-party programs and model data retain their own licenses; the root MIT license does not relicense them.

* Mapperatorinator: OliBomby and contributors, MIT. Pinned source commit `5884d1bcb57494f09fe3d74f3a6b58ca0d2960aa`. `vendor/mapperatorinator/event.py` and `template.osu` are unmodified upstream files. `assets/mapperatorinator-source.zip` is a selected-source archive of that commit, used by the optional inference backend. Source: https://github.com/OliBomby/Mapperatorinator
* Basic Pitch: Spotify AB and contributors, Apache-2.0. ONNX model from v0.4.0. The audio window sizes and model I/O conventions follow upstream; note extraction and sample-offset stitching are implemented in ManiaLab. Source: https://github.com/spotify/basic-pitch/tree/v0.4.0 . Model hash: `assets/model-manifest.json`. License: `licenses/BasicPitch-APACHE-2.0.txt`.
* Optional piano regression model: Qiuqiang Kong, Bochen Li, Xuchen Song, Yuan Wan and Yuxuan Wang, CC-BY-4.0. Original checkpoint: https://zenodo.org/records/4034264 . Inference source reference: https://github.com/qiuqiangkong/piano_transcription_inference at `0226e74cbc805660e34bbd6a8fed2083890ebb88`. ManiaLab exports the fixed weights to a batch-1, 10-second ONNX graph and implements overlap stitching and event decoding with release provenance. No fine-tuning was performed. Checkpoint and export hashes, changes and attribution are in `assets/piano-model.json`; model license: `licenses/PianoRegression-CC-BY-4.0.txt`. The model is disabled by default and applies only to isolated piano; pedal-extended intervals are not chart LN targets.
* Hybrid Demucs: Alexandre Défossez, Meta and contributors, MIT; the torchaudio implementation is BSD-2-Clause. The Windows package includes the official `HDEMUCS_HIGH_MUSDB_PLUS` weights and torchaudio 2.9.1 CPU runtime. Sources: https://github.com/facebookresearch/demucs and https://github.com/pytorch/audio/tree/v2.9.1 . Exact weight URL, size and SHA256 are in `assets/deep-models.json`; torchaudio notice is in `licenses/Torchaudio-BSD.txt`, PyTorch and dependency notices in `licenses/deep-runtime/`. Model output is four estimated sources and is not a set of isolated studio recordings.
* Beat This!: Francesco Foscarin, Jan Schlüter, Gerhard Widmer and contributors, MIT. Package version 1.1.0, checkpoint `final0`. Source: https://github.com/CPJKU/beat_this . Notice: `licenses/BeatThis-MIT.txt`; checkpoint URL and SHA256: `assets/deep-models.json`. Used without madmom/DBN, only for beat-unit evidence; measured audio attacks determine the millisecond phase.
* Six-source HTDemucs: Meta and contributors, Demucs 4.0.1, MIT. The additional official `5c90dfd2` checkpoint estimates drums, bass, other, vocals, guitar and piano. Its exact URL and full checksum are in `assets/deep-models.json`. Original library licenses, including Open-Unmix, Julius, Dora and OmegaConf, are retained in `licenses/deep-runtime/`. The optional six-source path uses Demucs's original overlap-add implementation; it does not contain NovaMSS's compiled modules or encrypted weights. NovaMSS was examined locally as a design reference only.
* uv: Astral and contributors, MIT / Apache-2.0. The Windows distribution includes uv 0.12.1 as a separate process to manage the optional Python environment. Source: https://github.com/astral-sh/uv/tree/0.12.1
* FFmpeg: the FFmpeg developers and the included codec-library authors. A separate executable supplied by imageio-ffmpeg is included for audio decoding. Its version/build configuration and source references are in `licenses/FFmpeg-build-info.txt`, with GPLv3 text in `licenses/FFmpeg-GPLv3.txt`. FFmpeg is invoked as a subprocess, not linked into ManiaLab's Python modules. Source project: https://ffmpeg.org/ ; build distributor: https://www.gyan.dev/ffmpeg/builds/ . Other library license notices are retained with the bundled runtime inventory.
* Python, NumPy, SciPy, ONNX Runtime, Flask, Waitress, imageio-ffmpeg and packaging dependencies: their notices are copied from the installed distributions into `licenses/runtime/` during packaging. `licenses/runtime-inventory.json` records exact package versions.

The complete Windows ZIP includes a separate CPU PyTorch audio worker for Hybrid Demucs and Beat This!. The optional Mapperatorinator backend installs its own environment into the user's application-data directory; its V32 model weights are not embedded in the ZIP. Its license metadata stays in that separate installation; the model repository README is downloaded with the weights.

User audio, beatmaps, API credentials, local projects and test downloads are not included in the source repository or Release.

Optional Tsumugi bass/other transcription and offline stitching audit: anime-song, MIT,
source commit `eba1c7d05e4d6a53cc4e8916d742d9623a3fdc38`,
https://github.com/anime-song/tsumugi . The lineage replay follows and checks
the upstream V1 stitcher; attribution is retained in `licenses/Tsumugi-MIT.txt`.
The optional Windows worker includes `bass_v2` and `other_v1_5` from
`https://huggingface.co/anime-song/instrument_agnostic_amt` at revision
`c6ee8a22626cd7185031ee71ce7cea27ec862781`, under the author's MIT model license.
It is disabled by default; predicted releases require a separate opt-in and
are not verified chart LN targets. The source/model hashes and packaged
real-audio checks are retained in `assets/instruments/manifest.json` and
`verification.json`; dependency notices are in `licenses/instrument-runtime/`.
The guitar, vocal and instrument-refinement models remain local research only
and are not bundled. Evaluation scope is recorded in
`docs/other-instrument-model-survey-2026-11-10.md`.

Design reference: Torosu, Copyright (c) 2026 Manolo-dl, MIT, commit b283791f46622b0dd1f47f53afa10bd62d700e66. Its explicit hand layouts and LN release-state costs informed the independently implemented ManiaLab allocator. License retained in licenses/Torosu-MIT.txt. Source: https://github.com/Manolo-dl/Torosu

The user-supplied Timing Analyzer v0.32.4 alpha and its BASS runtime were used only for local comparison. Neither executable is distributed in ManiaLab.

`assets/pattern-prior.json` is a small ManiaLab-trained lane-preference model. Its training sources, grouping, exclusions, validation limits and fingerprint are documented in `docs/training-report.md`. It does not contain audio or chart files. Publicly available chart data was used for research; the original charts retain their authorship and are not redistributed.

The explicit mapping-clock option is informed by local Timing Analyzer comparison and osu!'s public editor waveform discussion: https://github.com/ppy/osu/pull/26136 and https://github.com/ppy/osu/issues/21947 . osu!'s visual offset is 20ms; ManiaLab's adjustable 22.5ms initial setting matches the reference Timing Analyzer. Neither value is a claim about every device's playback latency. No osu! code is copied for this feature.
