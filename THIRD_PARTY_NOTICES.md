# Third-party notices

ManiaLab's own code is MIT licensed. Bundled third-party programs and model data retain their own licenses; the root MIT license does not relicense them.

* Mapperatorinator: OliBomby and contributors, MIT. Pinned source commit `5884d1bcb57494f09fe3d74f3a6b58ca0d2960aa`. `vendor/mapperatorinator/event.py` and `template.osu` are unmodified upstream files. `assets/mapperatorinator-source.zip` is a selected-source archive of that commit, used by the optional inference backend. Source: https://github.com/OliBomby/Mapperatorinator
* Basic Pitch: Spotify AB and contributors, Apache-2.0. ONNX model from v0.4.0. The audio window sizes and model I/O conventions follow upstream; note extraction and sample-offset stitching are implemented in ManiaLab. Source: https://github.com/spotify/basic-pitch/tree/v0.4.0 . Model hash: `assets/model-manifest.json`. License: `licenses/BasicPitch-APACHE-2.0.txt`.
* uv: Astral and contributors, MIT / Apache-2.0. The Windows distribution includes uv 0.12.1 as a separate process to manage the optional Python environment. Source: https://github.com/astral-sh/uv/tree/0.12.1
* FFmpeg: the FFmpeg developers and the included codec-library authors. A separate executable supplied by imageio-ffmpeg is included for audio decoding. Its version/build configuration and source references are in `licenses/FFmpeg-build-info.txt`, with GPLv3 text in `licenses/FFmpeg-GPLv3.txt`. FFmpeg is invoked as a subprocess, not linked into ManiaLab's Python modules. Source project: https://ffmpeg.org/ ; build distributor: https://www.gyan.dev/ffmpeg/builds/ . Other library license notices are retained with the bundled runtime inventory.
* Python, NumPy, SciPy, ONNX Runtime, Flask, Waitress, imageio-ffmpeg and packaging dependencies: their notices are copied from the installed distributions into `licenses/runtime/` during packaging. `licenses/runtime-inventory.json` records exact package versions.

The optional Mapperatorinator backend downloads CPU PyTorch and other upstream dependencies into the user's application-data directory. These packages and downloaded V32 model weights are not embedded in the base Windows ZIP. Their license metadata stays in that separate installation; the model repository README is downloaded with the weights.

User audio, beatmaps, API credentials, local projects and test downloads are not included in the source repository or Release.
