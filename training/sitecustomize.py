from __future__ import annotations

import os


if os.environ.get("MANIALAB_AUDIO_FFMPEG_COMPAT") == "1":
    from train_10k_lora import configure_audio_decoder

    configure_audio_decoder()
