import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from scripts.build_vocal_phrase_review import (
    build_review_bundle,
    clip_bounds_ms,
    pcm16,
    sha256_file,
)


def test_phrase_clip_bounds_keep_context_and_clamp_to_audio():
    assert clip_bounds_ms(500, 900, 5000) == (0, 2400)
    assert clip_bounds_ms(3000, 3400, 5000) == (1500, 4900)
    with pytest.raises(ValueError, match="extends beyond"):
        clip_bounds_ms(4900, 5100, 5000)


def test_pcm16_uses_supplied_global_gain_without_clipping():
    source = np.array([-1.4, -0.5, 0.0, 0.5, 1.4], dtype=np.float32)
    result = pcm16(source, 0.6)
    assert result.dtype == np.int16
    assert result[0] == -result[-1]
    assert np.max(np.abs(result.astype(np.int32))) < 32767
    with pytest.raises(ValueError, match="would clip"):
        pcm16(source, 1.0)


def _write_fixture(root: Path) -> tuple[Path, dict]:
    from scripts.build_vocal_phrase_review import PROTOCOL_PATH

    source = root / "test-data" / "sakura-benchmark"
    pipeline = source / "piano-full-pipeline-v1"
    pcm_root = source / "raw-pcm-audit-v1"
    stems = pipeline / "stems-provided" / "fixture"
    provenance_dir = root / "test-data" / "vocal-phrase-release-review-v1"
    for path in (pipeline, pcm_root, stems, provenance_dir, root / "docs"):
        path.mkdir(parents=True, exist_ok=True)

    rate = 22050
    time = np.arange(rate * 2, dtype=np.float32) / rate
    vocals = (0.4 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    mix = (0.6 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    vocals_path, mix_path = pcm_root / "vocals.wav", pcm_root / "mix.wav"
    wavfile.write(vocals_path, rate, vocals)
    wavfile.write(mix_path, rate, mix)

    placeholder_paths = [
        stems / "vocals-activations.npz",
        stems / "vocals.npy",
        pipeline / "instrument-analysis.json",
    ]
    for path in placeholder_paths:
        path.write_bytes(b"fixture")
    analysis_path = pipeline / "general-analysis.json"
    identity_path = pipeline / "identity.json"
    audit_path = pcm_root / "report.json"

    source_mix_hash = "1" * 64
    source_vocal_hash = "2" * 64
    analysis = {
        "duration": 2.0,
        "separation": {
            "audio_sha256": source_mix_hash,
            "sample_rate": rate,
            "duration_seconds": 2.0,
        },
        "segments": [
            {"label": "段 1", "start_ms": 0, "end_ms": 1000},
            {"label": "段 2", "start_ms": 1000, "end_ms": 2000},
        ],
        "phrase_candidates": [
            {
                "id": "vocals:p69:500",
                "start": 500,
                "end": 800,
                "pitch": 69,
                "phrase": True,
                "source": "pitch",
                "support": ["vocals:p69:500"],
                "support_coverage": 1.0,
            }
        ],
    }
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    identity = {
        "mix_sha256": source_mix_hash,
        "sources": {
            "vocals": source_vocal_hash,
        },
    }
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    audit = {
        "rows": [
            {
                "role": "vocals",
                "original_sha256": source_vocal_hash,
                "raw_float_sha256": sha256_file(vocals_path),
                "samples": len(vocals),
            },
            {
                "role": "mix",
                "original_sha256": source_mix_hash,
                "raw_float_sha256": sha256_file(mix_path),
                "samples": len(mix),
            },
        ]
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    analysis_sources = {
        stems / "vocals-activations.npz": sha256_file(stems / "vocals-activations.npz"),
        stems / "vocals.npy": sha256_file(stems / "vocals.npy"),
        vocals_path: sha256_file(vocals_path),
        mix_path: sha256_file(mix_path),
        analysis_path: sha256_file(analysis_path),
        pipeline / "instrument-analysis.json": sha256_file(pipeline / "instrument-analysis.json"),
    }
    report = {
        "real": {
            "inputs": {
                str(path.relative_to(root)).replace("/", "\\"): digest
                for path, digest in analysis_sources.items()
            },
            "frozen_geometry_matched": True,
            "phrase_geometry_unchanged": True,
            "phrase_count": 1,
        }
    }
    provenance_path = provenance_dir / "report.json"
    provenance_path.write_text(json.dumps(report), encoding="utf-8")
    (root / "docs" / "vocal-phrase-audio-review-protocol.md").write_text(
        PROTOCOL_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return root, {
        "analysis_path": analysis_path,
        "identity_path": identity_path,
        "audit_path": audit_path,
        "vocals_path": vocals_path,
        "mix_path": mix_path,
        "provenance_path": provenance_path,
    }


def test_review_bundle_checks_sources_and_preserves_existing_output(tmp_path):
    root, paths = _write_fixture(tmp_path)
    provenance_path = paths["provenance_path"]
    output = tmp_path / "review"
    report = build_review_bundle(output=output, root=root)
    assert report["candidate_count"] == 1
    assert report["clip_count"] == 2
    assert report["human_listening_completed"] is False
    assert (output / "clips" / "001-vocals.wav").is_file()
    assert (output / "clips" / "001-mix.wav").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["phrases"][0]["section_label"] == "段 1"
    html = (output / "index.html").read_text(encoding="utf-8")
    assert "候选声源判断" in html or "声源判断" in html

    with pytest.raises(ValueError, match="Preserve previous"):
        build_review_bundle(output=output, root=root)

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["real"]["phrase_geometry_unchanged"] = False
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen phrase geometry"):
        build_review_bundle(output=tmp_path / "second-review", root=root)
