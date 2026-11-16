import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from urllib.error import URLError
import zipfile

import numpy as np
import pytest

from manialab import piano
from scripts import build_provenance, fetch_piano_model


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def model_root(tmp_path, monkeypatch):
    identity = json.loads((ROOT / "assets/piano-model.json").read_text(encoding="utf-8"))
    payload = b"verified test graph"
    model_sha = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(piano, "MODEL_SHA256", model_sha)
    monkeypatch.setattr(piano, "MODEL_BYTES", len(payload))
    identity.update(sha256=model_sha, bytes=len(payload))

    license_path = tmp_path / identity["license_file"]
    license_path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / identity["license_file"], license_path)
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/piano-model.json").write_text(
        json.dumps(identity), encoding="utf-8"
    )
    model = tmp_path / "assets/deep" / identity["filename"]
    model.parent.mkdir()
    model.write_bytes(payload)
    return tmp_path, identity, model


def test_piano_packaging_identity_requires_fixed_model_and_attribution(model_root):
    root, identity, _ = model_root

    assert build_provenance.verify_piano_model(root) == identity


def test_piano_packaging_rejects_missing_or_changed_model(model_root):
    root, _, model = model_root
    model.unlink()

    with pytest.raises(ValueError, match="Required piano model is missing"):
        build_provenance.verify_piano_model(root)

    model.write_bytes(b"changed graph")
    with pytest.raises(ValueError, match="size or checksum"):
        build_provenance.verify_piano_model(root)


def test_verify_file_reports_actual_identity_on_checksum_mismatch(tmp_path):
    payload = b"actual model"
    target = tmp_path / "piano.onnx"
    target.write_bytes(payload)
    expected_sha256 = hashlib.sha256(b"expected model").hexdigest()
    actual_sha256 = hashlib.sha256(payload).hexdigest()

    with pytest.raises(ValueError) as error:
        fetch_piano_model.verify_file(
            target,
            byte_count=len(payload),
            sha256=expected_sha256,
        )

    assert expected_sha256 in str(error.value)
    assert actual_sha256 in str(error.value)
    assert f"bytes={len(payload)}" in str(error.value)


def test_conv_batch_norm_fusion_uses_fixed_float32_operation_order():
    weight = np.array([0.123456789, -0.7654321, 1.3333333], dtype=np.float32).reshape(
        3, 1, 1, 1
    )
    bias = np.array([0.25, -0.5, 0.125], dtype=np.float32)
    running_mean = np.array([0.1, -0.25, 0.5], dtype=np.float32)
    running_var = np.array([0.3, 1.7, 0.0001], dtype=np.float32)
    gamma = np.array([0.9, -1.2, 0.0005], dtype=np.float32)
    beta = np.array([-0.2, 0.7, 0.3], dtype=np.float32)

    fused_weight, fused_bias = fetch_piano_model.fuse_conv_batch_norm_parameters(
        weight, bias, running_mean, running_var, gamma, beta, 1e-5, np
    )

    assert fused_weight.reshape(-1).view(np.uint32).tolist() == [
        0x3E4FB9B2,
        0x3F34581D,
        0x3D822DEF,
    ]
    assert fused_bias.view(np.uint32).tolist() == [
        0x3D3E5868,
        0x3F6E1A4D,
        0x3E90725F,
    ]


def test_onnx_export_diagnostic_records_graph_and_initializer_identity(
    tmp_path, monkeypatch
):
    payload = b"exported graph"
    target = tmp_path / "model.onnx"
    target.write_bytes(payload)
    tensor_data = b"initializer data"
    tensor = SimpleNamespace(
        name="weight",
        dims=(2, 2),
        data_type=1,
        raw_data=tensor_data,
    )
    graph = SimpleNamespace(
        initializer=[tensor],
        input=[SimpleNamespace(name="audio")],
        output=[SimpleNamespace(name="notes")],
        node=[SimpleNamespace(name="conv", domain="", op_type="Conv")],
        SerializeToString=lambda deterministic: b"graph",
    )
    model = SimpleNamespace(
        graph=graph,
        ir_version=8,
        producer_name="pytorch",
        producer_version="2.9.1",
        opset_import=[SimpleNamespace(domain="", version=17)],
    )
    fake_onnx = SimpleNamespace(
        load=lambda *_args, **_kwargs: model,
        numpy_helper=SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)

    result = fetch_piano_model.onnx_export_diagnostic(target)

    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["graph_sha256"] == hashlib.sha256(b"graph").hexdigest()
    assert result["input_names"] == ["audio"]
    assert result["output_names"] == ["notes"]
    assert result["nodes"] == [dict(name="conv", domain="", op_type="Conv")]
    assert result["initializers"] == [
        dict(
            name="weight",
            dims=[2, 2],
            data_type=1,
            bytes=len(tensor_data),
            sha256=hashlib.sha256(tensor_data).hexdigest(),
        )
    ]


@pytest.mark.parametrize("damage", ["metadata", "license_missing", "license_changed"])
def test_piano_packaging_rejects_changed_identity_or_license(model_root, damage):
    root, identity, _ = model_root
    if damage == "metadata":
        identity["source_commit"] = "0" * 40
        (root / "assets/piano-model.json").write_text(
            json.dumps(identity), encoding="utf-8"
        )
    else:
        license_path = root / identity["license_file"]
        if damage == "license_missing":
            license_path.unlink()
        else:
            license_path.write_text("changed license", encoding="utf-8")

    with pytest.raises(ValueError, match="identity|license/attribution"):
        build_provenance.validate_piano_model_metadata(root)


def test_download_verified_uses_pinned_fallback_and_cleans_partial_file(tmp_path, monkeypatch):
    payload = b"piano artifact"
    digest = hashlib.sha256(payload).hexdigest()
    calls = []

    def open_url(url, timeout):
        calls.append(url)
        if url == "primary":
            raise URLError("temporary source failure")
        return io.BytesIO(payload)

    monkeypatch.setattr(fetch_piano_model.urllib.request, "urlopen", open_url)
    target = tmp_path / "source.bin"

    result = fetch_piano_model.download_verified(
        ["primary", "mirror"],
        target,
        byte_count=len(payload),
        sha256=digest,
    )

    assert result == target
    assert target.read_bytes() == payload
    assert calls == ["primary", "mirror"]
    assert not target.with_name(target.name + ".pending").exists()


def test_download_verified_never_overwrites_an_existing_bad_file(tmp_path, monkeypatch):
    target = tmp_path / "source.bin"
    target.write_bytes(b"keep this")
    monkeypatch.setattr(
        fetch_piano_model.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("existing artifact must not be replaced"),
    )

    with pytest.raises(ValueError, match="Unexpected file size"):
        fetch_piano_model.download_verified(
            ["source"],
            target,
            byte_count=4,
            sha256=hashlib.sha256(b"good").hexdigest(),
        )

    assert target.read_bytes() == b"keep this"


def test_download_verified_stops_oversized_response_before_writing_it(tmp_path, monkeypatch):
    target = tmp_path / "source.bin"
    monkeypatch.setattr(
        fetch_piano_model.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(b"too much data"),
    )

    with pytest.raises(RuntimeError, match="Download exceeds expected size"):
        fetch_piano_model.download_verified(
            ["source"],
            target,
            byte_count=4,
            sha256=hashlib.sha256(b"good").hexdigest(),
        )

    assert not target.exists()
    assert not target.with_name(target.name + ".pending").exists()


def test_extract_upstream_source_checks_archive_identity_and_expected_files(tmp_path):
    identity = json.loads((ROOT / "assets/piano-model.json").read_text(encoding="utf-8"))
    files = {
        "piano_transcription_inference/__init__.py": "from .config import sample_rate\n",
        "piano_transcription_inference/config.py": "sample_rate = 16000\n",
        "piano_transcription_inference/inference.py": "class PianoTranscription: pass\n",
        "piano_transcription_inference/models.py": "class Note_pedal: pass\n",
        "piano_transcription_inference/piano_vad.py": "pass\n",
        "piano_transcription_inference/pytorch_utils.py": "pass\n",
        "piano_transcription_inference/utilities.py": "pass\n",
    }
    archive_path = tmp_path / "source.zip"
    prefix = f"piano_transcription_inference-{identity['source_commit']}/"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, content in files.items():
            archive.writestr(prefix + name, content)
    identity = identity | {
        "source_archive_bytes": archive_path.stat().st_size,
        "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }

    destination = fetch_piano_model.extract_upstream_source(
        archive_path, tmp_path / "source", identity
    )

    assert {path.relative_to(destination).as_posix() for path in destination.rglob("*.py")} == set(files)
    assert (destination / "piano_transcription_inference/models.py").read_text(
        encoding="utf-8"
    ) == files["piano_transcription_inference/models.py"]


def test_extract_upstream_source_rejects_archive_path_traversal(tmp_path):
    identity = json.loads((ROOT / "assets/piano-model.json").read_text(encoding="utf-8"))
    archive_path = tmp_path / "source.zip"
    prefix = f"piano_transcription_inference-{identity['source_commit']}/"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(prefix + "piano_transcription_inference/../../outside.py", "pass\n")
    identity = identity | {
        "source_archive_bytes": archive_path.stat().st_size,
        "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }

    with pytest.raises(ValueError, match="Unsafe path"):
        fetch_piano_model.extract_upstream_source(
            archive_path, tmp_path / "source", identity
        )
