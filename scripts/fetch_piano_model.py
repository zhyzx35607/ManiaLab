"""Build the pinned piano ONNX export from its attributed upstream checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import validate_piano_model_metadata, verify_piano_model


def file_hash(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def verify_file(path, *, byte_count, sha256, md5=None):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Expected file is missing: {path.name}")
    actual_bytes = path.stat().st_size
    actual_sha256 = file_hash(path)
    if actual_bytes != byte_count:
        raise ValueError(
            f"Unexpected file size: {path.name}; expected={byte_count}, "
            f"actual={actual_bytes}, sha256={actual_sha256}"
        )
    if actual_sha256 != sha256:
        raise ValueError(
            f"Unexpected SHA-256: {path.name}; expected={sha256}, "
            f"actual={actual_sha256}, bytes={actual_bytes}"
        )
    if md5 is not None and file_hash(path, "md5") != md5:
        raise ValueError(f"Unexpected MD5: {path.name}")


def onnx_export_diagnostic(path):
    import onnx
    from onnx import numpy_helper

    path = Path(path)
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    initializers = []
    for tensor in graph.initializer:
        data = tensor.raw_data or numpy_helper.to_array(tensor).tobytes()
        initializers.append(
            dict(
                name=tensor.name,
                dims=list(tensor.dims),
                data_type=tensor.data_type,
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    initializers.sort(key=lambda item: item["name"])
    return dict(
        bytes=path.stat().st_size,
        sha256=file_hash(path),
        graph_sha256=hashlib.sha256(
            graph.SerializeToString(deterministic=True)
        ).hexdigest(),
        ir_version=model.ir_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        opset_imports=[
            dict(domain=item.domain, version=item.version)
            for item in model.opset_import
        ],
        input_names=[item.name for item in graph.input],
        output_names=[item.name for item in graph.output],
        nodes=[
            dict(name=item.name, domain=item.domain, op_type=item.op_type)
            for item in graph.node
        ],
        initializers=initializers,
    )


def download_verified(urls, target, *, byte_count, sha256, md5=None):
    target = Path(target)
    if target.exists():
        verify_file(target, byte_count=byte_count, sha256=sha256, md5=md5)
        return target
    temporary = target.with_name(target.name + ".pending")
    if temporary.exists():
        raise FileExistsError(f"Preserving existing partial download: {temporary}")
    target.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                with temporary.open("xb") as stream:
                    written = 0
                    while block := response.read(min(1024 * 1024, byte_count - written + 1)):
                        if written + len(block) > byte_count:
                            raise ValueError(f"Download exceeds expected size: {target.name}")
                        stream.write(block)
                        written += len(block)
            verify_file(temporary, byte_count=byte_count, sha256=sha256, md5=md5)
            temporary.replace(target)
            return target
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            failures.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError("Unable to fetch verified piano artifact: " + "; ".join(failures))


def extract_upstream_source(archive_path, destination, identity):
    verify_file(
        archive_path,
        byte_count=identity["source_archive_bytes"],
        sha256=identity["source_archive_sha256"],
    )
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Preserving existing source directory: {destination}")
    destination.mkdir(parents=True)
    prefix = f"piano_transcription_inference-{identity['source_commit']}/"
    expected = {
        "piano_transcription_inference/__init__.py",
        "piano_transcription_inference/config.py",
        "piano_transcription_inference/inference.py",
        "piano_transcription_inference/models.py",
        "piano_transcription_inference/piano_vad.py",
        "piano_transcription_inference/pytorch_utils.py",
        "piano_transcription_inference/utilities.py",
    }
    found = set()
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            if item.is_dir() or not item.filename.startswith(prefix):
                continue
            relative = item.filename[len(prefix):]
            path = PurePosixPath(relative)
            if (
                path.is_absolute()
                or not path.parts
                or any(part in ("", ".", "..") for part in path.parts)
                or "\\" in relative
            ):
                raise ValueError("Unsafe path in the pinned piano source archive")
            if relative.startswith("piano_transcription_inference/") and relative.endswith(".py"):
                target = destination.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item))
                found.add(relative)
    if found != expected:
        raise ValueError("Pinned piano source archive has an unexpected Python file set")
    return destination


def export_onnx(checkpoint_path, source_dir, output_path, identity):
    os.environ.update(TORCH_FORCE_WEIGHTS_ONLY_LOAD="1", MPLBACKEND="Agg")
    sys.path.insert(0, str(Path(source_dir).resolve()))
    import numpy as np
    import onnx
    import torch

    exporter = identity["export"]
    versions = {
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "numpy": np.__version__,
    }
    if any(versions.get(name) != version for name, version in exporter.items() if name in versions):
        raise RuntimeError(f"Pinned piano export environment mismatch: {versions}")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    from piano_transcription_inference.models import Note_pedal

    model = Note_pedal(frames_per_second=100, classes_num=88)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.eval()
    output_names = [
        "reg_onset_output",
        "reg_offset_output",
        "frame_output",
        "velocity_output",
        "reg_pedal_onset_output",
        "reg_pedal_offset_output",
        "pedal_frame_output",
    ]

    class Outputs(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, audio):
            result = self.model(audio)
            return tuple(result[name] for name in output_names)

    input_tensor = torch.zeros((1, identity["input_samples"]), dtype=torch.float32)
    torch.onnx.export(
        Outputs(model).eval(),
        input_tensor,
        str(output_path),
        input_names=["audio"],
        output_names=output_names,
        dynamo=False,
        opset_version=exporter["opset"],
    )
    onnx.checker.check_model(str(output_path))


def prepare(output, work, checkpoint=None, source_archive=None):
    identity = validate_piano_model_metadata(ROOT)
    output = Path(output).resolve()
    if output.exists():
        verify_file(
            output,
            byte_count=identity["bytes"],
            sha256=identity["sha256"],
        )
        if output == (ROOT / "assets/deep" / identity["filename"]).resolve():
            verify_piano_model(ROOT)
        return dict(status="already_verified", sha256=identity["sha256"], bytes=identity["bytes"])

    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(checkpoint).resolve() if checkpoint else None
    source_archive = Path(source_archive).resolve() if source_archive else None
    if checkpoint is None:
        checkpoint = download_verified(
            [
                identity["upstream_checkpoint_url"],
                "https://zenodo.org/records/4034264/files/"
                "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1",
            ],
            work / "upstream-checkpoint.pth",
            byte_count=identity["upstream_checkpoint_bytes"],
            sha256=identity["upstream_checkpoint_sha256"],
            md5=identity["upstream_checkpoint_md5"],
        )
    else:
        verify_file(
            checkpoint,
            byte_count=identity["upstream_checkpoint_bytes"],
            sha256=identity["upstream_checkpoint_sha256"],
            md5=identity["upstream_checkpoint_md5"],
        )
    if source_archive is None:
        source_archive = download_verified(
            [identity["source_archive_url"]],
            work / "upstream-source.zip",
            byte_count=identity["source_archive_bytes"],
            sha256=identity["source_archive_sha256"],
        )
    else:
        verify_file(
            source_archive,
            byte_count=identity["source_archive_bytes"],
            sha256=identity["source_archive_sha256"],
        )

    source_dir = extract_upstream_source(source_archive, work / "source", identity)
    exported = work / identity["filename"]
    export_onnx(checkpoint, source_dir, exported, identity)
    try:
        verify_file(exported, byte_count=identity["bytes"], sha256=identity["sha256"])
    except ValueError as exc:
        diagnostic = onnx_export_diagnostic(exported)
        diagnostic.update(
            expected_bytes=identity["bytes"],
            expected_sha256=identity["sha256"],
            verification_error=str(exc),
        )
        diagnostic_path = work / "export-diagnostic.json"
        diagnostic_path.write_text(
            json.dumps(diagnostic, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                dict(
                    status="model_identity_mismatch",
                    diagnostic=str(diagnostic_path),
                    bytes=diagnostic["bytes"],
                    sha256=diagnostic["sha256"],
                ),
                ensure_ascii=True,
            ),
            flush=True,
        )
        raise
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing model: {output}")
    temporary = output.with_name(output.name + ".pending")
    if temporary.exists():
        raise FileExistsError(f"Preserving existing partial model: {temporary}")
    shutil.copyfile(exported, temporary)
    verify_file(temporary, byte_count=identity["bytes"], sha256=identity["sha256"])
    temporary.replace(output)
    if output == (ROOT / "assets/deep" / identity["filename"]).resolve():
        verify_piano_model(ROOT)
    report = dict(
        status="built_and_verified",
        checkpoint_sha256=identity["upstream_checkpoint_sha256"],
        source_archive_sha256=identity["source_archive_sha256"],
        onnx_sha256=identity["sha256"],
        bytes=identity["bytes"],
        input="all-zero synthetic 10-second tensor used only to trace the fixed graph",
        musical_acceptance=False,
    )
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/deep/piano-regression.onnx")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Use a previously verified local upstream checkpoint")
    parser.add_argument("--source-archive", type=Path, help="Use a previously verified local source archive")
    args = parser.parse_args()
    if args.work is None:
        build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.work = ROOT / "build" / f"piano-model-{build_id}"
    result = prepare(args.output, args.work, args.checkpoint, args.source_archive)
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
