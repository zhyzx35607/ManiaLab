"""Pinned optional instrument assets, verifiable without a Git installation."""
import hashlib
import json
from pathlib import Path

SOURCE_COMMIT = 'eba1c7d05e4d6a53cc4e8916d742d9623a3fdc38'
MODEL_REVISION = 'c6ee8a22626cd7185031ee71ce7cea27ec862781'
SOURCE_MANIFEST_SHA256 = '4ab10eeec9e829722a88ce3f6fb603dcc506d0d64d39a738cda5f69448001dd8'
MODELS = {
    'bass_v2':'807eedb409c77a710357d57c9e47d17821c4518db3b09c78d8dabd205a2b604a',
    'other_v1_5':'e836d5b30a29ad2301726aedb9ae348db927fbb4813359cdfb2e304ba3084501',
}
VOICE_MODELS = {'bass':'bass_v2','other':'other_v1_5'}
RUNTIME_FILES = (
    'scripts/transcribe_instruments.py', 'scripts/research_tsumugi_transcription.py',
    'scripts/tsumugi_candidate_adapter.py', 'scripts/audit_tsumugi_releases.py',
    'scripts/recover_tsumugi_attacks.py', 'scripts/evaluate_guitar_vocal_benchmark.py',
    'scripts/evaluate_instrument_benchmark.py', 'scripts/probe_mapper_adapter.py',
    'scripts/worker_lifetime.py', 'manialab/instrument_bundle.py',
    'manialab/domain.py', 'vendor/mapperatorinator/event.py',
)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def source_manifest():
    path = Path(__file__).with_name('tsumugi-source.json')
    result = json.loads(path.read_text(encoding='utf-8'))
    canonical = json.dumps(result,sort_keys=True,separators=(',',':')).encode('utf-8')
    if hashlib.sha256(canonical).hexdigest() != SOURCE_MANIFEST_SHA256:
        raise ValueError('专用采音源码清单校验失败')
    if result['source_commit'] != SOURCE_COMMIT:
        raise ValueError('专用采音源码版本不符')
    return result


def verify_source(source):
    source = Path(source).resolve()
    result = source_manifest()['source_files']
    for name,digest in result.items():
        path = (source/name).resolve()
        if not path.is_relative_to(source) or not path.is_file() or sha(path) != digest:
            raise ValueError('专用采音源码文件校验失败：'+name)
    return dict(result)


def verify_model(models, model):
    if model not in MODELS:
        raise ValueError('此入口仅提供已对照的贝斯与其他伴奏采音模型')
    path = Path(models)/('best_model_'+model+'.pth')
    if not path.is_file() or sha(path) != MODELS[model]:
        raise ValueError('专用采音模型校验失败：'+model)
    return path


def runtime_identity(root):
    return {name:sha(Path(root)/name) for name in RUNTIME_FILES}
