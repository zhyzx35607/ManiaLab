"""Names and separation profiles shared by analysis, storage and the UI API."""
import hashlib
import json
import re
import shutil
from pathlib import Path

VOICES = {'drums':'鼓 / 打击', 'bass':'贝斯', 'vocals':'人声',
          'other':'其余伴奏', 'guitar':'吉他', 'piano':'钢琴'}
PROFILES = {
    'four': dict(directory='stems', filename='hdemucs_high_trained.pt',
        sha256='a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816',
        algorithm_revision=1, voices=('drums','bass','vocals','other'),
        label='四声部 · Hybrid Demucs'),
    'six': dict(directory='stems-six', filename='htdemucs_6s.th',
        sha256='34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd',
        algorithm_revision=1, voices=('drums','bass','vocals','other','guitar','piano'),
        label='六声部 · HTDemucs'),
}


def profile_for(value):
    if value not in PROFILES:
        raise ValueError('分离方案无效')
    return PROFILES[value]


def stem_path(project, analysis, voice, extension='wav'):
    separation = (analysis or {}).get('separation',{})
    if separation.get('source')=='provided':
        if voice not in separation_voices(separation) or extension not in ('wav','npy'):
            raise ValueError('当前导入声轨中没有这个声部')
        identity=separation.get('set_sha256','')
        if not re.fullmatch(r'[0-9a-f]{64}',identity):raise ValueError('导入声轨身份无效，请重新导入')
        return project/'stems-provided'/identity/f'{voice}.{extension}'
    profile = profile_for(separation.get('profile','four'))
    if voice not in profile['voices']:
        raise ValueError('当前分析没有这个声部，请切换分离方案并重新分析')
    review_identity = separation.get('review_audio_identity')
    if review_identity is not None:
        identity = separation_identity(separation)
        if review_identity != identity:
            raise ValueError('声部试听身份无效，请重新分析')
        return project/'stems-review'/identity/f'{voice}.{extension}'
    return project / profile['directory'] / f'{voice}.{extension}'


def separation_voices(separation):
    if separation.get('source')!='provided':
        return profile_for(separation.get('profile','four'))['voices']
    voices=separation.get('voices',[])
    if not voices or len(set(voices))!=len(voices) or any(v not in VOICES for v in voices):
        raise ValueError('导入声轨分类无效')
    return tuple(v for v in VOICES if v in voices)


def separation_label(separation):
    if separation.get('source')=='provided':return '已导入的独立声轨'
    return profile_for(separation.get('profile','four'))['label']


def separation_identity(separation):
    """Include input processing, not just weights, in downstream audio caches.

    Provided stems have their own verified content identity and do not pass
    through the model's input processing pipeline.
    """
    if separation.get('source') == 'provided':
        identity = dict(source='provided', audio=separation['audio_sha256'],
                        stems=separation['set_sha256'])
    else:
        identity = dict(source='model', audio=separation['audio_sha256'],
                        model=separation['model_sha256'], profile=separation.get('profile','four'),
                        input_processing=separation.get('input_processing'),
                        algorithm_revision=separation.get('algorithm_revision'))
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def archive_review_stems(work, stem_root, separation, voices):
    """Keep model playback WAVs addressable after the profile cache is replaced."""
    if separation.get('source') == 'provided':
        return separation
    identity = separation_identity(separation)
    folder = Path(work)/'stems-review'/identity
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for voice in voices:
        source = Path(stem_root)/f'{voice}.wav'
        target = folder/f'{voice}.wav'
        if not source.is_file():
            raise ValueError(f'{voice} 声部试听文件缺失，无法保存来源快照')
        if target.is_file():
            if _sha256_file(target) != _sha256_file(source):
                raise ValueError('已有声部来源快照与当前分析不一致')
        else:
            temporary = target.with_name(target.name+'.pending')
            try:
                shutil.copyfile(source, temporary)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        files[voice] = _sha256_file(target)

    manifest = folder/'review.json'
    record = dict(revision=1, identity=identity, files=files)
    if manifest.is_file():
        try:
            previous = json.loads(manifest.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError('声部来源快照清单无法读取') from exc
        if previous != record:
            raise ValueError('已有声部来源快照清单与当前分析不一致')
    else:
        temporary = manifest.with_suffix('.pending')
        try:
            temporary.write_text(json.dumps(record, sort_keys=True), encoding='utf-8')
            temporary.replace(manifest)
        finally:
            temporary.unlink(missing_ok=True)

    for voice in voices:
        (Path(stem_root)/f'{voice}.wav').unlink(missing_ok=True)
    return separation | {'review_audio_identity': identity, 'review_audio_files': files}


def candidate_stem_path(project, source_reference, voice):
    """Resolve only a candidate's saved source; never substitute current analysis."""
    if not isinstance(source_reference, dict):
        raise ValueError('此旧候选未保存声轨来源，不能准确回看声部；请重新分析并生成候选')
    separation = source_reference.get('separation')
    if not isinstance(separation, dict):
        raise ValueError('候选声轨来源记录无效')
    identity = separation_identity(separation)
    if source_reference.get('identity') != identity:
        raise ValueError('候选声轨来源身份不匹配')
    if separation.get('source') != 'provided' and separation.get('review_audio_identity') != identity:
        raise ValueError('此候选的模型声轨快照不可用；请重新分析并生成候选')
    path = stem_path(project, {'separation': separation}, voice)
    if not path.is_file():
        raise ValueError('此候选绑定的声轨文件不可用，不能借用当前分析声轨')
    if separation.get('source') == 'provided':
        decoded = separation.get('decoded_files')
        voice_files = decoded.get(voice) if isinstance(decoded, dict) else None
        expected_hash = voice_files.get('wav_sha256') if isinstance(voice_files, dict) else None
    else:
        files = separation.get('review_audio_files')
        expected_hash = files.get(voice) if isinstance(files, dict) else None
        manifest = path.parent/'review.json'
        try:
            record = json.loads(manifest.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError('此候选的模型声轨快照清单不可用') from exc
        if record != dict(revision=1, identity=identity, files=files):
            raise ValueError('此候选的模型声轨快照清单与来源不匹配')
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError('此候选没有声轨内容校验信息；请重新分析并生成候选')
    try:
        actual_hash = _sha256_file(path)
    except OSError as exc:
        raise ValueError('此候选绑定的声轨文件不可用，不能借用当前分析声轨') from exc
    if actual_hash != expected_hash:
        raise ValueError('此候选绑定的声轨文件内容已改变，不能借用其他声轨')
    return path


def _sha256_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
