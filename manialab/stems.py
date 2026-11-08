"""Names and separation profiles shared by analysis, storage and the UI API."""
VOICES = {'drums':'鼓 / 打击', 'bass':'贝斯', 'vocals':'人声',
          'other':'其余伴奏', 'guitar':'吉他', 'piano':'钢琴'}
PROFILES = {
    'four': dict(directory='stems', filename='hdemucs_high_trained.pt',
        sha256='a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816',
        voices=('drums','bass','vocals','other'), label='四声部 · Hybrid Demucs'),
    'six': dict(directory='stems-six', filename='htdemucs_6s.th',
        sha256='34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd',
        voices=('drums','bass','vocals','other','guitar','piano'), label='六声部 · HTDemucs'),
}


def profile_for(value):
    if value not in PROFILES:
        raise ValueError('分离方案无效')
    return PROFILES[value]


def stem_path(project, analysis, voice, extension='wav'):
    separation = (analysis or {}).get('separation',{})
    profile = profile_for(separation.get('profile','four'))
    if voice not in profile['voices']:
        raise ValueError('当前分析没有这个声部，请切换分离方案并重新分析')
    return project / profile['directory'] / f'{voice}.{extension}'
