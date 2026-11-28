"""Offline-first metadata defaults for osu!mania exports."""
from concurrent.futures import ThreadPoolExecutor, wait
import json
from pathlib import Path
import re
import subprocess
import unicodedata
from urllib.parse import urlencode
from urllib.request import urlopen

PRESETS = {
    "rice_stream": ("Rice Stream", "rice stream roll chordstream 切指 米"),
    "rice_jack": ("Rice Jack", "rice jack chordjack stamina 叠键 米"),
    "hybrid_adaptive": ("Hybrid", "hybrid rice LN 综合"),
    "ln_focus": ("LN Focus", "LN long note longnote noodle shield release coordination 协调 面条 放手 盾"),
    "torosu_remap": ("Torosu Remap", "remap torosu"),
}
TEXT_FIELDS = ("Title", "TitleUnicode", "Artist", "ArtistUnicode", "Creator", "Version", "Source", "Tags", "Background")
DIFFICULTY_FIELDS = ("OverallDifficulty", "HPDrainRate")
LANG_TAGS = {"zh": "中文 Chinese cpop Mandopop 华语", "ja": "Japanese J-Pop", "ko": "Korean K-Pop"}
GENERIC = {"unknown", "unknown artist", "unknown title", "untitled", "audio", "track"}

KANA = dict(zip(
    'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゔ',
    'a i u e o ka ki ku ke ko sa shi su se so ta chi tsu te to na ni nu ne no ha hi fu he ho ma mi mu me mo ya yu yo ra ri ru re ro wa o n ga gi gu ge go za ji zu ze zo da ji zu de do ba bi bu be bo pa pi pu pe po a i u e o vu'.split(),
))
DIGRAPHS = {k + y: stem + vowel for k, stem in zip('きぎしじちにひびぴみり',
    'ky gy sh j ch ny hy by py my ry'.split()) for y, vowel in zip('ゃゅょ', 'auo')}
DIGRAPHS.update({'てぃ': 'ti', 'でぃ': 'di', 'ふぁ': 'fa', 'ふぃ': 'fi', 'ふぇ': 'fe', 'ふぉ': 'fo',
    'うぃ': 'wi', 'うぇ': 'we', 'うぉ': 'wo', 'しぇ': 'she', 'じぇ': 'je', 'ちぇ': 'che',
    'ゔぁ': 'va', 'ゔぃ': 'vi', 'ゔぇ': 've', 'ゔぉ': 'vo'})
KO_INITIAL = 'g kk n d tt r m b pp s ss 0 j jj ch k t p h'.split()
KO_VOWEL = 'a ae ya yae eo e yeo ye o wa wae oe yo u wo we wi yu eu ui i'.split()
KO_FINAL = '0 k k k n n n t l k m l l l p l m p p t t ng t t k t p t'.split()


def _text(value, limit=200):
    return re.sub(r"[\x00-\x1f\x7f]", " ", value).strip()[:limit] if isinstance(value, str) else ""


def language(text):
    if re.search(r"[\u3041-\u309f\u30a1-\u30fa]", text):
        return "ja"
    if re.search(r"[가-힣]", text):
        return "ko"
    if re.search(r"[\u3400-\u9fff]", text):
        return "zh"
    return ""


def _hepburn(text):
    kana = ''.join(chr(ord(c) - 96) if 'ァ' <= c <= 'ヴ' else c for c in text)
    output = []
    while kana:
        c, rest = kana[0], kana[1:]
        word = DIGRAPHS.get(kana[:2])
        if word:
            rest = rest[1:]
        elif c == 'っ':
            following = DIGRAPHS.get(rest[:2], KANA.get(rest[:1], ''))
            word = 't' if following.startswith('ch') else following[:1]
        elif c == 'ん':
            following = KANA.get(rest[:1], '')
            word = "n'" if following and following[0] in 'aeiouy' else 'n'
        elif c == 'ー':
            word = next((v for v in reversed(output[-1]) if v in 'aeiou'), '') if output else ''
        elif c == '・':
            word = ' '
        else:
            word = KANA.get(c, c)
        output.append(word)
        kana = rest
    return ''.join(output)


def _hangul(text):
    output = []
    for char in text:
        if not '가' <= char <= '힣':
            output.append(char)
            continue
        initial, rest = divmod(ord(char) - 0xac00, 588)
        vowel, final = divmod(rest, 28)
        output.append((KO_INITIAL[initial] + KO_VOWEL[vowel] + KO_FINAL[final]).replace('0', ''))
    return ''.join(output)


def ascii_name(value, fallback, lang=None):
    value = _text(value)
    if value.isascii():
        return value or fallback
    original_lang = language(value)
    if original_lang == 'ja':
        value = _hepburn(unicodedata.normalize('NFKC', value)).title()
    elif original_lang == 'ko':
        value = _hangul(value).title()
    elif original_lang == 'zh' and lang != 'ja':
        from pypinyin import lazy_pinyin

        value = ' '.join(part.title() if part.isalpha() else part.strip()
                         for part in lazy_pinyin(value, errors=lambda chars: [chars]))
    ascii_value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    ascii_value = ' '.join(ascii_value.split())
    return ascii_value if any(c.isalnum() for c in ascii_value) else fallback


def _ffmetadata(stdout):
    fields = {}
    text = re.sub(r"\\\r?\n", " ", stdout)
    for line in text.splitlines():
        if line.startswith('['):
            break
        if line.startswith(('#', ';')) or '=' not in line:
            continue
        key, value = line.split('=', 1)
        fields[key.strip().lower()] = _text(re.sub(r"\\(.)", r"\1", value))
    return fields


def audio_metadata(path):
    from imageio_ffmpeg import get_ffmpeg_exe
    from .audio_io import run_hidden

    fields = {}
    try:
        raw = run_hidden([get_ffmpeg_exe(), '-v', 'error', '-i', str(path), '-f', 'ffmetadata', '-'],
                         capture_output=True, timeout=4, check=True).stdout
        fields = _ffmetadata(raw.decode('utf-8-sig', errors='replace'))
    except (OSError, RuntimeError, subprocess.SubprocessError):
        pass
    stem = Path(path).stem
    parts = re.split(r"\s+(?:-|–|—)\s+", stem, maxsplit=1)
    file_artist, file_title = parts if len(parts) == 2 else ('Unknown', stem)
    title = fields.get('title', '')
    artist = fields.get('artist', '')
    title = title if title and title.casefold() not in GENERIC else file_title
    artist = artist if artist and artist.casefold() not in GENERIC else file_artist
    lang = language(title + ' ' + artist)
    date = fields.get('date', '')
    return dict(Title=ascii_name(title, 'Untitled', lang), TitleUnicode=title,
                Artist=ascii_name(artist, 'Unknown', lang), ArtistUnicode=artist,
                Source=fields.get('album', ''), Date=date,
                Tags=next(iter(re.findall(r'\b(?:19|20)\d{2}\b', date)), ''))


def tags(*groups):
    unique = {}
    for group in groups:
        if isinstance(group, str):
            for word in group.split():
                unique.setdefault(word.casefold(), word)
    return ' '.join(unique.values())


def default_difficulty(notes, keys):
    holds = sum(note.hold for note in notes)
    return 8 if not notes or holds * 10 < len(notes) * 3 else 6 if keys <= 8 else 5


def override_metadata(payload):
    if not isinstance(payload, dict) or set(payload) - set(TEXT_FIELDS + DIFFICULTY_FIELDS):
        raise ValueError('Metadata 字段无效')
    result = {}
    for key, value in payload.items():
        if value is None:
            result[key] = None
        elif key in DIFFICULTY_FIELDS:
            if type(value) not in (int, float) or not 0 <= value <= 10:
                raise ValueError('OD / HP 必须在 0–10 之间')
            result[key] = value
        elif not isinstance(value, str) or len(value) > 200 or re.search(r'[\x00-\x1f\x7f]', value):
            raise ValueError(f'{key} 必须为不超过 200 字符的单行文字')
        elif key == 'Background' and (re.search(r'[<>:"/\\|?*]', value) or value.endswith((' ', '.'))):
            raise ValueError('背景图只接受安全文件名，不接受路径')
        elif key in ('Title', 'Artist') and not value.isascii():
            result[key] = ascii_name(value, 'Untitled' if key == 'Title' else 'Unknown')
        else:
            result[key] = value.strip()
    return result


def effective_metadata(state, variant=None, notes=(), keys=7, *, original=None, audio=None):
    """Resolve candidate defaults and explicit overrides without persisting derived values."""
    variant = variant or {}
    custom = variant.get('metadata') or {}
    settings = variant.get('settings') or state.get('settings') or {}
    selection = variant.get('selection') or {}
    preset = 'torosu_remap' if original is not None else variant.get('preset') or selection.get('preset')
    version, preset_tags = PRESETS.get(preset, PRESETS['hybrid_adaptive'])
    suffix = custom.get('version_suffix')
    if isinstance(suffix, str) and suffix.strip():
        version += ' ' + suffix.strip()
    if original is not None:
        from .domain import sections, values

        source = sections(original)
        result = values(source.get('Metadata', []))
        result.update({k: float(v) for k, v in values(source.get('Difficulty', [])).items()
                       if k in DIFFICULTY_FIELDS and re.fullmatch(r'\d+(?:\.\d+)?', v)})
        result['Version'] = f'{version} {keys}K'
        result['Tags'] = tags(result.get('Tags'), preset_tags, f'{keys}K')
        for line in source.get('Events', []):
            match = re.match(r'\s*0\s*,\s*0\s*,\s*"([^"]+)"', line)
            if match:
                result['Background'] = match[1]
                break
    else:
        result = dict(state.get('track_metadata') or {})
        title = result.get('TitleUnicode') or state.get('title') or 'Untitled'
        artist = result.get('ArtistUnicode') or state.get('artist') or 'Unknown'
        lang = language(title + ' ' + artist)
        if 'Title' not in result:
            result['Title'] = ascii_name(title, 'Untitled', lang)
        if 'Artist' not in result:
            result['Artist'] = ascii_name(artist, 'Unknown', lang)
        result.update(TitleUnicode=title, ArtistUnicode=artist, Creator=custom.get('creator') or 'ManiaLab',
                      Version=version, Background=Path(audio or state.get('audio') or 'audio.ogg').stem + '.png')
        ln_mode = settings.get('torosu_ln_mode') or selection.get('torosu_ln_mode')
        pinyin = ' '.join(ascii_name(text, '').lower() for text in (title, artist) if language(text) == 'zh') if lang == 'zh' else ''
        result['Tags'] = tags(preset_tags, 'FLN' if ln_mode == 'fln' else ln_mode if ln_mode == 'short_ln' else '',
                              f'{keys}K', LANG_TAGS.get(lang), pinyin, result.get('Tags'))
        result.update(dict.fromkeys(DIFFICULTY_FIELDS, default_difficulty(notes, keys)))
    result.update({key: value for key, value in (state.get('export_metadata') or {}).items()
                   if value is not None})
    return {field: result.get(field, '') for field in TEXT_FIELDS + DIFFICULTY_FIELDS}


def _itunes(query, country, timeout):
    url = 'https://itunes.apple.com/search?' + urlencode(dict(term=query, entity='song', country=country, limit=8))
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read(512 * 1024))['results']


_LOOKUP_TIMEOUT = 3.7
_LOOKUPS = ThreadPoolExecutor(max_workers=4, thread_name_prefix='metadata-lookup')


def online_metadata(fields):
    """Return only a matched catalog result; callers retain all local fallback values."""
    title, artist = (fields.get(key, '') for key in ('TitleUnicode', 'ArtistUnicode'))
    if not title:
        return {}
    artist = '' if artist.casefold() in GENERIC else artist
    query = ' '.join((title, artist)).strip()
    country = {'zh': 'tw', 'ja': 'jp', 'ko': 'kr'}.get(language(query), 'us')
    requests = {_LOOKUPS.submit(_itunes, query, c, _LOOKUP_TIMEOUT): c for c in dict.fromkeys((country, 'us'))}
    done, pending = wait(requests, timeout=_LOOKUP_TIMEOUT)
    for future in pending:
        future.cancel()
    results = {}
    for future in done:
        try:
            rows = future.result()
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if isinstance(rows, list):
            results[requests[future]] = [row for row in rows if isinstance(row, dict)]

    def canonical(value):
        return re.sub(r'\W+', '', unicodedata.normalize('NFKC', _text(value)).casefold())

    title_keys = {canonical(title), canonical(fields.get('Title'))} - {''}
    artist_keys = ({canonical(artist), canonical(fields.get('Artist'))} - {''}) if artist else set()
    matched = next((row for c in dict.fromkeys((country, 'us')) for row in results.get(c, [])
                    if canonical(row.get('trackName')) in title_keys
                    and (not artist_keys or canonical(row.get('artistName')) in artist_keys)), None)
    if not matched:
        return {}
    english = next((row for row in results.get('us', []) if matched.get('trackId')
                    and row.get('trackId') == matched['trackId']), matched)
    output = {}
    for key, catalog_key in (('Title', 'trackName'), ('Artist', 'artistName')):
        value = _text(english.get(catalog_key))
        if value and value.isascii():
            output[key] = value
    if source := _text(english.get('collectionName')):
        output['Source'] = source
    if extra := tags(_text(english.get('primaryGenreName')), _text(english.get('releaseDate'))[:4]):
        output['Tags'] = extra
    return output
