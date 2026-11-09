"""Make fixed original-audio evidence cards; never classify or modify notes."""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import sha256
from scripts.piano_band_evidence import analyze_window

EXPECTED_READINESS = 'f9e8a983055ff6c097fa18b6f78052be2c6c17f5ff035bcb57a2ddf5fcc606d2'
INPUT_KEYS = ('audio', 'activations', 'events', 'matches', 'harmonic')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def plot_card(event, lower, metadata, arrays, samples, rate, activations, name, path):
    import numpy as np
    import matplotlib.pyplot as plt

    t0 = event['onset']
    begin, end = t0-.25, t0+.45
    fig, axes = plt.subplots(6, 1, figsize=(13, 12), sharex=True,
                             gridspec_kw={'height_ratios': [.6, 1.3, 1.3, 1.3, 1, 1]})
    left, right = max(0, round(begin*rate)), min(len(samples), round(end*rate))
    axes[0].plot(np.arange(left, right)/rate, samples[left:right], lw=.4)
    axes[0].set_title(f'{name} | target MIDI {event["pitch"]} | raw PCM time; no note edits')
    axes[0].set_ylabel('PCM')
    low_pitch = min([event['pitch']]+[item['pitch'] for item in lower])-3
    high_pitch = min(135, event['pitch']+38)
    common_max = max(float(arrays[f'magnitude_{size}'].max()) for size in (512, 2048, 8192))
    for panel, size in zip(axes[1:4], (512, 2048, 8192)):
        f, t, mag = (arrays[f'{key}_{size}'] for key in ('frequencies', 'times', 'magnitude'))
        valid = f > 0
        db = 20*np.log10(np.maximum(mag[valid], 1e-12)/max(common_max, 1e-12))
        panel.pcolormesh(t, 69+12*np.log2(f[valid]/440), db,
                         shading='auto', cmap='magma', vmin=-75, vmax=0)
        panel.set_ylim(low_pitch, high_pitch)
        panel.set_ylabel(f'{size} FFT\nMIDI frequency')
        panel.set_title(f'{size/rate*1000:.2f}ms window; shared dB scale; hop is not attack resolution',
                        fontsize=8, loc='left')
        panel.scatter([t0], [event['pitch']], marker='x', color='cyan')
        for item in lower:
            panel.scatter([item['onset']], [item['pitch']], marker='+', color='lime', s=25)
    for partial in range(1, 9):
        key = f'band_2048_{event["pitch"]}_{partial}'
        if key in arrays:
            axes[4].plot(arrays['times_2048'], 20*np.log10(np.maximum(arrays[key], 1e-12)),
                         label=f'target partial {partial}', lw=.8)
    for pitch in sorted({item['pitch'] for item in lower}):
        key = f'band_2048_{pitch}_1'
        axes[4].plot(arrays['times_2048'], 20*np.log10(np.maximum(arrays[key], 1e-12)),
                     '--', label=f'lower {pitch}', lw=.9)
    axes[4].set_ylabel('Band dB\nabsolute PCM scale')
    axes[4].set_ylim(-100, 0)
    axes[4].legend(ncol=4, fontsize=7)
    clock = np.arange(len(activations['frame_output']))/100
    take = (clock >= begin) & (clock <= end)
    for key, label in (('reg_onset_output', 'onset'), ('frame_output', 'frame'),
                       ('reg_offset_output', 'offset')):
        axes[5].plot(clock[take], activations[key][take, event['pitch']-21], label=label)
    axes[5].set_ylim(0, 1.05)
    axes[5].set_ylabel('Saved activations')
    axes[5].set_xlabel('Original audio time / seconds')
    axes[5].legend(ncol=3)
    for panel in axes:
        panel.set_xlim(begin, end)
        panel.axvline(t0, color='#0891b2', lw=.8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summarize(records):
    import numpy as np
    output = {}
    for matched in (True, False):
        for coincident in (True, False):
            selected = [r for r in records if r['matched_existing_onset'] == matched
                        and bool(r['lower_partners']) == coincident]
            group = ('matched' if matched else 'unmatched')+'_'+('coincident' if coincident else 'no_coincidence')
            windows = {}
            for size in (512, 2048, 8192):
                fundamentals = [next(b for b in r['measurement']['windows'][str(size)]['bands']
                                     if b['pitch'] == r['pitch'] and b['partial'] == 1) for r in selected]
                values = [b['attack_pre_db'] for b in fundamentals if b['attack_pre_db'] is not None]
                windows[str(size)] = dict(
                    measurable_attack_pre=len(values),
                    attack_pre_db_quantiles=[float(v) for v in np.quantile(values, [0, .25, .5, .75, 1])] if values else None,
                    attack_full_support_frames=sum(b['segments']['attack']['count'] for b in fundamentals))
            output[group] = dict(count=len(selected), windows=windows)
    return output


def gallery(records, path):
    cards = []
    for r in records:
        stem = r['artifact_stem']
        pairing = '已有 MIDI 起音匹配' if r['matched_existing_onset'] else '当前容差下未匹配（不等于无声）'
        rows = []
        for size in (512, 2048, 8192):
            b = next(b for b in r['measurement']['windows'][str(size)]['bands']
                     if b['pitch'] == r['pitch'] and b['partial'] == 1)
            db = '不可测' if b['attack_pre_db'] is None else f'{b["attack_pre_db"]:.2f} dB'
            rows.append(f'<tr><td>{size}</td><td>{b["segments"]["pre"]["count"]}</td>'
                        f'<td>{b["segments"]["attack"]["count"]}</td><td>{db}</td></tr>')
        cards.append(f'<article id="{stem}" data-coincident="{int(bool(r["lower_partners"]))}" '
                     f'data-matched="{int(r["matched_existing_onset"])}">'
                     f'<h2>{html.escape(r["case"])} #{r["event_index"]} · MIDI {r["pitch"]} · '
                     f'{r["onset_seconds"]:.6f}s</h2><p>{pairing}；低音伙伴 {len(r["lower_partners"])} 个。</p>'
                     f'<audio controls preload="none" src="{stem}.wav"></audio>'
                     '<p>播放为原声起音前 250ms 至后 450ms，保留原始音量。起音约在片段 0.25s。</p>'
                     '<table><tr><th>FFT</th><th>pre完整窗</th><th>attack完整窗</th><th>基频attack/pre</th></tr>'
                     +''.join(rows)+f'</table><p>其他候选起音 {len(r["nearby_attacks"])} 个；'
                     '共享频带、比值和持续能量均不判定真音或 LN 音尾。</p>'
                     f'<details><summary>频谱、包络、模型激活与完整测量</summary>'
                     f'<img loading="lazy" src="{stem}.png" alt="Original PCM and spectral evidence">'
                     f'<p><a href="{stem}.json">测量 JSON</a></p></details></article>')
    prefix = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>原声采音证据</title>
<style>body{font:16px/1.65 system-ui;background:#f4f5f8;color:#162435;margin:24px auto;max-width:1180px;padding:0 16px}article{background:white;padding:20px;margin:20px 0;border-radius:12px}img{max-width:100%}td,th{padding:4px 18px;text-align:left}audio{width:100%}nav{position:sticky;top:0;background:#f4f5f8;padding:12px;border-bottom:1px solid #ccd}select{padding:6px}</style>
<h1>先看声音：47 个持续激活不足的候选</h1>
<p>固定范围的研究对照，不是新模型或质量验收。完整保留 10 个已有匹配与 37 个未匹配。
12 个频率重合案例中也有 3 个匹配，不能自动当泛音删去。助手未直接试听；播放器供人工复核。</p>
<p>FFT 只观察有限频带；长窗会模糊时间。只有完整窗落在统计区间内才参与均值，8192 的短区间显示不可测。
所有起音保持原时钟，没有制谱提前量或吸附拍线。音尾不作为 LN 放手依据。</p>
<nav><label>范围 <select id="coincident"><option value="">全部47例</option value="1">12个频率重合例</option><option value="0">其余35例</option></select></label>
<label> 起音配对 <select id="matched"><option value="">全部</option><option value="1">已有匹配</option><option value="0">未匹配</option></select></label>
<span id="count"></span></nav>'''
    script = '''<script>function filter(){let n=0;for(const card of document.querySelectorAll('article')){let show=true;for(const k of ['coincident','matched']){const v=document.getElementById(k).value;if(v!==''&&card.dataset[k]!==v)show=false;}card.hidden=!show;if(show)n++;}document.getElementById('count').textContent=' 显示 '+n+' 例';}for(const k of ['coincident','matched'])document.getElementById(k).addEventListener('change',filter);filter();</script></html>'''
    path.write_text(prefix+''.join(cards)+script, encoding='utf-8')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--plot-dependencies', type=Path, required=True)
    args = cli.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Use a new local test-data output directory')
    sys.path.insert(0, str(args.plot_dependencies.resolve()))
    import numpy as np
    import matplotlib
    import scipy
    matplotlib.use('Agg')
    from scipy.io import wavfile

    readiness_path = ROOT/'test-data/piano-acoustic-next-step-v1/readiness.json'
    if sha256(readiness_path) != EXPECTED_READINESS:
        raise ValueError('Frozen readiness changed')
    readiness = read(readiness_path)
    sources = {'readiness': readiness_path, 'runner': Path(__file__),
               'helper': ROOT/'scripts/piano_band_evidence.py',
               'protocol': ROOT/'docs/piano-band-evidence-protocol.md'}
    identities = {key: sha256(path) for key, path in sources.items()}
    inputs = {}
    for case in readiness['cases']:
        inputs[case['name']] = {key: case['inputs'][key] for key in INPUT_KEYS}
        for item in inputs[case['name']].values():
            if sha256(Path(item['path'])) != item['sha256']:
                raise ValueError('Changed original input: '+item['path'])
    output.mkdir(parents=True)
    for key, path in sources.items():
        shutil.copyfile(path, output/(key+'-snapshot'+path.suffix))
    records, targets, matched_count, coincident_count = [], 0, 0, 0
    for case in readiness['cases']:
        paths = {key: Path(item['path']) for key, item in inputs[case['name']].items()}
        events = read(paths['events'])['events']
        matches = read(paths['matches'])
        paired = {item['event_index'] for item in matches['original_pairs']}
        relations = {item['event_index']: item for item in read(paths['harmonic'])['events']}
        indices = [i for i, event in enumerate(events) if 0 <= event['onset'] < 60
                   and event['release_reason'] == 'frame_inactive']
        if indices != case['target_event_indices'] or len(paired) != len(matches['original_pairs']):
            raise ValueError('Frozen target/pair identities changed')
        if [i for i in indices if i in paired] != case['matched_target_event_indices']:
            raise ValueError('Frozen target matches changed')
        if [i for i in indices if relations[i]['lower_partners']] != case['coincident_target_event_indices']:
            raise ValueError('Frozen target relations changed')
        if not indices:
            continue
        rate, raw = wavfile.read(paths['audio'])
        if rate != 22050 or raw.dtype != np.int16 or raw.ndim != 1 or len(raw) < rate*70:
            raise ValueError('Unexpected original PCM')
        raw = raw[:rate*70]
        samples = raw.astype(np.float64)/32768
        with np.load(paths['activations'], allow_pickle=False) as loaded:
            activations = {key: loaded[key] for key in ('reg_onset_output', 'frame_output', 'reg_offset_output')}
        if any(a.shape != (7000, 88) or not np.isfinite(a).all() for a in activations.values()):
            raise ValueError('Unexpected frozen activation clock')
        for index in indices:
            event = events[index]
            relation = relations[index]
            lower = [dict(event_index=p['event_index'], **events[p['event_index']])
                     for p in relation['lower_partners']]
            measurement, arrays = analyze_window(samples, rate, event['onset'], event['pitch'],
                                                  [item['pitch'] for item in lower])
            stem = f'{case["name"]}-{index:04d}'
            left = max(0, round((event['onset']-.25)*rate))
            right = min(len(raw), round((event['onset']+.45)*rate))
            wavfile.write(output/(stem+'.wav'), rate, raw[left:right])
            clock = np.arange(7000)/100
            visible = (clock >= event['onset']-.25) & (clock <= event['onset']+.45)
            arrays['activation_times'] = clock[visible]
            for key, values in activations.items():
                arrays[key] = values[visible, event['pitch']-21]
            np.savez_compressed(output/(stem+'.npz'), **arrays)
            plot_card(event, lower, measurement, arrays, samples, rate, activations, stem, output/(stem+'.png'))
            # Existing pairing is attached only after the label-free acoustic measurement.
            row = dict(case=case['name'], event_index=index, pitch=event['pitch'],
                       onset_seconds=event['onset'], matched_existing_onset=index in paired,
                       output_interval_ms=1000*(event['offset']-event['onset']),
                       lower_partners=[dict(**p, pitch=events[p['event_index']]['pitch'],
                                            matched_existing_onset=p['event_index'] in paired)
                                       for p in relation['lower_partners']],
                       nearby_attacks=[dict(event_index=i, pitch=e['pitch'], onset_seconds=e['onset'],
                                            same_pitch=e['pitch'] == event['pitch'])
                                       for i, e in enumerate(events) if i != index
                                       and event['onset']-.18 <= e['onset'] <= event['onset']+.35],
                       clip=dict(first_sample=left, stop_sample=right, sample_rate=rate),
                       measurement=measurement, artifact_stem=stem,
                       independent_attack_established=False, note_filter_applied=False)
            save(output/(stem+'.json'), row)
            records.append(row)
            targets += 1
            matched_count += index in paired
            coincident_count += bool(relation['lower_partners'])
        print(f'{case["name"]}: {len(indices)} evidence cards', flush=True)
    if (targets, matched_count, coincident_count) != (47, 10, 12):
        raise ValueError('Cohort totals changed')
    if sum(r['matched_existing_onset'] and bool(r['lower_partners']) for r in records) != 3:
        raise ValueError('Coincident matches changed')
    for case_inputs in inputs.values():
        for item in case_inputs.values():
            if sha256(Path(item['path'])) != item['sha256']:
                raise ValueError('Input changed while processing')
    if any(sha256(path) != identities[key] for key, path in sources.items()):
        raise ValueError('Code or protocol changed while processing')
    gallery(records, output/'index.html')
    report = dict(identity=identities, inputs=inputs,
                  environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                   matplotlib=matplotlib.__version__),
                  targets=targets, matched=matched_count,
                  coincident=coincident_count, groups=summarize(records),
                  selected_plot_review=['bartok-0461', 'rachmaninoff-0678', 'bartok-0107'],
                  direct_listening_performed=False, model_rerun=False, labels_rematched=False,
                  notes_changed=False, application_enabled=False, musical_acceptance=False,
                  artifacts_sha256={p.name: sha256(p) for p in output.iterdir()})
    save(output/'report.json', report)
    print(json.dumps(report['groups'], allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
