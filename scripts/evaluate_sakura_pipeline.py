"""Run the actual provided-source application analysis on the full author song.

Reference events/bookmarks are used only after analysis for diagnostics.
No known timing, source masks or per-section expression is supplied to analysis.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.arrange import parameters
from manialab.audio import analyze_deep
from manialab.domain import parse_osu,note_from
from manialab.expression import propose_expression
from manialab.provided_stems import prepare_provided_stems,ProvidedStemSeparator
from manialab.stems import VOICES
from manialab.structure import reference_marks
from manialab.timing import grid_report
from scripts.probe_mapper_adapter import sha,event_similarity
from scripts.probe_sakura_sources import match_source
from scripts.evaluate_timing_grid_support import reference_residuals


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--output',type=Path,default=Path('test-data/sakura-benchmark/provided-pipeline'))
    cli.add_argument('--score-only',action='store_true',help='Score saved analysis without decoding or running models again')
    args=cli.parse_args();root=args.root;output=args.output;output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    source_paths={v:root/f'bundled/{v}.mp3' for v in ('mix',*VOICES)}
    expected={a['id']:a['sha256'] for a in manifest['audio']}
    if any(sha(path)!=expected[v] for v,path in source_paths.items()):raise ValueError('Source identity mismatch')
    last=[0.]
    def progress(p,message):
        if time.monotonic()-last[0]>15:
            print(json.dumps(dict(progress=p,message=message),ensure_ascii=False),flush=True);last[0]=time.monotonic()
    if args.score_only:
        analysis=json.loads((output/'analysis.json').read_text(encoding='utf-8'))
        separation=analysis['separation']
        if (separation.get('source')!='provided' or separation['audio_sha256']!=expected['mix'] or
                {v:s['sha256'] for v,s in separation['sources'].items()}!={v:expected[v] for v in VOICES}):
            raise ValueError('Saved analysis belongs to different source audio')
        plan=json.loads((output/'expression.json').read_text(encoding='utf-8'))
    else:
        separation=prepare_provided_stems(source_paths['mix'],{v:source_paths[v] for v in VOICES},output,progress)
        analysis=analyze_deep(source_paths['mix'],output,Path('assets/basic-pitch.onnx'),parameters(),
            ProvidedStemSeparator(separation),progress)
        plan=propose_expression(analysis,parameters())
        (output/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False),encoding='utf-8')
        (output/'expression.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    text=(root/'reference.osu').read_text(encoding='utf-8-sig')
    human=parse_osu(text)['notes'];predicted=[note_from(n) for n in analysis['candidates']]
    tempo=analysis['tempo'];offset=round(tempo['offset']-22.5)
    aligned=grid_report(human,dict(bpm=tempo['bpm'],offset=offset,division=tempo['division']))
    books=reference_marks(text)['bookmarks']
    boundaries=analysis['structure']['boundaries'][1:-1]
    comparisons={}
    for voice,lanes in dict(drums=range(4),piano=range(4,6),other=range(6,10)).items():
        sources=[voice] if voice!='other' else ['other','bass','guitar']
        comparisons[voice]=match_source([n for n in human if n.lane in lanes],
            [n for n in predicted if n.voice in sources],22.5)
    report=dict(source_hashes=expected,reference_sha256=sha(root/'reference.osu'),
        analysis_revision=analysis['revision'],engine=analysis['engine'],elapsed_seconds=analysis['elapsed_seconds'],
        timing=dict(bpm=tempo['bpm'],audio_offset=tempo['offset'],chart_offset=offset,division=tempo['division'],
            stable=tempo.get('stable'),evidence_source=tempo.get('evidence_source'),
            authored_events_on_automatically_chosen_grid=aligned,
            authored_grid_residuals=reference_residuals(human,tempo,22.5),
            notice='Grid diagnostic uses the final detected tempo with a fixed 22.5ms chart lead; no reference fitting.'),
        source_onsets=comparisons,
        sections=dict(reference_bookmark_count=len(books),detected_boundary_count=len(boundaries),
            within_1000ms=event_similarity([(t+22.5,) for t in books],[(t,) for t in boundaries],[1000]),
            notice='Author bookmarks are one interpretation, not exhaustive acoustic boundary truth.'),
        expression=[dict(start_ms=s['start_ms'],end_ms=s['end_ms'],main_voice=s['main_voice'],voices=s['voices']) for s in plan['sections']],
        limits=['Author supplied Nova stems; separation quality not evaluated.',
            'Default all-source plan may choose vocals, while author Reference deliberately excludes vocals.',
            'No application defaults were tuned from this run. Author similarity is diagnostic, not acceptance.'])
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(timing=report['timing'],sections=report['sections']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
