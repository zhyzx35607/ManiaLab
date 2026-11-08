"""Prepare paired-audio expression examples with community-compatible splits."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu
from manialab.stem_release_evidence import StemReleaseContext, STEM_FEATURES, EVIDENCE_REVISION
from scripts.paired_stem_inputs import prepare_observations
from scripts.probe_mapper_adapter import sha
from scripts.train_anchor_gestures import add_examples


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--pairs', type=Path, default=Path('test-data/mapper-training/mmrs/manifest.json'))
    cli.add_argument('--community', type=Path, default=Path('test-data/corpus-expanded/release-dataset-v2/manifest.json'))
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists(): raise ValueError('Preserve prior paired datasets')
    community = json.loads(args.community.read_text(encoding='utf-8'))
    index = {int(r['id']):r for r in community['records']}
    pairs = json.loads(args.pairs.read_text(encoding='utf-8'))['records']
    prepared = []; excluded = []
    args.output.mkdir(parents=True)
    for row in pairs:
        canonical = index.get(row['beatmap_id'])
        if canonical is None:
            excluded.append(dict(id=row['beatmap_id'], reason='No valid community expression group')); continue
        if canonical['sha256'] != row['chart_sha256']:
            raise ValueError('Pair and community chart bytes disagree')
        try:
            output = args.output/'observations'/str(row['beatmap_id'])
            evidence, identity = prepare_observations(row, output)
            chart = parse_osu(Path(row['chart']).read_text(encoding='utf-8-sig'))
            # Clock validity is checked before the record is accepted.
            StemReleaseContext([replace(n, end=n.start, lane=0, voice='mix', raw='') for n in chart['notes']],
                chart['keys'], chart['timing'], .35,
                evidence, identity['audio_start_ms'])
        except (FileNotFoundError, KeyError, ValueError) as error:
            excluded.append(dict(id=row['beatmap_id'], reason=str(error))); continue
        prepared.append(row|dict(old_split=row['split'], split=canonical['split'],
            group=canonical['group'], observations=str(output), identity=identity))
    groups = Counter((r['keys'], r['group']) for r in prepared)
    key_groups = Counter((k, next(r['split'] for r in prepared if r['group']==g)) for k,g in groups)
    arrays = defaultdict(lambda:dict(x=[], y=[], lengths=[], weights=[])); records=[]
    from scripts.paired_stem_inputs import load_observations
    for row in prepared:
        chart = parse_osu(Path(row['chart']).read_text(encoding='utf-8-sig'))
        notes = [replace(n, end=n.start, lane=0, voice='mix', raw='') for n in chart['notes']]
        ratio = sum(n.hold for n in chart['notes'])/len(notes)
        evidence = load_observations(Path(row['observations']), row['identity'])
        context = StemReleaseContext(notes, chart['keys'], chart['timing'], ratio, evidence, row['identity']['audio_start_ms'])
        start, end = row['identity']['supported_bounds']
        eligible = [i for i in range(len(context.times)) if context.contained(i, start, end)]
        if not eligible:
            excluded.append(dict(id=row['beatmap_id'], reason='No complete audio/candidate context')); continue
        indices = sorted(np.random.default_rng(row['beatmap_id']).choice(eligible, min(128,len(eligible)), replace=False).tolist())
        weight = 1/(len(indices)*groups[(row['keys'],row['group'])]*key_groups[(row['keys'],row['split'])])
        counts = add_examples(context, chart['notes'], indices, arrays[row['split']], weight)
        records.append(row|dict(rows=len(indices), labels=counts,
            score_bounds=[int(context.times[eligible[0]]),int(context.times[eligible[-1]])+1]))
        print(json.dumps(dict(prepared=row['beatmap_id'], split=row['split'], rows=len(indices))), flush=True)
    files={}
    for split, data in arrays.items():
        path=args.output/(split+'.npz')
        np.savez_compressed(path, x=np.concatenate(data['x']), y=np.concatenate(data['y']),
            lengths=np.asarray(data['lengths']), weights=np.asarray(data['weights']))
        files[split]=dict(sha256=sha(path), rows=len(data['lengths']), options=sum(data['lengths']))
    if set(files) != {'train','validation','test'}: raise ValueError('Missing an independent split')
    manifest=dict(revision=StemReleaseContext.revision, features=STEM_FEATURES, source='paired_stems',
        scope='Authored attacks/Timing/counts with frozen four-stem acoustic observations; development song-group split.',
        evidence_revision=EVIDENCE_REVISION, records=records, exclusions=excluded, files=files,
        pretraining_dataset_sha256=sha(args.community), pair_inventory_sha256=sha(args.pairs),
        feature_sha256=sha('manialab/anchor_gestures.py'), release_feature_sha256=sha('manialab/release_candidates.py'),
        stem_feature_sha256=sha('manialab/stem_release_evidence.py'), loader_sha256=sha('scripts/paired_stem_inputs.py'),
        script_sha256=sha(__file__), protocol_sha256=sha('docs/paired-stem-release-protocol.md'))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2), encoding='utf-8')
    print(json.dumps(dict(charts=len(records), files=files, exclusions=excluded)),flush=True)


if __name__=='__main__': main()
