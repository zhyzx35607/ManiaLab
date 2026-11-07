"""Exploratory song-balanced onset preference; not an application model.

Train only on the pinned development pairs. Evaluate known-reference-grid and
automatic-grid paths separately. Outputs and research weights stay local.
"""
import argparse,json,sys,hashlib
from pathlib import Path
from collections import defaultdict
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import note_from,parse_osu
from manialab.arrange import parameters,select_notes
from manialab.timing import chart_clock_notes,align_candidates
from manialab.voices import apply_phrase_evidence
from scripts.evaluate_audio import match_attacks,durations,LOADED_SOURCE
from scripts.audit_snapshot import assert_snapshot

VOICES=('drums','bass','vocals','other')
FEATURES=[f'{v}_{f}' for v in VOICES for f in ('count','max_strength','sum_strength','hold_share','duration','alignment')]+[
 'strong_beat','half_beat','offbeat','beat_seconds','rows_nearby','previous_gap','next_gap','density']

def dataset(folder, automatic=False, invariant=False):
 audit=json.loads((folder/'audit.json').read_text(encoding='utf-8'))
 if not automatic and not audit['reference']['constant_grid']:return None
 assert_snapshot(audit['source_fingerprint']['files'], LOADED_SOURCE['files'], Path(__file__).resolve().parents[1])
 analysis=json.loads((folder/'deep/analysis.json').read_text(encoding='utf-8'))
 if automatic and not analysis['tempo']['stable']:return None
 reference=parse_osu((folder/'reference.osu').read_text(encoding='utf-8-sig'))
 prefs=parameters({'keys':reference['keys']})
 raw,_=apply_phrase_evidence([note_from(n) for n in analysis['candidates']], [note_from(n) for n in analysis.get('phrase_candidates',[])],prefs['phrase_ln'])
 if automatic:
  tempo=analysis['tempo']|{'offset':int(np.floor(analysis['tempo']['offset']-22.5+.5))}
  offset,beat=tempo['offset'],60000/tempo['bpm']
 else:
  offset,beat=audit['reference']['timing_points'][0]
  tempo=dict(bpm=60000/beat,offset=offset,division=4,stable=True)
 aligned,_=align_candidates(chart_clock_notes(raw,22.5),tempo)
 groups=defaultdict(list)
 for n in aligned:groups[n.start].append(n)
 times=np.array(sorted(groups));rows=[]
 for index,time in enumerate(times):
  row=groups[int(time)];feature=[]
  for voice in VOICES:
   local=[n for n in row if n.voice==voice]
   feature.extend([min(4,len(local))/4,max((n.strength for n in local),default=0),sum(n.strength for n in local)/4,
    sum(n.end-n.start>=240 for n in local)/max(1,len(local)),
    np.log1p(max((n.end-n.start for n in local),default=0)/500),
    sum(abs(n.start-n.evidence_start)<=20 for n in local)/max(1,len(local))])
  tick=round((time-offset)/(beat/4))%4
  feature.extend([tick==0,tick==2,tick in (1,3),beat/1000,
   (np.searchsorted(times,time+1000)-np.searchsorted(times,time-1000))/16,
   min(4,(time-times[index-1])/beat) if index else 4,
   min(4,(times[index+1]-time)/beat) if index+1<len(times) else 4,prefs['density']/12])
  rows.append(feature)
 heads=np.array(sorted({n.start for n in reference['notes']}))
 where=np.searchsorted(heads,times)
 distance=np.minimum(abs(times-heads[np.clip(where,0,len(heads)-1)]),abs(times-heads[np.clip(where-1,0,len(heads)-1)]))
 features=np.asarray(rows,float)
 if invariant:
  gaps=np.full((len(times),2),4.)
  if len(times)>1:
   gaps[1:,0]=np.minimum(4,np.diff(times)/1000)
   gaps[:-1,1]=np.minimum(4,np.diff(times)/1000)
  features=np.column_stack([features[:,:24],features[:,28],gaps,features[:,31]])
 return dict(id=audit['beatmap_id'],folder=str(folder),keys=reference['keys'],x=features,y=(distance<=20).astype(float),
  aligned=aligned,times=times,reference=reference['notes'],prefs=prefs,tempo=tempo)

def fit(items):
 x=np.concatenate([r['x'] for r in items]);y=np.concatenate([r['y'] for r in items])
 weights=np.concatenate([np.full(len(r['y']),1/len(r['y'])/len(items)) for r in items])
 mean=np.average(x,axis=0,weights=weights);scale=np.maximum(.15,np.sqrt(np.average((x-mean)**2,axis=0,weights=weights)))
 z=np.column_stack([np.clip((x-mean)/scale,-5,5),np.ones(len(x))]);reg=.025
 def loss(w):
  margin=z@w
  return float(np.sum(weights*(np.logaddexp(0,margin)-y*margin))+.5*reg*np.sum(w[:-1]**2)),z.T@(weights*(expit(margin)-y))+np.r_[reg*w[:-1],0]
 result=minimize(loss,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':200})
 if not result.success:raise ValueError(result.message)
 return dict(mean=mean.tolist(),scale=scale.tolist(),weights=result.x.tolist())

def probability(item,model):
 z=np.column_stack([np.clip((item['x']-model['mean'])/model['scale'],-5,5),np.ones(len(item['x']))])
 return expit(z@model['weights'])

def compare(item,model):
 probabilities=probability(item,model);lookup=dict(zip(item['times'],probabilities));results={}
 for name,strength in [('baseline',0),('probe',.35)]:
  notes=[replace(n,strength=n.strength*(1+strength*(2*lookup[n.start]-1))) for n in item['aligned']]
  chosen,selection=select_notes(notes,item['prefs'])
  results[name]=dict(match=match_attacks([n.start for n in chosen],[n.start for n in item['reference']],20),holds=durations(chosen),
   attack_optimal=selection['attack_optimal'],voices=selection['selected_voices'])
 y=item['y'];ranks=rankdata(probabilities);p=y.sum();n=len(y)-p
 auc=float((np.sum(ranks*y)-p*(p+1)/2)/(p*n)) if p and n else None
 return dict(id=item['id'],keys=item['keys'],rows=len(y),positive=float(y.mean()),auc=auc,results=results)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--train',action='store_true');parser.add_argument('--evaluate',type=Path)
 parser.add_argument('--invariant',action='store_true',help='Omit strong/weak beat labels; use absolute time gaps')
 parser.add_argument('--automatic',action='store_true',help='Evaluate only the unassisted stable automatic timing path')
 parser.add_argument('--model',type=Path,default=Path('test-data/onset-preference-research.json'))
 parser.add_argument('--output',type=Path,default=Path('test-data/onset-preference-evaluation.json'))
 args=parser.parse_args();target=args.model
 if args.train and args.automatic:parser.error('Training protocol uses reference grids; --automatic is evaluation-only')
 if args.train:
  items=[]
  for group in ('audio-audit','audio-holdout'):
   for p in sorted(Path('test-data',group).glob('*/audit.json')):
    item=dataset(p.parent,invariant=args.invariant)
    if item:items.append(item)
  for r in items:r['fold']=int(hashlib.sha256(str(r['id']).encode()).hexdigest()[:8],16)%5
  evaluations=[]
  for fold in range(5):
   train=[r for r in items if r['fold']!=fold];model=fit(train)
   for r in [r for r in items if r['fold']==fold]:
    row=compare(r,model)|{'fold':fold};evaluations.append(row);print(json.dumps(row),flush=True)
  final=fit(items)
  features=FEATURES[:24]+['rows_nearby','previous_gap_seconds','next_gap_seconds','density'] if args.invariant else FEATURES
  target.parent.mkdir(parents=True,exist_ok=True)
  target.write_text(json.dumps(dict(model=final,features=features,invariant=args.invariant,training_ids=[r['id'] for r in items],
   source=LOADED_SOURCE,regularization=.025,strength=.35,notice='Exploratory human onset-row preference; no exhaustive sound labels, not enabled in application',cross_validation=evaluations),indent=2),encoding='utf-8')
 if args.evaluate:
  saved=json.loads(target.read_text(encoding='utf-8'));rows=[];skipped=[]
  invariant=saved.get('invariant',len(saved['features'])==28)
  for p in sorted(args.evaluate.glob('*/audit.json')):
   item=dataset(p.parent,automatic=args.automatic,invariant=invariant)
   if not item:skipped.append(p.parent.name);continue
   if item['id'] in saved['training_ids']:raise ValueError('Training song included in evaluation')
   row=compare(item,saved['model'])|{'bpm':item['tempo']['bpm'],'offset':item['tempo']['offset'],'division':item['tempo'].get('division',4)}
   rows.append(row);print(json.dumps(row),flush=True)
  args.output.parent.mkdir(parents=True,exist_ok=True)
  args.output.write_text(json.dumps(dict(automatic=args.automatic,invariant=invariant,rows=rows,skipped=skipped),indent=2),encoding='utf-8')

if __name__=='__main__':main()
