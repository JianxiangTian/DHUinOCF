"""Raw, matched-duration window-count and spectrum diagnostics.

Never remove or normalize by the instantaneous channel population.
"""
from pathlib import Path
import json,csv,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'open_v2'/'results'
RUNS=ROOT/'runs'
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
SEEDS=(31,62,93)
def read(prefix):
 m=json.loads(Path(str(prefix)+'_meta.json').read_text())
 a=np.fromfile(str(prefix)+'.bin',dtype='<f8').reshape(-1,m['columns'])
 assert len(a)==m['frames'] and np.isfinite(a).all()
 assert m['max_control_balance_error']==m['total_balance_error']==0
 return a,m
def calc(arrays):
 d=np.concatenate(arrays,axis=0)
 n=d[:,2]
 count=d[:,160:280].reshape(-1,5,24)[:,4,:] # 64 um full-width windows
 cmean=count.mean(0)
 v=np.mean((count-cmean)**2)
 den=cmean.mean()
 ncmean=n.mean();ncvar=n.var()
 cov=np.mean((count-cmean)*(n-ncmean)[:,None],axis=0)
 common=np.mean(cov**2/ncvar)/den
 residual=v/den-common
 amp=d[:,95]+1j*d[:,96]   # central rectangular q1
 s=np.mean(abs(amp-amp.mean())**2)/d[:,94].mean()
 return dict(frames=len(d),mean_Nc=float(ncmean),channel_Fano=float(ncvar/ncmean),
             Fano64=float(v/den),common_Fano64=float(common),
             residual_Fano64=float(residual),common_fraction=float(common/(v/den)),
             Srect1=float(s))
def record(stage,condition,seed,block,arrays):
 return dict(stage=stage,condition=condition,seed=seed,block=block,**calc(arrays))
def writecsv(path,rows):
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 original={}
 for condition in ['dt020','dt010']:
  typ='open_rep_on' if condition=='dt020' else 'open_rep_half_dt'
  original[condition]=[read(BASE/f'{typ}_{seed}')[0] for seed in SEEDS]
 matched={}
 for condition in ['dt020','dt010']:
  matched[condition]=[]
  for seed in SEEDS:
   pref=BASE/f'open_rep_on_long_{seed}' if condition=='dt020' else RUNS/f'open_rep_half_dt_long_{seed}'
   d,m=read(pref)
   assert m['burn']==3600 and m['duration']==18000 and m['frames']==3600
   assert m['dt']==(.02 if condition=='dt020' else .01)
   matched[condition].append(d)
 rows=[]
 for stage,series in [('original6000',original),('matched18000',matched)]:
  for condition,arrays in series.items():
   rows.append(record(stage,condition,'pooled','all',arrays))
   for seed,a in zip(SEEDS,arrays):rows.append(record(stage,condition,seed,'all',[a]))
   if stage=='matched18000':
    for block in range(3):
     starts=[a[block*1200:(block+1)*1200] for a in arrays]
     rows.append(record(stage,condition,'pooled',block+1,starts))
     for seed,a in zip(SEEDS,starts):rows.append(record(stage,condition,seed,block+1,[a]))
 writecsv(OUT/'matched_step_diagnostics.csv',rows)
 summary={}
 for stage in ['original6000','matched18000']:
  x={c:next(r for r in rows if r['stage']==stage and r['condition']==c and r['seed']=='pooled' and r['block']=='all') for c in ['dt020','dt010']}
  summary[stage]={c:x[c] for c in x}
  summary[stage]['difference']={key:x['dt020'][key]-x['dt010'][key] for key in ['Fano64','common_Fano64','residual_Fano64','channel_Fano','Srect1']}
  summary[stage]['fraction_of_Fano_gap_from_common']=summary[stage]['difference']['common_Fano64']/summary[stage]['difference']['Fano64']
 for condition in ['dt020','dt010']:
  blocks=[r for r in rows if r['stage']=='matched18000' and r['condition']==condition and r['seed']=='pooled' and isinstance(r['block'],int)]
  summary[condition+'_blocks']={key:[r[key] for r in blocks] for key in ['Fano64','common_Fano64','residual_Fano64','channel_Fano','Srect1']}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
