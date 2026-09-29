"""Resumable length scan; completed data are retained and source-checked."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,subprocess,time
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
def load(p):
 m=json.loads(Path(str(p)+'_meta.json').read_text()); d=np.fromfile(str(p)+'.bin',dtype='<f8').reshape(-1,m['columns'])
 assert d.shape[0]==m['frames'] and np.isfinite(d).all()
 assert m['max_control_balance_error']==m['total_balance_error']==0
 assert np.max(abs(d[:,426]-d[:,94]))==0
 assert np.max(abs(d[:,428:460]-d[:,95:127]))<1e-7
 return d,m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--lengths',nargs='+',type=int,default=[768,1024]);ap.add_argument('--seeds',nargs='+',type=int,default=[31,62,93]);ap.add_argument('--workers',type=int,default=6);ap.add_argument('--dt',type=float,default=.02);args=ap.parse_args()
 sha=hashlib.sha256((ROOT/'engine.cpp').read_bytes()).hexdigest()
 def task(L,seed):
  suffix='' if args.dt==.02 else '_dt'+str(args.dt)
  pref=OUT/f'on_L{L}_{seed}{suffix}'
  settings=dict(L=L,R=48,seed=seed,eps=20,kappa=.02,dt=args.dt,burn=max(3600.,25*(L+96)),duration=18000.*L/256,out=str(pref))
  request=dict(settings=settings,source_sha256=sha)
  manifest=Path(str(pref)+'_request.json')
  if manifest.exists() and json.loads(manifest.read_text())==request:load(pref);return pref.name,'cached'
  manifest.with_suffix('.pending.json').write_text(json.dumps(request,indent=2))
  cmd=[str(ROOT/'engine.exe')]
  for k,v in settings.items():cmd+=['--'+k,str(v)]
  t=time.time();subprocess.run(cmd,check=True,capture_output=True);d,m=load(pref)
  request['elapsed_seconds']=time.time()-t
  # Timing lives separately so cache identity does not depend on wall-clock time.
  (OUT/(pref.name+'_timing.json')).write_text(json.dumps({'elapsed_seconds':request.pop('elapsed_seconds')}))
  manifest.write_text(json.dumps(request,indent=2));manifest.with_suffix('.pending.json').unlink(missing_ok=True)
  return pref.name,round(time.time()-t,1)
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
  fs=[pool.submit(task,L,s) for L in args.lengths for s in args.seeds]
  for f in concurrent.futures.as_completed(fs):print(f.result(),flush=True)
 print('Requested particle runs complete.',flush=True)
if __name__=='__main__':main()
