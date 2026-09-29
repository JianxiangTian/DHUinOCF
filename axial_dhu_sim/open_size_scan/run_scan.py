"""Three lengths, two feedback conditions, three seeds, equal transport times.
The supplied engine.cpp is self-contained; prepare_engine.py is optional.
"""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,shutil,subprocess,time
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
def load(prefix):
 m=json.loads(Path(str(prefix)+'_meta.json').read_text());d=np.fromfile(str(prefix)+'.bin',dtype='<f8').reshape(-1,m['columns'])
 assert d.shape==(m['frames'],426) and np.isfinite(d).all()
 assert m['max_control_balance_error']==0 and m['total_balance_error']==0
 return d,m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
 exe=ROOT/('engine.exe' if os.name=='nt' else 'engine');compiler=shutil.which('g++')
 if compiler:
  cmd=[compiler,'-O3','-std=c++17',str(ROOT/'engine.cpp'),'-o',str(exe)]
  if os.name=='nt':cmd.insert(1,'-static')
  subprocess.run(cmd,check=True)
 elif not exe.exists():raise RuntimeError('A C++17 compiler or the supplied Windows executable is required.')
 subprocess.run([str(exe),'--selftest','1'],check=True)
 sha=hashlib.sha256((ROOT/'engine.cpp').read_bytes()).hexdigest()
 def task(L,case,seed):
  pref=OUT/f'{case}_L{L}_{seed}'
  # L/u is the controlled-channel advective traversal time. Retain >=5 whole
  # domain traversal times for initialization, then ~14 channel traversal times.
  burn=max(3600.,5*(L+96)/.2)
  duration=18000.*L/256
  settings=dict(L=L,R=48,seed=seed,eps=20,kappa=.02 if case=='on' else 0,
                burn=burn,duration=duration,out=str(pref))
  request=dict(settings=settings,source_sha256=sha)
  manifest=Path(str(pref)+'_request.json')
  if manifest.exists() and json.loads(manifest.read_text())==request:load(pref);return pref.name,'cached'
  cmd=[str(exe)]
  for k,v in settings.items():cmd+=['--'+k,str(v)]
  t=time.time();subprocess.run(cmd,check=True,capture_output=True);load(pref)
  manifest.write_text(json.dumps(request,indent=2),encoding='utf-8')
  return pref.name,round(time.time()-t,1)
 jobs=[(L,c,s) for L in [128,256,512] for c in ['on','off'] for s in [31,62,93]]
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
  for f in concurrent.futures.as_completed([pool.submit(task,*j) for j in jobs]):print(f.result(),flush=True)
 print('Completed all 18 size-scan records.',flush=True)
if __name__=='__main__':main()
