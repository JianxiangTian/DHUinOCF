from pathlib import Path
import subprocess, concurrent.futures, json, hashlib, time
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs'; OUT.mkdir(parents=True,exist_ok=True)
ENGINE=ROOT.parent/'extended_scan'/'engine.exe'
CASES={'gain_half':(.01,.2,.2),'baseline':(.02,.2,.2),'gain_double':(.04,.2,.2),'zero_flow':(.02,0,.2),'update_2s':(.02,.2,2.)}
def run(case,seed):
 k,u,update=CASES[case]; p=OUT/f'{case}_{seed}'
 opts=dict(L=256,N=430,periodic=1,eps=20,kappa=k,u=u,**{'control-dt':update},burn=3000,duration=15000,seed=seed,out=str(p))
 request=dict(settings=opts,engine_sha256=hashlib.sha256(ENGINE.read_bytes()).hexdigest())
 marker=Path(str(p)+'_request.json')
 if marker.exists() and json.loads(marker.read_text())==request:return case,seed,'cached'
 cmd=[str(ENGINE)]
 for key,value in opts.items():cmd.extend(['--'+key,str(value)])
 t=time.time();subprocess.run(cmd,check=True,capture_output=True,text=True)
 m=json.loads(Path(str(p)+'_meta.json').read_text())
 d=np.fromfile(str(p)+'.bin',dtype='<f8').reshape(-1,m['columns'])
 assert len(d)==m['frames'] and np.isfinite(d).all() and np.all(d[:,2]==430)
 assert m['total_balance_error']==m['max_control_balance_error']==0
 marker.write_text(json.dumps(request,indent=2),encoding='utf8')
 return case,seed,round(time.time()-t,2)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
  fs=[pool.submit(run,c,s) for c in CASES for s in [131,262,393]]
  for f in concurrent.futures.as_completed(fs):print(f.result(),flush=True)
 print('ALL_CONTROLS_COMPLETE',flush=True)
