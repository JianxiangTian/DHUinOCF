"""Longer records for the slowest open-system number modes.
Keeps the initial runs untouched; final analysis prefers these records.
"""
from pathlib import Path
import concurrent.futures,hashlib,json,subprocess,time,os
from run_suite import load,ROOT
OUT=ROOT/'results';EXE=ROOT/('engine.exe' if os.name=='nt' else 'engine')
def task(case,seed,eps,kappa,screen=0):
 prefix=OUT/(case+'_long_'+str(seed));sha=hashlib.sha256((ROOT/'engine.cpp').read_bytes()).hexdigest()
 settings=dict(seed=seed,eps=eps,kappa=kappa,screen=screen,burn=3600,duration=18000,out=str(prefix))
 request=dict(settings=settings,source_sha256=sha);manifest=Path(str(prefix)+'_request.json')
 if manifest.exists() and json.loads(manifest.read_text())==request:
  load(prefix);return prefix.name,'cached'
 cmd=[str(EXE)]
 for k,v in settings.items():cmd+=['--'+k,str(v)]
 t=time.time();subprocess.run(cmd,check=True,capture_output=True);load(prefix)
 manifest.write_text(json.dumps(request,indent=2),encoding='utf-8')
 return prefix.name,round(time.time()-t,1)
if __name__=='__main__':
 jobs=[(name,s,e,k,sc) for name,e,k,sc in [('open_ideal_gas',0,0,0),('open_rep_off',20,0,0),('open_rep_on',20,.02,0),('open_rep_screened',20,.02,.125)] for s in [31,62,93]]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for future in concurrent.futures.as_completed([pool.submit(task,*j) for j in jobs]):print(future.result(),flush=True)
