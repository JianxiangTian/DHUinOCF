"""Matched-duration dt=.01 controls using the archived open_v2 engine.

The existing dt=.02 long runs use the same engine, seeds and schedule.
Running this file is resumable; existing outputs are verified before reuse.
"""
from pathlib import Path
import concurrent.futures,hashlib,json,subprocess,time
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
ENGINE=PROJECT/'open_v2'/'engine.exe'
SOURCE=PROJECT/'open_v2'/'engine.cpp'
BASE=PROJECT/'open_v2'/'results'
RUNS=ROOT/'runs';RUNS.mkdir(parents=True,exist_ok=True)
SEEDS=(31,62,93)
SOURCE_SHA=hashlib.sha256(SOURCE.read_bytes()).hexdigest()

def one(seed):
    old=json.loads((BASE/f'open_rep_on_long_{seed}_request.json').read_text())
    assert old['source_sha256']==SOURCE_SHA, 'Engine source differs from baseline.'
    p=RUNS/f'open_rep_half_dt_long_{seed}'
    opts=dict(seed=seed,L=256,R=48,eps=20,kappa=.02,screen=0,
              dt=.01,burn=3600,duration=18000,out=str(p))
    request=dict(settings=opts,source_sha256=SOURCE_SHA,
                 engine_sha256=hashlib.sha256(ENGINE.read_bytes()).hexdigest())
    manifest=Path(str(p)+'_request.json')
    if manifest.exists() and json.loads(manifest.read_text())==request:
        meta=json.loads(Path(str(p)+'_meta.json').read_text())
        data=np.fromfile(str(p)+'.bin',dtype='<f8')
        assert len(data)==meta['frames']*meta['columns']
        assert meta['frames']==3600 and meta['dt']==.01
        assert meta['max_control_balance_error']==meta['total_balance_error']==0
        return seed,'cached'
    args=[str(ENGINE)]
    for key,value in opts.items():args.extend(['--'+key,str(value)])
    start=time.time()
    proc=subprocess.run(args,capture_output=True,text=True,check=True)
    meta=json.loads(Path(str(p)+'_meta.json').read_text())
    data=np.fromfile(str(p)+'.bin',dtype='<f8')
    assert len(data)==meta['frames']*meta['columns']
    assert meta['frames']==3600 and meta['dt']==.01
    assert meta['max_control_balance_error']==meta['total_balance_error']==0
    manifest.write_text(json.dumps(request,indent=2),encoding='utf8')
    (RUNS/f'open_rep_half_dt_long_{seed}_timing.json').write_text(
        json.dumps(dict(elapsed_seconds=time.time()-start),indent=2))
    return seed,round(time.time()-start,1)

if __name__=='__main__':
    assert ENGINE.exists()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for future in concurrent.futures.as_completed([pool.submit(one,s) for s in SEEDS]):
            print(future.result(),flush=True)
    print('ALL_MATCHED_RUNS_VALIDATED',flush=True)
