"""Build and run v2. Python orchestrates a small C++17 particle engine.
No OpenMP, external particle package, or internet connection is needed.
"""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, os, shutil, subprocess, time
import numpy as np
ROOT=Path(__file__).resolve().parent

def load(prefix):
    meta=json.loads(Path(str(prefix)+'_meta.json').read_text())
    data=np.fromfile(str(prefix)+'.bin',dtype='<f8').reshape(-1,meta['columns'])
    assert len(data)==meta['frames'] and np.isfinite(data).all()
    assert meta['max_control_balance_error']==0 and meta['total_balance_error']==0
    return data,meta

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--quick',action='store_true');ap.add_argument('--extra',action='store_true');args=ap.parse_args()
    out=ROOT/('quick_results' if args.quick else 'results');out.mkdir(exist_ok=True)
    compiler=shutil.which('g++'); exe=ROOT/('engine.exe' if os.name=='nt' else 'engine')
    if compiler:
        cmd=[compiler,'-O3','-std=c++17',str(ROOT/'engine.cpp'),'-o',str(exe)]
        if os.name=='nt':cmd.insert(1,'-static')
        subprocess.run(cmd,check=True)
    elif not exe.exists():raise RuntimeError('Install a C++17 compiler (g++), or use the supplied Windows executable.')
    subprocess.run([str(exe),'--selftest','1'],check=True)
    seeds=[31] if args.quick else [31,62,93]
    source_sha=hashlib.sha256((ROOT/'engine.cpp').read_bytes()).hexdigest()
    def task(name,seed,**opts):
        pref=out/f'{name}_{seed}';settings=dict(seed=seed,out=str(pref),**opts)
        if args.quick:settings.update(burn=100,duration=300)
        manifest=Path(str(pref)+'_request.json'); signature=dict(settings=settings,source_sha256=source_sha)
        if manifest.exists() and json.loads(manifest.read_text())==signature:
            load(pref);return name,seed,'cached'
        cmd=[str(exe)]
        for k,v in settings.items():cmd.extend(['--'+k.replace('_','-'),str(v)])
        start=time.time();result=subprocess.run(cmd,capture_output=True,text=True,check=True)
        data,meta=load(pref);manifest.write_text(json.dumps(signature,indent=2),encoding='utf-8')
        return name,seed,round(time.time()-start,1)
    def batch(jobs):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(task,name,seed,**kw) for name,seed,kw in jobs]
            for f in concurrent.futures.as_completed(futures):print(f.result(),flush=True)
    open_cases=[('open_ideal_gas',dict(eps=0,kappa=0)),
                ('open_rep_off',dict(eps=20,kappa=0)),
                ('open_rep_on',dict(eps=20,kappa=.02)),
                ('open_rep_screened',dict(eps=20,kappa=.02,screen=.125))]
    batch([(name,s,kw) for name,kw in open_cases for s in seeds])
    # Match periodic reference density to the OPEN baseline, not nominal activity.
    baseline=[load(out/f'open_rep_off_{s}')[0] for s in seeds]
    Nmatch=round(float(np.mean([d[:,2].mean() for d in baseline])))
    (out/'periodic_density_match.json').write_text(json.dumps(dict(N=Nmatch,method='round(mean open-control N, feedback OFF)'),indent=2))
    batch([(name,s,dict(periodic=1,N=Nmatch,eps=20,kappa=k))
           for name,k in [('periodic_rep_off',0),('periodic_rep_on',.02)] for s in seeds])
    if args.extra and not args.quick:
        # Sensitivity cases, three seeds for time step and longer reservoir;
        # bath-rate check one seed, explicitly not a full convergence study.
        jobs=[]
        for seed in seeds:
            jobs += [('open_rep_half_dt',seed,dict(eps=20,kappa=.02,dt=.01)),
                     ('open_rep_long_reservoir',seed,dict(eps=20,kappa=.02,R=96,burn=3600))]
        jobs += [('open_rep_fast_bath',31,dict(eps=20,kappa=.02,bath_rate=80)),
                 ('open_rep_stiffer',31,dict(eps=40,kappa=.02,dt=.01))]
        batch(jobs)
    print('Finished and validated all requested runs.',flush=True)

if __name__=='__main__':main()
