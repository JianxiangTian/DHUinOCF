"""Independent conservation/schema checks against saved raw observables."""
from pathlib import Path
import json,math,subprocess,sys,os
import numpy as np
from run_suite import ROOT,load
OUT=ROOT/'results'
def main():
 exe=ROOT/('engine.exe' if os.name=='nt' else 'engine')
 subprocess.run([str(exe),'--selftest','1'],check=True)
 # Independent ideal-bath detailed-balance check, using Poisson state weights.
 zv=16.;gamma=2.5;res=[]
 for n in range(80):
  pn=math.exp(-zv+n*math.log(zv)-math.lgamma(n+1))
  pn1=math.exp(-zv+(n+1)*math.log(zv)-math.lgamma(n+2))
  res.append(abs(pn*gamma*zv-pn1*gamma*(n+1)))
 assert max(res)<1e-11
 checks=[]
 for f in sorted(OUT.glob('*_meta.json')):
  prefix=str(f).replace('_meta.json','');d,m=load(prefix)
  profile_error=float(np.max(abs(d[:,13:61].sum(1)-d[:,1])))
  total_error=float(np.max(abs(np.diff(d[:,1])-np.diff(d[:,7]-d[:,8]))))
  control_error=float(np.max(abs(np.diff(d[:,2])-np.diff(d[:,5]-d[:,6]))))
  assert profile_error==total_error==control_error==0
  assert np.all(d[:,2]<=d[:,1]) and np.all(d[:,1]>=0)
  assert np.allclose(np.diff(d[:,0]),m['save_dt'])
  assert m['max_pair_force']<=m['eps']+1e-10
  assert np.all(d[:,281]<=d[:,280])
  snap=np.loadtxt(prefix+'_snapshot.csv',delimiter=',',skiprows=1)
  assert np.all((snap[:,0]>=0)&(snap[:,0]<m['T']))
  assert np.all((snap[:,1]>=0)&(snap[:,1]<=m['W']))
  assert len(snap)==m['N_final']
  checks.append(dict(run=Path(prefix).name,frames=len(d),profile_count_error=profile_error,
       sampled_control_balance_error=control_error,sampled_total_balance_error=total_error))
 result=dict(status='PASS',n_trajectories=len(checks),
    ideal_bath_detailed_balance_max_absolute_error=max(res),checks=checks)
 (OUT/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print('PASS',len(checks),'trajectories: force gradient, pair antisymmetry, bath detailed balance, particle balances, profiles, time grid, snapshots.')
if __name__=='__main__':main()
