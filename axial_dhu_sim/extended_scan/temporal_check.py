"""Time-correlation and cross-mode block-bootstrap diagnostics, not new runs."""
import json
import numpy as np
from analyze import ROOT,OUT,get,moments,fit,SEEDS,writecsv
def tau_power(v,dt=5.):
 v=np.asarray(v)-np.mean(v);n=len(v);size=1<<(2*n-1).bit_length();f=np.fft.rfft(v,size)
 ac=np.fft.irfft(f*f.conj(),size)[:n]/np.arange(n,0,-1)
 if ac[0]<=0:return 0
 ac/=ac[0];pairs=ac[1:1+2*((n-1)//2)].reshape(-1,2).sum(1)
 stop=np.where(pairs<=0)[0];k=int(stop[0]) if len(stop) else len(pairs)
 return float(dt*(1+2*pairs[:k].sum()))
def main():
 summary=json.loads((OUT/'particle_summary.json').read_text());rows=[];stationarity=[];aa,mm=moments(np.arange(2,9));shapes=np.log(mm);shapes-=shapes.mean(1)[:,None]
 rng=np.random.default_rng(20260916)
 for info in summary['lowest']:
  L=info['Lc'];runs=[get(L,s) for s in SEEDS];hs=[.5*z[:,2:9]-.25*(z[:,1:8]+z[:,3:10]) for d,z,m in runs]
  mean=np.concatenate(hs).mean(0);powers=[abs(h-mean)**2 for h in hs]
  taus=[tau_power(p[:,0]) for p in powers]
  halves=[]
  for late in [False,True]:
   vals=np.array([(p[len(p)//2:] if late else p[:len(p)//2]).mean(0) for p in powers]);q=4*np.pi*np.arange(2,9)/L
   halves.append(fit(np.arange(2,9),vals.mean(0),vals,q,'Hann'))
  stationarity.append(dict(Lc=L,alpha_first_half=halves[0]['alpha'],SEM_first_half=halves[0]['alpha_SEM'],alpha_second_half=halves[1]['alpha'],SEM_second_half=halves[1]['alpha_SEM'],alpha_difference=halves[1]['alpha']-halves[0]['alpha'],power_IAT_max_seconds=max(taus)))
  for nb in [10,20,40]:
   blocks=np.concatenate([np.array([v.mean(0) for v in np.array_split(p,nb)]) for p in powers])
   # Resample whole vector blocks, preserving inter-mode correlations.
   ix=rng.integers(0,len(blocks),size=(2000,len(blocks)));logs=np.log(blocks[ix].mean(1));logs-=logs.mean(1)[:,None]
   losses=np.mean(logs**2,axis=1)[:,None]+np.mean(shapes**2,axis=1)[None,:]-2*logs@shapes.T/7
   jj=losses.argmin(1);alpha=aa[jj].copy();valid=(jj>0)&(jj<len(aa)-1);ii=np.arange(len(jj))[valid];j=jj[valid]
   alpha[valid]+=(aa[1]-aa[0])*.5*(losses[ii,j-1]-losses[ii,j+1])/(losses[ii,j-1]-2*losses[ii,j]+losses[ii,j+1])
   low,high=np.quantile(alpha,[.025,.975]);block_s=info['duration']/nb
   rows.append(dict(Lc=L,blocks_per_seed=nb,block_seconds=block_s,max_power_IAT_seconds=max(taus),block_over_IAT=block_s/max(taus),alpha_bootstrap_p025=float(low),alpha_bootstrap_p975=float(high),bootstrap_SD=float(alpha.std(ddof=1)),n_bootstrap=2000,note='Conditional block-bootstrap interval; excludes window/range/finite-size model uncertainty.'))
 writecsv('temporal_block_diagnostics.csv',rows)
 writecsv('early_late_spectrum_check.csv',stationarity)
 print(json.dumps([r for r in rows if r['blocks_per_seed']==20],indent=2))
if __name__=='__main__':main()
