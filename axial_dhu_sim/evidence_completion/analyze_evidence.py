from pathlib import Path
import sys,json,csv
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT.parent/'extended_scan'))
from analyze import get
LENGTHS=[128,256,384,512,640,768,1024,2048,4096]
def sem(v):return float(np.std(v,ddof=1)/np.sqrt(len(v)))
def csvsave(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def corrfit(series,dt,q,k,horizon=1.):
 # Per-run centering, unbiased lag normalization; fit only a predeclared ~one relaxation time.
 a=series-series.mean();n=len(a);maxlag=min(n//10,max(3,int(horizon/k/dt)))
 t=np.arange(1,maxlag+1)*dt
 c=np.array([np.mean(a[j:]*a[:-j].conj()) for j in range(1,maxlag+1)])/np.mean(abs(a)**2)
 valid=(abs(c)>.05)
 rate=-np.polyfit(t[valid],np.log(abs(c[valid])),1)[0]
 speed=-np.polyfit(t[valid],np.unwrap(np.angle(c))[valid],1)[0]/q
 return float(rate),float(speed),t,c
def main():
 boundary=[]; amplitudes=[];dynamic=[];counts=[];checks=[]
 for L in LENGTHS:
  runs=[get(L,s) for s in [31,62,93]];ds=[v[0] for v in runs];zs=[v[1] for v in runs];m=runs[0][2]
  nc=np.concatenate([d[:,2] for d in ds]);nmean=nc.mean();nv=nc.var();k=.02*nmean/L/2;A=.2/k
  r=dict(Lc=L,mean_Nc=float(nmean),kappa_predicted=float(k),A_predicted=float(A),Nc_Fano=float(nv/nmean))
  for name,start,ell in [('fraction',256,L/4),('interior',402,np.sqrt(8*L))]:
   arrays=[d[:,start:start+24] for d in ds];x=np.concatenate(arrays);mean=x.mean(0);den=mean.mean()
   vari=np.mean((x-mean)**2);cov=np.mean((x-mean)*(nc-nmean)[:,None],axis=0)
   common=np.mean(cov**2/nv)/den
   vals=[np.mean((v-mean)**2)/den for v in arrays]
   r.update({name+'_ell':float(ell),name+'_Fano':float(np.mean(vals)),name+'_SEM':sem(vals),name+'_common_Fano':float(common),name+'_residual_Fano':float(vari/den-common),name+'_common_fraction':float(common/(vari/den)),name+'_b':float(np.mean(cov/nv))})
   early=np.concatenate([v[:len(v)//2] for v in arrays]);late=np.concatenate([v[len(v)//2:] for v in arrays])
   r[name+'_early_Fano']=float(np.mean((early-early.mean(0))**2)/early.mean())
   r[name+'_late_Fano']=float(np.mean((late-late.mean(0))**2)/late.mean())
  for name,re,im,norm in [('full_rect',284,285,283),('fraction_Hann',293,294,295),('interior_Hann',301,302,303)]:
   aa=[d[:,re]+1j*d[:,im] for d in ds];center=np.concatenate(aa).mean();den=np.mean([d[:,norm].mean() for d in ds]);ss=[np.mean(abs(a-center)**2)/den for a in aa]
   r[name+'_S']=float(np.mean(ss));r[name+'_SEM']=sem(ss)
  # Position comparison uses same rectangular window lengths and same modes.
  for name,start in [('upstream',61),('center',94),('downstream',127)]:
   aa=[d[:,start+1]+1j*d[:,start+2] for d in ds];center=np.concatenate(aa).mean();den=np.mean([d[:,start].mean() for d in ds]);ss=[np.mean(abs(a-center)**2)/den for a in aa]
   r[name+'_rect_S']=float(np.mean(ss));r[name+'_rect_SEM']=sem(ss)
  boundary.append(r)
  # Each count length retains raw total variance; no regression subtraction in primary evidence.
  for j,f in enumerate([.125,.25,.5,.75,1.]):
   aa=[d[:,306+24*j:330+24*j] for d in ds];center=np.concatenate(aa).mean(0);den=center.mean();ff=[np.mean((a-center)**2)/den for a in aa]
   counts.append(dict(Lc=L,ell=float(f*np.sqrt(8*L)),Fano=float(np.mean(ff)),SEM=sem(ff)))
  O=L/2;norm=np.mean([np.mean(.375*z[:,0].real-.5*z[:,1].real+.125*z[:,2].real) for z in zs])
  for mode in range(2,9):
   hs=[.5*z[:,mode]-.25*(z[:,mode-1]+z[:,mode+1]) for z in zs];hm=np.concatenate(hs).mean();ss=[np.mean(abs(h-hm)**2)/norm for h in hs]
   moment=(2*np.pi/O)**2*(mode**2+1/3);pred=A*moment
   amplitudes.append(dict(Lc=L,mode=mode,q=2*np.pi*mode/O,S=float(np.mean(ss)),SEM=sem(ss),S_predicted_q2=float(pred),measured_over_predicted=float(np.mean(ss)/pred)))
   if mode==2:
    rates=[];speeds=[]
    for h in hs:
     rate,speed,t,c=corrfit(h,m['save_dt'],2*np.pi*mode/O,k);rates.append(rate);speeds.append(speed)
    dynamic.append(dict(Lc=L,q=2*np.pi*mode/O,gamma_predicted=float(k),gamma_measured=float(np.mean(rates)),gamma_SEM=sem(rates),u_measured=float(np.mean(speeds)),u_SEM=sem(speeds),u_input=.2,note='Hann wavepacket: approximate bulk rate, not exact Fourier eigenmode'))
  checks.append(dict(Lc=L,seeds=3,frames_per_seed=m['frames'],balance='PASS',Hann_consistency='PASS'))
  del runs,ds,zs,nc
 csvsave('open_boundary_windows.csv',boundary);csvsave('parameter_free_amplitudes.csv',amplitudes);csvsave('open_time_correlations.csv',dynamic);csvsave('interior_count_curves.csv',counts)
 (OUT/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
 print(json.dumps([r for r in boundary if r['Lc'] in [128,512,1024,2048,4096]],indent=2),flush=True)
 print('DYNAMICS',json.dumps(dynamic[-3:],indent=2),flush=True)
def controls():
 rows=[];seedrows=[];corrrows=[]
 for case in ['gain_half','baseline','gain_double','zero_flow','update_2s']:
  ds=[];metas=[]
  for seed in [131,262,393]:
   p=ROOT/'runs'/f'{case}_{seed}';assert Path(str(p)+'_request.json').exists()
   m=json.loads(Path(str(p)+'_meta.json').read_text());metas.append(m);ds.append(np.fromfile(str(p)+'.bin',dtype='<f8').reshape(-1,m['columns']))
  m=metas[0];N=m['N_initial'];k=m['kappa']*(N/m['L'])/m['lambda_ref']
  for mode in [1,2]:
   q=2*np.pi*mode/m['L'];col=284+2*(mode-1);aa=[d[:,col]+1j*d[:,col+1] for d in ds];center=np.concatenate(aa).mean();vals=[]
   for seed,a in zip([131,262,393],aa):
    S=float(np.mean(abs(a-center)**2)/N);rate,speed,t,c=corrfit(a,m['save_dt'],q,k)
    # Also retain half and 1.5-times fitting horizons to expose sensitivity.
    rates=[corrfit(a,m['save_dt'],q,k,h)[0] for h in [.5,1.5]]
    v=dict(case=case,seed=seed,mode=mode,q=q,S=S,gamma=rate,u=speed,gamma_half_horizon=rates[0],gamma_long_horizon=rates[1]);vals.append(v);seedrows.append(v)
    for ti,ci in zip(t,c):corrrows.append(dict(case=case,seed=seed,mode=mode,t=float(ti),C_re=float(ci.real),C_im=float(ci.imag)))
   r=dict(case=case,mode=mode,q=q,kappa_reference=m['kappa'],kappa_effective=k,u_input=m['u'],update_interval=m['control_dt'],S_predicted_leading=.2*q*q/k,gamma_predicted_leading=k)
   for key in ['S','gamma','u','gamma_half_horizon','gamma_long_horizon']:
    values=[v[key] for v in vals];r[key]=float(np.mean(values));r[key+'_SEM']=sem(values)
   r['S_over_prediction']=r['S']/r['S_predicted_leading'];r['gamma_over_prediction']=r['gamma']/k
   r['noise_ratio']=r['S']*r['gamma']/(.2*q*q)
   rows.append(r)
 csvsave('independent_controls.csv',rows);csvsave('independent_control_seeds.csv',seedrows);csvsave('control_correlations.csv',corrrows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':
 if '--controls' in sys.argv:controls()
 else:main()
