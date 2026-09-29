"""Particle-only analysis. A fitted exponent is not imposed to be two.
Hann forward fits include the known finite-window spectral convolution.
"""
from pathlib import Path
import json,csv,sys
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
OLD=ROOT.parent/'open_size_scan'/'results'
SEEDS=[31,62,93]
def writecsv(name,rows):
 if not rows:return
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sem(a):return float(np.std(a,ddof=1)/np.sqrt(len(a))) if len(a)>1 else float('nan')
def get(L,seed):
 p=(OLD if (OLD/f'on_L{L}_{seed}_request.json').exists() else OUT)/f'on_L{L}_{seed}'
 m=json.loads(Path(str(p)+'_meta.json').read_text());d=np.fromfile(str(p)+'.bin',dtype='<f8').reshape(-1,m['columns'])
 assert len(d)==m['frames'] and np.isfinite(d).all()
 assert m['max_control_balance_error']==m['total_balance_error']==0
 assert np.max(abs(np.diff(d[:,2])-np.diff(d[:,5]-d[:,6])))==0
 assert np.max(abs(np.diff(d[:,1])-np.diff(d[:,7]-d[:,8])))==0
 if m['columns']>426:
  z=d[:,426::2]+1j*d[:,427::2]
  assert np.max(abs(z[:,0].real-d[:,94]))==0
  assert np.max(abs(z[:,1:17]-(d[:,95:127:2]+1j*d[:,96:127:2])))<1e-7
 else:z=np.column_stack([d[:,94],d[:,95:127:2]+1j*d[:,96:127:2]])
 return d,z,m
_moment_cache={}
def moments(ms):
 """Hann convolution of a local power law; numerical alpha is freely fitted."""
 key=tuple(ms)
 if key in _moment_cache:return _moment_cache[key]
 a=np.linspace(0,3.5,701);r=np.arange(-128.,128.0001,.025)
 E=lambda x:np.exp(-1j*np.pi*x)*np.sinc(x)
 kernel=abs(.5*E(r)-.25*(E(r-1)+E(r+1)))**2
 out=[]
 for m in ms:
  x=abs(r+m)
  val=np.trapezoid(np.power(x[None,:],a[:,None])*kernel,r,axis=1)/(3/8)
  assert abs(val[0]-1)<1e-7
  assert abs(val[400]-(m*m+1/3))<1e-5
  out.append(val)
 out=np.array(out).T
 _moment_cache[key]=(a,out)
 return a,out
def fit(ms,y,seed_y,q,kind):
 if len(ms)<3:return dict(alpha=None,alpha_SEM=None,alpha_naive=None,log_RMSE=None,boundary=False)
 naive=float(np.polyfit(np.log(q),np.log(y),1)[0])
 if kind=='rect':
  vals=[np.polyfit(np.log(q),np.log(s),1)[0] for s in seed_y]
  pred=np.polyval(np.polyfit(np.log(q),np.log(y),1),np.log(q))
  return dict(alpha=naive,alpha_SEM=sem(vals),alpha_naive=naive,log_RMSE=float(np.sqrt(np.mean((pred-np.log(y))**2))),boundary=False)
 aa,mm=moments(ms); shapes=np.log(mm);shapes-=shapes.mean(1)[:,None]
 def one(s):
  logs=np.log(s);logs-=logs.mean();loss=np.mean((shapes-logs)**2,axis=1);j=int(np.argmin(loss))
  # Parabolic interpolation removes the discretization of the alpha grid.
  alpha=aa[j]
  if 0<j<len(aa)-1:
   alpha+=(aa[1]-aa[0])*.5*(loss[j-1]-loss[j+1])/(loss[j-1]-2*loss[j]+loss[j+1])
  return float(alpha),float(np.sqrt(loss[j])),j in [0,len(aa)-1]
 alpha,err,bound=one(y);vals=[one(s)[0] for s in seed_y]
 return dict(alpha=alpha,alpha_SEM=sem(vals),alpha_naive=naive,log_RMSE=err,boundary=bound)
def main():
 lengths=sorted({128,256,512}|{int(p.name.split('_')[1][1:]) for p in OUT.glob('on_L*_31_meta.json') if '_dt' not in p.name})
 tables=[];spectra=[];fits=[];checks=[];seed_spectra=[]
 for L in lengths:
  if not all(any((base/f'on_L{L}_{s}_request.json').exists() for base in [OLD,OUT]) for s in SEEDS):continue
  runs=[get(L,s) for s in SEEDS];ds=[r[0] for r in runs];zs=[r[1] for r in runs];z=np.concatenate(zs);d=np.concatenate(ds);O=L/2
  mean=z.mean(0);norm=z[:,0].real.mean()
  rp=np.array([np.mean(abs(v-mean)**2,axis=0)/norm for v in zs]);rs=rp.mean(0)
  hs=[.5*v[:,2:-1]-.25*(v[:,1:-2]+v[:,3:]) for v in zs]
  h=np.concatenate(hs);hm=h.mean(0);hn=np.mean(.375*z[:,0].real-.5*z[:,1].real+.125*z[:,2].real)
  hp=np.array([np.mean(abs(v-hm)**2,axis=0)/hn for v in hs]);ss=hp.mean(0)
  # Existing independently implemented m=2 taper must match exactly in power.
  ref=d[:,293]+1j*d[:,294];assert np.max(abs(ref-h[:,0]))<1e-7
  assert abs(hn-d[:,295].mean())<1e-9
  aa=d[:,284]+1j*d[:,285];am=aa.mean();den=d[:,283].mean()
  fvals=[np.mean(abs(v[:,284]+1j*v[:,285]-am)**2)/den for v in ds]
  row=dict(Lc=L,n_seeds=3,qmin_central=2*np.pi/O,Smin_central=float(rs[1]),SEM_central=sem(rp[:,1]),qmin_full=2*np.pi/L,Smin_full=float(np.mean(fvals)),SEM_full=sem(fvals),q_Hann_m2=4*np.pi/O,S_Hann_m2=float(ss[0]),SEM_Hann_m2=sem(hp[:,0]),mean_Nc=float(d[:,2].mean()),burn=runs[0][2]['burn'],duration=runs[0][2]['duration'])
  profile=d[:,13:61].mean(0);centers=(np.arange(48)+.5)*(L+96)/48
  interior=profile[(centers>48+L/4)&(centers<48+3*L/4)]
  row['central_profile_relative_range']=float(np.ptp(interior)/interior.mean())
  tables.append(row)
  for kind,ms,ys,ys_seed in [('rect',np.arange(1,z.shape[1]),rs[1:],rp[:,1:]),('Hann',np.arange(2,z.shape[1]-1),ss,hp)]:
   qs=2*np.pi*ms/O
   for j,(m,q,s,err) in enumerate(zip(ms,qs,ys,[sem(ys_seed[:,i]) for i in range(len(ms))])):
    spectra.append(dict(Lc=L,window=kind,m=int(m),q=float(q),S=float(s),SEM=err))
    for seed,v in zip(SEEDS,ys_seed[:,j]):seed_spectra.append(dict(Lc=L,window=kind,seed=seed,m=int(m),q=float(q),S=float(v)))
   bands=[('m2_to_m8',(ms>=2)&(ms<=8))]+[(f'qmax_{cap}',(ms>=2)&(qs<=cap)) for cap in [.05,.08,.1,.15]]
   for label,mask in bands:
    # Prevent hundreds of high-order forward convolutions; lowest 24 modes are adequate.
    mask=mask&(ms<=24);mm=ms[mask];qq=qs[mask]
    f=fit(mm,ys[mask],ys_seed[:,mask],qq,kind)
    fits.append(dict(Lc=L,window=kind,band=label,n_modes=len(mm),q_min=float(qq.min()) if len(qq) else None,q_max=float(qq.max()) if len(qq) else None,**f))
  for dd,zz,meta in runs:
   checks.append(dict(Lc=L,seed=meta['seed'],frames=meta['frames'],conservation='PASS',Hann_transform='PASS',early_late_N_ratio=float(dd[len(dd)//2:,2].mean()/dd[:len(dd)//2,2].mean())))
 writecsv('particle_lowest_modes.csv',tables);writecsv('particle_spectra.csv',spectra);writecsv('particle_alpha.csv',fits)
 writecsv('particle_seed_spectra.csv',seed_spectra)
 extrap=[]
 for key in ['Smin_central','Smin_full','S_Hann_m2']:
  for n in [3,min(4,len(tables)),len(tables)]:
   rows=tables[-n:];x=np.log([r['Lc'] for r in rows]);y=np.log([r[key] for r in rows]);b,a=np.polyfit(x,y,1)
   extrap.append(dict(observable=key,n_points=n,power_minus_slope=float(-b),estimated_Lc_at_1e_4=float(np.exp((np.log(1e-4)-a)/b))))
 writecsv('threshold_extrapolations.csv',extrap)
 leakage=[]
 for key in ['Smin_central','Smin_full']:
  ls=np.array([r['Lc'] for r in tables]);ys=np.array([r[key] for r in tables]);mat=np.column_stack([1/ls,1/ls**2]);a,b=np.linalg.lstsq(mat,ys,rcond=None)[0]
  disc=a*a+4e-4*b
  leakage.append(dict(observable=key,A_over_L=float(a),B_over_L2=float(b),estimated_Lc_at_1e_4=float((a+np.sqrt(disc))/2e-4) if disc>0 else None,note='Planning sensitivity only, not a validated asymptotic fit.'))
 writecsv('threshold_window_sensitivity.csv',leakage)
 (OUT/'particle_summary.json').write_text(json.dumps(dict(lowest=tables,alpha=fits,extrapolations=extrap,leakage_extrapolations=leakage,validation=checks),ensure_ascii=False,indent=2))
 print(json.dumps(dict(lowest=tables,primary_alpha=[r for r in fits if r['window']=='Hann' and r['band'] in ['m2_to_m8','qmax_0.08']]),indent=2),flush=True)
if __name__=='__main__':main()
