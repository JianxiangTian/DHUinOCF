"""Check periodic order of the z-projected particle positions in archived frames."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.stats import gamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':12,'xtick.labelsize':11.5,'ytick.labelsize':11.5,'legend.fontsize':10.5})

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'axial_order'
OUT.mkdir(exist_ok=True)
BASE=Path(__file__).resolve().parent/'data'/'order_snapshots'
SEEDS=(31,62,93);L=4096.;R=48.;KMAX=20.
zsets=[];nsets=[];gaps=[]
for seed in SEEDS:
 xy=np.loadtxt(BASE/f'on_L4096_{seed}_snapshot.csv',delimiter=',',skiprows=1)
 z=np.sort(xy[(xy[:,0]>=R)&(xy[:,0]<R+L),0]);zsets.append(z);nsets.append(len(z))
 gaps.append(np.diff(z[(z>=R+64)&(z<R+L-64)]))
print('projected N',nsets,flush=True)

spectra={};max_rows=[]
for O in (128,256,512):
 m=np.arange(1,int(np.floor(KMAX*O/(2*np.pi)))+1)
 q=2*np.pi*m/O
 powers=[];nblocks=[]
 for z in zsets:
  for b in range(1,int(L/O)-1):
   left=R+b*O;local=z[(z>=left)&(z<left+O)]-left
   if len(local)<10:continue
   amp=np.exp(-1j*np.outer(local,q)).sum(axis=0)
   powers.append(np.abs(amp)**2/len(local));nblocks.append(len(local))
 powers=np.asarray(powers)
 S=powers.mean(axis=0)
 sm=gaussian_filter1d(S,sigma=max(1,.1/(2*np.pi/O)))
 select=(q>=.5)&(q<=KMAX)
 j=np.where(select)[0][np.argmax(S[select])]
 js=np.where(select)[0][np.argmax(sm[select])]
 spectra[O]=(q,S,sm)
 ntested=int(select.sum()); bcount=len(powers)
 null_p=float(1-gamma.cdf(S[j],a=bcount,scale=1/bcount)**ntested)
 null_q95=float(gamma.ppf(.95**(1/ntested),a=bcount,scale=1/bcount))
 max_rows.append(dict(window_um=O,blocks=bcount,mean_N=float(np.mean(nblocks)),
                      raw_peak=float(S[j]),raw_peak_q=float(q[j]),
                      smooth_peak=float(sm[js]),smooth_peak_q=float(q[js]),
                      median_S=float(np.median(S[select])),
                      S_near_density_period=float(S[np.argmin(abs(q-2*np.pi*np.mean(nsets)/L))]),
                      tested_modes=ntested,approx_poisson_95pct_max=null_q95,
                      approx_poisson_peak_exceedance_p=null_p))
 print('window',O,'blocks',len(powers),'meanN',np.mean(nblocks),'raw peak',S[j],q[j],
       'smooth peak',sm[js],q[js],flush=True)

gap=np.concatenate(gaps)
mean_gap=float(gap.mean());gap_cv=float(gap.std(ddof=1)/gap.mean())
rho=float(np.mean(nsets)/L)

# One-dimensional pair correlation with exact rectangular-overlap correction.
edges=np.linspace(0,10,201);hist=np.zeros(len(edges)-1)
for z in zsets:
 zz=z[(z>=R+64)&(z<R+L-64)]
 pairs=cKDTree(zz[:,None]).query_pairs(10,output_type='ndarray')
 dist=np.abs(zz[pairs[:,0]]-zz[pairs[:,1]])
 hist+=np.histogram(dist,edges)[0]
rr=(edges[:-1]+edges[1:])/2;dr=np.diff(edges)
ell=L-128
expected=sum(len(z[(z>=R+64)&(z<R+L-64)]) for z in zsets)*rho*(1-rr/ell)*dr
g=hist/expected

fig,ax=plt.subplots(1,3,figsize=(13.8,3.6),layout='constrained')
for O,(q,S,sm) in spectra.items():
 ax[0].plot(q,sm,lw=1.5,label=rf'$L_{{\mathrm{{w}}}}={O}\,\mu\mathrm{{m}}$')
ax[0].axvline(2*np.pi*rho,color='gray',ls=':',label=r'$2\pi\bar\lambda$')
ax[0].set(xlim=(.5,20),ylim=(0,2),xlabel=r'$q_z$ ($\mu$m$^{-1}$)',ylabel=r'$P_{L_{\mathrm{w}}}(q_z)$',title='No stable Bragg peak')
ax[0].legend(frameon=False,ncol=2)
ax[1].plot(rr,g,color='#d95f02');ax[1].axhline(1,color='gray',ls='--',lw=.8)
ax[1].set(xlim=(0,10),ylim=(0,1.6),xlabel=r'axial separation $r$ ($\mu$m)',ylabel=r'$g_z(r)$',title='Projected pair correlation')
h,b=np.histogram(gap,bins=np.linspace(0,4,80),density=True)
ax[2].stairs(h,b,label='projected gaps',color='#1f77b4')
xx=np.linspace(0,4,300)
ax[2].plot(xx,np.exp(-xx/mean_gap)/mean_gap,color='gray',ls='--',label='exponential guide')
ax[2].set(xlim=(0,4),xlabel=r'consecutive projected gap ($\mu$m)',ylabel='probability density',title=f'Gap CV = {gap_cv:.3f}')
ax[2].legend(frameon=False)
fig.savefig(OUT/'axial_order_diagnostics.png',dpi=180);plt.close(fig)

# Captioned version for Supplemental Material.
fig,ax=plt.subplots(1,3,figsize=(9.2,3.2),layout='constrained')
for O,(q,S,sm) in spectra.items():ax[0].plot(q,sm,lw=1.35,label=rf'$L_{{\mathrm{{w}}}}={O}\,\mu\mathrm{{m}}$')
ax[0].axvline(2*np.pi*rho,color='gray',ls=':',lw=.9)
ax[0].set(xlim=(.5,20),ylim=(0,1.7),xlabel=r'$q_z$ ($\mu$m$^{-1}$)',ylabel=r'$P_{L_{\mathrm{w}}}(q_z)$')
ax[0].legend(frameon=False,ncol=1,loc='upper right',fontsize=10)
ax[1].plot(rr,g,color='#d95f02',lw=1.3);ax[1].axhline(1,color='gray',ls='--',lw=.8)
ax[1].set(xlim=(0,10),ylim=(0,1.5),xlabel=r'$r_z$ ($\mu$m)',ylabel=r'$g_z(r_z)$')
ax[2].stairs(h,b,color='#1f77b4',lw=1.2,label='data')
ax[2].plot(xx,np.exp(-xx/mean_gap)/mean_gap,color='gray',ls='--',lw=1,label='exponential')
ax[2].set(xlim=(0,4),xlabel=r'gap $\Delta z$ ($\mu$m)',ylabel=r'Gap density ($\mu\mathrm{m}^{-1}$)')
ax[2].legend(frameon=False,fontsize=10)
for tag,axis in zip('abc',ax):
 axis.text(.03,.97,f'({tag})',transform=axis.transAxes,va='top',ha='left',weight='bold',
           bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
 for spine in axis.spines.values():spine.set_visible(True)
 axis.tick_params(which='both',direction='in')
fig.savefig(Path(__file__).resolve().parent/'figures'/'v10'/'figS4.png',dpi=220)
fig.savefig(Path(__file__).resolve().parent/'figures'/'v10'/'figS4.pdf')
plt.close(fig)

report=dict(source='Three archived final 2D configurations at Lc=4096 um, projected onto z',
 N_control=nsets,line_density_per_um=rho,mean_projected_gap_um=mean_gap,gap_cv=gap_cv,
 spectral_windows=max_rows,
 pair_correlation_samples={str(r):float(g[np.argmin(abs(rr-r))]) for r in (.25,.6,1,2,3,5,8,10)},
 caveat='Full z-coordinate time series was not archived, only one final configuration per seed. Spectra use independent axial subwindows within each seed, which are not independent simulation runs. Results rule out obvious Bragg order at sampled q but not every possible finite-size/quasiperiodic ordering.')
(OUT/'axial_order_diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
for O,(q,S,sm) in spectra.items():np.savetxt(OUT/f'spectrum_O{O}.csv',np.c_[q,S,sm],delimiter=',',header='q_per_um,S_raw_mean,S_smooth_0p1_per_um',comments='')
print(json.dumps(report,indent=2),flush=True)
