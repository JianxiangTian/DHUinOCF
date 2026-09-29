"""Post-hoc 2D order diagnostics from archived final-coordinate snapshots.

This deliberately does not infer a thermodynamic absence of crystalline order from
three final frames. It checks whether the frames used by the axial study display
Bragg-like peaks or coherent sixfold bond order.
"""
from pathlib import Path
import sys, json
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':12,'xtick.labelsize':11.5,'ytick.labelsize':11.5,'legend.fontsize':11})
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'two_dimensional_order'
OUT.mkdir(exist_ok=True)
DATA=Path(__file__).resolve().parent/'data'/'order_snapshots'
SEEDS=(31,62,93)
L=4096.; R=48.; W=16.; d=1.
BLOCK=64.; mvals=np.arange(-20,21); nvals=np.arange(0,121)
kx=2*np.pi*mvals/W; kz=2*np.pi*nvals/BLOCK

def load(seed):
 p=DATA/f'on_L4096_{seed}_snapshot.csv'
 xy=np.loadtxt(p,delimiter=',',skiprows=1)
 xy=xy[(xy[:,0]>=R)&(xy[:,0]<R+L)].copy()
 return xy

def orient(xy,length=L):
 # Six nearest bonds, with the order parameter averaged only well inside walls.
 tree=cKDTree(xy)
 _,idx=tree.query(xy,k=7)
 dr=xy[idx[:,1:]]-xy[:,None,:]
 ang=np.arctan2(dr[:,:,1],dr[:,:,0])
 local=np.mean(np.exp(6j*ang),axis=1)
 core=(xy[:,1]>3)&(xy[:,1]<W-3)&(xy[:,0]>R+3)&(xy[:,0]<R+length-3)
 return float(abs(np.mean(local[core]))),float(np.mean(abs(local[core]))),int(core.sum())

def spectrum(xy):
 sums=np.zeros((len(nvals),len(mvals)))
 blocks=0; ns=[]
 for i in range(2,int(L/BLOCK)-2):
  z0=R+i*BLOCK
  seg=xy[(xy[:,0]>=z0)&(xy[:,0]<z0+BLOCK)]
  n=len(seg)
  if n<5:continue
  ez=np.exp(-1j*np.outer(seg[:,0]-z0,kz))
  ex=np.exp(-1j*np.outer(seg[:,1],kx))
  val=(ez.T@ex)
  sums+=(val.real**2+val.imag**2)/n
  blocks+=1;ns.append(n)
 return sums,blocks,ns

bins=np.linspace(0,6,121);hist=np.zeros(len(bins)-1);ns=[];psis=[];localpsis=[]
Ssum=np.zeros((len(nvals),len(mvals)));blocks=0;block_ns=[];seed_spectra=[]
snapshots={}
for seed in SEEDS:
 xy=load(seed);snapshots[seed]=xy;ns.append(len(xy))
 psi,local,ncore=orient(xy);psis.append(psi);localpsis.append(local)
 pairs=cKDTree(xy).query_pairs(6,output_type='ndarray')
 rr=np.linalg.norm(xy[pairs[:,0]]-xy[pairs[:,1]],axis=1)
 hist+=np.histogram(rr,bins)[0]
 part,bn,counts=spectrum(xy);Ssum+=part;blocks+=bn;block_ns+=counts;seed_spectra.append(part/bn)
 print('seed',seed,'N',len(xy),'psi6_global',psi,'psi6_local_abs_mean',local,'core_N',ncore,'blocks',bn,flush=True)

S=Ssum/blocks;S[0,np.where(mvals==0)[0][0]]=np.nan
np.savez_compressed(OUT/'spectrum_2d.npz',S=S,kz=kz,kx=kx,seed_spectra=np.asarray(seed_spectra))
kr=np.hypot(kz[:,None],kx[None,:]);valid=(kr>=0.5)&(kr<=12)&np.isfinite(S)
imax=np.nanargmax(np.where(valid,S,np.nan));ii,jj=np.unravel_index(imax,S.shape)
dr=np.diff(bins);mid=(bins[1:]+bins[:-1])/2
rho=np.mean(ns)/(L*W)
# Leading rectangle edge correction; for r<6 the remaining z correction is tiny.
edge=(1-2*mid/(np.pi*W))*(1-2*mid/(np.pi*L))
expected=0.5*sum(ns)*rho*2*np.pi*mid*dr*edge
gr=hist/expected

fig,axs=plt.subplots(1,3,figsize=(13.8,3.4),layout='constrained')
xy=snapshots[31];sel=xy[(xy[:,0]>=R+1000)&(xy[:,0]<R+1064)]
axs[0].scatter(sel[:,0]-(R+1000),sel[:,1],s=9,facecolors='none',edgecolors='#1f77b4',lw=.7)
axs[0].set(xlim=(0,64),ylim=(0,W),xlabel=r'$z$ in 64 $\mu$m strip',ylabel=r'$x$ ($\mu$m)',title='One archived snapshot')
img=axs[1].pcolormesh(kz,kx,S.T,shading='auto',vmin=0,vmax=2.2,cmap='viridis')
axs[1].set(xlim=(0,12),ylim=(-8,8),xlabel=r'$k_z$ ($\mu$m$^{-1}$)',ylabel=r'$k_x$ ($\mu$m$^{-1}$)',title=r'2D $S(k_z,k_x)$, 180 blocks')
fig.colorbar(img,ax=axs[1],label=r'$S_{2D}$')
axs[2].plot(mid,gr,color='#d95f02',lw=1.7)
axs[2].axhline(1,color='gray',lw=.8,ls='--')
axs[2].axvline(1,color='gray',lw=.8,ls=':')
axs[2].set(xlim=(0,6),ylim=(0,max(2,np.nanmax(gr[5:]))),xlabel=r'$r$ ($\mu$m)',ylabel=r'$g(r)$',title='Radial pair correlation')
fig.savefig(OUT/'order_diagnostics.png',dpi=180);plt.close(fig)

# Compact manuscript version with panel labels and captions supplied by Word/TeX.
fig,axs=plt.subplots(1,3,figsize=(11.4,3.15),layout='constrained')
axs[0].scatter(sel[:,0]-(R+1000),sel[:,1],s=7,facecolors='none',edgecolors='#1f77b4',lw=.65)
axs[0].set(xlim=(0,64),ylim=(0,W),xlabel=r'$z$ ($\mu$m)',ylabel=r'$x$ ($\mu$m)')
img=axs[1].pcolormesh(kz,kx,S.T,shading='auto',vmin=0,vmax=2.2,cmap='viridis')
axs[1].set(xlim=(0,12),ylim=(-8,8),xlabel=r'$k_z$ ($\mu$m$^{-1}$)',ylabel=r'$k_x$ ($\mu$m$^{-1}$)')
fig.colorbar(img,ax=axs[1],label=r'$S_{2D}$')
axs[2].plot(mid,gr,color='#d95f02',lw=1.5)
axs[2].axhline(1,color='gray',lw=.8,ls='--');axs[2].axvline(1,color='gray',lw=.8,ls=':')
axs[2].set(xlim=(0,6),ylim=(0,1.7),xlabel=r'$r$ ($\mu$m)',ylabel=r'$g(r)$')
for tag,axis in zip('abc',axs):
 axis.text(.03,.97,f'({tag})',transform=axis.transAxes,va='top',ha='left',weight='bold',
           bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
fig.savefig(Path(__file__).resolve().parent/'figures'/'v10'/'figS3.png',dpi=220)
fig.savefig(Path(__file__).resolve().parent/'figures'/'v10'/'figS3.pdf')
plt.close(fig)

report=dict(source='Three final snapshots at Lc=4096 um; 60 disjoint 64-um interior axial blocks per seed for S2D',
 seeds=SEEDS,N_control=ns,number_density_per_um2=rho,nominal_area_fraction=rho*np.pi*d*d/4,
 spectral_blocks=blocks,mean_particles_per_block=float(np.mean(block_ns)),
 S2D_peak=float(S[ii,jj]),peak_kz=float(kz[ii]),peak_kx=float(kx[jj]),
 S2D_median=float(np.nanmedian(S[valid])),S2D_p99=float(np.nanpercentile(S[valid],99)),
 axial_kz_values={str(int(n)):float(S[n,np.where(mvals==0)[0][0]]) for n in (1,2,3,4)},
 transverse_kx_values={str(int(m)):float(S[0,np.where(mvals==m)[0][0]]) for m in (1,2,3,4)},
 global_psi6=psis,local_abs_psi6=localpsis,
 g_r_values={str(r):float(gr[np.argmin(abs(mid-r))]) for r in (0.5,0.95,1.25,1.75,2.5,3.5,5.0)},
 caveat='Only final configurations were archived as full 2D coordinates. This is evidence against obvious Bragg order in these snapshots, not a finite-size scaling proof of a fully disordered 2D thermodynamic phase.')
(OUT/'order_diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2),flush=True)

# A size check using all 27 archived final snapshots. A crystal with a common
# orientation would retain nonzero global bond order as N grows.
size_rows=[]
for length in (128,256,384,512,640,768,1024,2048,4096):
 for seed in SEEDS:
  coords=np.loadtxt(DATA/f'on_L{length}_{seed}_snapshot.csv',delimiter=',',skiprows=1)
  coords=coords[(coords[:,0]>=R)&(coords[:,0]<R+length)]
  # The same interior definition across all lengths.
  val,local,ncore=orient(coords,length)
  size_rows.append((length,seed,len(coords),ncore,val,local))
np.savetxt(OUT/'sixfold_size_scan.csv',size_rows,delimiter=',',header='Lc_um,seed,N_control,N_core,global_abs_psi6,mean_local_abs_psi6',comments='')
sizes=np.asarray(size_rows)
fig,ax=plt.subplots(figsize=(5.4,3.5),layout='constrained')
for i,length in enumerate((128,256,384,512,640,768,1024,2048,4096)):
 rows=sizes[sizes[:,0]==length]
 ax.errorbar(rows[:,2].mean(),rows[:,4].mean(),yerr=rows[:,4].std(ddof=1)/np.sqrt(3),fmt='o',color='#1f77b4')
nn=np.linspace(sizes[:,2].min(),sizes[:,2].max(),200)
ax.plot(nn,0.55/np.sqrt(nn),ls='--',color='gray',label=r'guide $N^{-1/2}$')
ax.set(xscale='log',yscale='log',xlabel='Particles in control region',ylabel=r'$|\langle\psi_6\rangle|$',title='Sixfold order across 27 final snapshots')
ax.xaxis.set_major_locator(FixedLocator([250,1000,4000,7000]))
ax.xaxis.set_major_formatter(FixedFormatter(['250','1000','4000','7000']))
ax.xaxis.set_minor_locator(NullLocator())
ax.legend(frameon=False)
fig.savefig(OUT/'sixfold_size_scan.png',dpi=180);plt.close(fig)
print('size scan',[(int(L),round(float(sizes[sizes[:,0]==L,4].mean()),4)) for L in (128,256,384,512,640,768,1024,2048,4096)],flush=True)

# Consolidated four-panel figure used by the manuscript.
fig,aa=plt.subplots(2,2,figsize=(8.9,6.1),layout='constrained')
aa[0,0].scatter(sel[:,0]-(R+1000),sel[:,1],s=6,facecolors='none',edgecolors='#1f77b4',lw=.6)
aa[0,0].set(xlim=(0,64),ylim=(0,W),xlabel=r'$z$ ($\mu$m)',ylabel=r'$x$ ($\mu$m)')
img=aa[0,1].pcolormesh(kz,kx,S.T,shading='auto',vmin=0,vmax=2.2,cmap='viridis')
aa[0,1].set(xlim=(0,12),ylim=(-8,8),xlabel=r'$k_z$ ($\mu$m$^{-1}$)',ylabel=r'$k_x$ ($\mu$m$^{-1}$)')
fig.colorbar(img,ax=aa[0,1],label=r'$P_B$')
aa[1,0].plot(mid,gr,color='#d95f02',lw=1.4)
aa[1,0].axhline(1,color='gray',lw=.8,ls='--');aa[1,0].axvline(1,color='gray',lw=.8,ls=':')
aa[1,0].set(xlim=(0,6),ylim=(0,1.7),xlabel=r'$r$ ($\mu$m)',ylabel=r'$g(r)$')
for length in (128,256,384,512,640,768,1024,2048,4096):
 rows=sizes[sizes[:,0]==length]
 aa[1,1].errorbar(rows[:,2].mean(),rows[:,4].mean(),yerr=rows[:,4].std(ddof=1)/np.sqrt(3),fmt='o',color='#1f77b4',ms=4)
aa[1,1].plot(nn,0.55/np.sqrt(nn),ls='--',color='gray',lw=1,label=r'$N^{-1/2}$ guide')
aa[1,1].set(xscale='log',yscale='log',xlabel=r'$N_c$',ylabel=r'$|\langle\psi_6\rangle|$')
aa[1,1].xaxis.set_major_locator(FixedLocator([250,1000,4000,7000]))
aa[1,1].xaxis.set_major_formatter(FixedFormatter(['250','1000','4000','7000']))
aa[1,1].xaxis.set_minor_locator(NullLocator())
aa[1,1].legend(frameon=False,fontsize=11)
for tag,axis in zip('abcd',aa.flat):
 axis.text(.03,.97,f'({tag})',transform=axis.transAxes,va='top',ha='left',weight='bold',
           bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
 for spine in axis.spines.values():spine.set_visible(True)
 axis.tick_params(which='both',direction='in')
fig.savefig(Path(__file__).resolve().parent/'figures'/'v10'/'figS3.png',dpi=220)
fig.savefig(Path(__file__).resolve().parent/'figures'/'v10'/'figS3.pdf')
plt.close(fig)
