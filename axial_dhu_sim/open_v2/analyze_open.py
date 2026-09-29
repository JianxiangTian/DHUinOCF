"""Open-system statistics. Never normalize per frame or condition on N(t)."""
from pathlib import Path
import json,sys,csv
import numpy as np
try:
 import matplotlib
 if not hasattr(matplotlib,'use'):raise ImportError()
except ImportError:
 sys.path.append(str(Path(__file__).resolve().parents[2]/'analysis'/'packages'))
 if 'matplotlib' in sys.modules:del sys.modules['matplotlib']
 import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from run_suite import load
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
plt.rcParams.update({'font.size':10,'figure.dpi':145,'savefig.bbox':'tight',
                    'axes.spines.top':False,'axes.spines.right':False})
CASES=['open_ideal_gas','open_rep_off','open_rep_on','open_rep_screened',
       'periodic_rep_off','periodic_rep_on','open_rep_half_dt','open_rep_long_reservoir',
       'open_rep_fast_bath','open_rep_stiffer']
LABELS={'open_ideal_gas':'Open, no repulsion / no feedback','open_rep_off':'Open, repulsion / feedback OFF',
 'open_rep_on':'Open, repulsion / global feedback','open_rep_screened':'Open, repulsion / screened feedback',
 'periodic_rep_off':'Periodic, repulsion / feedback OFF','periodic_rep_on':'Periodic, repulsion / feedback ON',
 'open_rep_half_dt':'Half time step (3 seeds)','open_rep_long_reservoir':'Longer reservoirs (3 seeds)',
 'open_rep_fast_bath':'Faster bath (1 seed)','open_rep_stiffer':'Stronger repulsion (1 seed)'}
COL={'open_ideal_gas':'#888888','open_rep_off':'#3d3e48','open_rep_on':'#007eaa','open_rep_screened':'#d88720',
     'periodic_rep_off':'#9e599d','periodic_rep_on':'#48a977','open_rep_half_dt':'#a4578b',
     'open_rep_long_reservoir':'#57a34d','open_rep_fast_bath':'#bd6e2f','open_rep_stiffer':'#5d67a8'}
stats={};summary={};summary['definitions']={
 'spectrum':'Connected raw coordinate Fourier power divided by pooled mean observation N. Ensemble mean subtracted once, not per-frame detrending.',
 'window_variance':'Temporal+seed variance at each fixed window location, then average over locations. No per-frame normalization.',
 'uncertainty':'One SEM between independent seeds. Only 3 seeds normally; 1-seed checks have no error estimate.',
 'no_infinite_limit_claim':'Open finite windows mix modes; periodic HU formula is not fitted to these data.'}
for name in CASES:
 longfiles=[OUT/f'{name}_long_{s}_meta.json' for s in [31,62,93]]
 files=longfiles if all(f.exists() for f in longfiles) else [OUT/f'{name}_{s}_meta.json' for s in [31,62,93] if (OUT/f'{name}_{s}_meta.json').exists()]
 if not files:continue
 runs=[load(str(f).replace('_meta.json','')) for f in files]
 ds=[r[0] for r in runs]; meta=runs[0][1]; ns=len(ds);d=np.concatenate(ds)
 mean=d.mean(0);O=meta['L']/2; lengths=np.array([4,8,16,32,meta['L']/4]);q=2*np.pi*np.arange(1,17)/O
 def sem(v):return np.std(v,axis=0,ddof=1)/np.sqrt(len(v)) if len(v)>1 else np.full(np.shape(v)[1:],np.nan)
 spectra=[];sem_spec=[];obsfanos=[]
 for region in range(3):
  j=61+33*region;nbar=mean[j];amp=d[:,j+1:j+33:2]+1j*d[:,j+2:j+33:2];abar=amp.mean(0)
  per=np.array([np.mean(abs(v[:,j+1:j+33:2]+1j*v[:,j+2:j+33:2]-abar)**2,0)/nbar for v in ds])
  spectra.append(per.mean(0));sem_spec.append(sem(per));obsfanos.append(d[:,j].var()/nbar)
 counts=d[:,160:280].reshape(-1,5,24);cm=counts.mean(0)
 vper=np.array([np.mean((v[:,160:280].reshape(-1,5,24)-cm)**2,axis=(0,2)) for v in ds])
 var=vper.mean(0);den=cm.mean(1);fano=var/den
 Nc=d[:,2];Ncbar=Nc.mean();Ncvar=Nc.var();cov=np.mean((counts-cm)*(Nc-Ncbar)[:,None,None],axis=0)
 global_component=np.mean(cov**2/max(Ncvar,1e-30),axis=1)
 Nc_fano=Ncvar/Ncbar;Nc_fano_per=np.array([np.mean((v[:,2]-Ncbar)**2)/Ncbar for v in ds])
 Nt=d[:,1];flow=np.array([(v[-1,5:7]-v[0,5:7])/(v[-1,0]-v[0,0]) for v in ds])
 # Mean density outside active baths: exclude terminalmost profile bins.
 zp=(np.arange(48)+.5)*meta['T']/48;profile=d[:,13:61].mean(0)/(meta['T']/48)
 lo=0 if meta['periodic'] else meta['R']; hi=lo+meta['L']
 mid=(zp>lo+meta['L']/4)&(zp<lo+3*meta['L']/4)
 relgradient=(profile[mid].max()-profile[mid].min())/profile[mid].mean()
 early=np.concatenate([v[:len(v)//2] for v in ds]);late=np.concatenate([v[len(v)//2:] for v in ds])
 deep=float(mean[281]/max(mean[280],1e-30));current=flow.mean(0)
 row=dict(seeds=ns,N_control_mean=float(Ncbar),N_control_fano=float(Nc_fano),
  N_control_fano_SEM=float(sem(Nc_fano_per[:,None])[0]),N_observation_mean=float(mean[94]),
  S_first=float(spectra[1][0]),S_first_SEM=float(sem_spec[1][0]),
  observation_N_fano=float(obsfanos[1]),fano_window64=float(fano[-1]),
  fano_window64_SEM=float(sem(vper)[-1]/den[-1]),
  global_number_fraction_window64=float(global_component[-1]/var[-1]),
  flux_left_per_s=float(current[0]),flux_right_per_s=float(current[1]),
  control_density=float(Ncbar/meta['L']),actual_linearized_kappa=float(meta['kappa']*Ncbar/meta['L']/meta['lambda_ref']),
  mean_central_profile_range_relative=float(relgradient),
  early_N_control=float(early[:,2].mean()),late_N_control=float(late[:,2].mean()),
  close_pairs_per_particle=float(2*mean[280]/mean[94]),deep_fraction_of_close_pairs=deep,
  min_recorded_separation_um=float(d[:,282].min()),
  max_control_balance_error=max(m['max_control_balance_error'] for _,m in runs),
  max_total_balance_error=max(abs(m['total_balance_error']) for _,m in runs),
  files=[f.name for f in files],meta=meta)
 summary[name]=row;stats[name]=dict(runs=runs,mean=mean,S=np.array(spectra),Ssem=np.array(sem_spec),
  fano=fano,fsem=sem(vper)/den,var=var,lengths=lengths,q=q,profile=profile,zp=zp,
  global_part=global_component/den,residual=(var-global_component)/den,data=d)
 with (OUT/f'{name}_observables.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(['q_um^-1','S_upstream','S_center','SEM_center','S_downstream']);w.writerows(zip(q,spectra[0],spectra[1],sem_spec[1],spectra[2]))
 with (OUT/f'{name}_counts.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(['length_um','variance','mean_count','Fano','SEM_Fano','linear_projection_global_N_contribution_to_Fano','residual_Fano'])
  w.writerows(zip(lengths,var,den,fano,sem(vper)/den,global_component/den,(var-global_component)/den))

def save(fig,name):fig.savefig(OUT/(name+'.png'));plt.close(fig)
def qticks(ax):ax.set_xticks([.05,.1,.2,.5],['0.05','0.1','0.2','0.5']);ax.xaxis.set_minor_formatter(NullFormatter())
main=[c for c in CASES[:4] if c in stats]
fig,ax=plt.subplots(figsize=(8,4.8))
for c in main:
 s=stats[c];ax.errorbar(s['q'],s['S'][1],yerr=s['Ssem'][1],fmt='o-',ms=3,color=COL[c],label=LABELS[c])
ax.set(xscale='log',yscale='log',xlabel=r'$q\ (\mu m^{-1})$',ylabel=r'$S_{\mathrm{window}}(q)$',title='Open reservoirs: connected finite-window axial spectrum')
qticks(ax);ax.axhline(1,color='#bbbbbb',ls=':');ax.legend(fontsize=8);save(fig,'open_spectra')

fig,ax=plt.subplots(figsize=(8,4.8))
for c in main:
 s=stats[c];ax.errorbar(s['lengths'],s['fano'],yerr=s['fsem'],fmt='o-',color=COL[c],label=LABELS[c])
ax.set(xlabel=r'Window length $\ell\ (\mu m)$',ylabel=r'$\mathrm{Var}(N_\ell)/\langle N_\ell\rangle$',title='Open-system counting: particle number is allowed to fluctuate')
ax.axhline(1,color='#bbbbbb',ls=':');ax.legend(fontsize=8);save(fig,'open_counts')

fig,ax=plt.subplots(figsize=(8,4.7))
for c in main:ax.plot(stats[c]['zp'],stats[c]['profile'],label=LABELS[c],color=COL[c])
for z in (48,304):ax.axvline(z,color='grey',ls='--',lw=1)
ax.axvspan(0,8,color='grey',alpha=.15);ax.axvspan(344,352,color='grey',alpha=.15)
ax.set(xlabel=r'Global $z\ (\mu m)$',ylabel=r'Mean line density $(\mu m^{-1})$',title='Explicit upstream reservoir | controlled channel | downstream reservoir')
ax.legend(fontsize=8);save(fig,'density_profile')
fig,ax=plt.subplots(figsize=(8,4.5))
for c in ['open_rep_off','open_rep_on']:
 if c not in stats:continue
 d,m=stats[c]['runs'][0];ax.plot(d[:,0]-m['burn'],d[:,2],label=LABELS[c],color=COL[c],lw=.7)
ax.set(xlabel='Time after burn-in (s)',ylabel='Particles in controlled channel',title='The controlled-channel total number is not fixed')
ax.legend(fontsize=8);save(fig,'number_trace')

fig,ax=plt.subplots(figsize=(8,4.5))
for c in ['open_rep_on','open_rep_screened']:
 if c not in stats:continue
 s=stats[c];ax.plot(s['lengths'],s['fano'],'o-',color=COL[c],label=LABELS[c]+': total')
 ax.plot(s['lengths'],s['global_part'],'--',color=COL[c],label=c.replace('open_rep_','')+': correlated with channel N')
ax.set(xlabel=r'$\ell\ (\mu m)$',ylabel='Contribution to window Fano factor',title='Exact linear-regression variance decomposition (not causal attribution)')
ax.legend(fontsize=8);save(fig,'global_number_component')

fig,axs=plt.subplots(1,2,figsize=(11,4.4))
for c in ['open_rep_off','open_rep_on','periodic_rep_off','periodic_rep_on']:
 if c not in stats:continue
 s=stats[c];axs[0].loglog(s['q'],s['S'][1],'o-',ms=3,color=COL[c],label=LABELS[c]);axs[1].plot(s['lengths'],s['fano'],'o-',color=COL[c],label=LABELS[c])
qticks(axs[0]);axs[0].set(xlabel=r'$q\ (\mu m^{-1})$',ylabel='Finite-window spectrum');axs[1].set(xlabel=r'$\ell\ (\mu m)$',ylabel='Window Fano factor')
axs[0].legend(fontsize=7);fig.suptitle('Density-matched periodic controls: boundaries and feedback kernel both differ');fig.tight_layout();save(fig,'periodic_comparison')

fig,ax=plt.subplots(figsize=(8,4.7))
for j,txt in enumerate(['upstream','center','downstream']):
 s=stats['open_rep_on'];ax.loglog(s['q'],s['S'][j],'o-',ms=3,label=txt)
qticks(ax);ax.set(xlabel=r'$q\ (\mu m^{-1})$',ylabel='Finite-window spectrum',title='Position dependence inside the open controlled channel');ax.legend();save(fig,'position_dependence')

fig,axs=plt.subplots(1,2,figsize=(11,4.4))
for c in ['open_rep_on','open_rep_half_dt','open_rep_long_reservoir','open_rep_fast_bath','open_rep_stiffer']:
 if c not in stats:continue
 s=stats[c];axs[0].errorbar(s['q'],s['S'][1],yerr=s['Ssem'][1] if len(s['runs'])>1 else None,fmt='o-',ms=3,label=LABELS[c],color=COL[c]);axs[1].plot(s['lengths'],s['fano'],'o-',color=COL[c])
axs[0].set(xscale='log',yscale='log',xlabel=r'$q\ (\mu m^{-1})$',ylabel='Finite-window spectrum');qticks(axs[0]);axs[1].set(xlabel=r'$\ell\ (\mu m)$',ylabel='Window Fano factor');axs[0].legend(fontsize=7);fig.suptitle('Time step, reservoir size, bath kinetics, and repulsion sensitivity');fig.tight_layout();save(fig,'sensitivity_open')

fig,axs=plt.subplots(3,1,figsize=(11,5.4),sharex=True)
for ax,c in zip(axs,['open_ideal_gas','open_rep_off','open_rep_on']):
 path=OUT/summary[c]['files'][0].replace('_meta.json','_snapshot.csv');xy=np.loadtxt(path,delimiter=',',skiprows=1)
 ax.scatter(xy[:,0],xy[:,1],s=4,color=COL[c]);ax.axvspan(0,48,color='grey',alpha=.13);ax.axvspan(304,352,color='grey',alpha=.13)
 ax.axvline(48,color='grey',lw=.7);ax.axvline(304,color='grey',lw=.7);ax.set(ylabel=r'$x\ (\mu m)$',ylim=(0,16),title=LABELS[c])
axs[-1].set(xlabel=r'$z\ (\mu m)$',xlim=(0,352));fig.tight_layout();save(fig,'open_snapshots')

summary['comparisons']={
 'on_off_S_first_ratio':summary['open_rep_on']['S_first']/summary['open_rep_off']['S_first'],
 'on_off_Fano64_ratio':summary['open_rep_on']['fano_window64']/summary['open_rep_off']['fano_window64'],
 'repulsion_deep_overlap_ratio_relative_to_ideal':summary['open_rep_off']['deep_fraction_of_close_pairs']/summary['open_ideal_gas']['deep_fraction_of_close_pairs']}
summary['software']=dict(python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__)
def strict_json(obj):
 if isinstance(obj,dict):return {k:strict_json(v) for k,v in obj.items()}
 if isinstance(obj,list):return [strict_json(v) for v in obj]
 if isinstance(obj,float) and not np.isfinite(obj):return None
 return obj
(OUT/'summary.json').write_text(json.dumps(strict_json(summary),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
columns=['case','seeds','N_control_mean','N_control_fano','N_observation_mean','S_first','S_first_SEM','fano_window64','fano_window64_SEM','global_number_fraction_window64','flux_left_per_s','flux_right_per_s','deep_fraction_of_close_pairs']
with (OUT/'case_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
 for c in stats:w.writerow({k:c if k=='case' else summary[c][k] for k in columns})
print(json.dumps({k:{key:summary[k][key] for key in columns[1:]} for k in stats},indent=2,ensure_ascii=False))

def fmt(c,k):return f"{summary[c][k]:.4g}" if np.isfinite(summary[c][k]) else '—'
rows=''.join('<tr><td>'+LABELS[c]+'</td><td>'+str(summary[c]['seeds'])+'</td>'+''.join('<td>'+fmt(c,k)+'</td>' for k in ['S_first','S_first_SEM','fano_window64','N_control_fano'])+'</tr>' for c in stats)
html=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>开放储液区与短程排斥：模拟v2</title><style>body{{max-width:1100px;margin:35px auto;padding:0 24px;font:17px/1.8 system-ui;color:#23364b}}h1,h2{{line-height:1.4}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #bdcbd5;padding:8px}}.note{{background:#eef4f7;padding:18px}}a{{color:#0878a1}}</style>
<h1>短程排斥与开放储液区：模拟实验v2</h1><p>由粒子坐标动力学直接生成的实际结果；Python组织运行和分析，C++执行邻居搜索与粒子更新。</p>
<p class="note">新增软排斥势U(r)=ε(1−r/d)²/2（r&lt;d），ε=20 kBT、d=1 μm；控制通道长256 μm、宽16 μm，两端各有48 μm显式储液区。只有最远端8 μm的浴区允许粒子交换。主通道严格通过截面输运改变粒子数，不固定N(t)，不把出口粒子送回入口。</p>
<h2>直接比较结果</h2><table><tr><th>条件</th><th>种子数</th><th>中央窗口最低模S</th><th>种子间SEM</th><th>64 μm窗口Fano</th><th>通道总数Fano</th></tr>{rows}</table>
<p>中央观测区长128 μm，最低非零波数约0.0491 μm⁻¹。S是开放有限窗口的连通谱，不等于无限体相S；所有归一化用跨帧、跨种子的平均粒子数。误差是种子间一倍SEM，3个种子不足以作高精度判断。</p>
<img src="open_spectra.png"><img src="open_counts.png">
<h2>密度、流动与收支</h2><img src="density_profile.png"><img src="number_trace.png"><p>全部已读取轨道逐步满足ΔN控制区=左端净入流−右端净出流；整个模拟域的粒子数变化严格等于远端浴的插入数减移除数。浴采用连续时间生灭过程，位置不动时满足局部详细平衡；理想气体极限的插入为独立泊松过程，删除为逐粒子独立过程。</p>
<h2>为何要保留通道总数涨落</h2><img src="global_number_component.png"><p>虚线为窗口计数对通道总数的线性投影方差Cov(N窗口,N通道)²/Var(N通道)，实线为总方差。该分解用于识别共同涨落，并未从报告的原始数据中扣除它；它是统计关联，不是已证明的因果来源。全通道零模式未被反馈直接恢复，故局部长波抑制不自动意味着开放窗口的无限尺度超均匀性。</p>
<h2>边界、位置和数值敏感性</h2><img src="periodic_comparison.png"><p>周期参照的N取开放无反馈条件的平均N四舍五入。比较同时改变了边界和反馈核的边界条件，不能把所有差异只归因于储液区。</p><img src="position_dependence.png"><img src="sensitivity_open.png"><p>减半步长及加长储液区各3个种子；加倍浴交换速率及增强排斥各1个种子，只是初步敏感性检查。浴活动度相同不保证有无排斥、开关反馈后的局部密度完全相同。</p>
<h2>瞬时构型</h2><img src="open_snapshots.png"><p>灰色区域是显式储液区。颗粒仍有有限的软重叠概率，本模型不是硬球；详情见重叠统计、源代码和README。</p>
<h2>结论边界</h2><p>本轮直接检验“加入软排斥和开放粒子交换后，反馈能否在观测区压低轴向涨落”。没有把周期理论公式直接拟合到开放谱，也没有宣称严格DHU已经验证。仍未包含有限厚度、多体水动力、压力驱动剪切、成像误差和执行器饱和。</p>
<p>详细参数、理论、复现和限制：<a href="../README_v2.md">README_v2.md</a>；数据摘要：<a href="case_summary.csv">case_summary.csv</a>。</p></html>'''
(OUT/'report.html').write_text(html,encoding='utf-8')
