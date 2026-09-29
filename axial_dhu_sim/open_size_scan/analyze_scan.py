"""Finite-size analysis: open sharp windows, smooth probes, and count variances."""
from pathlib import Path
import csv,json,sys
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
from run_scan import load,ROOT,OUT
plt.rcParams.update({'font.size':10,'figure.dpi':150,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
records=[];checks=[]
def sem(a):return float(np.std(a,ddof=1)/np.sqrt(len(a)))
for case in ['on','off']:
 for L in [128,256,512]:
  runs=[load(OUT/f'{case}_L{L}_{seed}') for seed in [31,62,93]]
  ds=[v[0] for v in runs];d=np.concatenate(ds);m=runs[0][1]
  def power(re,im,norm):
   a=d[:,re]+1j*d[:,im];mean=a.mean();den=d[:,norm].mean()
   vals=np.array([np.mean(abs(v[:,re]+1j*v[:,im]-mean)**2)/den for v in ds])
   return float(vals.mean()),sem(vals),vals.tolist()
  nc=d[:,2];nbar=nc.mean();nvar=nc.var();nper=[np.mean((v[:,2]-nbar)**2)/nbar for v in ds]
  r=dict(case=case,Lc=L,R=m['R'],N_control_mean=float(nbar),N_control_Fano=float(nvar/nbar),
   N_control_Fano_SEM=sem(nper),burn=m['burn'],duration=m['duration'],n_seeds=3,
   control_density=float(nbar/L),kappa_effective=m['kappa']*float(nbar/L)/m['lambda_ref'])
  for tag,re,im,norm in [('full',284,285,283),('fraction_rect',291,292,290),
        ('fraction_taper',293,294,295),('meso_rect',299,300,298),('meso_taper',301,302,303)]:
   r['S_'+tag],r['SEM_'+tag],r['seeds_'+tag]=power(re,im,norm)
  for tag,col in [('neumann1',288),('neumann2',289)]:
   mean=d[:,col].mean();vals=[2*np.mean((v[:,col]-mean)**2)/nbar for v in ds]
   r['S_'+tag]=float(np.mean(vals));r['SEM_'+tag]=sem(vals)
  for tag,start in [('fraction',256),('meso',402)]:
   c=d[:,start:start+24];cm=c.mean(0);den=cm.mean()
   vals=[np.mean((v[:,start:start+24]-cm)**2)/den for v in ds]
   var=np.mean((c-cm)**2);cov=np.mean((c-cm)*(nc-nbar)[:,None],axis=0)
   component=np.mean(cov**2/nvar)
   r['Fano_'+tag]=float(np.mean(vals));r['Fano_SEM_'+tag]=sem(vals)
   r['global_fraction_'+tag]=float(component/var)
   r['global_Fano_'+tag]=float(component/den)
   r['residual_Fano_'+tag]=float((var-component)/den)
   r['regression_b_'+tag]=float(np.mean(cov/nvar))
   r['seeds_Fano_'+tag]=[float(v) for v in vals]
  r.update(Lo_fraction=L/2,Lo_meso=float(np.sqrt(32*L)),ell_fraction=L/4,
           ell_meso=float(np.sqrt(8*L)),q_full=2*np.pi/L,q_fraction=4*np.pi/L,
           q_taper_fraction=8*np.pi/L,q_meso=float(2*np.pi/np.sqrt(32*L)),
           q_taper_meso=float(4*np.pi/np.sqrt(32*L)))
  r['early_late_N_ratio']=float(np.concatenate([v[len(v)//2:] for v in ds])[:,2].mean()/np.concatenate([v[:len(v)//2] for v in ds])[:,2].mean())
  records.append(r)
  for v,meta in runs:
   assert np.max(abs(v[:,283]-v[:,2]))==0
   assert np.max(abs(v[:,290]-v[:,94]))==0
   assert np.max(abs(v[:,291:293]-v[:,95:97]))<1e-9
   assert np.max(abs(v[:,13:61].sum(1)-v[:,1]))==0
   assert np.max(abs(np.diff(v[:,2])-np.diff(v[:,5]-v[:,6])))==0
   assert np.max(abs(np.diff(v[:,1])-np.diff(v[:,7]-v[:,8])))==0
   assert meta['max_control_balance_error']==meta['total_balance_error']==0
   if L==128:
    assert np.max(abs(v[:,290:298]-v[:,298:306]))<1e-9
    assert np.max(abs(v[:,256:280]-v[:,402:426]))==0
   checks.append(dict(case=case,L=L,seed=meta['seed'],frames=len(v),conservation='PASS',probe_consistency='PASS'))

on=[r for r in records if r['case']=='on'];off=[r for r in records if r['case']=='off'];Ls=np.array([r['Lc'] for r in on])
slopes={}
for key in ['S_full','S_fraction_rect','S_fraction_taper','S_meso_rect','S_meso_taper','Fano_fraction','Fano_meso']:
 slopes[key]=float(-np.polyfit(np.log(Ls),np.log([r[key] for r in on]),1)[0])
summary=dict(records=records,descriptive_exponents_vs_Lc=slopes,
    note='Only three lengths and three seeds per condition. Exponents are descriptive, not asymptotic proofs. Error bars are one seed-level SEM.',
    validation=checks,software=dict(numpy=np.__version__,matplotlib=matplotlib.__version__,python=sys.version))
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
def csv_out(name,fields,rows):
 with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r[k] for k in fields} for r in rows])
csv_out('最低非零模式与通道长度.csv',['Lc','N_control_mean','Lo_fraction','q_fraction','S_fraction_rect','SEM_fraction_rect','q_full','S_full','SEM_full'],on)
csv_out('all_conditions.csv',[k for k in records[0] if not k.startswith('seeds_')],records)
csv_out('数方差与极限路径.csv',['case','Lc','ell_fraction','Fano_fraction','Fano_SEM_fraction','global_fraction_fraction','ell_meso','Fano_meso','Fano_SEM_meso','global_fraction_meso','N_control_Fano'],records)

def finish(fig,name):fig.savefig(OUT/(name+'.png'));plt.close(fig)
def lticks(ax):ax.set_xticks(Ls,[str(int(v)) for v in Ls]);ax.xaxis.set_minor_formatter(NullFormatter())
fig,ax=plt.subplots(figsize=(8,4.8))
for rs,label,color in [(on,'Global feedback ON','#087fac'),(off,'Feedback OFF','#696969')]:
 ax.errorbar(Ls,[r['S_fraction_rect'] for r in rs],yerr=[r['SEM_fraction_rect'] for r in rs],fmt='o-',color=color,label=label)
ref=on[1]['S_fraction_rect'];ax.plot(Ls,ref*(Ls/256)**-1.,'--',color='#cf9132',label='L^-1 guide');ax.plot(Ls,ref*(Ls/256)**-2.,':',color='#ab5796',label='L^-2 guide')
ax.set(xscale='log',yscale='log',xlabel=r'$L_c\ (\mu m)$',ylabel=r'$S_{O}(2\pi/L_o)$',title='Requested length scan: central rectangular window, Lo = Lc/2');lticks(ax);ax.legend(fontsize=8);finish(fig,'lowest_mode_vs_length')

fig,ax=plt.subplots(figsize=(8,4.8))
for key,se,label,c in [('S_full','SEM_full','Entire control interval, q=2pi/Lc','#915598'),('S_fraction_rect','SEM_fraction_rect','Central sharp window, q=4pi/Lc','#087fac'),('S_fraction_taper','SEM_fraction_taper','Central smooth probe, q=8pi/Lc','#168d62')]:
 ax.errorbar(Ls,[r[key] for r in on],yerr=[r[se] for r in on],fmt='o-',color=c,label=label)
ref=on[1]['S_fraction_taper'];ax.plot(Ls,ref*(Ls/256)**-2.,':',color='#168d62',label='L^-2 guide for smooth probe')
ax.set(xscale='log',yscale='log',xlabel=r'$L_c\ (\mu m)$',ylabel='Connected spectral power',title='Different windows probe different finite-size effects');lticks(ax);ax.legend(fontsize=8);finish(fig,'window_comparison')

fig,axs=plt.subplots(1,2,figsize=(11,4.5))
for rs,c,lab in [(on,'#087fac','Feedback ON'),(off,'#696969','Feedback OFF')]:
 for ax,tag,title in [(axs[0],'fraction','Counting length ell=Lc/4'),(axs[1],'meso','Counting length ell=sqrt(8 um * Lc)')]:
  ax.errorbar(Ls,[r['Fano_'+tag] for r in rs],yerr=[r['Fano_SEM_'+tag] for r in rs],fmt='o-',color=c,label=lab)
  ax.set(xscale='log',xlabel=r'$L_c\ (\mu m)$',ylabel='Count variance / mean',title=title);lticks(ax)
for ax in axs:ax.legend(fontsize=8)
fig.suptitle('Counts distinguish a fixed-fraction limit from an interior bulk limit');fig.tight_layout();finish(fig,'count_limits')

fig,axs=plt.subplots(1,2,figsize=(11,4.5))
for ax,tag in zip(axs,['fraction','meso']):
 ax.plot(Ls,[r['Fano_'+tag] for r in on],'o-',label='Total window Fano',color='#087fac')
 ax.plot(Ls,[r['global_Fano_'+tag] for r in on],'o--',label='Linear projection onto channel N',color='#cf9132')
 ax.plot(Ls,[r['residual_Fano_'+tag] for r in on],'o:',label='Orthogonal residual',color='#168d62')
 ax.set(xscale='log',xlabel=r'$L_c\ (\mu m)$',ylabel='Fano contribution',title=tag+' counting windows');lticks(ax);ax.legend(fontsize=8)
fig.tight_layout();finish(fig,'number_mode_decomposition')

fig,ax=plt.subplots(figsize=(8,4.8))
for rs,c,lab in [(on,'#087fac','Feedback ON'),(off,'#696969','Feedback OFF')]:
 ax.errorbar(Ls,[r['S_meso_taper'] for r in rs],yerr=[r['SEM_meso_taper'] for r in rs],fmt='o-',color=c,label=lab)
ref=on[1]['S_meso_taper'];ax.plot(Ls,ref*(Ls/256)**-1.,'--',label='L^-1 guide (q^2 when Lo proportional to sqrt(Lc))',color='#cf9132')
ax.set(xscale='log',yscale='log',xlabel=r'$L_c\ (\mu m)$',ylabel='Smooth interior spectral power',title='Interior window grows while its fraction of the channel shrinks');lticks(ax);ax.legend(fontsize=8);finish(fig,'mesoscopic_spectrum')

# Exact window convolution benchmark for a hypothetical infinite HU bulk.
O=np.geomspace(32,4096,200);xi=np.sqrt(.2/.02);q=2*np.pi/O
bulk=(q*xi)**2/(1+(q*xi)**2)
rect=bulk+xi/O*(1-np.exp(-O/xi))*(1-(q*xi)**2)/(1+(q*xi)**2)**2
fig,ax=plt.subplots(figsize=(8,4.8));ax.loglog(O,bulk,label='True infinite-bulk S(q), q=2pi/Lo',color='#168d62');ax.loglog(O,rect,label='Exact rectangular-window power at the same q',color='#087fac')
ax.set(xlabel=r'$L_o\ (\mu m)$',ylabel='Spectral power',title='Analytic illustration only: a sharp window can change L^-2 to L^-1');ax.legend(fontsize=8);finish(fig,'analytic_window_effect')

# Meaningful independent check of taper normalization and derivative moment.
x=np.linspace(0,1,100001);w=np.sin(np.pi*x)**2;f=w*np.exp(-4j*np.pi*x)
fp=(np.pi*np.sin(2*np.pi*x)-4j*np.pi*w)*np.exp(-4j*np.pi*x)
assert abs(np.trapezoid(f,x))<1e-10
assert abs(np.trapezoid(abs(f)**2,x)-3/8)<1e-10
assert abs(np.trapezoid(abs(fp)**2,x)/np.trapezoid(abs(f)**2,x)-52*np.pi**2/3)<1e-8
(OUT/'validation.json').write_text(json.dumps(dict(status='PASS',runs=checks,
  taper='PASS zero integral, squared norm, derivative moment',
  note='These checks validate implementation and normalization, not the physical thermodynamic limit.'),indent=2),encoding='utf-8')
print(json.dumps(dict(on=[{k:r[k] for k in ['Lc','S_fraction_rect','SEM_fraction_rect','S_fraction_taper','SEM_fraction_taper','Fano_fraction','Fano_meso','N_control_Fano']} for r in on],exponents=slopes),indent=2))

rows=''.join(f"<tr><td>{r['Lc']}</td><td>{r['Lo_fraction']:.0f}</td><td>{r['q_fraction']:.5f}</td><td>{r['S_fraction_rect']:.5f} ± {r['SEM_fraction_rect']:.5f}</td><td>{r['S_full']:.5f} ± {r['SEM_full']:.5f}</td></tr>" for r in on)
html=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>开放系统通道长度扫描</title><style>body{{max-width:1080px;margin:35px auto;padding:0 24px;font:17px/1.8 system-ui;color:#23384c}}img{{max-width:100%}}td,th{{border:1px solid #bdcbd5;padding:9px}}table{{border-collapse:collapse;width:100%}}.note{{background:#eef4f7;padding:18px}}</style>
<h1>开放软排斥系统：全域反馈的通道长度扫描</h1><p>控制长度128、256、512 μm；固定每端储液区48 μm、活动度、驱动、排斥、反馈和数值参数。每条件3个种子，反馈开启/关闭各一组，共18条记录。预热至少5个整个显式域的穿越时间，采样约14个控制区穿越时间。</p>
<h2>您要求的数据表</h2><table><tr><th>Lc / μm</th><th>中央Lo / μm</th><th>中央qmin / μm⁻¹</th><th>中央最低窗口谱 ± SEM</th><th>全控制区最低窗口谱 ± SEM</th></tr>{rows}</table>
<p>最后一列采用整个控制区，波数为2π/Lc；中央窗口采用2π/Lo。二者波数和窗口不同，不应混用。误差是一倍种子间标准误，只有3个种子。</p><img src="lowest_mode_vs_length.png">
<h2>为何必须区分窗口和极限</h2><p class="note">有限尺寸下降不能证明严格无限极限。矩形窗口有谱泄漏，体相S∝q²也可能导致最低窗口模式主要按1/Lo下降。通道总数涨落则会影响大计数窗口；因此补测平滑谱探针和两条计数极限路径，而不通过每帧固定N或扣除低频来制造超均匀。</p>
<img src="window_comparison.png"><img src="analytic_window_effect.png"><p>解析示意图不是模拟拟合，使用已知超均匀体相的精确窗口卷积。平滑谱探针采用sin²窗和m=2载波，它与均匀密度模式正交；不能把它叫作原始矩形窗口的最低模式。</p>
<h2>数方差与共同总数涨落</h2><img src="count_limits.png"><img src="number_mode_decomposition.png"><p>第一条路径ℓ=Lc/4，窗口占比固定；第二条ℓ=√(8 μm·Lc)，窗口增长但ℓ/Lc→0。线性投影方差没有从报告的总方差中删除，也不是因果归因。</p><img src="mesoscopic_spectrum.png">
<h2>结论与复现</h2><p>具体数值、描述性指数、局限和条件化判断见<a href="结果解读.md">结果解读</a>。所有原始观测量、参数、CSV和校验记录均在本目录。完整定义、有限窗口解析式及运行方法见上一级README.md。只凭三个长度不能证明Lc→∞时的严格DHU。</p></html>'''
(OUT/'report.html').write_text(html,encoding='utf-8')
