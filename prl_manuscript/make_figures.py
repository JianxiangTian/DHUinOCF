from pathlib import Path
import sys,csv,json,shutil,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent;PROJ=ROOT.parent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from matplotlib.ticker import FixedLocator,FixedFormatter,NullFormatter
OUT=ROOT/'figures'/'v10';OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'data';DATA.mkdir(exist_ok=True)
E=PROJ/'axial_dhu_sim'/'evidence_completion'/'results';X=PROJ/'axial_dhu_sim'/'extended_scan'/'results';V=PROJ/'axial_dhu_sim'/'open_v2'/'results'
sources={}
def load(path):
 actual=path if path.exists() else DATA/path.name
 sources[str(path.relative_to(PROJ))]=hashlib.sha256(actual.read_bytes()).hexdigest()
 if actual.resolve()!=(DATA/path.name).resolve():shutil.copy2(actual,DATA/path.name)
 return list(csv.DictReader(actual.open(encoding='utf-8-sig')))
def vals(rs,key):return np.array([float(r[key]) for r in rs])
def panel(ax,label):ax.text(-.16,1.05,label,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
def ticks(ax):ax.set_xscale('log');ax.xaxis.set_major_locator(FixedLocator([128,512,4096]));ax.xaxis.set_major_formatter(FixedFormatter(['128','512','4096']));ax.xaxis.set_minor_formatter(NullFormatter());ax.tick_params(axis='x',labelsize=10)
def finish(fig,name):
 def frame(ax):
  if ax.axison:
   ax.set_box_aspect(1)
   for spine in ax.spines.values():
    spine.set_visible(True);spine.set_linewidth(.7)
   ax.tick_params(which='both',direction='in')
  for child in ax.child_axes:frame(child)
 for ax in fig.axes:frame(ax)
 fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');fig.savefig(OUT/f'{name}.png',dpi=300,bbox_inches='tight');plt.close(fig)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.labelsize':10.5,'legend.fontsize':9,'xtick.labelsize':10,'ytick.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'lines.linewidth':1.2,'errorbar.capsize':2})
blue='#0072B2';orange='#D55E00';green='#009E73';grey='#777777';purple='#7B4EA3'
b=load(E/'open_boundary_windows.csv');amp=load(E/'parameter_free_amplitudes.csv');sp=load(X/'particle_spectra.csv');alph=load(X/'particle_alpha.csv');controls=load(E/'independent_controls.csv');corr=load(E/'control_correlations.csv');dyn=load(E/'open_time_correlations.csv')
for name in ['independent_control_seeds.csv','interior_count_curves.csv','validation.json','new_run_validation.json']:
 p=E/name;actual=p if p.exists() else DATA/name;sources[str(p.relative_to(PROJ))]=hashlib.sha256(actual.read_bytes()).hexdigest()
 if actual.resolve()!=(DATA/name).resolve():shutil.copy2(actual,DATA/name)
for name in ['particle_lowest_modes.csv','temporal_block_diagnostics.csv','early_late_spectrum_check.csv']:
 if (X/name).exists():load(X/name)
fig=plt.figure(figsize=(3.45,5.5));gs=fig.add_gridspec(2,1,height_ratios=[1.25,2.8],hspace=.62);ax=fig.add_subplot(gs[0]);panel(ax,'(a)');ax.set(xlim=(0,10),ylim=(-1.0,3.1));ax.axis('off')
for a,w,c,lab in [(0,2,'#eeeeee','Bath'),(2,6,'#e5f3fa','Control region'),(8,2,'#eeeeee','Bath')]:
 ax.add_patch(Rectangle((a,0),w,1.25,fc=c,ec='#555555',lw=.8));ax.text(a+w/2,.65,lab,ha='center',va='center',fontsize=9)
for a in [0,9.55]:ax.add_patch(Rectangle((a,0),.45,1.25,fc='#aaaaaa',ec='none'))
for a in [2,8]:ax.annotate('',(a+.6,1.52),(a-.6,1.52),arrowprops={'arrowstyle':'<->','lw':1})
ax.annotate('',(6.7,2.28),(3.3,2.28),arrowprops={'arrowstyle':'->','color':blue,'lw':1.6});ax.text(5,2.4,r'Mean flow $u\hat{\mathbf{z}}$',ha='center')
ax.text(5,-.48,'Projected density controls the feedback force',ha='center',fontsize=9)
ax=fig.add_subplot(gs[1]);panel(ax,'(b)')
for name,label,color in [('open_rep_off','Feedback off',grey),('open_rep_on','Unscreened',blue),('open_rep_screened','Screened',orange)]:
 r=load(V/(name+'_observables.csv'));ax.errorbar(vals(r,'q_um^-1'),vals(r,'S_center'),yerr=vals(r,'SEM_center'),fmt='o-',ms=3,color=color,label=label)
ax.set(xscale='log',yscale='log',xlabel=r'$q\ (\mu\mathrm{m}^{-1})$',ylabel=r'$S_{\rm rect}(q)$');ax.xaxis.set_major_locator(FixedLocator([.05,.1,.2,.5]));ax.xaxis.set_major_formatter(FixedFormatter(['0.05','0.1','0.2','0.5']));ax.xaxis.set_minor_formatter(NullFormatter());ax.set_ylim(.045,1.3);ax.legend(loc='lower right');ax.text(.03,.98,r'$L_c=256\,\mu\mathrm{m}$, $L_{\mathrm{w}}=L_c/2$',transform=ax.transAxes,va='top',fontsize=9);finish(fig,'fig1')
fig,axs=plt.subplots(1,3,figsize=(7.05,2.42),gridspec_kw={'wspace':.6});cols=[grey,green,orange,blue]
for L,color,marker in zip([128,512,2048,4096],cols,['o','s','D','^']):
 r=[r for r in sp if r['window']=='Hann' and int(r['Lc'])==L and int(r['m'])<=16];axs[0].plot(vals(r,'q'),vals(r,'S'),marker+'-',ms=3,color=color,label=str(L))
axs[0].set(xscale='log',yscale='log',xlabel=r'$q\ (\mu\mathrm{m}^{-1})$',ylabel=r'$S_H(q)$')
r=[r for r in alph if r['window']=='Hann' and r['band']=='m2_to_m8'];axs[1].errorbar(vals(r,'Lc'),vals(r,'alpha'),yerr=vals(r,'alpha_SEM'),fmt='o-',ms=3,color=blue);axs[1].axhline(2,ls='--',color='black',lw=.8);ticks(axs[1]);axs[1].set(xlabel=r'$L_c$ ($\mu\mathrm{m}$)',ylabel=r'Fitted $\alpha$',ylim=(.5,2.15))
r=[r for r in amp if r['Lc']=='4096'];axs[2].errorbar(vals(r,'q'),vals(r,'measured_over_predicted'),yerr=vals(r,'SEM')/vals(r,'S_predicted_q2'),fmt='o',ms=3,color=blue);axs[2].axhline(1,ls='--',color='black',lw=.8);axs[2].set(xlabel=r'$q\ (\mu\mathrm{m}^{-1})$',ylabel=r'$S_H/S_H^{\mathrm{pred}}$',ylim=(.95,1.05));axs[2].text(.05,.95,r'$L_c=4096\,\mu\mathrm{m}$',transform=axs[2].transAxes,va='top',fontsize=9)
for ax,lab in zip(axs,['(a)','(b)','(c)']):panel(ax,lab)
finish(fig,'fig2')
fig,axs=plt.subplots(1,3,figsize=(7.05,2.55),gridspec_kw={'width_ratios':[.9,1.15,1.15],'wspace':.9});ax=axs[0];ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
for y,label,f in [(.75,r'$\ell=L_c/4$',.25),(.3,r'$\ell=\sqrt{8\,\mu\mathrm{m}\,L_c}$',.13)]:
 ax.add_patch(Rectangle((0,y),1,.12,facecolor='#eeeeee',edgecolor='#777777'));ax.add_patch(Rectangle(((1-f)/2,y),f,.12,facecolor=blue,alpha=.8));ax.text(.5,y+.19,label,ha='center',fontsize=9)
ax.text(.5,.04,'Growing interior window\nwith vanishing fraction',ha='center',fontsize=9)
for tag,label,color in [('fraction',r'$\ell=L_c/4$',blue),('interior',r'$\ell\propto\sqrt{L_c}$',orange)]:axs[1].errorbar(vals(b,'Lc'),vals(b,tag+'_Fano'),yerr=vals(b,tag+'_SEM'),fmt='o-',ms=3,color=color,label=label)
axs[1].set(yscale='log',xlabel=r'$L_c$ ($\mu\mathrm{m}$)',ylabel=r'$\mathrm{Var}(N_\ell)/\langle N_\ell\rangle$');ticks(axs[1]);axs[1].legend(loc='lower left')
for key,label,color in [('fraction_Fano','Total',blue),('fraction_common_Fano','Population',orange),('fraction_residual_Fano','Residual',green)]:axs[2].plot(vals(b,'Lc'),vals(b,key),'o-',ms=3,color=color,label=label)
axs[2].set(yscale='log',xlabel=r'$L_c$ ($\mu\mathrm{m}$)',ylabel='Fano contribution');ticks(axs[2]);axs[2].legend(loc='lower left',fontsize=8.2);axs[2].set_title(r'$\ell=L_c/4$',fontsize=9)
for ax,lab in zip(axs,['(a)','(b)','(c)']):panel(ax,lab)
finish(fig,'fig3')
fig,axs=plt.subplots(1,2,figsize=(7.05,3.0),gridspec_kw={'wspace':.55});names=['gain_half','baseline','gain_double']
for name,color in zip(names,[green,blue,orange]):
 r=[r for r in corr if r['case']==name and r['mode']=='1'];ts=sorted(set(float(v['t']) for v in r));cs=[];es=[]
 for t in ts:
  v=[complex(float(x['C_re']),float(x['C_im'])) for x in r if float(x['t'])==t];cs.append(abs(np.mean(v)));es.append(np.std(abs(np.array(v)),ddof=1)/np.sqrt(3))
 cr=next(x for x in controls if x['case']==name and x['mode']=='1');k=float(cr['kappa_effective']);axs[0].errorbar(ts,cs,yerr=es,fmt='o',ms=2.5,color=color,label=cr['kappa_reference']);tt=np.linspace(0,max(ts),120);axs[0].plot(tt,np.exp(-k*tt),color=color)
axs[0].set(xlabel=r'$t\ (\mathrm{s})$',ylabel=r'$|C_q(t)/C_q(0)|$',yscale='log');axs[0].legend(title=r'$\kappa_{\mathrm{ref}}$ ($\mathrm{s}^{-1}$)',title_fontsize=7)
r=[r for r in controls if r['case'] in names and r['mode']=='1'];axs[1].errorbar(vals(r,'kappa_reference'),vals(r,'gamma'),yerr=vals(r,'gamma_SEM'),fmt='o',ms=4,color=blue);axs[1].plot(vals(r,'kappa_reference'),vals(r,'gamma_predicted_leading'),color='black',lw=1);axs[1].set(xlabel=r'$\kappa_{\mathrm{ref}}$ ($\mathrm{s}^{-1}$)',ylabel=r'$\Gamma(q_1)$ ($\mathrm{s}^{-1}$)')
ins=axs[1].inset_axes([.19,.55,.31,.35]);ins.errorbar(vals(r,'kappa_reference'),vals(r,'S'),yerr=vals(r,'S_SEM'),fmt='o',ms=2,color=blue);ins.plot(vals(r,'kappa_reference'),vals(r,'S_predicted_leading'),color='black',lw=.8);ins.set(title=r'$S(q_1)$',xticks=[.01,.04]);ins.tick_params(labelsize=5);ins.title.set_size(6);ins.text(.04,.08,r'$\kappa_{\rm ref}$',fontsize=6,transform=ins.transAxes)
for ax,lab in zip(axs,['(a)','(b)']):panel(ax,lab)
finish(fig,'fig4')
# Supplementary figures reproduce raw diagnostics rather than selected passing runs.
fig,axs=plt.subplots(1,2,figsize=(7.05,2.6))
for tag,label,color in [('fraction','Fixed fraction',blue),('interior','Interior',orange)]:
 for half,marker in [('early','o'),('late','s')]:axs[0].plot(vals(b,'Lc'),vals(b,f'{tag}_{half}_Fano'),marker+'-',ms=3,color=color,label=f'{label}, {half}')
axs[0].set(xlabel=r'$L_c$ ($\mu\mathrm{m}$)',ylabel='Half-record Fano');ticks(axs[0]);axs[0].legend(fontsize=9,loc='lower left')
for tag,color in [('upstream',green),('center',blue),('downstream',orange)]:axs[1].plot(vals(b,'Lc'),vals(b,tag+'_rect_S'),'o-',ms=3,color=color,label=tag)
axs[1].set(yscale='log',xlabel=r'$L_c$ ($\mu\mathrm{m}$)',ylabel=r'$S_{\rm rect}(q_1)$');ticks(axs[1]);axs[1].legend();panel(axs[0],'(a)');panel(axs[1],'(b)');fig.tight_layout();finish(fig,'figS1')
fig,axs=plt.subplots(1,2,figsize=(7.05,2.6));rs=[r for r in controls if r['mode']=='1'];labels=['Half gain','Baseline','Double gain','Zero flow','2 s update']
axs[0].errorbar(np.arange(5),vals(rs,'S_over_prediction'),yerr=vals(rs,'S_SEM')/vals(rs,'S_predicted_leading'),fmt='o',color=blue);axs[0].axhline(1,color='black',ls='--');axs[0].set_xticks(range(5),labels,rotation=25,ha='right');axs[0].set(ylabel=r'$S(q_1)/S^{\mathrm{pred}}(q_1)$')
axs[1].errorbar(vals(dyn,'Lc'),vals(dyn,'gamma_measured'),yerr=vals(dyn,'gamma_SEM'),fmt='o',color=blue);axs[1].plot(vals(dyn,'Lc'),vals(dyn,'gamma_predicted'),color='black');ticks(axs[1]);axs[1].set(xlabel=r'$L_c$ ($\mu\mathrm{m}$)',ylabel=r'Open Hann decay rate ($\mathrm{s}^{-1}$)');panel(axs[0],'(a)');panel(axs[1],'(b)');fig.tight_layout();finish(fig,'figS2')
(OUT/'plot_input_manifest.json').write_text(json.dumps(sources,indent=2),encoding='utf8')
print('Four main and two supplemental figures generated; source hashes recorded.')
