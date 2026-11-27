from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
R=read(S/'CRITICAL87.json');Q=read(S/'SUMMARY87.json');E=read(S/'ERRATUM86.json');figs=[];LS=[13,15,17,19,21];BETA=[0.,10/3,8.,16.]
def save(fig,name,ids,scope):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=180);fig.savefig(str(p)+'.pdf');plt.close(fig)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,scope=scope,png_sha=sha(a),pdf_sha=sha(b)))
fig,ax=plt.subplots(1,3,figsize=(13,4))
for bi,b in enumerate(BETA+[None]):
 q=[x for x in Q['metrics'] if x['bi']==bi];h=[x['h'] for x in q];label='log L^2' if b is None else f'beta={b:.3g}'
 ax[0].plot(h,[x['raw_trace'] for x in q],'o-',label=label)
 ax[1].plot(h,[x['trace'] for x in q],'o-',label=label)
 ax[2].plot(h,[x['uncentered_innovation_trace'] for x in q],'o-',label=label)
for a,t in zip(ax,['Root-centered raw covariance','Predictable innovation covariance','Sample compensated second moment']):a.set(xlabel='prefix backups',ylabel='normalized trace',title=t);a.legend(fontsize=7)
ax[1].axhline(1,color='k',ls=':')
save(fig,'bounded_innovation_vs_raw_scaling',['L17','L26','L38','Q32'],'Fresh R87 equal-stratum descriptive means, actual full-action per-selection bracket, raw root-centered variance vs compensated moment. Prefixes not separate budgets.')
fig,ax=plt.subplots(2,3,figsize=(13,8))
for arm in (0,1):
 for L in LS:
  a=[x for x in R['strata'] if x['L']==L and x['cohort']==2 and x['arm']==arm and x['h']==64 and x['bi']<4]
  a=sorted(a,key=lambda x:x['bi']);ax[arm,0].plot(BETA,[x['ratio'] for x in a],'o-',label=str(L));ax[arm,1].plot(BETA,[x['mean'][0] for x in a],'o-',label=str(L))
 z=[x for x in R['gates'] if x['arm']==arm]
 for j,k in enumerate(('smooth','crossover','critical')):ax[arm,2].bar(np.arange(len(z))+(j-1)*.25,[x['errors'][k] for x in z],.25,label=k)
 ax[arm,2].set_xticks(range(len(z)),[x['observable']+'\nR'+('77' if x['cohort']==1 else '87') for x in z]);ax[arm,2].legend(fontsize=7)
 for a in ax[arm,:2]:a.set(xlabel='beta');a.legend(title='L',fontsize=7)
 ax[arm,0].set(title=('fixed8' if arm==0 else 'dense')+' | low/high ratio',ylabel='dimensionless ratio')
 ax[arm,1].set(title='Full-action predictable trace',ylabel='trace')
 ax[arm,2].set(title='L21 heldout prediction',ylabel='RMSE')
save(fig,'controlled_criticality_model_comparison',['L17','L26','L36','L37','L38','Q32'],'Actual fresh size curves and R76-selected heldout L21 fits; not established RG invariant or exponent.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
for n in (100,1000,10000,100000,1000000):
 q=[x for x in R['entropy_null'] if x['n']==n];ax[0].plot([x['s'] for x in q],[x['pmax'] for x in q],'o-',label=str(n))
ax[0].set(xlabel='beta*gap - log(n-1)',ylabel='top probability',title='Noninteracting exact collapse');ax[0].legend(fontsize=7)
for arm in (0,1):
 for bi in (1,3,4):
  z=[x for x in R['frozen'] if x['arm']==arm and x['bi']==bi];lbl=('fixed' if arm==0 else 'dense')+' '+['0','10/3','8','16','logL²'][bi];ax[1].plot(LS,[x['frozen_mean'][3] for x in z],'o--',label=lbl+' frozen');ax[1].plot(LS,[x['actual_H32_mean_pmax'] for x in z],'x-',label=lbl+' adaptive')
ax[1].set(xlabel='L',ylabel='top probability',title='H32 frozen vs adaptive');ax[1].legend(fontsize=5,ncol=2)
z=Q['corrected_R86'];x=np.arange(3)
for j,k in enumerate(['predictable_old','predictable_corrected','compensated']):ax[2].bar(x+(j-1)*.24,[r[k] for r in z],.24,label=k.replace('_',' '))
ax[2].set_xticks(x,['0','10/3','16']);ax[2].set(xlabel='beta',ylabel='normalized radial trace',title='R86 startup erratum');ax[2].legend(fontsize=7)
save(fig,'entropy_competition_and_startup_correction',['L17','L26','L36','L37','L38'],'Exact independent categorical null, actual frozen score controls, R86 startup bracket correction; no causal identification or physical criticality.')
assert len(figs)==3 and len({x['png_sha'] for x in figs})==3
dump(S/'PLOT_MANIFEST87.json',dict(figures=figs,animations=[],sources={f:sha(S/f) for f in ['CRITICAL87.json','SUMMARY87.json','ERRATUM86.json','INDEPENDENT87.json','CONVERGENCE87.json','plot87.py']}))
print('PLOTS',len(figs),flush=True)
