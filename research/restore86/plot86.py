from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
Q=read(S/'SUMMARY86.json');M=read(S/'MEMORY86.json');W=read(S/'COLLISION_WITNESSES86.json');F=['g0','g3','Rapfi'];figs=[]
def save(fig,name,ids,scope):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=180);fig.savefig(str(p)+'.pdf');plt.close(fig)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,scope=scope,png_sha=sha(a),pdf_sha=sha(b)))
fig,ax=plt.subplots(1,2,figsize=(11,4))
for r in M['results']:
 z=r['memory_profile'][1:];ax[0].plot([v['delay']*8 for v in z],[v['mean_weighted_L1'] for v in z],'o-',label=r['family'])
for i,r in enumerate(Q['gates']):
 v=r['gain_ci'];ax[1].errorbar(i,v[0],yerr=[[v[0]-v[1]],[v[2]-v[0]]],fmt='o',capsize=3)
ax[0].set(title='Exact empirical signed memory coefficients',xlabel='delay (backups)',ylabel='weighted L1 coefficient norm');ax[0].legend()
ax[1].set_xticks(range(6),[r['family']+'\nR'+('84' if r['group']==3 else '86') for r in Q['gates']]);ax[1].axhline(0,color='k',lw=.7);ax[1].set(title='32-backup memory, 64-backup forecast',ylabel='Markov minus memory joint Brier')
save(fig,'nonstationary_memory_kernel_and_transfer',['L16','L37','L38','L39'],'Exact two-time empirical projection and heldout endpoint Brier; signed kernel coefficients not physical RG flow.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for i,r in enumerate(W['witnesses']):
 x=r['prob'];ax[0].plot([i,i],x,'o-');ax[1].bar(i,r['binomial_TV'])
for a in ax:
 a.set_xticks(range(9),[r['family']+'\n'+['dev','R84','R86'][r['cohort']] for r in W['witnesses']],rotation=25);a.set_ylim(0,1)
ax[0].set(title='Same root, clock and coarse state',ylabel='actual next source-pair mass');ax[1].set(title='Exact conditional wave law difference',ylabel='Binomial(4,p) TV')
save(fig,'coarse_state_nonlumpability',['L16','L37','L39'],'Nine authenticated actual worst coarse-state witnesses; one-wave binomial law differs. No full NW collision observed.')
fig,ax=plt.subplots(2,3,figsize=(13,7))
for i,r in enumerate(Q['scale']):
 z=r['blocks'];h=[4*v['h'] for v in z];ax[0,i].plot(h,[v['raw_mean'] for v in z],'o-',label='pooled raw count variance');ax[0,i].plot(h,[v['compensated_mean'] for v in z],'s-',label='compensated moment');ax[0,i].plot(h,[v['predictable_mean'] for v in z],'k--',label='predictable variance');ax[0,i].set(xscale='log',title='beta='+str(round(r['beta'],3)),xlabel='block backups',ylabel='normalized trace');ax[0,i].legend(fontsize=7)
 v=r['lag'];ax[1,i].plot([4*q['h'] for q in v],[q['raw'] for q in v],'o-',label='raw centered');ax[1,i].plot([4*q['h'] for q in v],[q['compensated'] for q in v],'s-',label='compensated');ax[1,i].axhline(0,color='k',lw=.7);ax[1,i].set(xlabel='lag backups',ylabel='normalized cross-moment');ax[1,i].legend(fontsize=7)
save(fig,'compensated_dyadic_scale_and_correlation',['L38','L39'],'R77 heldout size/density strata, equal descriptive stratum means. Compensated predictable variation identity vs raw pooled variance; no stationary ACF or critical exponent.')
assert len(figs)==3 and len({f['png_sha'] for f in figs})==3
dump(S/'PLOT_MANIFEST86.json',dict(figures=figs,animations=[],sources={f:sha(S/f) for f in ['SUMMARY86.json','MEMORY86.json','SCALE86.json','AUTOCORRELATION86.json','COLLISION_WITNESSES86.json','INDEPENDENT86.json','CONVERGENCE86.json','plot86.py']}))
print('PLOTS',len(figs),flush=True)
