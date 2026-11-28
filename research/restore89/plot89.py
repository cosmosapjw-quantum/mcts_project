from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
Q=read(S/'QEC89.json');A=read(S/'SUMMARY89.json');O=B/'mqft_restore88_20261128';G=read(O/'GAUGE88.json');T=read(O/'TOPOLOGY88.json')
F=['g0','g3','Rapfi'];figs=[]
def save(fig,id,name,scope):
 fig.tight_layout(rect=(0,0,1,.95));b=S/'plots'/f'{id}_{name}';fig.savefig(str(b)+'.png',dpi=175);fig.savefig(str(b)+'.pdf');plt.close(fig)
 p=Path(str(b)+'.png');q=Path(str(b)+'.pdf')
 with Image.open(p) as im:im.verify()
 subprocess.run(['pdfinfo',str(q)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=b.name,ids=[id],scope=scope,png_sha=sha(p),pdf_sha=sha(q)))
def qec(ax):
 # original value error distribution; same logical carrier draw budget, no hardware time equivalence
 for g,name in [(0,'g0'),(3,'g3')]:
  for arm,style in [('single_iid','--'),('rep_iid','-')]:
   r=next(x for x in A['root_rows'] if x['generation']==g and x['arm']==arm);v=np.sort(r['per_search_value_MAE'])
   ax[0].step(v,np.arange(1,len(v)+1)/len(v),where='post',ls=style,label=name+' '+arm)
 ax[0].set(xlabel='per-search value MAE',ylabel='empirical CDF',title='Actual value transport')
 for k,mu in enumerate([0.,.5,1.]):
  r=next(x for x in Q['posteriors'] if x['p']==.12 and x['mu']==mu)
  ax[1].plot([1,3],[1-r['single_error'],1-r['majority_error']],'o-',label=f'model mu={mu:g}')
  for g in (0,3):
   arm=['rep_iid','rep_mix','rep_common'][k];p=next(x for x in A['root_rows'] if x['generation']==g and x['arm']==arm)
   ax[1].scatter(3+.045*(1 if g==3 else -1),1-p['metrics']['sign_error']['mean'],marker='x' if g==0 else '+',s=30)
 ax[1].set(xticks=[1,3],xlabel='used sign carriers',ylabel='decoded sign success',title='Declared noise + actual decoded (×/+)')
 for k,arm in enumerate(['rep_iid','rep_mix','rep_common']):
  z=[x for x in A['root_rows'] if x['generation']==3 and x['arm']==arm];r=z[0]
  for j,s in enumerate(r['syndromes']):
   if s['n']:
    ax[2].plot(k*5+j,s['model_residual'],'o',color='black',ms=5)
    ax[2].plot(k*5+j,s['residual'],'x',color='tab:blue',ms=6)
   else:ax[2].text(k*5+j,.01,'NA',rotation=90,fontsize=7,ha='center')
 ax[2].set(xticks=[0,1,2,3,5,6,7,8,10,11,12,13],xticklabels=['00','10','11','01']*3,xlabel='syndrome | iid, mix, common (g3)',ylabel='residual probability',title='Calibrated posterior (model ○, data ×)')
 for a in ax[:2]:a.legend(fontsize=7)
fig,ax=plt.subplots(2,3,figsize=(14,8));fig.suptitle('L23 | Explicit QEC and actual classical sign recovery',fontsize=15)
qec(ax[0])
for k,arm in enumerate(['ordinary','pauli','known_angle','estimate_plus','estimate_minus']):
 ax[1,0].plot([x['theta'] for x in Q['coherent']],[x['results'][arm]['100'] for x in Q['coherent']],label=arm)
ax[1,0].set(title='Declared quantum code, 100 cycles',xlabel='physical rotation theta (rad)',ylabel='logical entanglement fidelity',ylim=(0,1.04));ax[1,0].legend(fontsize=7)
for g,name in [(0,'g0'),(3,'g3')]:
 v=[x for x in A['prior_R67_contrasts'] if x['generation']==g and x['right']=='single_iid'][0]
 for j,k in enumerate(['top5_diff','rapfi9_diff']):
  z=v['metrics'][k];m=z['mean'];lo,hi=z['ci95'];ax[1,1].errorbar(j+(g/3-.5)*.12,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=3,label=name if j==0 else None)
ax[1,1].axhline(0,color='k',lw=.7);ax[1,1].set(xticks=[0,1],xticklabels=['g3 B4096\ntop5 proxy','Rapfi9\nmove proxy'],ylabel='recovery minus single',title='R67 decision proxy (not Elo)');ax[1,1].legend(fontsize=8)
x=A['game_results']
for i,r in enumerate(x):
 lo,hi=r['bounds'];ax[1,2].plot([i,i],[lo,hi],lw=3);ax[1,2].plot(i,(lo+hi)/2,'o',ms=3)
ax[1,2].set(xticks=range(len(x)),xticklabels=[z['match'].replace('internal_','int_').replace('ext_','ext\n').replace('recovery','rec') for z in x],ylabel='all registered score bounds',title='R68 games incl. failed outcomes',ylim=(0,1));ax[1,2].tick_params(axis='x',labelsize=7,rotation=25)
save(fig,'L23','quantum_error_correction','Actual classical sign recovery distribution, declared redundancy and conditional confidence, coherent channel and utility; no physical device or general strength claim.')
fig,ax=plt.subplots(3,3,figsize=(14,11));fig.suptitle('L21 | Advanced phenomena: corrected original 3 × 3 report',fontsize=15)
for g,name in enumerate(F):
 z=[x for x in G['contexts'] if x['family']==g]
 ax[0,0].hist([np.mean([v[0] for v in x['reconstructed']['SU2']['wilson']]) for x in z],bins=np.linspace(-1,1,21),alpha=.45,label=name)
 ax[0,1].scatter([max(x['reconstructed']['policy_gauge_error']) for x in z],[x['reconstructed']['policy_stability'][-1] for x in z],label=name)
 ax[0,2].scatter([x['reconstructed']['nodes'] for x in z],[x['reconstructed']['edges'] for x in z],label=name)
ax[0,0].set(title='Declared SU(2): Wilson trace',xlabel='mean normalized trace',ylabel='contexts')
ax[0,1].set(title='Declared covariant field update',xlabel='gauge readout residual',ylabel='policy stability (1−TV)')
ax[0,2].set(title='Actual legal graph',xlabel='vertices',ylabel='edges')
for a in ax[0]:a.legend(fontsize=7)
qec(ax[1])
for g,name in enumerate(F):
 z=[x['topology']['cubical'] for x in T['contexts'] if x['kind']=='R78' and x['family']==g]
 ax[2,0].hist([len(x['critical_thresholds']) for x in z],bins=12,alpha=.5,label=name)
 ax[2,1].scatter([x['plateau_min'] for x in z],[x['plateau_max'] for x in z],label=name)
 ax[2,2].scatter([x['final_betti'][1] for x in z if x['finite_mean_lifetime'] is not None],[x['finite_mean_lifetime'] for x in z if x['finite_mean_lifetime'] is not None],label=name)
ax[2,0].set(title='Actual score sublevel filtration',xlabel='persistent critical levels',ylabel='contexts')
ax[2,1].set(title='Discrete extrema (not flow stability)',xlabel='minimum plateaus',ylabel='maximum plateaus')
ax[2,2].set(title='Cubical topology and persistence',xlabel='final H1 rank',ylabel='mean positive finite lifetime')
for a in ax[2]:a.legend(fontsize=7)
save(fig,'L21','advanced_quantum_phenomena','Exact original three-by-three layout intent; R88 gauge/topology and R89 actual sign transport and declared QEC kept separate. No joint physical state or hardware claim.')
assert len(figs)==2 and len({x['png_sha'] for x in figs})==2
dump(S/'PLOT_MANIFEST89.json',dict(figures=figs,animations=[],sources={f:sha(S/f) for f in ['QEC89.json','SUMMARY89.json','INDEPENDENT89.json','plot89.py']},prior_sources={str((O/f).relative_to(B)):sha(O/f) for f in ['GAUGE88.json','TOPOLOGY88.json','ITEM_RECONSTRUCTION88.json']}))
print('PLOTS',len(figs),flush=True)
