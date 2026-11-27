from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
N0=read(S/'NATIVE85.json');T=read(S/'THERMAL85.json');Q=read(S/'SUMMARY85.json');F=read(S/'STATIONARY85.json');names=['g0','g3','Rapfi'];figs=[]
def save(fig,name,ids,scope):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=180);fig.savefig(str(p)+'.pdf');plt.close(fig)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,scope=scope,png_sha=sha(a),pdf_sha=sha(b)))
fig,ax=plt.subplots(1,3,figsize=(13,4))
for i,n in enumerate(names):
 r=[x for x in N0['contexts'] if x['family']==i];s=np.array([x['s_reverse'] for x in r]);mc=np.array([x['mc_ift'] for x in r]);rev=np.array([x['mc_reverse_support'] for x in r])
 ax[0].scatter(s,mc,s=8,alpha=.6,label=n);ax[1].scatter(s,rev,s=8,alpha=.6,label=n)
 ax[2].bar(i-.15,s.mean(),.3,label='exact' if i==0 else None);ax[2].bar(i+.15,mc.mean(),.3,label='forward MC' if i==0 else None)
for a in ax[:2]:a.plot([0,1],[0,1],'k:',lw=.8);a.set(xlabel='exact reverse support mass',ylabel='sample estimate');a.legend(fontsize=8)
ax[0].set_title('Conditioned forward IFT');ax[1].set_title('Independent endpoint deletion')
ax[2].set_xticks(range(3),names);ax[2].set(ylim=(0,1),title='Not the unconditional native law',ylabel='support-aware IFT');ax[2].legend()
save(fig,'native_reverse_support_and_ledger',['L18','Q19','Q21','Q23'],'All 768 H32 actual wave settings; independently sampled forward/reverse auxiliary conditional model; native full ledger inverse audited separately.')
fig,ax=plt.subplots(2,2,figsize=(12,8));eps=[0.,.05,.2,1.]
for row,protocol in enumerate(['R73_fine_detector','R83_binary_detector']):
 for i,n in enumerate(names):
  z=[x for x in Q['groups'] if x['protocol']==protocol and x['family']==n and x['alpha']==1.]
  if not z:continue
  z=sorted(z,key=lambda x:x['eps']);x=np.arange(4)+(i-1)*.06
  ex=[q['mean']['exact_ift'] for q in z];mc=[q['mean']['mc_ift'] for q in z];ax[row,0].plot(x,ex,'o--',label=n+' exact');ax[row,0].plot(x,mc,'x',label=n+' MC')
  ax[row,1].plot(x,[q['mean']['mc_ess'] for q in z],'o-',label=n)
 for a in ax[row]:a.set_xticks(range(4),eps);a.set_xlabel('detector error-mixture epsilon');a.legend(fontsize=8)
 ax[row,0].set(title=protocol.replace('_',' ')+' | full refresh',ylabel='generalized IFT');ax[row,1].set(title='Finite-sample weight degeneracy',ylabel='mean ESS / 2048 (count)')
save(fig,'feedback_absolute_irreversibility',['L08','L18','L32','Q19','Q20','Q23'],'Actual branch-energy reference, alpha=1: exact support correction, Monte Carlo IFT and finite ESS. Other alpha retained in data.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
for i,n in enumerate(names):
 z=[x for x in Q['groups'] if x['protocol']=='R83_binary_detector' and x['family']==n and x['alpha']==1.]
 z=sorted(z,key=lambda x:x['eps']);ax[0].plot(range(4),[q['mean']['mean_work'] for q in z],'o-',label=n+' W');ax[0].plot(range(4),[q['mean']['mean_heat'] for q in z],'s--',label=n+' Q')
 ax[1].plot(range(4),[q['mean']['mean_sigma'] for q in z],'o-',label=n+' Sigma');ax[1].plot(range(4),[q['mean']['jensen_lower'] for q in z],':',label=n+' bound')
 v=[x['stationary_entropy_production'] for x in F['contexts'] if x['family']==i];ax[2].scatter(np.full(len(v),i)+np.linspace(-.16,.16,len(v)),v,s=6,alpha=.65)
for a in ax[:2]:a.set_xticks(range(4),eps);a.set_xlabel('detector error-mixture epsilon');a.legend(fontsize=7,ncol=2)
ax[0].set(title='Declared reference cycle',ylabel='energy / E*');ax[1].set(title='Support-corrected Jensen bound',ylabel='entropy production (nat)')
ax[2].set_xticks(range(3),names);ax[2].set(title='Same Gibbs marginal, distinct kernel',ylabel='cycle entropy production (nat/step)');ax[2].axhline(0,color='k',ls=':',label='iid baseline');ax[2].legend(fontsize=8)
save(fig,'feedback_energy_and_stationary_identification',['L08','L18','L32','Q20','Q21','Q23'],'Declared feedback energy ledger and stationary nonreversible flux construction, no physical MCTS heat inference.')
assert len(figs)==3 and len({f['png_sha'] for f in figs})==3
dump(S/'PLOT_MANIFEST85.json',dict(figures=figs,animations=[],sources={f:sha(S/f) for f in ['NATIVE85.json','THERMAL85.json','SUMMARY85.json','STATIONARY85.json','INDEPENDENT85.json','CONVERGENCE85.json','plot85.py']}))
print('PLOTS',len(figs),flush=True)
