from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
A=read(S/'RESPONSE74.json');P=read(S/'PETZ74.json');V=read(S/'INDEPENDENT74.json');U=read(S/'AUDIT74.json')
families=A['results'];contexts=P['contexts'];theta=P['reference_theta'];names=[f['family'] for f in families];wave=np.arange(32);figs=[];anims=[]
def save(fig,name,ids,scope):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=150);fig.savefig(str(p)+'.pdf');plt.close(fig)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 assert b.read_bytes()[:4]==b'%PDF'
 figs.append(dict(figure=name,ids=ids,scope=scope,png_sha=sha(a),pdf_sha=sha(b)))
def curve(ax,y,label,shade=False):
 y=np.asarray(y);ax.plot(wave,y[0],label=label)
 if shade:ax.fill_between(wave,y[1],y[2],alpha=.12)
for k,title in enumerate(['source selection fraction','signed root value mark']):
 fig,ax=plt.subplots(1,3,figsize=(14,4),sharey=True)
 for a,f in zip(ax,families):
  for key,lab in [('direct_fd','direct ± pulse'),('reweighted_fd','exact finite-h reweighting'),('score_derivative','infinitesimal score')]:
   curve(a,np.asarray(f[key])[:,:,k],lab,key=='direct_fd')
  a.axvline(8,ls=':',color='k');a.axhline(0,lw=.6,color='k');a.set(title=f['family'],xlabel='wave index');a.legend(fontsize=7)
 ax[0].set_ylabel('response / logit');fig.suptitle(title+' | pointwise root-block intervals')
 save(fig,'time_response_'+str(k),['G11','G12','G10','L18'],'Finite native pulse vs reweighting and score derivative; not equilibrium physical FDT.')
fig,ax=plt.subplots(2,3,figsize=(14,7),sharex=True)
for i,f in enumerate(families):
 for j in range(2):
  a=ax[j,i]
  for key,lab in [('count_covariance','count covariance'),('predictable_drift_covariance','predictable term'),('score_derivative','difference = response')]:curve(a,np.asarray(f[key])[:,:,j],lab,key=='predictable_drift_covariance')
  a.axvline(8,ls=':',color='k');a.axhline(0,lw=.6,color='k');a.set_title(f['family']+' | '+['source fraction','root mark'][j]);a.set_xlabel('wave');a.legend(fontsize=7)
save(fig,'response_decomposition',['G11','G12','L18'],'Adaptive predictable term versus naive count covariance; estimates not all significant.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for i,f in enumerate(families):
 for j,key in enumerate(['baseline_Lplus','baseline_Lminus','plus_inverse_L','minus_inverse_L']):
  m,lo,hi=f[key];ax[0].errorbar(i+(j-1.5)*.18,m,yerr=[[m-lo],[hi-m]],fmt='o',label=key if i==0 else None)
 for j,key in enumerate(['plus_mean_logLR','minus_mean_logLR']):
  m,lo,hi=f[key];ax[1].errorbar(i+(j-.5)*.2,m,yerr=[[m-lo],[hi-m]],fmt='o',label=key if i==0 else None)
for a in ax:a.set_xticks(range(3),names);a.legend(fontsize=7)
ax[0].axhline(1,ls=':');ax[0].set(title='Native likelihood normalization',ylabel='mean ratio');ax[1].axhline(0,ls=':');ax[1].set(title='Same-order protocol KL estimate',ylabel='mean log likelihood ratio (nat)')
save(fig,'native_path_ratio',['L18','Q07','Q05'],'Same-order native path likelihood ratio; no physical reversal.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
for f in families:
 p=np.asarray(f['baseline_profiles'])
 for j,k in enumerate([3,4,5]):curve(ax[j],p[:,:,k],f['family'],True)
for a,t in zip(ax,['conditional Gibbs capacity / kB','score-law Shannon entropy (nat)','participation 1 / sum(p²)']):a.set(title=t,xlabel='wave');a.legend(fontsize=8)
save(fig,'score_gibbs_observables',['G04','G05','G15'],'Dimensionless conditional Gibbs score law; not measured physical heat capacity.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for f in families:
 p=np.asarray(f['baseline_profiles'])[0]
 ax[0].plot(wave,p[:,6],label=f['family'])
 ax[1].plot(wave,np.cumsum(p[:,8]),label=f['family']+' quench')
 ax[1].plot(wave,np.cumsum(p[:,9]),'--',label=f['family']+' redistribution')
ax[0].set(title='Conditional score-energy mean',ylabel='E / declared E*',xlabel='wave');ax[1].set(title='Exact score ledger components',ylabel='cumulative change / E*',xlabel='wave')
for a in ax:a.legend(fontsize=7)
save(fig,'conditional_score_balance',['G05','L18','Q08'],'Exact algebraic score-energy balance; not native heat/work.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
for j,key in enumerate(['quantum_rel_loss','TPM_mean_EP','recovery_TD']):
 a=ax[j];vals=[[q['reference'][i][key] for q in contexts] for i in range(3)];a.boxplot(vals,tick_labels=[str(x) for x in theta]);a.set(title=['quantum relative entropy loss','TPM log ratio mean','Petz recovery trace distance'][j],xlabel='reference theta',ylabel=['nat','nat','trace distance'][j])
save(fig,'petz_reference_dependence',['L18','Q05','Q07','Q38'],'Reference-dependent quantum contraction; TPM measured protocol separate.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for i,t in enumerate(theta):
 rows=[q['reference'][i] for q in contexts]
 for k,key in enumerate(['reverse_completeness_max_error','reference_recovery_error','TPM_ratio_max_error']):
  ax[0].scatter(i+(k-1)*.2,max(1e-18,max(q[key] for q in rows)),label=key if i==0 else None)
 ax[1].scatter(np.full(len(rows),i),[q['TPM_IFT']-1 for q in rows],s=8,alpha=.5)
ax[0].set_yscale('log');ax[0].legend(fontsize=7);ax[0].set(ylabel='max absolute error',title='Reverse channel audit');ax[1].set(title='Exact finite TPM fluctuation identity',ylabel='E exp(-sigma) − 1')
for a in ax:a.set_xticks(range(3),theta);a.set_xlabel('reference theta')
save(fig,'petz_reverse_audit',['Q05','Q07','Q08'],'Exact reference recovery and normalized reverse TPM path sum, not inverse tree.')
fig,ax=plt.subplots(1,2,figsize=(12,4))
for name,key,ls in [('native driven channel','switch_ledger','-'),('identity free control','identity_free_control','--')]:
 e=np.array([q[key]['energy'] for q in contexts]);s=np.array([q[key]['conditional_entropy'] for q in contexts]);ax[0].plot(range(6),e.mean(0),ls,marker='o',label=name);ax[1].plot(range(4),s.mean(0),ls,marker='o',label=name)
ax[0].set_xticks(range(6),['start','meas A','switch Z','free','switch P','meas B'],rotation=25);ax[1].set_xticks(range(4),['start','meas A','free','meas B'],rotation=25);ax[0].set(ylabel='E / E*',title='Hamiltonian returns; state need not');ax[1].set(ylabel='conditional entropy (nat)',title='Conditional ensemble entropy')
for a in ax:a.legend(fontsize=8)
save(fig,'switch_hamiltonian_ledger',['L18','Q06','Q07','Q08','Q38'],'Explicit switched H model vs identity-free control, not closed ideal thermodynamic cycle.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
w=np.array([q['switch_ledger']['switch_work'] for q in contexts]);ent=np.array([q['switch_ledger']['free_entropy_gain'] for q in contexts])
ax[0].scatter(w[:,0],w[:,1],s=18);ax[0].axhline(0,lw=.6,color='k');ax[0].set(xlabel='first switch work / E*',ylabel='second switch work / E*',title='Protocol work per context')
ax[1].scatter(w.sum(1),ent,s=18);ax[1].set(xlabel='net switch work / E*',ylabel='free entropy gain (nat)',title='Zero free energy exchange, nonzero entropy change')
save(fig,'work_entropy_relation',['G05','L18','Q07','Q38'],'Explicit model Hamiltonians; no physical bath heat inference.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
th=np.linspace(0,7,200);ax[0].plot(th,th**2/(2+2*np.cosh(th)));ax[0].scatter(theta,[t*t/(2+2*np.cosh(t)) for t in theta]);ax[0].set(xlabel='declared theta = E*/(kB Tmodel)',ylabel='C / kB',title='Auxiliary equilibrium qubit capacity')
q=[r['record_exchange_KL'] for r in contexts];ax[1].hist(q,bins=14);ax[1].set(xlabel='KL(P(i,j) || P(j,i)) (nat)',ylabel='contexts',title='Ordered-record asymmetry, not heat')
save(fig,'reference_capacity_record_arrow',['G04','G05','Q07','Q38'],'Auxiliary equilibrium reference capacity and exact record-order KL.')
# Actual data-driven frame animation, with explicit noncyclic state and model units.
frames=[]
for k in range(6):
 fig,a=plt.subplots(figsize=(7,4))
 for name,key,ls in [('adaptive conditional channel','switch_ledger','-'),('identity-free control','identity_free_control','--')]:
  e=np.array([q[key]['energy'] for q in contexts]);a.plot(range(k+1),e.mean(0)[:k+1],ls,marker='o',label=name)
 a.set_xticks(range(6),['start','meas A','switch Z','free','switch P','meas B'],rotation=20);a.set_xlim(-.2,5.2);a.set_ylim(-.45,.45);a.set_ylabel('mean E / E*');a.set_title('Switched Hamiltonian loop: state not a closed cycle');a.legend(fontsize=8);fig.tight_layout()
 b=BytesIO();fig.savefig(b,format='png',dpi=100);plt.close(fig);b.seek(0);frames.append(Image.open(b).convert('RGB').copy())
p=S/'plots'/'switch_protocol.gif';frames[0].save(p,save_all=True,append_images=frames[1:],duration=850,loop=0)
with Image.open(p) as im:
 assert im.n_frames==6
 for k in range(im.n_frames):im.seek(k);im.load()
anims.append(dict(file=p.name,ids=['Q06'],scope='Six-stage actual-conditioned auxiliary Hamiltonian loop vs identity-free control; not a physical ideal cycle.',sha256=sha(p),frames=6))
assert len(figs)==11 and len({q['png_sha'] for q in figs})==11
dump(S/'PLOT_MANIFEST74.json',dict(figures=figs,animations=anims,sources={f:sha(S/f) for f in ['RESPONSE74.json','PETZ74.json','INDEPENDENT74.json','AUDIT74.json','plot74.py']}))
print('PLOTS',len(figs),'ANIMATIONS',len(anims),flush=True)
