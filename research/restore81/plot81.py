from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
Q=read(S/'SUMMARY81.json');F=Q['families'];names=[x['family'] for x in F];arms=['plus81','minus81','eraseplus81','eraseminus81','swap81','base81'];figs=[]
def save(fig,name,ids,scope):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=180);fig.savefig(str(p)+'.pdf');plt.close(fig)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,scope=scope,png_sha=sha(a),pdf_sha=sha(b)))
def comp(f,name):return next(x for x in f['comparisons'] if x['name']==name)
def eb(a,x,q,l=None):
 q=np.asarray(q);a.errorbar(x,q[0],yerr=[[q[0]-q[1]],[q[2]-q[0]]],fmt='o',capsize=3,label=l)
fig,ax=plt.subplots(2,2,figsize=(11,8))
for i,f in enumerate(F):
 c=comp(f,'phase_reversal');q=np.asarray(c['direct_ci'])
 eb(ax[0,0],i-.12,f['first_law_signed_pred_ci'],'conditional prediction' if i==0 else None);eb(ax[0,0],i+.12,q[:,0],'actual paired draw' if i==0 else None)
 for j,k in enumerate(['phase_reversal','erase_plus','erase_minus','order_swap']):
  eb(ax[0,1],i+(j-1.5)*.15,np.asarray(comp(f,k)['direct_ci'])[:,1],k if i==0 else None)
 ax[1,0].plot(range(32,44),f['same_history_phase_qa_diff'],'o-',label=f['family'])
 ax[1,1].plot(range(32,44),f['same_history_swap_TV'],'o-',label=f['family'])
for a in ax[0]:a.set_xticks(range(3),names);a.axhline(0,color='k',lw=.6);a.legend(fontsize=8)
ax[0,0].set(ylabel='E[a−b], plus minus minus',title='First intervention: exact conditional law vs draws')
ax[0,1].set(ylabel='paired signed-count fraction difference',title='Actual 12 active selections')
for a in ax[1]:
 a.axvline(35.5,color='k',ls=':',lw=.7);a.axvline(39.5,color='k',ls=':',lw=.7);a.set(xlabel='selection index',xticks=[32,35,36,39,40,43]);a.legend(fontsize=8)
ax[1,0].set(ylabel='mean same-history q_a(+) − q_a(−)',title='Phase signal across Y,Z,X')
ax[1,1].set(ylabel='mean same-history grouped TV',title='Order swap after common first wave')
save(fig,'operational_phase_and_order',['L09','L20','Q24','Q42'],'Actual paired phase and erasure contrasts, noncommuting axis order and same-history conditional laws. Pointwise root-block intervals; not physical hardware.')
fig,ax=plt.subplots(2,2,figsize=(12,8))
for i,f in enumerate(F):
 q=np.array(f['importance_weight_ci']);a=ax[0,0];a.errorbar(np.arange(5)+(i-1)*.18,q[0,:5],yerr=[q[0,:5]-q[1,:5],q[2,:5]-q[0,:5]],fmt='o',capsize=3,label=f['family'])
 ax[0,1].bar(np.arange(5)+(i-1)*.24,f['importance_ESS'][:5],.24,label=f['family'])
 c=comp(f,'phase_reversal');eb(ax[1,0],i-.12,np.array(c['direct_ci'])[:,1],'actual direct' if i==0 else None);eb(ax[1,0],i+.12,np.array(c['importance_ci'])[:,1],'base path likelihood' if i==0 else None)
 eb(ax[1,1],i,np.array(c['direct_ci'])[:,7],f['family'])
for a in ax[0]:a.set_xticks(range(5),['plus','minus','erase+','erase−','swap']);a.legend(fontsize=8)
ax[0,0].axhline(1,color='k',ls=':');ax[0,0].set(title='Likelihood ratio normalization',ylabel='mean P_arm / P_base')
ax[0,1].set(title='Finite importance ESS (192 base histories)',ylabel='effective histories')
for a in ax[1]:a.set_xticks(range(3),names);a.axhline(0,color='k',lw=.6)
ax[1,0].set(title='Direct vs same-history likelihood transport',ylabel='phase reversal active signed fraction');ax[1,0].legend(fontsize=8)
ax[1,1].set(title='No consistent final mean-mark gain',ylabel='B128 mean signed mark difference')
save(fig,'causal_path_likelihood_and_final_mark',['L16','Q43','Q42'],'Exact likelihood on identical fine history under declared register. Finite IS and actual paired effects; no physical action or game utility.')
fig,ax=plt.subplots(2,2,figsize=(11,8))
for i,f in enumerate(F):
 r=f['record'];ax[0,0].bar(np.arange(3)+(i-1)*.24,[r['phase_reverse_TV'],r['phase_erase_TV'],r['swap_TV']],.24,label=f['family'])
 ax[0,1].bar(i,r['record_QMI'],label=f['family'])
 for a,k,y in [(ax[1,0],'mean_entropy','vN entropy (nat)'),(ax[1,1],'mean_coherence','transverse Bloch norm')]:
  a.plot(range(32,44),f['register_trajectories']['plus81'][k],'o-',label=f['family']+' plus')
  a.plot(range(32,44),f['register_trajectories']['eraseplus81'][k],'--',label=f['family']+' erase')
  a.set(xlabel='selection index',ylabel=y)
ax[0,0].set_xticks(range(3),['phase ±','phase erase','axis swap']);ax[0,0].set(ylabel='joint probability TV',title='Frozen three-operation model');ax[0,0].legend(fontsize=8)
ax[0,1].set_xticks(range(3),names);ax[0,1].set(title='Frozen first/rest QMI; PPT in all 72 cases',ylabel='quantum mutual information (nat)')
for a in ax[1]:a.axvline(35.5,color='k',ls=':',lw=.7);a.axvline(39.5,color='k',ls=':',lw=.7);a.legend(fontsize=7,ncol=2)
save(fig,'three_time_record_and_register',['L09','L15','L20','Q24','Q25','Q35','Q37','Q42','Q45'],'Actual conditional qubit trajectories and separately declared frozen three-qutrit record density; PPT not separability and not native adaptive coherent recombination.')
assert len(figs)==3 and len({f['png_sha'] for f in figs})==3
dump(S/'PLOT_MANIFEST81.json',dict(figures=figs,animations=[],sources={f:sha(S/f) for f in ['SUMMARY81.json','EFFECTS81.json','RECORD81.json','AUDIT81.json','INDEPENDENT81.json','CONVERGENCE81.json','plot81.py']}))
print('PLOTS',len(figs),flush=True)
