from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
N=read(S/'NATIVE94.json');R=read(S/'RECORD94.json');F=N['families'];rec=R['summary'];t=np.arange(32,44);ar=N['arms'];figs=[]
def save(fig,name,ids):
 fig.tight_layout();b=S/'plots'/name
 fig.savefig(str(b)+'.png',dpi=160);fig.savefig(str(b)+'.pdf');plt.close(fig)
 p=Path(str(b)+'.png');q=Path(str(b)+'.pdf')
 with Image.open(p) as im:im.verify()
 subprocess.run(['pdfinfo',str(q)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,png_sha=sha(p),pdf_sha=sha(q)))
def band(ax,x,a,label):
 a=np.asarray(a);ax.plot(x,a[0],label=label,lw=1.3);ax.fill_between(x,a[1],a[2],alpha=.10)
def clocks(a):
 for x in [32,36,40]:a.axvline(x,ls=':',color='gray',lw=.7)
def zero(a):a.axhline(0,lw=.6,color='gray')
# G13 actual poststate coherence, offdiagonal relative entropy, phases and readout.
fig,ax=plt.subplots(3,3,figsize=(13,10));fig.suptitle('G13 | recorded conditional software qubit coherence (not native physical bath)')
for i,f in enumerate(F):
 x=np.array(f['features_ci'])
 for j,k in enumerate([2,5,20]):
  for a in [0,1,2,4,5]:band(ax[i,j],t,x[:,a,:,k],ar[a])
  clocks(ax[i,j]);ax[i,j].set(xlabel='Selection clock',ylabel=N['fields'][k],title=f['family']);ax[i,j].legend(fontsize=6)
save(fig,'G13_coherence_evolution',['G13'])
# G14 entropy evolution conditional vs fixed-history unselective and Jensen gap.
fig,ax=plt.subplots(3,3,figsize=(13,10));fig.suptitle('G14 | selective entropy and local unselective entropy; nats')
for i,f in enumerate(F):
 x=np.array(f['features_ci'])
 for a in [0,1,2,4,5]:band(ax[i,0],t,x[:,a,:,8],ar[a])
 for k in [7,9,10,8]:band(ax[i,1],t,x[:,0,:,k],N['fields'][k])
 for k in [11,12,14,18]:band(ax[i,2],t,x[:,0,:,k],N['fields'][k])
 for j in range(3):
  clocks(ax[i,j]);ax[i,j].set(xlabel='Selection clock',ylabel='Entropy / change (nat)',title=f['family']);ax[i,j].legend(fontsize=6)
 zero(ax[i,2])
save(fig,'G14_entropy_growth',['G14'])
# Explicit causal prefix versus final postselection; success and reduced entropy.
fig,ax=plt.subplots(2,3,figsize=(13,7));fig.suptitle('G14/G17 | declared S-record coherent prefix; not a final-success marginal')
for j,g in enumerate(['g0','g3','Rapfi']):
 for r in [x for x in rec if x['family']==g]:
  q=np.asarray(r['plus_ci']);p=r['protocol'];ax[0,j].semilogy(range(4),r['success_median'],'o-',label=p)
  for k,lab in [(1,'S(S)'),(2,'S(R)'),(3,'S(SR)'),(10,'S(stopped S)')]:
   ax[1,j].plot(range(4),q[0,:,k],'o-' if p=='native_repeat' else 's--',label=p+' '+lab,ms=3)
 ax[0,j].set(xlabel='Completed waves',ylabel='Median success probability',title=g);ax[0,j].legend(fontsize=6)
 ax[1,j].set(xlabel='Completed waves',ylabel='Mean entropy (nat)',title=g);ax[1,j].legend(fontsize=5,ncol=2)
save(fig,'G14_record_prefix_entropy',['G14','G17'])
# G15 participation: actual full legal probabilities, base and occupancy are different objects.
fig,ax=plt.subplots(2,3,figsize=(13,7));fig.suptitle('G15 | native conditional participation vs realized occupancy')
for j,f in enumerate(F):
 a=np.asarray(f['participation_ci'])
 for k in [0,1,2]:band(ax[0,j],np.arange(1,129),a[:,0,:,k],N['participation_fields'][k])
 for arm in [0,1,2,4,5]:band(ax[1,j],t,a[:,arm,32:44,0],ar[arm])
 ax[0,j].axvspan(33,44,color='gray',alpha=.1);ax[0,j].set(xlabel='Draw count',ylabel='Inverse participation',title=f['family']);ax[0,j].legend(fontsize=7)
 clocks(ax[1,j]);ax[1,j].set(xlabel='Selection clock',ylabel='Conditional participation',title='Active window');ax[1,j].legend(fontsize=6)
save(fig,'G15_participation_ratio',['G15'])
# G16 pointer stability under frozen settings, local distinguishability and independently retained fragments.
fig,ax=plt.subplots(2,3,figsize=(14,7));fig.suptitle('G16 | local pointer stability, frozen sequence and record information')
for j,f in enumerate(F):
 x=np.asarray(f['features_ci']);band(ax[0,j],t,x[:,0,:,16],'Readout TV=m lambda');band(ax[0,j],t,x[:,0,:,17],'Transverse infidelity');band(ax[0,j],t,np.asarray(f['pointer_mi_ci'])[:,0,:],'Equiprobable pointer MI (nat)');clocks(ax[0,j]);ax[0,j].set(xlabel='Selection clock',ylabel='Distinguishability / disturbance',title=f['family']);ax[0,j].legend(fontsize=6)
 s=np.asarray(f['frozen_stability_ci']);a=ax[1,j];a.bar(np.arange(4),s[0,0,[0,6,7,8]],yerr=[s[0,0,[0,6,7,8]]-s[1,0,[0,6,7,8]],s[2,0,[0,6,7,8]]-s[0,0,[0,6,7,8]]],capsize=2);a.set(xticks=range(4),xticklabels=['optimal','X','Y','Z'],ylabel='Final Bloch norm for pure input',ylim=(0,1.1),title='Frozen observed settings only')
save(fig,'G16_pointer_states',['G16'])
fig,ax=plt.subplots(2,3,figsize=(13,7));fig.suptitle('G16/G17 | declared record fragments and actual phase decision comparison')
for j,f in enumerate(F):
 g=f['family'];rs=[r for r in rec if r['family']==g]
 for k,r in enumerate(rs):
  a=np.asarray(r['fragment_ci']);ax[0,j].bar(np.arange(2)+(k-.5)*.35,a[0],.35,yerr=[a[0]-a[1],a[2]-a[0]],label=r['protocol'],capsize=2)
 ax[0,j].axhline(.9,color='gray',ls='--');ax[0,j].set(xticks=range(2),xticklabels=['Fragment2','Fragment3'],ylim=(0,1.05),ylabel='Holevo / first-bit H',title=g);ax[0,j].legend(fontsize=6)
 a=ax[1,j];x=np.asarray(f['phase_first_signed_ci']);y=np.asarray(f['phase_active_signed_ci']);z=np.array([x,y]);a.errorbar([0,1],z[:,0],yerr=[z[:,0]-z[:,1],z[:,2]-z[:,0]],fmt='o',capsize=3);zero(a);a.set(xticks=[0,1],xticklabels=['First draw','Active window'],ylabel='Actual signed plus-minus contrast',title=f"Final recommendation differs {f['recommend_disagreement']}/192")
save(fig,'G16_pointer_fragments',['G16','G17'])
fig,ax=plt.subplots(2,3,figsize=(14,7));fig.suptitle('G17 | adaptive decoherence summary; software instrument and declared record separate')
for j,f in enumerate(F):
 x=np.asarray(f['features_ci'])
 for k in [2,5,8]:band(ax[0,j],t,x[:,0,:,k],N['fields'][k])
 clocks(ax[0,j]);ax[0,j].set(xlabel='Selection clock',ylabel='Coherence / entropy',title=f['family']);ax[0,j].legend(fontsize=7)
 for r in [v for v in rec if v['family']==f['family']]:
  q=np.asarray(r['plus_ci']);a=ax[1,j]
  a.semilogy(np.arange(1,4),np.maximum(q[0,1:,8],1e-40),'o-',label=r['protocol']+' conditional')
  a.semilogy(np.arange(1,4),np.maximum(q[0,1:,9],1e-40),'s--',label=r['protocol']+' success-weighted')
 ax[1,j].set(xlabel='Completed waves',ylabel='Hadamard TV (log scale)',title='Rare-event suppression');ax[1,j].legend(fontsize=6)
save(fig,'G17_decoherence_summary',['G17'])
dump(S/'PLOT_MANIFEST94.json',dict(figures=figs,sources={k:sha(S/k) for k in ['NATIVE94.json','NATIVE94.npz','RECORD94.json','INDEPENDENT94.json','ORIGINAL_SOURCE94.json']},scope='Root-block pointwise intervals; coherent record model explicit, not physical quantum identification.'))
print('PLOTS',len(figs),flush=True)
