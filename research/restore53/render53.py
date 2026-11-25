"""Grouped original physics figures, native vs conditional evidence."""
from pathlib import Path
import json,hashlib,statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S=Path(__file__).resolve().parent;r=json.loads((S/'PHYSICS_RESULTS.json').read_text())
out=S/'plots/grouped';out.mkdir(exist_ok=False);reg={}
Ls=(13,21);laws=('iid','pl','reserve');G=[x for x in r['groups'] if x['campaign']!='pilot'];T=[x for x in G if x['campaign'] in ('g0','g1')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(node,fig,scope):
 for e in ('png','pdf'):fig.savefig(out/(node+'.'+e),dpi=160,bbox_inches='tight')
 plt.close(fig);reg[node]=dict(scope=scope,files={node+'.'+e:sha(out/(node+'.'+e)) for e in ('png','pdf')})
def mean(a,k):return st.mean(x[k] for x in a)
def grid(ax):ax.grid(alpha=.2)
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for L in Ls:
 a=[x for x in r['groups'] if x['L']==L and x['campaign']=='pilot']
 b=[x for x in G if x['L']==L]
 d={(x['position'],x['law']):x['fisher_eigen'][0] for x in b}
 axs[0].scatter([x['fisher_eigen'][0] for x in a],[d[x['position'],x['law']] for x in a],label=f'L{L}')
axs[0].plot([0,5],[0,5],ls=':',color='gray');axs[0].set(xlabel='Pilot minimum Fisher eigenvalue',ylabel='Independent baseline minimum eigenvalue');axs[0].legend()
axs[1].bar(Ls,[x['plies'] for x in r['games']]);axs[1].set(xlabel='Board side',ylabel='Natural-terminal game plies')
fig.suptitle('Native vector field, new geometry and separate pilot');save('REPLAN-01',fig,'native')
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for L in Ls:
 for law in laws:
  a=[x for x in r['temperature'] if x['L']==L and x['law']==law];ks=sorted({x['k'] for x in a})
  axs[0].plot(ks,[mean([x for x in a if x['k']==k],'H') for k in ks],marker='.',label=f'L{L} {law}')
  axs[1].plot(ks,[st.mean(x['beta_status']=='finite' for x in a if x['k']==k) for k in ks],marker='.',label=f'L{L} {law}')
for ax,y in zip(axs,('Occupancy entropy (nats)','Finite inverse-score fit fraction')):ax.set(xscale='log',xlabel='Completed backups',ylabel=y);ax.legend(fontsize=7);grid(ax)
fig.suptitle('T: native entropy and inverse score scale, not physical temperature');save('REPLAN-02',fig,'native')
fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for k,l in (('leading_error','Leading'),('one_loop_error','Lattice determinant')):
 axs[0].plot((8,16,32),[np.median([v[k] for x in r['loops'] for v in x['values'] if v['M']==M]) for M in (8,16,32)],marker='o',label=l)
axs[0].set(xscale='log',yscale='log',xlabel='Independent frozen waves',ylabel='Absolute log mass error');axs[0].legend()
mods=('zero','frozen','score','LR');pred=r['prediction']
axs[1].bar(range(4),[st.mean(x['errors'][m]['sector'] for x in pred) for m in mods]);axs[1].set_xticks(range(4),mods);axs[1].set_ylabel('Heldout sector response MSE')
fig.suptitle('A: exact restricted action versus predictive native response');save('REPLAN-03',fig,'exact_reference_and_heldout_native')
fig,axs=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
for i,L in enumerate(Ls):
 for j,l in enumerate(laws):
  a=[x for x in T if x['L']==L and x['law']==l];z=np.mean([x['retarded'] for x in a],0)
  # Frobenius response matrix norm, output 3 tangent components + mark
  v=np.linalg.norm(z,axis=(-1,-2));im=axs[i,j].imshow(v,origin='lower');axs[i,j].set(title=f'L{L} {l}',xlabel='Source block',ylabel='Observable block');fig.colorbar(im,ax=axs[i,j],shrink=.8)
fig.suptitle('R: native retarded block-matrix norm; upper triangle causal mask');save('REPLAN-04.R',fig,'native_score')
fig,axs=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
for law in laws:
 for k,ax in zip(('cov_trace','S_low'),axs[:2]):
  ax.plot(Ls,[mean([x for x in G if x['L']==L and x['law']==law],k) for L in Ls],marker='o',label=law)
 a=[x for x in G if x['law']==law];axs[2].plot(Ls,[st.mean(x['fisher_eigen'][0] for x in a if x['L']==L) for L in Ls],marker='o',label=law)
for ax,y in zip(axs,('Connected covariance trace','Lowest Fourier mode','Minimum tangent Fisher eigenvalue')):ax.set(xlabel='Board side',ylabel=y);ax.legend();grid(ax)
fig.suptitle('C: native finite-size diagnostics, fixed B16; two core templates');save('REPLAN-04.C',fig,'native')
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);q=r['quantum'];ths=sorted({x['angle'] for x in q})
for l in laws:
 a=[x for x in q if x['law']==l]
 axs[0].plot(ths,[st.mean(abs(x['order_gap']) for x in a if x['angle']==t) for t in ths],marker='o',label=l)
 axs[1].plot(ths,[st.mean(x['correlations']['NM'] for x in a if x['angle']==t) for t in ths],marker='o',label=l)
for ax,y in zip(axs,('Mean absolute order gap','Past/future signed correlation after NM')):ax.set(xlabel='Nonorthogonal measurement angle (rad)',ylabel=y);ax.legend();grid(ax)
fig.suptitle('D: explicit conditional environment measurement order');save('REPLAN-05',fig,'conditional_quantum')
fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True);a=r['thermal'];ts=(4.,16.,64.)
samples=np.load(S/'WORK_SAMPLES.npz')
allw=np.concatenate([samples[k] for k in samples.files]);bins=np.linspace(allw.min(),allw.max(),81)
for tau in ts:
 b=[x for x in a if x['tau']==tau];w=np.concatenate([samples[f"{x['position']}_{x['law']}_{tau:g}"] for x in b])
 axs[0].hist(w,bins=bins,density=False,weights=np.ones(len(w))/len(w),histtype='step',label=f'tau{tau:g}')
 axs[1].plot(b[0]['k'],np.mean([x['cf_re'] for x in b],0),marker='o',label=f'ODE {tau:g}');axs[1].plot(b[0]['k'],np.mean([x['mc_re'] for x in b],0),ls=':',label=f'MC {tau:g}')
axs[0].set(xlabel='Work / E*',ylabel='Bin probability');axs[0].legend(fontsize=7)
axs[1].set(xlabel='Characteristic frequency',ylabel='Real CF');axs[1].legend(fontsize=7)
axs[2].plot(ts,[mean([x for x in a if x['tau']==t],'no_transition_prob') for t in ts],marker='o',label='Exact no transition')
axs[2].plot(ts,[mean([x for x in a if x['tau']==t],'mc_no_transition') for t in ts],marker='s',ls=':',label='MC')
axs[2].set(xscale='log',xlabel='Duration',ylabel='Probability');axs[2].legend(fontsize=7)
fig.suptitle('N: declared continuous Gibbs work distribution, ODE and exact-clock MC');save('REPLAN-06',fig,'conditional_thermal')
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
a=[x for x in q if abs(x['angle']-np.pi/4)<1e-12];z=np.zeros((5,5));n=np.zeros((5,5))
for x in a:z[x['n1'],x['n2']]+=x['order_gap'];n[x['n1'],x['n2']]+=1
z=np.divide(z,n,out=np.full_like(z,np.nan),where=n>0);lim=max(np.nanmax(abs(z)),1e-12)
im=axs[0].imshow(z,origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim);fig.colorbar(im,ax=axs[0]);axs[0].set(xlabel='Last-block count',ylabel='First-block count',title='Order gap at pi/4')
axs[1].scatter([x['analytic'] for x in q],[x['order_gap'] for x in q],s=10,alpha=.4);axs[1].set(xlabel='Analytic Bloch calculation',ylabel='Direct complex matrix result')
fig.suptitle('Q: noncommuting conditional channels, not native quantum witness');save('X-REP',fig,'conditional_quantum')
fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for i,L in enumerate(Ls):
 for j,k in enumerate((0,1)):
  ax=axs[i,j]
  if k==0:
   a=[x for x in r['temporal'] if x['L']==L and x['i']==x['k'] and x['j']==3]
   for l in laws:
    b=[x for x in a if x['law']==l]
    lr=[st.mean(x['values'][1][3][d] for x in b if x['i']==d) for d in range(3)]
    fd=[st.mean(x['values'][2][3][d] for x in b if x['i']==d) for d in range(3)]
    ax.plot(range(3),lr,marker='o',ls=':',label=l+' LR');ax.plot(range(3),fd,marker='s',label=l+' direct')
   ax.set(xlabel='Tangent direction',ylabel='Diagonal first finite response',title=f'L{L}')
  else:
   a=[x for x in r['temporal'] if x['L']==L and x['i']!=x['k']]
   for l in laws:
    b=[x for x in a if x['law']==l];v=np.mean([x['values'] for x in b],0)
    ax.plot(range(4),v[5,3],marker='o',ls=':',label=l+' LR');ax.plot(range(4),v[6,3],marker='s',label=l+' direct')
   ax.set(xlabel='Observable (3 tangent coordinates, mark)',ylabel='Mixed finite second response',title=f'L{L}')
  ax.axhline(0,color='grey',ls=':');ax.legend(fontsize=6)
fig.suptitle('Memory: direct multicomponent and mixed pulses, finite sample');save('X-MEMORY',fig,'native_direct_response')
(out/'manifest.json').write_text(json.dumps(dict(source_sha256=sha(S/'PHYSICS_RESULTS.json'),work_samples_sha256=sha(S/'WORK_SAMPLES.npz'),groups=reg),indent=2)+'\n')
print('grouped figures',len(reg))
