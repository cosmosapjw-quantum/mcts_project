"""Original grouped plot axes, actual and conditional separately."""
from pathlib import Path
import json,hashlib,statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S=Path(__file__).resolve().parent;r=json.loads((S/'PHYSICS_RESULTS.json').read_text());crn=json.loads((S/'CRN_CALIBRATION.json').read_text())
out=S/'plots/grouped';out.mkdir(exist_ok=False);reg={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(node,fig,scope):
 for e in ('png','pdf'):fig.savefig(out/(node+'.'+e),dpi=160,bbox_inches='tight')
 plt.close(fig);reg[node]=dict(scope=scope,files={node+'.'+e:sha(out/(node+'.'+e)) for e in ('png','pdf')})
def grid(ax):ax.grid(alpha=.25)
def mean(a,k):return st.mean(x[k] for x in a)
g=next(x for x in r['groups'] if x['campaign']=='precision');rows=g['rows'];ls=(13,21)
fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
ax[0].bar(ls,[sum(x['jobs'] for n,x in r['counts'][f'L{L}'].items() if n!='game') for L in ls]);ax[0].set(xlabel='Board side',ylabel='Actual root histories')
ax[1].bar(ls,[x['plies'] for x in r['games']]);ax[1].set(xlabel='Board side',ylabel='Natural-terminal plies')
fig.suptitle('Native v5 common-MixNet campaigns; games not strength estimates');save('REPLAN-01',fig,'native')
fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for L in ls:
 for B in (16,32,64):
  a=[x for x in r['temperature'] if x['L']==L and x['campaign']==f'scale{B}'];ks=sorted({x['k'] for x in a})
  ax[0].plot(ks,[mean([x for x in a if x['k']==k],'H') for k in ks],marker='.',label=f'L{L} B{B}')
  ax[1].plot(ks,[st.mean(x['beta_status']=='finite' for x in a if x['k']==k) for k in ks],marker='.',label=f'L{L} B{B}')
for a,y in zip(ax,('Occupancy entropy (nats)','Finite inverse-score fit fraction')):
 a.set(xscale='log',xlabel='Completed backups',ylabel=y);a.legend(fontsize=7);grid(a)
fig.suptitle('T: score-scale diagnostics, not physical thermometry');save('REPLAN-02',fig,'native')
fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for key,label in (('leading_error','Leading rate'),('one_loop_error','One-loop determinant'),('corrected_error','Stirling correction')):
 ax[0].plot((16,32,64,128),[st.median(v[key] for x in r['exact_count'] for v in x['rows'] if v['N']==N and v['interior']) for N in (16,32,64,128)],marker='o',label=label)
ax[0].set(xscale='log',yscale='log',xlabel='Frozen independent draws',ylabel='Absolute log-mass error');ax[0].legend();grid(ax[0])
a=[x['observed_log_mass'] for x in r['exact_count']];ax[1].hist([-v for v in a],bins=16);ax[1].set(xlabel='Exact conditional four-count action',ylabel='Native baseline histories')
fig.suptitle('A: full sector multinomial and controlled frozen one-loop');save('REPLAN-03',fig,'exact_conditional_iid')
fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
ax[0].hist([x['F_retained_trace'] for x in r['exact_count']],bins=14);ax[0].set(xlabel='Binary pushforward Fisher trace / full',ylabel='IID pre-wave states')
for L in ls:
 a=[x for x in r['groups'] if x['L']==L and x['campaign'].startswith('scale')]
 ax[1].plot((16,32,64),[next(x['fisher_eigen'][0] for x in a if x['B']==B) for B in (16,32,64)],marker='o',label=f'L{L}')
ax[1].set(xlabel='Backup budget',ylabel='Minimum full-history Fisher eigenvalue');ax[1].legend();grid(ax[1])
fig.suptitle('R: exact conditional information loss and native path Fisher');save('REPLAN-04.R',fig,'native_and_exact_conditional')
fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for L in ls:
 a=sorted([x for x in r['groups'] if x['L']==L and x['campaign'].startswith('scale')],key=lambda x:x['B'])
 ax[0].plot([x['B'] for x in a],[x['cov_trace'] for x in a],marker='o',label=f'L{L}')
 ax[1].plot([x['B'] for x in a],[x['S_low'] for x in a],marker='o',label=f'L{L}')
 for h,sty in ((.15,':'),(.3,'-')):
  v=[next(y for y in x['rows'] if y['label']=='d0' and y['h']==h) for x in a]
  ax[2].plot([x['B'] for x in a],[x['conditional'][0] for x in v],marker='o',ls=sty,label=f'L{L} h{h}')
for a,y in zip(ax,('Connected covariance trace','Lowest Fourier mode','Conditional diagonal response')):
 a.set(xlabel='Backup budget B',ylabel=y);a.legend(fontsize=7);grid(a)
fig.suptitle('C: same local core, separate size and budget, n4/condition');save('REPLAN-04.C',fig,'native_and_exact_conditional')
fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
q=r['quantum'];modes=('coherent','dephase','reset')
for mode in modes:
 ax[0].scatter([x['transfer'] for x in q],[x['models'][mode]['Favg'] for x in q],s=14,alpha=.6,label=mode)
 ax[1].scatter([x['transfer'] for x in q],[x['models'][mode]['negativity'] for x in q],s=14,alpha=.6,label=mode)
ax[0].axhline(2/3,color='gray',ls=':',label='EB bound');ax[0].set(xlabel='Transfer amplitude',ylabel='Six-state fidelity');ax[0].legend(fontsize=7)
ax[1].set(xlabel='Transfer amplitude',ylabel='Normalized Choi negativity');ax[1].legend(fontsize=7)
fig.suptitle('D: declared retained-environment quantum memory and matched controls');save('REPLAN-05',fig,'conditional_quantum')
fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
w=np.load(S/'WORK_SAMPLES.npz');allw=np.concatenate([w[k] for k in w.files]);bins=np.linspace(min(allw.min(),-allw.max()),max(allw.max(),-allw.min()),81)
a=r['thermal']
for tau in (4.,16.,64.):
 b=[x for x in a if x['tau']==tau]
 f=np.concatenate([w[f"{x['position']}_{x['law']}_{tau:g}_F"] for x in b]);rev=-np.concatenate([w[f"{x['position']}_{x['law']}_{tau:g}_R"] for x in b])
 ax[0].hist(f,bins=bins,weights=np.exp(-f)/len(f),histtype='step',label=f'F weighted {tau:g}')
 ax[0].hist(rev,bins=bins,weights=np.ones(len(rev))/len(rev),histtype='step',ls=':',label=f'R reflected {tau:g}')
 ax[1].plot([x['k'] for x in b[0]['weighted_checks']],[st.mean(abs(complex(x['weighted_checks'][i]['diff_re'],x['weighted_checks'][i]['diff_im'])) for x in b) for i in range(5)],marker='o',label=f'tau{tau:g}')
ax[0].set(xlabel='Dimensionless work',ylabel='Weighted bin mass');ax[0].legend(fontsize=6)
ax[1].set(xlabel='Characteristic frequency',ylabel='MC Crooks absolute difference');ax[1].legend(fontsize=7)
ax[2].plot((4,16,64),[st.mean(x['forward']['variance'] for x in a if x['tau']==t) for t in (4,16,64)],marker='o',label='forward')
ax[2].plot((4,16,64),[st.mean(x['reverse']['variance'] for x in a if x['tau']==t) for t in (4,16,64)],marker='s',ls=':',label='reverse');ax[2].set(xscale='log',xlabel='Duration',ylabel='Work variance');ax[2].legend()
fig.suptitle('N: declared continuous-time Crooks distribution, independent exact-clock paths');save('REPLAN-06',fig,'conditional_thermal')
fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
ax[0].scatter([x['transfer'] for x in q],[x['phase_probe_Y'] for x in q],s=15,label='coherent')
ax[0].scatter([x['transfer'] for x in q],[x['dephased_probe_Y'] for x in q],s=15,label='dephase')
ax[0].set(xlabel='Transfer amplitude',ylabel='Fixed Y probe after phase pi/2');ax[0].legend()
states=np.load(S/'CHOI_STATES.npz');best=max(q,key=lambda x:x['phase_imag_norm']);v=states[best['job']+'_phase_pi2'];lim=max(abs(v.imag).max(),1e-12)
im=ax[1].imshow(v.imag,cmap='RdBu_r',vmin=-lim,vmax=lim);fig.colorbar(im,ax=ax[1]);ax[1].set(title='Declared Choi imaginary part',xlabel='Column',ylabel='Row')
fig.suptitle('Q: explicit complex channel, phase-sensitive readout');save('X-REP',fig,'conditional_quantum')
fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
for x in rows:
 if x['label'] not in ('d0','m02','m12') or x['h']!=.3:continue
 k=x['i'];a=x['prefix_precision'];ax[0].plot([z['n'] for z in a],[z['se'][k] for z in a],marker='o',label=x['label']+' direct')
 ax[0].plot([z['n'] for z in a],[z['se'][12+k] for z in a],marker='s',ls=':',label=x['label']+' conditional')
ax[0].set(yscale='log',xlabel='Nested paired seeds',ylabel='SE');ax[0].legend(fontsize=7);grid(ax[0])
x=np.arange(len(crn['labels']));ax[1].bar(x,crn['z']);ax[1].axhline(0,color='gray');ax[1].set_xticks(x,crn['labels'],rotation=35,ha='right');ax[1].set(ylabel='Exact conditional standardized difference',title=f"Joint calibration tail={crn['max_stat_tail_MC']:.3f}")
fig.suptitle('Memory/response: exact IID conditional variance, direct CRN comparison');save('X-MEMORY',fig,'native_and_exact_conditional')
(out/'manifest.json').write_text(json.dumps(dict(source_sha256=sha(S/'PHYSICS_RESULTS.json'),states_sha256=sha(S/'CHOI_STATES.npz'),work_samples_sha256=sha(S/'WORK_SAMPLES.npz'),crn_sha256=sha(S/'CRN_CALIBRATION.json'),groups=reg),indent=2)+'\n')
print('grouped figures',len(reg))
