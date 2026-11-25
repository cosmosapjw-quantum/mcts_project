"""Grouped plots, original longitudinal output contracts and physics references."""
from pathlib import Path
import json,statistics as st,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S=Path(__file__).resolve().parent;r=json.loads((S/'PHYSICS_RESULTS.json').read_text())
out=S/'plots/grouped';out.mkdir(exist_ok=False);reg={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(node,fig,scope,items):
 for e in ('png','pdf'):fig.savefig(out/(node+'.'+e),dpi=160,bbox_inches='tight')
 plt.close(fig);reg[node]=dict(scope=scope,original_items=items,files={node+'.'+e:sha(out/(node+'.'+e)) for e in ('png','pdf')})
def grid(a):a.grid(alpha=.22)
def mean(a,k):return st.mean(x[k] for x in a)
games=r['games'];long=r['longitudinal'];laws=('iid','pl','reserve')
def traj(g):return [x for x in long if x['game']==g['job']]
def label(g):return f"L{g['L']} B{g['B']} {g['opening_id']}"
def time(g):return np.arange(len(traj(g)))/max(len(traj(g))-1,1)
fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
for g in games:
 a=traj(g);ax[0].plot(time(g),[x['black_q'] for x in a],marker='.',label=label(g))
 ax[1].plot(time(g),[x['support'] for x in a],marker='.',label=label(g))
for a,y in zip(ax,('Mean return (black frame)','Visited root support')):a.set(xlabel='Normalized searched game ply',ylabel=y);grid(a);a.legend(fontsize=6,ncol=2)
fig.suptitle('Actual full self-play histories (each curve one game)');save('REPLAN-01',fig,'native complete games',['L02'])
fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for g in games:
 a=traj(g);t=time(g)
 ax[0,0].plot(t,[x['H'] for x in a],marker='.',label=label(g))
 for k in ('Q_all','PUCT_all'):
  x=np.array([v['fits'][k]['beta'] if v['fits'][k]['status']=='finite' else np.nan for v in a],float)
  ax[0,1 if k=='Q_all' else 0] if False else None
  ax[0,1].plot(t,x,ls='-' if k=='Q_all' else ':',marker='.',label=label(g)+' '+k)
ax[0,0].set(xlabel='Normalized game ply',ylabel='Occupancy entropy (nats)');ax[0,0].legend(fontsize=6)
ax[0,1].set(xlabel='Normalized game ply',ylabel='Finite inverse score fit beta');ax[0,1].legend(fontsize=5,ncol=2)
fits=('Q_all','Q_visited','PUCT_all');stt=sorted({x['fits'][f]['status'] for x in long for f in fits})
bottom=np.zeros(3)
for s in stt:
 vals=[sum(x['fits'][f]['status']==s for x in long) for f in fits];ax[1,0].bar(range(3),vals,bottom=bottom,label=s);bottom+=vals
ax[1,0].set_xticks(range(3),fits);ax[1,0].set_ylabel('Search plies');ax[1,0].legend()
for B in (16,32):
 a=[x for x in long if x['B']==B and x['fits']['Q_all']['status']=='finite']
 if a:
  ax[1,1].hist([1/x['fits']['Q_all']['beta'] for x in a],bins=16,histtype='step',label=f'B{B}')
ax[1,1].set(xlabel='Reciprocal fitted Q score scale (finite fits only)',ylabel='Plies');ax[1,1].legend()
fig.suptitle('T: original Q fit vs visited-only / PUCT; not physical temperature');save('REPLAN-02',fig,'native descriptive Q-score fit',['L03','L25','L04','L26'])
fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for k,l in (('leading_error','Leading'),('one_loop_error','Lattice determinant')):
 ax[0].plot((8,16,32),[st.median(v[k] for x in r['loops'] for v in x['values'] if v['M']==M) for M in (8,16,32)],marker='o',label=l)
ax[0].set(xscale='log',yscale='log',xlabel='Independent frozen waves',ylabel='Absolute log-mass error');ax[0].legend()
for g in games:
 a=traj(g);ax[1].plot(time(g),[x['kkt_gap'] for x in a],marker='.',label=label(g))
ax[1].set(xlabel='Normalized game ply',ylabel='Frozen score objective gap');ax[1].legend(fontsize=6)
fig.suptitle('A: exact restricted wave and separate actual frozen objective gap');save('REPLAN-03',fig,'restricted exact reference and native diagnostic',['L02'])
fig,ax=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
for g in games:
 a=traj(g);ds=sorted({d['depth'] for x in a for d in x['depth']})
 ax[0].plot(ds,[st.mean(d['mean'] for x in a for d in x['depth'] if d['depth']==k) for k in ds],marker='.',label=label(g))
 ax[1].plot(ds,[st.mean(d['between'] for x in a for d in x['depth'] if d['depth']==k) for k in ds],marker='.',label=label(g))
 ds=sorted({d['depth'] for x in a for d in x['flow']})
 ax[2].plot(ds,[st.mean(d['beta_depth'] for x in a for d in x['flow'] if d['depth']==k) for k in ds],marker='.',label=label(g))
for a,y in zip(ax,('Mean mark (surviving paths)','Between-prefix mark variance','Matched-survival variance increment')):a.set(xlabel='Actual tree depth',ylabel=y);grid(a);a.legend(fontsize=6)
fig.suptitle('R: original depth-scale panels, no autonomous RG fixed point');save('REPLAN-04.R',fig,'native actual prefix tree',['L35','L29','L38','Q09','Q11'])
fig,ax=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
for B in (16,32):
 for L in (13,21):
  a=[x for x in r['phase'] if x['L']==L and x['B']==B]
  for k,axs in zip(('H_fraction','variance','order'),ax):
   axs.plot(range(3),[mean([x for x in a if x['phase']==j],k) for j in range(3)],marker='o',label=f'L{L} B{B}')
for a,y in zip(ax,('Entropy / finite support ceiling','Return variance','Top-two occupancy gap')):a.set_xticks(range(3),('early','middle','late'));a.set_ylabel(y);a.legend(fontsize=7);grid(a)
fig.suptitle('C: equal-game phase means, two openings per size/budget');save('REPLAN-04.C',fig,'native finite size/budget, not criticality',['L07','Q31'])
fig,ax=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
for g in (.2,.5,.8):
 a=[x for x in r['quantum'] if x['g']==g];ms=range(9)
 for k,sty in (('QMI','-'),('Holevo','--'),('basis_MI',':')):
  ax[0].plot(ms,[st.mean(x['rows'][m][k]/x['H'] for x in a) for m in ms],ls=sty,marker='.',label=f'{g} {k}')
 ax[1].plot(ms,[st.mean(x['rows'][m]['coherence'] for x in a) for m in ms],marker='o',label=f'g={g}')
ax[0].axhline(.9,ls=':',color='grey');ax[0].set(xlabel='Environment fragments',ylabel='Information / pointer entropy');ax[0].legend(fontsize=6,ncol=2)
ax[1].set(xlabel='Traced environment fragments',ylabel='Coherence L1');ax[1].legend()
a=r['thermal'];ts=[x['t'] for x in a[0]['pointer']['rows']]
ax[2].plot(ts,[st.mean(x['pointer']['rows'][i]['entropy'] for x in a) for i in range(len(ts))],marker='o',label='vN entropy')
ax[2].plot(ts,[st.mean(x['pointer']['rows'][i]['purity'] for x in a) for i in range(len(ts))],marker='s',label='purity');ax[2].set(xlabel='Declared gamma t',ylabel='Reduced state diagnostic');ax[2].legend()
fig.suptitle('D: declared fragment records and pointer decoherence on actual game propensities');save('REPLAN-05',fig,'conditional record model, not native quantum data',['L19','L20','Q24','Q25','Q26'])
fig,ax=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
for g in games:
 a=traj(g)
 ax[0].plot(time(g),[sum(v['field'] for v in x['increments']) for x in a],marker='.',label=label(g))
 ax[0].plot(time(g),[sum(v['redistribution'] for v in x['increments']) for x in a],ls=':',alpha=.55)
ax[0].set(xlabel='Game ply',ylabel='Same-root score change components');ax[0].legend(fontsize=6)
a=r['thermal'];ts=[x['t'] for x in a[0]['equilibrium']['rows']]
ax[1].plot(ts,[st.mean(x['equilibrium']['rows'][i]['response'][0] for x in a) for i in range(len(ts))],marker='o',label='Analytic response')
ax[1].plot(ts,[st.mean(x['equilibrium']['rows'][i]['C0_minus_Ct'][0] for x in a) for i in range(len(ts))],ls=':',marker='s',label='C(0)-C(t)')
ax[1].set(xlabel='Declared time',ylabel='Step susceptibility');ax[1].legend()
ts=[x['t'] for x in a[0]['nonstationary']['rows']]
ax[2].plot(ts,[mean([x['nonstationary']['rows'][i] for x in a],'gap') for i in range(len(ts))],marker='o')
ax[2].set(xlabel='Declared time',ylabel='Naive nonequilibrium FDT error')
fig.suptitle('N: native score balance (left), declared Gibbs FDT (right)');save('REPLAN-06',fig,'native score proxy and conditional equilibrium response',['L33','L06','Q20'])
fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
a=r['thermal'];ts=[x['t'] for x in a[0]['pointer']['rows']]
ax[0].plot(ts,[st.mean(x['pointer']['rows'][i]['coherence'] for x in a) for i in range(len(ts))],marker='o');ax[0].set(xlabel='Declared gamma t',ylabel='Complex coherence L1')
v=[x['swap_residual'] for y in a for x in y['pointer']['envariance']]
ax[1].hist(v,bins=16);ax[1].set(xlabel='Compensated swap norm (orthogonal Schmidt model)',ylabel='Pairs')
fig.suptitle('Q: pointer-projector Lindblad and exact conditional envariance diagnostic');save('X-REP',fig,'conditional complex state',['L19','L20','Q24','Q26'])
fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True);a=r['control']
for k,axs in enumerate(ax.ravel()):
 if k<2:
  for l in laws:
   b=[x for x in a if x['law']==l and x['source']==0]
   for method,sty in ((0,'-'),(1,':')):
    v=[st.mean(sum(x['variance'][method][:3]) if k==0 else x['variance'][method][3] for x in b if x['output']==j) for j in range(4)]
    axs.plot(range(4),v,marker='o',ls=sty,label=l+(' control' if method else ' direct'))
  axs.set(xlabel='Output block',ylabel='Direct '+('sector variance trace' if k==0 else 'mark variance'))
 else:
  for l in laws:
   b=[x for x in a if x['law']==l and x['source']==0 and x['output']>0]
   for method,sty in ((2,'-'),(3,':')):
    v=[st.mean(sum(x['variance'][method][:3]) if k==2 else x['variance'][method][3] for x in b if x['output']==j) for j in range(1,4)]
    axs.plot(range(1,4),v,marker='o',ls=sty,label=l+(' control' if method==3 else ' LR'))
  axs.set(xlabel='Future block',ylabel='LR '+('sector variance trace' if k==2 else 'mark variance'))
 axs.legend(fontsize=6);grid(axs)
fig.suptitle('Memory: actual sequential martingale controls; held-out geometry mark proxy');save('X-MEMORY',fig,'native paired finite response and sequential controls',[])
(out/'manifest.json').write_text(json.dumps(dict(source_sha256=sha(S/'PHYSICS_RESULTS.json'),states_sha256=sha(S/'CONDITIONAL_STATES.npz'),groups=reg),indent=2)+'\n')
print('grouped figures',len(reg))
