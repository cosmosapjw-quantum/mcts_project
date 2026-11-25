"""Ten scoped figures: nine grouped topic outputs + Q19 original panel."""
from pathlib import Path
import json,hashlib,statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S=Path(__file__).resolve().parent;F=S.parent/'mqft_restore54_20261126'
r=json.loads((S/'PHYSICS_RESULTS.json').read_text())
out=S/'plots/grouped';out.mkdir(exist_ok=False);orig=S/'plots/original';orig.mkdir(exist_ok=False)
reg={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(node,fig,scope,items,folder=out):
 for e in ('png','pdf'):fig.savefig(folder/(node+'.'+e),dpi=160,bbox_inches='tight')
 plt.close(fig);reg[node]=dict(scope=scope,original_items=items,files={str((folder/(node+'.'+e)).relative_to(S)):sha(folder/(node+'.'+e)) for e in ('png','pdf')})
def grid(ax):ax.grid(alpha=.22)
def mean(a,k):return st.mean(x[k] for x in a)
long=r['longitudinal'];games=r['games'];allg=r['combined_game_summary'];L0=(13,21);Bs=(16,32,64)
def traj(g):return [x for x in long if x['game']==g['job']]
def phase(a):return np.arange(len(a))/max(len(a)-1,1)
fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for i,L in enumerate(L0):
 for B in Bs:
  z=[x for x in games if x['L']==L and x['B']==B];ax[i,0].scatter([B]*len(z),[x['searched'] for x in z],label=f'B{B}',alpha=.7)
  a=[x for x in r['phase'] if x['L']==L and x['B']==B]
  ax[i,1].plot(range(3),[mean([x for x in a if x['phase']==j],'black_q') for j in range(3)],marker='o',label=f'B{B}')
 ax[i,0].set(xlabel='Backup budget',ylabel=f'L{L} searched plies');ax[i,0].set_xticks(Bs)
 ax[i,1].set_xticks(range(3),('early','middle','late'));ax[i,1].set_ylabel(f'L{L} black-frame return');ax[i,1].legend();grid(ax[i,1])
fig.suptitle('Actual matched-opening games, six new core templates per size/budget');save('REPLAN-01',fig,'native complete games',['L02'])
fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for L in L0:
 for B in Bs:
  a=[x for x in r['combined_phase'] if x['L']==L and x['B']==B]
  ax[0,0].plot(range(3),[mean([x for x in a if x['phase']==j],'H_fraction') for j in range(3)],marker='o',label=f'L{L} B{B}')
  q=[x for x in allg if x['L']==L and x['B']==B]
  ax[0,1].scatter([B+(.6 if L==21 else -.6)]*len(q),[x['finite_Q_all'] for x in q],alpha=.5,label=f'L{L} B{B}')
ax[0,0].set_xticks(range(3),('early','middle','late'));ax[0,0].set_ylabel('Entropy / finite support ceiling');ax[0,0].legend(fontsize=7,ncol=2)
ax[0,1].set(xlabel='Backup budget',ylabel='Per-game finite Q-fit fraction');ax[0,1].set_xticks(Bs);ax[0,1].legend(fontsize=6,ncol=2)
for L in L0:
 for B in Bs:
  z=[v for x in long if x['L']==L and x['B']==B for v in x['cuts']]
  ks=sorted({x['k'] for x in z})
  ax[1,0].plot(ks,[st.mean(x['Q_all']['status']=='finite' for x in z if x['k']==k) for k in ks],marker='o',label=f'L{L} B{B}')
  ax[1,1].plot(ks,[st.mean(x['PUCT_all']['status']=='finite' for x in z if x['k']==k) for k in ks],marker='o',label=f'L{L} B{B}')
for a,y in zip(ax[1],('Finite Q-fit fraction','Finite PUCT-fit fraction')):
 a.set(xscale='log',xlabel='Completed backup cut',ylabel=y);a.legend(fontsize=7,ncol=2);grid(a)
fig.suptitle('T: longitudinal and intra-search score fit; no physical thermometry');save('REPLAN-02',fig,'native descriptive score fit',['L03','L04','L25','L26','L45','G02','G03'])
fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for k,l in (('leading_error','Leading'),('one_loop_error','Lattice determinant')):
 ax[0].plot((8,16,32),[st.median(v[k] for x in r['loops'] for v in x['values'] if v['M']==M) for M in (8,16,32)],marker='o',label=l)
ax[0].set(xscale='log',yscale='log',xlabel='Frozen independent wave repetitions',ylabel='Absolute log-mass error');ax[0].legend()
for L in L0:
 for B in Bs:
  z=[x['kkt_gap'] for x in games if x['L']==L and x['B']==B]
  ax[1].scatter([B+(.7 if L==21 else -.7)]*len(z),z,label=f'L{L} B{B}',alpha=.7)
ax[1].set(xlabel='Budget',ylabel='Per-game frozen objective gap');ax[1].legend(fontsize=6,ncol=2);ax[1].set_xticks(Bs)
fig.suptitle('A: restricted reference and distinct native frozen objective gap');save('REPLAN-03',fig,'conditional exact restricted wave and native diagnostic',['L02'])
fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for L in L0:
 for B in Bs:
  z=[x for x in long if x['L']==L and x['B']==B];ds=sorted({d['depth'] for x in z for d in x['depth']})
  for a,k in zip(ax[:2],('between','survivors')):
   # Equal game weighting at each observed depth, not pooled plies.
   vv=[]
   for d in ds:
    by={}
    for x in z:
     for q in x['depth']:
      if q['depth']==d:by.setdefault(x['game'],[]).append(q[k])
    vv.append(st.mean(st.mean(v) for v in by.values()))
   a.plot(ds,vv,marker='o',label=f'L{L} B{B}')
  ds=sorted({d['depth'] for x in z for d in x['flow']})
  v=[]
  for d in ds:
   by={}
   for x in z:
    for q in x['flow']:
     if q['depth']==d:by.setdefault(x['game'],[]).append(q['beta_depth'])
   v.append(st.mean(st.mean(y) for y in by.values()))
  ax[2].plot(ds,v,marker='o',label=f'L{L} B{B}')
for a,y in zip(ax,('Between-prefix variance','Surviving paths per search','Matched-survival variance increment')):
 a.set(xlabel='Observed tree depth',ylabel=y);a.legend(fontsize=6);grid(a)
fig.suptitle('R: observed depth flow and survival control; no autonomous beta function');save('REPLAN-04.R',fig,'native tree depth',['L29','L35','L38','Q09','Q11'])
fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for L in L0:
 for a,k in zip(ax[:2],('H_fraction','variance')):
  y=[mean([x for x in allg if x['L']==L and x['B']==B],k) for B in Bs];a.plot(Bs,y,marker='o',label=f'L{L}')
for i,k in enumerate(('H_fraction','variance')):
 z=[x for x in r['paired'] if x['observable']==k]
 for j,x in enumerate(z):
  ax[2].errorbar(j+4*i,x['mean'],yerr=[[max(0,x['mean']-x['interval'][0])],[max(0,x['interval'][1]-x['mean'])]],fmt='o',capsize=3)
ax[2].axhline(0,ls=':',color='gray');ax[2].set_xticks(range(8),[f"{x['L']}:{x['low']}→{x['high']}" for k in ('H_fraction','variance') for x in r['paired'] if x['observable']==k],rotation=70,fontsize=7)
ax[2].set_ylabel('Paired differences (Hfrac first, variance second)')
for a,y in zip(ax[:2],('Equal-game normalized entropy','Equal-game return variance')):
 a.set(xlabel='Backup budget',ylabel=y);a.legend();grid(a)
fig.suptitle('C: opening-block comparisons (8 at B16/32, 6 at B64)');save('REPLAN-04.C',fig,'finite budget and size comparisons; no scaling exponent',['L07','Q31'])
fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for g in (.2,.5,.8):
 a=[x for x in r['quantum'] if x['g']==g];m=range(9)
 for k,sty in (('Holevo','-'),('PGM_MI','--'),('flag_MI',':')):
  ax[0].plot(m,[st.mean(x['rows'][i][k]/x['H'] for x in a) for i in m],marker='.',ls=sty,label=f'{g} {k}')
 ax[1].plot(m,[st.mean(x['rows'][i]['recovered_fraction'] for x in a) for i in m],marker='o',label=f'g={g}')
 ax[2].plot(m,[st.mean(x['rows'][i]['entropy'] for x in a) for i in m],marker='o',label=f'g={g}')
for a,y,x in zip(ax,('Record information / pointer H','Recovered coherence fraction','System vN entropy'),('Read fragments','Erased fragments','Erased fragments')):
 a.set(xlabel=x,ylabel=y);a.legend(fontsize=6);grid(a)
fig.suptitle('D: declared nonorthogonal records, accessible measurement and Fourier erasure');save('REPLAN-05',fig,'conditional quantum records',['L20','Q24','Q25','Q26'])
fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for L in L0:
 for B in Bs:
  a=[x for x in r['thermal'] if x['L']==L and x['B']==B]
  ts=[v['t'] for v in a[0]['rows']]
  for k,q in zip(('energy','rate'),ax[:2]):
   q.plot(ts,[st.mean(x['rows'][i][k][0] for x in a) for i in range(len(ts))],marker='.',label=f'L{L} B{B}')
  ax[2].plot(ts,[st.mean(np.linalg.norm(np.asarray(x['rows'][i]['pi'])-x['rows'][i]['mu']) for x in a) for i in range(len(ts))],marker='.',label=f'L{L} B{B}')
for a,y in zip(ax,('Energy response tangent 0','Rate response tangent 0','Lag norm ||pi-mu||')):
 a.set(xlabel='Declared normalized protocol time',ylabel=y);a.legend(fontsize=6);grid(a)
fig.suptitle('N: driven nonequilibrium reset, independent energy and rate perturbations');save('REPLAN-06',fig,'conditional nonstationary response',['L06','L33','Q20'])
fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
q=max([x for x in r['quantum'] if x['g']==.5],key=lambda x:x['H']);states=np.load(S/'CONDITIONAL_STATES.npz')
key=f"{q['game']}_g{q['g']:g}"
for a,m in zip(ax,(0,8)):
 z=states[key+'_'+str(m)];v=max(abs(states[key+'_8'].imag).max(),1e-12)
 im=a.imshow(z.imag,cmap='RdBu_r',vmin=-v,vmax=v);a.set(title=f'{m} fragments coherently erased',xlabel='Column',ylabel='Row');fig.colorbar(im,ax=a)
fig.suptitle('Q: explicit phase-corrected complex state recovery');save('X-REP',fig,'conditional complex state',['Q24','Q26'])
fig,ax=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
p=r['mark_prediction'];modes=('zero','sector','action')
ax[0].bar(range(3),[st.mean(x['mse'][i] for x in p) for i in range(3)]);ax[0].set_xticks(range(3),modes);ax[0].set_ylabel('Heldout mark prediction MSE')
for a,method in zip(ax[1:],('direct','LR')):
 z=[x for x in r['control'] if x['method']==method];labs=z[0]['names']
 for i,l in enumerate(('iid','pl','reserve')):
  v=[st.mean(x['variance'][k] for x in z if x['law']==l) for k in range(len(labs))]
  a.plot(range(len(labs)),v,marker='o',label=l)
 a.set_xticks(range(len(labs)),labs,rotation=45,ha='right',fontsize=7);a.set_ylabel(f'{method} mark variance');a.legend();grid(a)
fig.suptitle('Memory: prior-only trained predictable action control on independent cores');save('X-MEMORY',fig,'native predictive mark control',[])
# Dedicated Q19: earlier exact conditional work samples, explicit cyclic protocol.
fig,ax=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
w=np.load(F/'WORK_SAMPLES.npz');q=r['q19'];bins0=np.concatenate([w[k] for k in w.files]);lim=max(abs(bins0));bins=np.linspace(-lim,lim,81)
for tau in (4.,16.,64.):
 z=[x for x in q if x['tau']==tau];f=np.concatenate([w[x['key']+'_F'] for x in z]);rev=-np.concatenate([w[x['key']+'_R'] for x in z])
 ax[0,0].hist(f,bins=bins,weights=np.ones(len(f))/len(f),histtype='step',label=f'tau={tau:g}')
 ax[0,1].hist(f,bins=bins,weights=np.exp(-f)/len(f),histtype='step',label=f'F weighted {tau:g}')
 ax[0,1].hist(rev,bins=bins,weights=np.ones(len(rev))/len(rev),histtype='step',ls=':',label=f'R reflected {tau:g}')
 for k,a,y in (('JE',ax[0,2],'E exp(-w)'),('deltaF_est',ax[1,0],'Delta F estimate')):
  a.plot(z[0]['prefix_n'],np.mean([x[k] for x in z],0),marker='o',label=f'tau={tau:g}');a.set(xscale='log',xlabel='Prefix MC sample count',ylabel=y);a.legend(fontsize=7)
ax[0,0].set(xlabel='Forward dimensionless work',ylabel='Bin mass');ax[0,0].legend(fontsize=7)
ax[0,1].set(xlabel='Work / reflected reverse work',ylabel='Weighted bin mass');ax[0,1].legend(fontsize=6)
ax[0,2].axhline(1,ls=':',color='gray');ax[1,0].axhline(0,ls=':',color='gray')
ts=(4.,16.,64.)
for k in ('Var_F','Var_R'):
 ax[1,1].plot(ts,[mean([x for x in q if x['tau']==t],k) for t in ts],marker='o',label=k)
for k in ('W_F','W_R'):
 ax[1,2].plot(ts,[mean([x for x in q if x['tau']==t],k) for t in ts],marker='o',label=k)
for a,y in zip(ax[1,1:],('Work variance','Cyclic mean dissipated work')):
 a.set(xscale='log',xlabel='Declared duration',ylabel=y);a.legend()
fig.suptitle('Q19 replot: conditional cyclic Crooks/Jarzynski, reused verified R54 work paths');save('Q19_jarzynski_verification',fig,'conditional cyclic work reference, not native heat', ['Q19'],orig)
(S/'plots/manifest.json').write_text(json.dumps(dict(source_sha256=sha(S/'PHYSICS_RESULTS.json'),states_sha256=sha(S/'CONDITIONAL_STATES.npz'),groups=reg),indent=2)+'\n')
print('figures',len(reg),'grouped',len(list(out.glob('*.png'))),'dedicated',len(list(orig.glob('*.png'))))
