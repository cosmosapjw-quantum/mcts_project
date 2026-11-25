"""Dedicated original critical, transport and structure plots; explicit scope."""
from pathlib import Path
import json,hashlib,statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
S=Path(__file__).resolve().parent;r=json.loads((S/'PHYSICS_RESULTS.json').read_text())
out=S/'plots/original';out.mkdir(exist_ok=False);reg={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,fig,scope,ids):
 for e in ('png','pdf'):fig.savefig(out/(name+'.'+e),dpi=150,bbox_inches='tight')
 plt.close(fig);p=out/(name+'.png')
 with Image.open(p) as im:im.verify()
 reg[name]=dict(scope=scope,original_items=ids,files={str((out/(name+'.'+e)).relative_to(S)):sha(out/(name+'.'+e)) for e in ('png','pdf')})
def mean(a,k):return st.mean(x[k] for x in a)
Ls=(13,15,17,19,21);bs=(1.,10/3,8.);sp=r['spatial']
def group(B,b,L):return [x for x in sp if x['B']==B and x['beta']==b and x['L']==L]
def curve(B,b,k):return [mean(group(B,b,L),k) for L in Ls]
def grid(a):a.grid(alpha=.2)
fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for B in (16,32):
 for b in bs:
  for a,k in zip(ax,('order','H_fraction','central_mass')):
   a.plot(Ls,curve(B,b,k),marker='o',label=f'B{B} beta={b:.2g}',ls='-' if B==16 else ':')
for a,y in zip(ax,('Top-two occupancy gap','Normalized entropy','Central 5x5 mass')):
 a.set(xlabel='Board size L',ylabel=y);a.legend(fontsize=7);grid(a)
fig.suptitle('G06: observed finite-budget concentration, not spontaneous symmetry breaking')
save('G06_order_parameter',fig,'native finite-budget order proxy',['G06','L07'])
fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for B in (16,32):
 for b in bs:
  a=[x for x in sp if x['B']==B and x['beta']==b]
  ax[0].scatter([x['source_variance'] for x in a],[x['source_response'] for x in a],label=f'B{B} beta={b:.2g}',s=18)
for a,k in zip(ax[1:],('occupancy','detection')):
 for key,sty in (('direct','o-'),('conditional','s:')):
  a.plot(Ls,[st.mean(x[k][key] for x in r['response'] if x['L']==L) for L in Ls],sty,label=key)
 a.set(xlabel='Board size L',ylabel='Finite source contrast: '+k);a.legend();grid(a)
ax[0].set(xlabel='Whole-history source occupancy variance',ylabel='Whole-history score covariance response');ax[0].legend(fontsize=7);grid(ax[0])
fig.suptitle('G07/Q32: distinct fluctuation and response; exact conditional final IID wave')
save('G07_Q32_susceptibility',fig,'native finite source response, no equilibrium FDT or critical peak',['G07','Q32'])
fig,ax=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
for i,B in enumerate((16,32)):
 for j,b in enumerate(bs):
  a=ax[i,j]
  for L in Ls:
   rows=group(B,b,L);rr=sorted(set.intersection(*[set(y['r'] for y in x['radial']) for x in rows]))
   c=[st.mean(next(y['C'] for y in x['radial'] if y['r']==k) for x in rows) for k in rr]
   a.plot(rr,c,marker='.',ms=3,label=f'L{L}')
  a.set(title=f'B={B}, beta={b:.3g}',xlabel='Open-boundary distance bin',ylabel='Connected C(r)',yscale='symlog',ylim=None);a.axhline(0,color='gray',lw=.7);a.legend(fontsize=7);grid(a)
fig.suptitle('G08/Q29: legal-pair connected covariance; fixed-total zero-mode constraint')
save('G08_Q29_spatial_correlation',fig,'native open spatial covariance; temporal correlations and physical xi not claimed',['G08','Q29'])
fig,ax=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
for i,B in enumerate((16,32)):
 for j,b in enumerate(bs):
  a=ax[i,j]
  for k,sty in (('total','o-'),('martingale_sample','s:'),('drift','^-'),('cross','d--'),('predictable_trace','x:')):
   a.plot(Ls,curve(B,b,k),sty,label=k,ms=4)
  a.set(title=f'B={B}, beta={b:.3g}',xlabel='Board size L',ylabel='Covariance trace');a.axhline(0,color='gray',lw=.7);a.legend(fontsize=7);grid(a)
fig.suptitle('Q31/G09: finite-size variation separated into counting, adaptive drift and cross terms')
save('Q31_G09_finite_size',fig,'native finite-size covariance decomposition, no critical exponent or energy gap',['Q31','G09','Q16'])
fig,ax=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
e=r['example'];L=e['L'];v=np.array([np.nan if x is None else x for x in e['root_k8_score']]).reshape(L,L)
im=ax[0].imshow(v,cmap='viridis',origin='lower');path=e['barrier']['path'];ax[0].plot([x%L for x in path],[x//L for x in path],'w.-',label='Minimax path');ax[0].legend(fontsize=7);ax[0].set(title='Preselected L17 k8 score landscape');fig.colorbar(im,ax=ax[0],shrink=.7)
for B in (16,32):
 for b in bs:
  a=[x for x in r['survival'] if x['B']==B and x['beta']==b]
  ax[1].step(range(B),np.mean([x['survival'] for x in a],0),where='post',label=f'B{B} beta={b:.2g}')
ax[1].set(xlabel='Root selection index (0-based)',ylabel='No detection fraction');ax[1].legend(fontsize=7)
a=[x for x in r['barriers'] if x['amplitude']==0 and x['survived_k8'] and x['connected']]
for hit in (0,1):
 z=[x for x in a if x['hit_after_k8']==hit];ax[2].scatter([x['excess'] for x in z],[x['source_probability_k8'] for x in z],label='later hit' if hit else 'censored',s=15,alpha=.6)
ax[2].set(xlabel='Geometric excess barrier',ylabel='Actual k8 source probability',yscale='log');ax[2].legend();grid(ax[2])
fig.suptitle('Barrier diagnostics: geometric graph reference vs actual first detection (not tunneling)')
save('G18_G19_G20_transport',fig,'native first detection and minimax geometric score landscape; no WKB',['G18','G19','G20','L10'])
fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for q,sty in ((e['topology_k8'],'-'),(e['topology_visited'],':')):
 a=q['filtration'];ax[0].step([x['level'] for x in a],[x['b0'] for x in a],where='post',ls=sty,label='k8 score' if sty=='-' else 'visited Q');ax[1].step([x['level'] for x in a],[x['b1'] for x in a],where='post',ls=sty,label='k8 score' if sty=='-' else 'visited Q')
ax[0].set(xlabel='Sublevel threshold',ylabel='Betti 0');ax[1].set(xlabel='Sublevel threshold',ylabel='Betti 1')
for a in ax[:2]:a.legend();grid(a)
for L in Ls:
 a=[x for x in r['topology'] if x['L']==L and x['amplitude']==0];ax[2].scatter([x['positive_H0'] for x in a],[x['mean_persistence'] for x in a],s=12,label=f'L{L}',alpha=.5)
ax[2].set(xlabel='Positive H0 intervals',ylabel='Mean persistence');ax[2].legend(fontsize=7)
fig.suptitle('L24: lower-star cubical score filtration (not quantum topology)')
save('L24_topological_analysis',fig,'native score and visited-Q cubical topology, exact DFS cross-check',['L24'])
fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
ax[0].bar(['Single fault\ncorrected','Double fault\ndetected only'],[r['fault_counts']['single_corrected'],r['fault_counts']['double_detected_uncorrectable']]);ax[0].set_ylabel('Offline injected cases')
ax[1].bar(['Maximum reconstruction error'],[r['fault_max_error']]);ax[1].set_ylabel('Root W-vector absolute error');ax[1].set_ylim(0,1e-12)
fig.suptitle('L23: classical parity recovery on copies of actual backup arrays')
save('L23_backup_fault_recovery',fig,'classical offline numerical recovery, not quantum error correction or live engine repair',['L23'])
(S/'plots/manifest.json').write_text(json.dumps(dict(source_sha256=sha(S/'PHYSICS_RESULTS.json'),groups=reg),indent=2)+'\n')
print('rendered',len(reg),'dedicated pairs')
