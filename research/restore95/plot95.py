from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
A=read(S/'ENTANGLE95.json');Q=read(S/'SUMMARY95.json')['groups'];F=['g0','g3','Rapfi'];P=['native_repeat','pair_diagnostic'];figs=[]
def get(g,p,d,xi=1.,mode='plus81'):
 return next(x for x in Q if (x['family'],x['protocol'],x['depth'],x['xi'],x['mode'])==(g,p,d,xi,mode))
def save(fig,name,ids):
 fig.tight_layout();b=S/'plots'/name
 fig.savefig(str(b)+'.png',dpi=160);fig.savefig(str(b)+'.pdf');plt.close(fig)
 p=Path(str(b)+'.png');q=Path(str(b)+'.pdf')
 with Image.open(p) as im:im.verify()
 subprocess.run(['pdfinfo',str(q)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,png_sha=sha(p),pdf_sha=sha(q)))
def mat(x,k):
 n=len(x['labels']);a=np.zeros((n,n))
 for (i,j),v in zip(x['pairs'],x['metrics_ci'][0]):a[i,j]=a[j,i]=v[k]
 return a
def lab(x):
 return ['-'.join(x['labels'][i] for i in v) for v in x['pairs']]
# G23: QMI vs classical Z MI; same diagonal, not entanglement.
fig,ax=plt.subplots(3,4,figsize=(13,10));fig.suptitle('G23 | two-qubit mutual information, current depth 3, plus arm (nat)')
for i,g in enumerate(F):
 for j,(p,k) in enumerate([(P[0],0),(P[0],1),(P[1],0),(P[1],1)]):
  a=ax[i,j];x=get(g,p,3);m=mat(x,k);im=a.imshow(m,vmin=0,vmax=np.log(2)*2,cmap='viridis')
  for ii,jj in np.ndindex(m.shape):
   if ii!=jj:a.text(jj,ii,f'{m[ii,jj]:.2f}',ha='center',va='center',fontsize=8,color='white' if m[ii,jj]<.8 else 'black')
  a.set(xticks=range(4),yticks=range(4),xticklabels=x['labels'],yticklabels=x['labels'],title=f'{g} {p.replace("_"," ")}\n'+('QMI' if k==0 else 'Z classical MI'));fig.colorbar(im,ax=a,shrink=.65)
save(fig,'G23_mutual_info_matrix',['G23'])
# G24 network: mean conditional negativity; link annotation QMI separate, zero edge not graph disconnect evidence.
fig,ax=plt.subplots(3,2,figsize=(10,11));fig.suptitle('G24 | declared pairwise negativity network; mean of 24 contexts')
xy=np.array([[0,1],[1,1],[1,0],[0,0]],float)
for i,g in enumerate(F):
 for j,p in enumerate(P):
  a=ax[i,j];x=get(g,p,3)
  for (u,v),m in zip(x['pairs'],x['metrics_ci'][0]):
   a.plot(xy[[u,v],0],xy[[u,v],1],color='tab:red' if m[2]>1e-10 else '0.75',lw=.6+8*m[2],ls='-' if m[2]>1e-10 else ':',zorder=1)
   mid=.5*(xy[u]+xy[v]);delta=np.array([-.04,.055]) if (u,v)==(0,2) else np.array([.04,-.055]) if (u,v)==(1,3) else np.array([0,.075])
   a.text(*(mid+delta),f'N={m[2]:.3f}',ha='center',fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
  a.scatter(xy[:,0],xy[:,1],s=700,color='tab:blue',zorder=2)
  for (u,v),s in zip(xy,x['labels']):a.text(u,v,s,color='white',ha='center',va='center',weight='bold',zorder=3)
  a.set(xlim=(-.25,1.25),ylim=(-.25,1.25),title=f'{g} | {p}');a.set_axis_off()
save(fig,'G24_entanglement_network',['G24'])
fig,ax=plt.subplots(3,2,figsize=(12,10));fig.suptitle('G24 | bipartition negativity, depth 3, not just pairwise entanglement')
for i,g in enumerate(F):
 for j,p in enumerate(P):
  a=ax[i,j];x=get(g,p,3);cuts=[''.join(x['labels'][k] for k in c)+' | '+''.join(x['labels'][k] for k in range(4) if k not in c) for c in x['cuts']]
  for k,xi in enumerate([0,.5,1.]):
   y=np.array(get(g,p,3,xi)['cut_ci']);a.bar(np.arange(len(cuts))+(k-1)*.24,y[0],.24,label=f'overlap {xi:g}',yerr=[y[0]-y[1],y[2]-y[0]],capsize=1)
  a.set(xticks=range(len(cuts)),xticklabels=cuts,ylabel='Negativity',title=f'{g} | {p}');a.tick_params(axis='x',labelrotation=35,labelsize=7);a.legend(fontsize=7)
save(fig,'G24_bipartition_negativity',['G24','G27'])
# G25 fixed pairs vs time; no absent pair manufactured as zero
fig,ax=plt.subplots(3,2,figsize=(12,10));fig.suptitle('G25 | current-prefix concurrence evolution; missing subsystems omitted')
for i,g in enumerate(F):
 for j,p in enumerate(P):
  a=ax[i,j]
  for pair in [('R1','S'),('R1','R2'),('R2','S'),('R1','R3'),('R2','R3'),('R3','S')]:
   xs=[];ys=[]
   for d in [1,2,3]:
    x=get(g,p,d);names=[tuple(x['labels'][i] for i in pa) for pa in x['pairs']]
    if pair in names:
     xs.append(d);ys.append(np.array(x['metrics_ci'])[:,names.index(pair),3])
   if xs:
    v=np.array(ys);a.errorbar(xs,v[:,0],yerr=[v[:,0]-v[:,1],v[:,2]-v[:,0]],fmt='o-',label='-'.join(pair),capsize=2)
  a.set(xticks=[1,2,3],xlabel='Completed waves',ylabel='Concurrence',title=f'{g} | {p}');a.legend(fontsize=7,ncol=2)
save(fig,'G25_concurrence_evolution',['G25'])
# G26 explicit Pauli CHSH, all pairs and tag controls
fig,ax=plt.subplots(3,2,figsize=(12,10));fig.suptitle('G26 | theoretical maximal Pauli CHSH; no operational Bell experiment')
for i,g in enumerate(F):
 for j,p in enumerate(P):
  a=ax[i,j];x=get(g,p,3);rows=[r for r in A['rows'] if r['family']==i and r['protocol']==p and r['mode']=='plus81' and r['depth']==3 and r['xi']==1]
  z=np.array([r['metrics'] for r in rows]);a.boxplot(z[:,:,5],tick_labels=lab(x),showfliers=True);a.axhline(2,ls='--',color='gray',label='local bound');a.axhline(2*np.sqrt(2),ls=':',color='black',label='Tsirelson bound')
  a.set(ylabel='Maximal Pauli CHSH',title=f'{g} | {p}');a.tick_params(axis='x',labelrotation=25,labelsize=8);a.legend(fontsize=7)
save(fig,'G26_bell_test_results',['G26'])
fig,ax=plt.subplots(2,3,figsize=(13,7));fig.suptitle('G26 | conditional violations versus success-weighted indicators')
for j,g in enumerate(F):
 for p in P:
  for xi in [0,.5,1.]:
   x=[get(g,p,d,xi) for d in [1,2,3]]
   ax[0,j].plot([1,2,3],[v['pair_CHSH']/(24*len(v['pairs'])) for v in x],'o-',label=f'{p} xi={xi:g}')
   ax[1,j].semilogy([1,2,3],[max(1e-40,v['max_weighted_CHSH']) for v in x],'o-',label=f'{p} xi={xi:g}')
 ax[0,j].set(xticks=[1,2,3],ylabel='Fraction of pairs above 2',title=g);ax[1,j].set(xticks=[1,2,3],xlabel='Completed waves',ylabel='max s × positive CHSH excess (0 at floor)');ax[0,j].legend(fontsize=6);ax[1,j].legend(fontsize=6)
save(fig,'G26_tag_and_success_controls',['G26','G27'])
# G27 contrasts: explicit counts, nonlocality and QMI distinguish.
fig,ax=plt.subplots(2,3,figsize=(13,7));fig.suptitle('G27 | entanglement summary; plus arm at current depth 3')
for j,g in enumerate(F):
 for k,p in enumerate(P):
  x=get(g,p,3);n=24*len(x['pairs']);cutn=24*len(x['cuts'])
  ax[0,j].bar(np.arange(3)+(k-.5)*.35,[x['pair_NPT']/n,x['pair_CHSH']/n,x['cut_NPT']/cutn],.35,label=p)
  for xi in [0,.5,1.]:
   y=get(g,p,3,xi);ax[1,j].scatter(np.array(y['metrics_ci'])[0,:,0],np.array(y['metrics_ci'])[0,:,2],s=22,label=p+f' xi={xi:g}',marker=['o','s','^'][[0,.5,1.].index(xi)])
 ax[0,j].set(xticks=range(3),xticklabels=['Pair NPT','Pair CHSH','Cut NPT'],ylim=(0,1.05),ylabel='Fraction of evaluated partitions',title=g);ax[0,j].legend(fontsize=7)
 ax[1,j].set(xlabel='Mean QMI (nat)',ylabel='Mean negativity',title='Information is not entanglement');ax[1,j].legend(fontsize=5,ncol=2)
save(fig,'G27_entanglement_summary',['G27'])
dump(S/'PLOT_MANIFEST95.json',dict(figures=figs,sources={k:sha(S/k) for k in ['ENTANGLE95.json','JOINT95.npz','SUMMARY95.json','INDEPENDENT95.json','ORIGINAL_SOURCE95.json']},scope='Actual branch-derived declared record model; current prefix, pointwise root intervals, no physical Bell test.'))
print('PLOTS',len(figs),flush=True)
