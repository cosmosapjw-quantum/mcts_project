from therm100 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io,subprocess
N=read(S/'NATIVE100.json');T=read(S/'THERMAL100.json');Q=read(S/'SUMMARY100.json');C=T['cases'];cycles=T['cycles'];G=N['groups'];figs=[];animations=[]
def save(fig,name,ids):
 fig.tight_layout();b=S/'plots'/name;fig.savefig(str(b)+'.png',dpi=145);fig.savefig(str(b)+'.pdf');plt.close(fig)
 a=Path(str(b)+'.png');p=Path(str(b)+'.pdf')
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(p)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,png_sha=sha(a),pdf_sha=sha(p)))
def band(a,x,z,label):
 z=np.asarray(z);a.plot(x,z[:,0],'o-',ms=3,label=label);a.fill_between(x,z[:,1],z[:,2],alpha=.12)
def legend(a):a.legend(fontsize=7)
def get(L,arm,bi,h):return next(g for g in G if (g['L'],g['cohort'],g['arm'],g['bi'],g['budget'],g['h'])==(L,2,arm,bi,64,h))
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q01/Q07 | actual native information ledger, not thermal entropy')
for k in range(2):
 for L in LS:
  for j,m in enumerate([0,1,2]):band(ax[k,j],HS,[np.array(get(L,1,3,h)['ci'])[:,k,m] for h in HS],str(L))
 for j,l in enumerate(['Realized log P/Q (nat)','Predictable KL sum (nat)','Centered martingale sum (nat)']):ax[k,j].set(xlabel='Backup prefix',ylabel=l,title=N['comparators'][k]);legend(ax[k,j])
save(f,'Q01_native_information_ledger',['Q01','Q07'])
f,ax=plt.subplots(2,2,figsize=(10,7));f.suptitle('Native likelihood IFT: exact conditional normalization vs finite sampled rare weights')
for k in range(2):
 for bi in range(5):
  g=[get(21,1,bi,h) for h in HS];l='beta index '+str(bi)
  ax[k,0].plot(HS,[v['log_sample_IFT'][k] for v in g],'o-',label=l);ax[k,1].plot(HS,[v['ess'][k]/v['samples'] for v in g],'o-',label=l)
 ax[k,0].axhline(0,color='gray',lw=.5);ax[k,0].set(xlabel='Backup prefix',ylabel='log sample mean exp(-Sigma)',title=N['comparators'][k]);ax[k,1].set(xlabel='Backup prefix',ylabel='ESS/N',ylim=(0,1.04));legend(ax[k,0]);legend(ax[k,1])
save(f,'Q01_native_likelihood_sampling',['Q01','Q07'])
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q01 | explicitly driven reference: two quenches and two relaxations')
for j,r in enumerate(RS):
 g=[v for v in C if v['refresh']==r];st=np.array([v['stages'] for v in g]);z=st.mean(0)
 a=ax[0,j]
 for k,l in [(0,'U'),(2,'cum W on'),(3,'cum Q in')]:a.plot(range(5),z[:,k],'o-',label=l)
 a.set(xticks=range(5),xlabel='Declared stage',ylabel='Score-energy units',title=f'r={r}');legend(a)
 a=ax[1,j];s=np.array([v['sigma'] for v in g]);a.bar([1,2,3],s.mean(0),tick_label=['Relax 1','Relax 2','Final KL']);a.set(ylabel='Dimensionless dissipation',title=f'r={r}')
save(f,'Q01_reference_driven_protocol',['Q01','Q07','Q23','Q38'])
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q19 | independent F and R: exact and sampled work laws (first registered context)')
for j,r in enumerate(RS):
 g=next(v for v in C if v['context']==0 and v['refresh']==r);b=np.array(g['hist_edges']);x=(b[:-1]+b[1:])/2
 for k,l in [('exact_F','Exact F(W)'),('exact_R_minus','Exact R(-W)'),('MC_F','Sample F(W)'),('MC_R_minus','Sample R(-W)')]:ax[0,j].plot(x,g[k],'-' if k.startswith('exact') else 'o',ms=2,label=l)
 ax[0,j].axvline(g['df'],color='gray',ls=':');ax[0,j].set(xlabel='W (score-energy units)',ylabel='Bin mass',title=f'r={r}');legend(ax[0,j])
 # Exact unbinned Crooks: use actual pair weights, plotted log ratios; no bin-center approximation.
 c=read(R85/'BRANCH_ENERGY85.json')['contexts'][0];e=np.array(c['e0']);e1=np.array(c['em'][0]);em=(e+e1)/2;d=em-e;A=K(gib(em),r);pf=gib(e)[:,None]*A;pr=gib(e1)[None,:]*A.T;w=d[:,None]+d[None,:]
 ax[1,j].scatter(w.ravel(),np.log(pf/pr).ravel(),s=2,alpha=.2,label='Exact pair law');xx=np.array([w.min(),w.max()]);ax[1,j].plot(xx,BETA*(xx-g['df']),'k:',label='beta(W-DeltaF)');ax[1,j].set(xlabel='W',ylabel='log F/R',title='Paired support');legend(ax[1,j])
save(f,'Q19_reference_Crooks_work_partial',['Q19','Q01','Q23'])
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q19/Q22 | finite sample Jarzynski convergence; all context errors retained')
for j,r in enumerate(RS):
 g=[v for v in C if v['refresh']==r]
 for k,lab in [('F','Forward'),('R','Reverse')]:
  e=np.array([[z['estimate']-(v['df'] if k=='F' else -v['df']) for z in v[k]] for v in g]);ess=np.array([[z['ess']/z['n'] for z in v[k]] for v in g])
  for a,z in [(ax[0,j],abs(e)),(ax[1,j],ess)]:
   a.plot(NS,np.median(z,axis=0),'o-',label=lab);a.fill_between(NS,np.quantile(z,.1,axis=0),np.quantile(z,.9,axis=0),alpha=.12)
 for i,l in enumerate(['Absolute DeltaF error','ESS / N']):ax[i,j].set(xscale='log',xticks=NS,xticklabels=NS,xlabel='Independent samples',ylabel=l,title=f'r={r}');legend(ax[i,j])
save(f,'Q19_Q22_Jarzynski_sampling',['Q19','Q22','Q23'])
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q20 | exact reference work cumulants and finite tilted dual')
for j,k in enumerate([0,1,2,3]):
 a=ax.flat[j]
 for r in RS:
  g=[v for v in C if v['refresh']==r];a.plot(range(len(g)),[v['forward_cumulants'][k] for v in g],'.',ms=2,label=str(r))
 a.set(xlabel='Fixed spectrum context',ylabel=['Mean','Variance','Central third','Fourth cumulant'][k]);legend(a)
for r in RS:
 g=next(v for v in C if v['context']==0 and v['refresh']==r);ax.flat[4].plot(g['tilt']['mean'],g['tilt']['dual'],label=str(r));ax.flat[5].plot(g['tilt']['lam'],g['tilt']['psi'],label=str(r))
ax.flat[4].set(xlabel='Tilted mean work',ylabel='Finite convex dual (nat)');ax.flat[5].set(xlabel='Tilt source',ylabel='log E exp(lambda W)');legend(ax.flat[4]);legend(ax.flat[5])
save(f,'Q20_reference_work_cumulants_partial',['Q20','Q23'])
f,ax=plt.subplots(1,3,figsize=(12,4));f.suptitle('Q20 | actual reference sample work trajectories, not synthetic random walks')
with np.load(S/'THERMAL_SAMPLES100.npz') as z:
 src=read(R85/'BRANCH_ENERGY85.json')['contexts'][0];d=(np.array(src['em'][0])-src['e0'])/2
 for j,r in enumerate(RS):
  a=ax[j];i,k,l=z['indices'][j,0,:48].T;v=np.c_[np.zeros(len(i)),d[i],d[i],d[i]+d[k],d[i]+d[k]]
  a.plot(np.arange(5),v.T,alpha=.12);a.plot(np.arange(5),v.mean(0),'k-o',label='48 sample mean');a.set(xlabel='Exact stage',ylabel='Accumulated W on',title=f'r={r}');legend(a)
save(f,'Q20_reference_work_trajectories_partial',['Q20','Q01'])
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q03 | periodic two-reservoir scaled-spectrum reference cycle')
for j,key in enumerate(['W_out','Qh','sigma']):
 for a0 in [1.5,2.]:
  g=[v for v in Q['cycles'] if v['scale']==a0];ax[0,j].plot([v['refresh'] for v in g],[v[key] for v in g],'o-',label=f'a={a0}')
 ax[0,j].set(xlabel='Refresh fraction (both baths)',ylabel=key);legend(ax[0,j])
for j,r in enumerate([.25,.75,1.]):
 a=ax[1,j]
 for a0 in [1.5,2.]:
  g=next(v for v in cycles if v['context']==0 and v['scale']==a0 and v['refresh']==r);v=np.array(g['stages']);a.plot(v[:,1],v[:,0],'o-',label=f'a={a0}')
 a.set(xlabel='S/kB (nat)',ylabel='U (score-energy units)',title=f'r={r}, periodic');legend(a)
save(f,'Q03_periodic_Otto_reference',['Q03','Q04','Q07'])
f,ax=plt.subplots(1,3,figsize=(13,4));f.suptitle('Q06 partial | reference finite vs full refresh; native score changes separately')
r=np.array([.25,.75,1.]);ax[0].plot(r,r/(2-r),'o-',label='Finite/ideal output');ax[0].set(xlabel='Refresh fraction',ylabel='Output ratio');legend(ax[0])
for a0 in [1.5,2.]:
 g=[v for v in Q['cycles'] if v['scale']==a0];ax[1].plot(r,[v['eta'] for v in g],'o-',label=f'a={a0}')
ax[1].axhline(1-.3/.9,color='gray',ls=':',label='Carnot bound');ax[1].set(xlabel='Refresh fraction',ylabel='Reference efficiency');legend(ax[1])
for k,l in [(6,'Score switch'),(7,'Action change'),(8,'Net score-energy change')]:ax[2].plot(LS,[np.array(get(L,1,3,64)['ci'])[0,0,k] for L in LS],'o-',label=l)
ax[2].set(xlabel='Native board L',ylabel='Native score units, not heat/work');legend(ax[2])
save(f,'Q06_noncomparable_native_reference_partial',['Q06','Q07'])
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q38 | explicit finite spectrum Gibbs entropy and density of states')
v=np.array([x['curve'] for x in T['capacity']]);th=T['theta']
for j,k in enumerate([2,3,1,0]):
 a=ax.flat[j];a.plot(th,v[:,:,k].mean(0));a.fill_between(th,np.quantile(v[:,:,k],.1,axis=0),np.quantile(v[:,:,k],.9,axis=0),alpha=.18);a.set(xscale='log',xlabel='theta=kB T (score-energy units)',ylabel=['U','S/kB','C/kB','F'][k])
for a,g in zip(ax.flat[4:],(T['capacity'][0],T['capacity'][-1])):
 a.bar([(z['low']+z['high'])/2 for z in g['dos']],[z['S'] for z in g['dos']],width=.09);a.set(xlabel='Fixed energy bin center',ylabel='ln occupied bin count',title=f"Context {g['context']}, n={g['n']}")
save(f,'Q38_Gibbs_capacity_entropy_DOS',['Q38','Q07'])
# Actual exact stage animation: no fabricated interpolation. Use identical scales each frame.
a0=2.;rr=.75;cy=next(v for v in cycles if v['context']==0 and v['scale']==a0 and v['refresh']==rr);cy1=next(v for v in cycles if v['context']==0 and v['scale']==a0 and v['refresh']==1)
def cycle_frame(j):
 f,ax=plt.subplots(1,2,figsize=(10,4));f.suptitle('Q04 | declared periodic cycle: '+['cold start','scale up','hot relaxation','scale down','cold relaxation'][j])
 for g,l in [(cy,'Finite refresh'),(cy1,'Full refresh')]:
  v=np.array(g['stages']);ax[0].plot(v[:j+1,1],v[:j+1,0],'o-',label=l);ax[1].plot(range(j+1),v[:j+1,2],'o-',label=l+' W');ax[1].plot(range(j+1),v[:j+1,3],'s--',label=l+' Q')
 vs=np.concatenate([np.array(g['stages']) for g in (cy,cy1)])
 for a,k,l in [(ax[0],0,'U'),(ax[1],2,'Cumulative W/Q')]:
  if k==0:a.set(xlim=(vs[:,1].min()-.1,vs[:,1].max()+.1),ylim=(vs[:,0].min()-.1,vs[:,0].max()+.1),xlabel='S/kB',ylabel=l)
  else:a.set(xlim=(-.1,4.1),ylim=(vs[:,2:].min()-.1,vs[:,2:].max()+.1),xlabel='Stage',ylabel=l)
  legend(a)
 return f
def je_frame(j):
 f,ax=plt.subplots(1,2,figsize=(10,4));f.suptitle(f'Q22 | independent reference samples N={NS[j]}')
 for r in RS:
  g=[v for v in C if v['refresh']==r]
  for k,style in [('F','o-'),('R','s--')]:
   e=np.array([[abs(z['estimate']-(v['df'] if k=='F' else -v['df'])) for z in v[k]] for v in g]);ess=np.array([[z['ess']/z['n'] for z in v[k]] for v in g])
   ax[0].plot(NS[:j+1],np.median(e,axis=0)[:j+1],style,label=f'{k}, r={r}');ax[1].plot(NS[:j+1],np.median(ess,axis=0)[:j+1],style,label=f'{k}, r={r}')
 for a,l in zip(ax,['Median absolute DeltaF error','Median ESS/N']):a.set(xscale='log',xlim=(50,5000),xticks=NS,xticklabels=NS,xlabel='Sample count',ylabel=l);legend(a)
 ax[0].set_ylim(0,.06);ax[1].set_ylim(0,1.04);return f
def gif(name,id0,frames,fn,clock):
 ims=[]
 for j in frames:
  f=fn(j);f.tight_layout();b=io.BytesIO();f.savefig(b,format='png',dpi=95);b.seek(0);ims.append(Image.open(b).convert('RGB'));plt.close(f)
 p=S/'plots'/name;ims[0].save(p,save_all=True,append_images=ims[1:],duration=800,loop=0)
 with Image.open(p) as im:
  assert im.n_frames==len(frames);hs=[]
  for i in range(im.n_frames):im.seek(i);hs.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
  assert len(set(hs))==len(frames)
 animations.append(dict(id=id0,path='plots/'+p.name,sha256=sha(p),frames=len(frames),frame_rgb_sha256=hs,clock=clock))
gif('Q04_periodic_reference_cycle.gif','Q04',list(range(5)),cycle_frame,['cold start','scale up','hot relaxation','scale down','cold relaxation'])
gif('Q22_actual_reference_Jarzynski.gif','Q22',list(range(4)),je_frame,NS)
dump(S/'PLOT_MANIFEST100.json',dict(figures=figs,animations=animations,sources={k:sha(S/k) for k in ['NATIVE100.json','NATIVE100.npz','THERMAL100.json','THERMAL_SAMPLES100.npz','SUMMARY100.json','INDEPENDENT100.json','ORIGINAL_SOURCE100.json']},scope='Native information ledger, independent declared thermal reference, no native calibrated work/heat.'))
print('PLOTS',len(figs),'GIFS',[(a['id'],a['frames']) for a in animations],flush=True)
