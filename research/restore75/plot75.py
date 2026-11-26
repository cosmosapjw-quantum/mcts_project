from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
R=read(S/'FIRST_RESULTS75.json');Q=read(S/'TRANSPORT75.json');G=read(S/'GAMES75.json');I=read(S/'INDEPENDENT75.json')
names=['g0','g3','Rapfi'];fs=R['results'];ctx=Q['contexts'];figs=[]
def save(fig,n,ids,scope):
 fig.tight_layout();p=S/'plots'/n;fig.savefig(str(p)+'.png',dpi=150);fig.savefig(str(p)+'.pdf');plt.close(fig)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 assert b.read_bytes()[:4]==b'%PDF'
 figs.append(dict(figure=n,ids=ids,scope=scope,png_sha=sha(a),pdf_sha=sha(b)))
fig,ax=plt.subplots(1,3,figsize=(12,4))
for k,a in enumerate(ax):
 g=next(q for q in R['contexts'] if q['family']==k and q['core']==0);L=g['L'];v=np.full(L*L,np.nan);v[g['component']]=g['potential'];im=a.imshow(v.reshape(L,L),cmap='coolwarm');p=np.array(g['path'])
 a.plot(p%L,p//L,'k.-',lw=1);a.scatter([p[0]%L,p[-1]%L],[p[0]//L,p[-1]//L],marker='s',s=40,facecolors='none',edgecolors='black')
 a.set_title(names[k]+f" | b={g['height']:.3f}, width={g['width']}");fig.colorbar(im,ax=a,shrink=.7,label='shifted score V')
save(fig,'actual_score_minimax_graph',['G20','G21','G22'],'Registered core 0 contexts; actual score graph and shortest minimax path, not native adjacency constraint.')
t=np.arange(1,97)
fig,ax=plt.subplots(1,3,figsize=(13,4))
for a,r in zip(ax,fs):
 y=np.array(r['cum_hit']);a.plot(t,y[0],label='native hit');a.fill_between(t,y[1],y[2],alpha=.15)
 # Use root-level means of frozen curves directly from array.
 with np.load(S/'FIRST75.npz') as z:m=z['meta'];v=z['features']
 k=names.index(r['family']);f=np.mean([v[(m[:,0]==k)&(m[:,1]==c),:,4].mean(0) for c in range(32)],0);a.plot(t,f,'--',label='frozen hazard')
 a.set(title=r['family'],xlabel='post-H32 selections',ylabel='cumulative passage');a.set_ylim(0,1.03);a.legend(fontsize=8)
save(fig,'native_first_passage',['L10','G19','G22'],'First subsequent hit and fixed initial hazard with complete risk set, no causal comparison.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
for a,r in zip(ax,fs):
 y=np.array(r['compensator_residual']);a.plot(t,y[0]);a.fill_between(t,y[1],y[2],alpha=.16);a.axhline(0,color='k',lw=.6);a.set(title=r['family'],xlabel='post-H32 selections',ylabel='hit minus compensator')
save(fig,'native_hazard_martingale',['G19','G22'],'Pointwise root block intervals; predictable compensator check, not universal equivalence.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
for k,a in enumerate(ax):
 z=[q for q in R['paths'] if q['family']==k];v=np.array([[q['height'],q['width'],q['event']] for q in z]);e=v[:,2]>0
 a.scatter(v[~e,0],v[~e,1],s=12,alpha=.3,label='censored');a.scatter(v[e,0],v[e,1],s=8,alpha=.17,label='hit')
 a.set(title=names[k],xlabel='minimax excess barrier',ylabel='path width (edges)');a.legend(fontsize=8)
save(fig,'native_barrier_risk',['G19','G20','G22'],'All eligible paths including censored; descriptive selection-dependent landscape.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for i,r in enumerate(fs):
 y=np.array(r['RMST'])
 for j,lab in enumerate(['native','frozen']):ax[0].errorbar(i+(j-.5)*.22,y[0,j],yerr=[[y[0,j]-y[1,j]],[y[2,j]-y[0,j]]],fmt='o',label=lab if i==0 else None)
 ax[1].bar(i,r['nonneighbor_fraction'])
for a in ax:a.set_xticks(range(3),names)
ax[0].set(ylabel='restricted mean selections (96 cap)',title='Right-censored first hit');ax[0].legend();ax[1].set(ylabel='fraction of hit events',title='Non-neighbor previous action');ax[1].set_ylim(0,1)
save(fig,'native_rmst_and_nonlocality',['L10','G20','G22'],'RMST and direct nonlocality witness; geometric barrier is not required native path.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for h in [.06,.1,.25,.5,1.]:
 v=[q for q in Q['rectangular_controls'] if q['height']==h];n=[q['n'] for q in v];ax[0].plot(n,[q['logT'] for q in v],'o-',label=f'V={h}');ax[0].plot(n,[q['discrete_exponent'] for q in v],':',color=ax[0].lines[-1].get_color())
 ax[1].plot(n,[q['logT']-q['discrete_exponent'] for q in v],'o-',label=f'V={h}')
ax[0].set(xlabel='barrier sites',ylabel='log transmission',title='Exact (solid), lattice action (dotted)');ax[1].set(xlabel='barrier sites',ylabel='log T + 2 sum kappa',title='Finite interface/prefactor residual')
for a in ax:a.legend(fontsize=7)
save(fig,'rectangular_wkb',['G21','G22'],'Exact rectangular control and lattice asymptotic slopes; not a native passage formula.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for w in [1.,2.,4.,8.]:
 z=[q for q in Q['continuum_controls'] if q['width']==w];a=np.array([q['lattice_spacing'] for q in z]);e=np.array([q['error'] for q in z]);ax[0].loglog(a,e,'o-',label=f'width={w}');ax[1].plot(a,[q['logT_lattice'] for q in z],'o-',label=f'width={w}');ax[1].axhline(z[0]['logT_continuum'],ls=':',color=ax[1].lines[-1].get_color())
ax[0].set(xlabel='declared grid spacing a',ylabel='absolute logT error',title='Controlled continuum discretization');ax[1].set(xlabel='a',ylabel='log T',title='Exact continuum slab (dotted)')
for a in ax:a.legend(fontsize=8)
save(fig,'continuum_wkb_limit',['G21'],'Fixed width and physical model coefficient; convergence of lattice scattering, not calibrated native hbar.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
for k,a in enumerate(ax):
 z=[q['scattering'][1] for q in ctx if q['family']==k];x=np.array([v['path']['T'] for v in z]);y=np.array([v['full']['T'] for v in z]);a.loglog(x,y,'o',ms=4);lo=min(x.min(),y.min());hi=max(x.max(),y.max());a.plot([lo,hi],[lo,hi],'k:',lw=.7)
 a.set(title=names[k]+' E=.15',xlabel='single minimax path exact T',ylabel='full graph exact T')
save(fig,'graph_vs_path_transmission',['G21','G22'],'Actual frozen score graphs; graph and single-path transmissions can differ by orders of magnitude.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
for k,a in enumerate(ax):
 z=[q['scattering'][1] for q in ctx if q['family']==k];x=np.array([np.log(v['path']['T']) for v in z]);a.scatter(x,[-2*v['action_discrete'] for v in z],s=22,label='lattice local action');a.scatter(x,[-2*v['action_continuum'] for v in z],s=14,marker='x',label='continuum exponent')
 lo=min(x.min(),min(-2*v['action_continuum'] for v in z));hi=max(x.max(),0);a.plot([lo,hi],[lo,hi],'k:');a.set(title=names[k],xlabel='exact logT along path',ylabel='action-only logT proxy');a.legend(fontsize=7)
save(fig,'native_path_wkb_error',['G21'],'Unfitted local evanescent actions on actual minimax paths; no uniform WKB quantitative validity.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for k in range(3):
 z=[q['scattering'][1] for q in ctx if q['family']==k];x=np.array([v['full']['T'] for v in z]);y=np.array([v['phase_T'] for v in z]);ax[0].loglog(x,y,'o',ms=4,label=names[k]);ax[1].scatter(np.full(len(x),k),y-x,s=15,alpha=.6)
ax[0].plot([1e-17,1],[1e-17,1],'k:');ax[0].set(xlabel='zero-flux graph T',ylabel='phase-flux graph T',title='Declared plaquette flux perturbation');ax[0].legend()
ax[1].set_xticks(range(3),names);ax[1].axhline(0,color='k',lw=.6);ax[1].set(ylabel='T(phi) − T(0)',title='Phase-sensitive changes, either sign')
save(fig,'graph_phase_transport',['G21','G22'],'Defined link phase, gauge test and time-reversal reciprocity; not native measured phase.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
for k,a in enumerate(ax):
 z=[q for q in ctx if q['family']==k]
 for key,field,lab in [('detector','coherent_f','coherent'),('detector','dephased_f','fully dephased'),('phase_detector','coherent_f','coherent + flux')]:
  y=np.mean([np.cumsum(q[key][field]) for q in z],0);a.plot(np.arange(1,65),y,label=lab)
 a.set(title=names[k],xlabel='monitoring step (model time)',ylabel='cumulative first detection');a.legend(fontsize=7)
save(fig,'graph_detector_coherence',['G19','G22'],'Same H and cadence, coherent vs exact fully dephased classical chain; not stationary T or native time.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
r=next(q for q in G['results'] if q['threshold']==.2)
for q in r['rows']:
 if not q['eligible']:continue
 a=np.array(q['values']);v=a[q['entry_index']:,1]-q['peak'];ax[0].plot(np.arange(len(v)),v,alpha=.65,ls='-' if q['event'] else '--')
ax[0].axhline(0,color='k',lw=.6);ax[0].set(xlabel='same-player decision age after entry',ylabel='value relative to fixed entry peak',title='R68 historical drawdown entries')
for i,q in enumerate(G['results']):
 for j,key in enumerate(['events','censored','no_entry']):ax[1].bar(i+(j-1)*.25,q[key],.24,label=key if i==0 else None)
ax[1].set_xticks(range(3),[str(q['threshold']) for q in G['results']]);ax[1].set(xlabel='value drawdown threshold',ylabel='games',title='All 16 historical games');ax[1].legend(fontsize=8)
save(fig,'across_game_drawdown',['L10','G19','G20','G22'],'Historical actual same-side root value recovery; informative terminal censoring, no population probability claim.')
assert len(figs)==12 and len({q['png_sha'] for q in figs})==12
dump(S/'PLOT_MANIFEST75.json',dict(figures=figs,animations=[],sources={f:sha(S/f) for f in ['FIRST_RESULTS75.json','TRANSPORT75.json','GAMES75.json','INDEPENDENT75.json','AUDIT75.json','plot75.py']}))
print('PLOTS',len(figs),flush=True)
