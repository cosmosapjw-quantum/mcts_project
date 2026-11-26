from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
R=read(S/'SCALE_RESULTS72.json');I=read(S/'INDEPENDENT72.json');M=read(S/'MEMORY72.json')
L=np.array([13,15,17,19,21]);K=[4,8,16,32];rows=R['groups'];labels=['fixed 8 stones','density near .06'];figs=[];anims=[]
def save(fig,name,ids,scope):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=150);fig.savefig(str(p)+'.pdf');plt.close(fig)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 assert b.read_bytes()[:4]==b'%PDF'
 figs.append(dict(figure=name,ids=ids,scope=scope,png_sha=sha(a),pdf_sha=sha(b)))
def vals(L,arm,k,field):
 return np.array([q[field] for q in sorted(rows,key=lambda q:q['core']) if q['L']==L and q['arm']==arm and q['k']==k])
def means(field,arm,k):return np.array([vals(l,arm,k,field).mean(0) for l in L])
def ci(y):return 2.364624251*y.std(0,ddof=1)/np.sqrt(8)
fig,ax=plt.subplots(1,2,figsize=(11,4))
for a,arm in zip(ax,(0,1)):
 for k in K:
  y=np.stack([vals(l,arm,k,'mass')[:,0] for l in L]);a.errorbar(L,y.mean(1),yerr=2.365*y.std(1,ddof=1)/np.sqrt(8),marker='o',label='prefix '+str(k))
 a.set(title=labels[arm],xlabel='board side L',ylabel='central 3×3 visit mass');a.legend()
save(fig,'common_core_mass',['L13','Q11','L14'],'Fixed common spatial observable, root-block intervals, finite prefixes.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for a,arm in zip(ax,(0,1)):
 for k in K:
  y=np.stack([vals(l,arm,k,'coarse13')[:,-1] for l in L]);a.errorbar(L,y.mean(1),yerr=2.365*y.std(1,ddof=1)/np.sqrt(8),marker='o',label='prefix '+str(k))
 a.set(title=labels[arm],xlabel='board side L',ylabel='outside central 13×13 probability');a.legend()
save(fig,'outside_mass_flow',['L13','Q09','Q11'],'Exact crop-plus-outside pushforward, finite geometry flow not RG beta function.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for a,arm in zip(ax,(0,1)):
 for t in (0,7,15,31):
  x=[q for q in I['cross_conditional'] if q['arm']==arm and q['t']==t]
  a.plot(L[:-1],[np.mean([v['conditional_mean_TV'] for v in x if v['L']==l]) for l in L[:-1]],'o-',label='conditional t='+str(t))
  if t==31:a.plot(L[:-1],[np.mean([.5*(v['half_split_noise_L']+v['half_split_noise_next']) for v in x if v['L']==l]) for l in L[:-1]],'k--',label='t31 within-size split noise')
 a.set(title=labels[arm],xlabel='adjacent size starting L',ylabel='TV in common 13+outside space');a.legend(fontsize=8)
save(fig,'cross_size_TV',['L13','Q13'],'Conditional law across distinct native sizes; split-noise control, no equivalence claim.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
w=R['witnesses'];ax[0].boxplot([[v['tv'] for v in w if v['L']==l] for l in L],tick_labels=L);ax[0].set(xlabel='L',ylabel='TV at identical coarse counts',title='112 observed collisions')
top=w[0];x=np.arange(4);ax[1].bar(x-.18,top['p1'],.36,label='history 1');ax[1].bar(x+.18,top['p2'],.36,label='history 2');ax[1].set_xticks(x,['3×3','7-ring','13-ring','outside']);ax[1].set(ylabel='next group probability',title=f"L={top['L']}, t={top['t']}, TV={top['tv']:.3f}");ax[1].legend()
save(fig,'nonlumpability_witness',['L13','Q13','L16'],'Same root/clock/coarse counts, exact conditional laws differ.')
fig,ax=plt.subplots(2,2,figsize=(12,7))
for i,arm in enumerate((0,1)):
 for j,field in enumerate(('cov_trace','low_mode')):
  a=ax[j,i]
  for k in K:
   y=np.stack([vals(l,arm,k,field) for l in L]);a.errorbar(L,y.mean(1),yerr=2.365*y.std(1,ddof=1)/np.sqrt(8),marker='o',label='prefix '+str(k))
  a.set(title=labels[arm],xlabel='L',ylabel=field);a.legend(fontsize=8)
save(fig,'finite_size_covariance',['L14','L27','Q16','Q18'],'Open-grid low-mode covariance and full trace, finite budget and size separated.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for a,arm in zip(ax,(0,1)):
 for field in ('cov_trace','bracket','drift_trace','martingale_trace','cross_trace'):
  a.plot(K,[means(field,arm,k).mean() for k in K],'o-',label=field)
 a.set(title=labels[arm],xlabel='prefix backup k',ylabel='covariance trace');a.set_xscale('log',base=2);a.legend(fontsize=8)
save(fig,'martingale_drift',['Q16','L14','L16'],'Predictable drift/martingale/cross decomposition, empirical finite sample.')
fig,ax=plt.subplots(2,2,figsize=(11,7))
for a,q in zip(ax.flat,R['fits']):
 x=np.arange(3);a.bar(x,[v['rmse'] for v in q['models']]);a.set_xticks(x,['constant','offset+1/L²','power']);a.set(title=labels[q['arm']]+' | '+q['observable'],ylabel='leave-one-size-out RMSE')
save(fig,'scaling_alternatives',['L28','Q14','Q18'],'Finite-size functional alternatives; not a critical exponent estimate.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for a,arm in zip(ax,(0,1)):
 for k in K:
  a.plot(L,k*means('cov_trace',arm,k),'o-',label='k='+str(k))
 a.set(title=labels[arm],xlabel='L',ylabel='k × covariance trace');a.legend()
save(fig,'budget_collapse_diagnostic',['Q15','Q30','L14'],'Noncritical finite-budget noise rescaling, no critical collapse claimed.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for a,k in zip(ax,(16,32)):
 for beta in (1.,10/3,8.):
  v=[q for q in R['prior_R58_spatial'] if q['B']==k and abs(q['beta']-beta)<1e-6]
  a.plot(L,[k*np.mean([q['trace'] for q in v if q['L']==l]) for l in L],'o-',label=f'beta={beta:.3g}')
 a.set(title=f'R58 independent B={k}, 2 cores',xlabel='L',ylabel='B × covariance trace');a.legend()
save(fig,'historical_beta_scan',['Q10','L27','Q16','Q18'],'Historical separate cohort, finite beta/crossover diagnostics, not phase boundary.')
fig,ax=plt.subplots(2,3,figsize=(13,7));ax=ax.flat
for a,f in zip(ax,M['results']):
 for v,lab in enumerate(['uniform roots','initial value','value + geometry']):
  z=[q for q in f['results'] if q['variant']==v];a.plot([q['H'] for q in z],[q['W_rmse'] for q in z],'o-',label=lab)
 a.axhline(f['baseline_R71_lag_RMSE'],color='k',ls='--',label='R71 state+lag');a.set(title=f['label'],xlabel='block H (waves)',ylabel='W RMSE');a.set_xscale('log',base=2)
ax[0].legend(fontsize=7);ax[-1].axis('off')
save(fig,'block_memory_W',['L16','Q21'],'Existing R71 split reused, fixed block alternatives; no new heldout claim.')
fig,ax=plt.subplots(2,3,figsize=(13,7));ax=ax.flat
for a,f in zip(ax,M['results']):
 for v,lab in enumerate(['uniform roots','initial value','value + geometry']):
  z=[q for q in f['results'] if q['variant']==v];a.plot([q['H'] for q in z],[q['mark_W_rmse'] for q in z],'o-',label=lab);a.plot([q['H'] for q in z],[q['count_W_rmse'] for q in z],':',color=a.lines[-1].get_color())
 a.set(title=f['label'],xlabel='block H',ylabel='mark (solid), count (dotted) RMSE');a.set_xscale('log',base=2)
ax[0].legend(fontsize=7);ax[-1].axis('off')
save(fig,'block_memory_mark_count',['L16','Q21'],'Source-type decomposition, no claim of exact memory kernel.')
# Actual finite scale-flow animation: same 21 board and root ensemble, exact crop.
from io import BytesIO
def gif(frames,name,ids,scope):
 p=S/'plots'/name;frames[0].save(p,save_all=True,append_images=frames[1:],duration=750,loop=0)
 with Image.open(p) as im:
  assert im.n_frames==len(frames);n=im.n_frames
 anims.append(dict(file=name,ids=ids,scope=scope,sha256=sha(p),frames=n))
def frame(fig):
 b=BytesIO();fig.tight_layout();fig.savefig(b,format='png',dpi=90);plt.close(fig);b.seek(0)
 return Image.open(b).convert('RGB').copy()
A=np.load(S/'SCALE_L21.npz');ids=(A['meta'][:,0]==0)&(A['meta'][:,1]==0);cnt=np.bincount(A['actions'][ids].ravel(),minlength=441)/(4*32);frames=[]
for m in [21,19,17,15,13,7,3]:
 h=(21-m)//2;z=cnt.reshape(21,21)[h:h+m,h:h+m];fig,a=plt.subplots(figsize=(5,4));a.imshow(z,vmin=0,vmax=cnt.max(),cmap='viridis');a.set_title(f'Exact crop {m}×{m} | outside mass={1-z.sum():.3f}');a.set_axis_off();frames.append(frame(fig))
gif(frames,'finite_spatial_pushforward.gif',['Q12'],'One actual finite ensemble under exact geometric projection, not autonomous RG trajectory.')
frames=[]
for k in K:
 fig,a=plt.subplots(figsize=(6,4))
 for arm in (0,1):a.plot(L,k*means('cov_trace',arm,k),'o-',label=labels[arm])
 a.set(xlabel='L',ylabel='k × covariance trace',title=f'Finite prefix k={k} (not critical scaling)');a.set_ylim(0,2);a.legend();frames.append(frame(fig))
gif(frames,'finite_budget_scaling.gif',['Q17'],'Finite-budget evolution animation, no critical phase transition established.')
assert len(figs)==11 and len({q['png_sha'] for q in figs})==11
dump(S/'PLOT_MANIFEST72.json',dict(figures=figs,animations=anims,sources={f:sha(S/f) for f in ['SCALE_RESULTS72.json','INDEPENDENT72.json','MEMORY72.json','plot72.py']}))
print('PLOTS',len(figs),'ANIMATIONS',len(anims),flush=True)
