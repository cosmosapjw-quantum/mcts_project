from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
r=read(S/'PHYSICS_RESULTS.json');d=read(S/'DIAGNOSTICS61.json')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'figure.dpi':120})
manifest=[];colors={0:'#436f9c',1:'#96985c',2:'#ae7950',3:'#24856f'}
def save(fig,name,ids):
 fig.tight_layout();p=S/'plots'/name
 fig.savefig(str(p)+'.png',dpi=175);fig.savefig(str(p)+'.pdf');plt.close(fig)
 with Image.open(str(p)+'.png') as im:im.verify()
 assert Path(str(p)+'.pdf').read_bytes().startswith(b'%PDF')
 manifest.append(dict(figure=name,ids=ids,png_sha=sha(Path(str(p)+'.png')),pdf_sha=sha(Path(str(p)+'.pdf'))))
# Native matched trajectories. R60 deterministic, no intervention, opening held fixed.
fig,axs=plt.subplots(2,3,figsize=(12.8,7.2))
for ax,key,label in zip(axs.ravel(),['Hvisit','Hprior','E_Q','var','within','between'],['Visit entropy (nat)','Prior entropy (nat)','Negative mean root mark','Mark variance','Within-action variance','Between-action variance']):
 for g in range(4):
  z=[x for x in r['matched_prefix'] if x['budget']==128 and x['generation']==g]
  ax.plot([x['k'] for x in z],[x['mean'][key] for x in z],'o-',c=colors[g],label=f'g{g}')
 ax.set(xscale='log',xlabel='Completed backups',ylabel=label);ax.set_xticks([4,8,16,32,64,128],labels=[4,8,16,32,64,128])
axs[0,0].legend(ncol=2)
fig.suptitle('R60 deterministic MCTS: 48 matched initial roots per checkpoint; B=128',fontsize=13)
save(fig,'G01_G03_L04_trained_native_prefix',['G01','G03','L04'])
# New stochastic selector source; no direct comparison with deterministic across root populations.
fig,axs=plt.subplots(2,2,figsize=(10,7))
for ax,(g,w) in zip(axs.ravel(),[(0,1),(0,4),(3,1),(3,4)]):
 z=[x for x in r['responses'] if x['generation']==g and x['width']==w];xx=np.array([x['k'] for x in z])
 for key,ls in [('fd','o-'),('score','s--'),('reweighted_fd','^:')]:
  v=[x['observables']['C'][key] for x in z];y=np.array([x['mean'] for x in v]);ci=np.array([x['ci95'] for x in v])
  ax.errorbar(xx,y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt=ls,capsize=2,label=key.replace('_',' '))
 ax.set(xscale='log',xlabel='Completed backups',ylabel='Source count response',title=f'g{g}, width {w}');ax.set_xticks(xx,labels=xx);ax.axhline(0,c='gray',ls=':')
axs[0,0].legend(fontsize=8)
fig.suptitle('32 independent geometry blocks × 8 seeds; h = ±0.35; finite contrast vs derivative',fontsize=13)
save(fig,'G07_Q32_source_response',['G07','Q32','L16'])
fig,axs=plt.subplots(1,3,figsize=(12.4,4.2))
z=[x for x in r['responses'] if x['k']==128];xx=np.arange(4);lab=[f"g{x['generation']}/w{x['width']}" for x in z]
for j,key in enumerate(['count_variance','expected_fisher','noise_variance']):
 v=[x[key] for x in z];y=np.array([a['mean'] for a in v]);ci=np.array([a['ci95'] for a in v]);axs[0].errorbar(xx+(j-1)*.14,y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt=['o','s','^'][j],capsize=3,label=key.replace('_',' '))
axs[0].set(xticks=xx,xticklabels=lab,ylabel='Count units squared',title='Variance is not Fisher');axs[0].legend(fontsize=7)
for j,key in enumerate(['noise_variance','adaptive_drift_variance','cross_term']):
 axs[1].bar(xx+(j-1)*.25,[a[key]['mean'] for a in z],width=.24,label=key.replace('_',' '))
axs[1].axhline(0,c='gray');axs[1].set(xticks=xx,xticklabels=lab,title='Exact sample variance split',ylabel='Contribution');axs[1].legend(fontsize=7)
v=[a['score2_minus_fisher'] for a in z];y=np.array([a['mean'] for a in v]);ci=np.array([a['ci95'] for a in v])
axs[2].errorbar(xx,y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=3);axs[2].axhline(0,c='gray',ls=':');axs[2].set(xticks=xx,xticklabels=lab,title='Score-square identity residual',ylabel='E[G²] − predicted Fisher')
save(fig,'G07_Q32_adaptive_variance_fisher',['G07','Q32'])
fig,axs=plt.subplots(2,2,figsize=(10.5,7.2))
for (g,w),ax in zip([(0,1),(0,4),(3,1),(3,4)],axs.ravel()):
 q=[x for x in r['coarse_cgf'] if x['generation']==g and x['width']==w];c=[x for x in d['coarse_cgf_mixture_vs_conditional'] if x['generation']==g and x['width']==w];l=[x['lambda_'] for x in q]
 ax.plot(l,[x['W'] for x in q],label='empirical mixture');ax.plot(l,[x['gaussian'] for x in q],'--',label='Gaussian truncation');ax.plot(l,[x['mean_conditional_empirical_W'] for x in c],':',label='mean per-root empirical CGF')
 ax.set(xlabel='Count tilt λ',ylabel='Log empirical MGF',title=f'g{g}, width {w}')
axs[0,0].legend(fontsize=8)
fig.suptitle('Finite empirical count CGF; 8 samples per root; not native continuum 1PI',fontsize=13)
save(fig,'L16_empirical_count_action',['L16'])
fig,axs=plt.subplots(1,3,figsize=(12,4.2))
for g,w,ls in [(0,1,'-'),(0,4,'--'),(3,1,'-'),(3,4,'--')]:
 q=[x for x in r['coarse_cgf'] if x['generation']==g and x['width']==w]
 axs[0].plot([x['tilted_count'] for x in q],[x['legendre'] for x in q],ls,c=colors[g],label=f'g{g}/w{w}')
 c=[x for x in d['coarse_cgf_mixture_vs_conditional'] if x['generation']==g and x['width']==w]
 axs[1].plot([x['lambda_'] for x in c],[x['mixture_ESS'] for x in c],ls,c=colors[g],label=f'g{g}/w{w}')
 z=[x for x in r['responses'] if x['generation']==g and x['width']==w]
 axs[2].plot([x['k'] for x in z],[x['path_action_per_backup']['mean'] for x in z],ls,marker='o',c=colors[g],label=f'g{g}/w{w}')
axs[0].set(xlabel='Tilted empirical mean count',ylabel='λ C − W(λ)',title='Restricted empirical Legendre')
axs[1].set(xlabel='Count tilt λ',ylabel='Effective sample size',title='Finite-sample tail limitation')
axs[2].set(xscale='log',xlabel='Completed backups',ylabel='−log P / B',title='Actual path action');axs[2].set_xticks([4,8,16,32,64,128],labels=[4,8,16,32,64,128]);axs[0].legend(fontsize=8)
save(fig,'L16_action_legendre_ess',['L16'])
fig,axs=plt.subplots(1,3,figsize=(12,4))
for ax,key,label in zip(axs[:2],['policy_TV','value_range'],['Policy TV vs base orientation','Value range over D4']):
 for j,g in enumerate((0,3)):
  v=[x for x in r['symmetry_core'] if x['generation']==g]
  ax.scatter(np.full(len(v),j-.12),[x[key+'_single'] for x in v],alpha=.6,s=14,c=colors[g]);ax.scatter(np.full(len(v),j+.12),[x[key+'_d4'] for x in v],alpha=.6,s=14,c='#bb8d3d')
 ax.set(xticks=[0,1],xticklabels=['g0','g3'],ylabel=label,title='Single (left) / D4 (right)')
for j,g in enumerate((0,3)):
 q=next(x for x in r['symmetry'] if x['generation']==g)
 for h in (0,1):
  a=q[f'search_TV_d4_{h}']['by_core'];axs[2].scatter(np.full(len(a),j+(h-.5)*.24),a,s=22,alpha=.7,c=colors[g] if h==0 else '#bb8d3d')
axs[2].set(xticks=[0,1],xticklabels=['g0','g3'],ylabel='Visit TV (2 seeds/orientation)',title='Native search residual')
fig.suptitle('D4 averaged evaluator: exact mapped probabilities; native ordering may break CRN coupling',fontsize=12)
save(fig,'L16_D4_readout_and_search',['L16'])
fig,axs=plt.subplots(2,3,figsize=(12.7,7.4))
for ax,key,label in zip(axs.ravel(),['mean_depth','visited_prefix_edges','Hvisit','KL','E_Q','var'],['Mean path depth','Visited prefix edges','Visit entropy (nat)','KL(visit || prior) (nat)','Negative mean root mark','Mark variance']):
 for g,w,ls in [(0,1,'-'),(0,4,'--'),(3,1,'-'),(3,4,'--')]:
  z=[x for x in r['responses'] if x['generation']==g and x['width']==w]
  ax.plot([x['k'] for x in z],[x['native_means'][key]['mean'] for x in z],ls,c=colors[g],marker='o',label=f'g{g}/w{w}')
 ax.set(xscale='log',xlabel='Completed backups',ylabel=label);ax.set_xticks([4,8,16,32,64,128],labels=[4,8,16,32,64,128])
axs[0,0].legend(fontsize=8,ncol=2)
fig.suptitle('Controlled stochastic MCTS: matched root geometry, h=0; finite budget flow, not RG fixed points',fontsize=13)
save(fig,'L13_L29_L38_trained_budget_flow',['L13','L29','L38','Q09','Q11'])
fig,axs=plt.subplots(1,3,figsize=(12.8,4.5))
flow=[]
for g,w,ls in [(0,1,'-'),(0,4,'--'),(3,1,'-'),(3,4,'--')]:
 z=[x for x in r['responses'] if x['generation']==g and x['width']==w]
 k=np.array([x['k'] for x in z]);H=np.array([x['native_means']['Hvisit']['mean'] for x in z]);E=np.array([x['native_means']['E_Q']['mean'] for x in z]);v=np.array([x['native_means']['var']['mean'] for x in z])
 x=np.sqrt(k[1:]*k[:-1]);dh=np.diff(H)/np.diff(np.log(k));de=np.diff(E)/np.diff(np.log(k))
 axs[0].plot(x,dh,ls,c=colors[g],marker='o',label=f'g{g}/w{w}');axs[1].plot(x,de,ls,c=colors[g],marker='o')
 axs[2].plot(H,v,ls,c=colors[g],marker='o',label=f'g{g}/w{w}')
 for j in [0,len(H)-1]:
  if w==1:axs[2].annotate(str(k[j]),(H[j],v[j]),xytext=(-10,8 if g==0 else -12),textcoords='offset points',fontsize=8,c=colors[g])
 for j in range(len(x)):flow.append(dict(generation=g,width=w,k_low=int(k[j]),k_high=int(k[j+1]),dH_dlogB=float(dh[j]),dE_dlogB=float(de[j])))
for ax,key in zip(axs[:2],['ΔH / Δlog B','ΔE_Q / Δlog B']):ax.set(xscale='log',xlabel='Geometric mean budget',ylabel=key);ax.axhline(0,c='gray',ls=':')
axs[0].legend(fontsize=8);axs[2].set(xlabel='Visit entropy (nat)',ylabel='Mark variance',title='Finite-budget trajectories');axs[2].legend(fontsize=8)
fig.suptitle('Measured dyadic flow increments and trajectories; no coupling-space vector field',fontsize=12)
dump(S/'BUDGET_FLOW61.json',dict(scope='actual finite log-budget increments, not autonomous RG beta functions',rows=flow))
save(fig,'Q09_Q11_L35_L36_finite_flow',['Q09','Q11','L35','L36'])
dump(S/'PLOT_MANIFEST61.json',dict(figures=manifest,sources={p:sha(S/p) for p in ['PHYSICS_RESULTS.json','DIAGNOSTICS61.json','CONTROL61.npz','R60_PREFIX61.npz']}))
imgs=[]
for x in manifest:
 with Image.open(S/'plots'/(x['figure']+'.png')) as im:
  i=im.convert('RGB');i.thumbnail((650,430));imgs.append(i.copy())
sheet=Image.new('RGB',(1300,430*((len(imgs)+1)//2)),'white')
for j,im in enumerate(imgs):sheet.paste(im,(650*(j%2),430*(j//2)))
sheet.save(S/'logs/plot_qa.jpg',quality=85)
print('PLOT_PAIRS',len(manifest))
