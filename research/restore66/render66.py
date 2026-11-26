from base61 import *
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from PIL import Image
r=read(S/'PHYSICS_RESULTS.json');tr=read(S/'THERMAL66.json');er=read(S/'ERASURE66.json');fb=read(S/'FEEDBACK66.json')
BETA=10/3
colors=['#326c95','#da8d45','#25866e','#994c69'];conds=[(0,.25),(0,1.),(3,.25),(3,1.)];lab=[f'g{g}/α={a:g}' for g,a in conds]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.dpi':120})
manifest=[];animations=[]
def save(fig,name,ids):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=160);fig.savefig(str(p)+'.pdf');plt.close(fig)
 with Image.open(str(p)+'.png') as im:im.verify()
 assert Path(str(p)+'.pdf').read_bytes().startswith(b'%PDF')
 manifest.append(dict(figure=name,ids=ids,png_sha=sha(Path(str(p)+'.png')),pdf_sha=sha(Path(str(p)+'.pdf'))))
def ci(ax,xx,ss,**kw):
 y=np.array([v['mean'] for v in ss]);c=np.array([v['ci95'] for v in ss]);ax.errorbar(xx,y,yerr=np.maximum(0,np.vstack([y-c[:,0],c[:,1]-y])),capsize=3,**kw)
def thermal(g,a):return next(x for x in r['thermal'] if (x['generation'],x['alpha'])==(g,a))['metrics']
def series(key,g,steps=None):
 return [x for x in r[key] if x['generation']==g]
# Native same-order intervention vs counter: actual native likelihood, not time reversal.
fig,axs=plt.subplots(1,3,figsize=(12.4,4.1))
for i,(g,arm) in enumerate([(0,'forward'),(0,'counter'),(3,'forward'),(3,'counter')]):
 q=[x for x in r['native_curves'] if (x['generation'],x['arm'])==(g,arm)];k=[x['budget'] for x in q];tag=f'g{g}/{arm[0].upper()}'
 for ax,key in zip(axs[:2],('log_ratio','ift')):
  y=[x[key]['mean'] for x in q];c=np.array([x[key]['ci95'] for x in q]);ax.plot(k,y,'o-',ms=3,c=colors[i],label=tag);ax.fill_between(k,c[:,0],c[:,1],color=colors[i],alpha=.08)
 axs[2].plot(k,[x['ess_fraction'] for x in q],'o-',ms=3,c=colors[i],label=tag)
axs[1].axhline(1,c='gray',ls=':');axs[0].axhline(0,c='gray',ls=':')
for a,y in zip(axs,('Mean own/other log ratio','Mean exp(−log ratio)','ESS / paths')):a.set(xlabel='Completed native backups',ylabel=y,xscale='log',xticks=[4,8,16,32,64,128],xticklabels=[4,8,16,32,64,128])
axs[0].legend(fontsize=8);fig.suptitle('Actual native schedule contrast (same order; no physical time reversal)')
save(fig,'Q05_native_schedule_path_ratio',['Q05'])
fig,axs=plt.subplots(1,3,figsize=(11.8,4.2));q=r['native'];xx=np.arange(4);labs=[f"g{x['generation']}/{x['arm'][0].upper()}" for x in q]
for j,key in enumerate(('native_entropy','native_surprisal')):
 ci(axs[0],xx+(j-.5)*.14,[x[key] for x in q],fmt=['o','s'][j],label=['conditional H','observed surprisal'][j])
axs[0].set(ylabel='Nats / draw',title='Native conditional entropy');axs[0].legend(fontsize=8)
ci(axs[1],xx,[x['entropy_residual'] for x in q],fmt='o');axs[1].axhline(0,c='gray',ls=':');axs[1].set(ylabel='Surprisal − H',title='Native entropy diagnostic')
for j,key in enumerate(('binary_record_entropy','pair_record_entropy')):
 ci(axs[2],xx+(j-.5)*.14,[x[key] for x in q],fmt=['o','s'][j],label=['a vs other','a vs b, conditional'][j])
axs[2].set(ylabel='Nats / record',title='Binary record entropy');axs[2].legend(fontsize=8)
for a in axs:a.set(xticks=xx,xticklabels=labs)
save(fig,'L08_L32_native_record_entropy',['L08','L32','Q38'])
# True auxiliary F/R samples, binned Crooks relation; compare reverse-weighted counts.
fig,axs=plt.subplots(2,2,figsize=(10.8,7.5))
for i,x in enumerate(r['workhist']):
 a=axs.ravel()[i];b=np.array(x['bins']);mid=(b[:-1]+b[1:])/2;n=x['n'];width=np.diff(b)
 for key,name,ls in [('forward','forward','-'),('reverse','reversed reverse','--'),('reverse_weighted','reverse × exp(Σ)','-.')]:
  y=np.asarray(x[key])/n/width;y[y<=0]=np.nan;a.plot(mid,y,ls,lw=1.3,label=name)
 a.set(yscale='log',xlabel='Σ = β(W−ΔF)',ylabel='Density / bin',title=f"g{x['generation']}, α={x['alpha']:g}");a.legend(fontsize=7)
fig.suptitle('Independent finite thermal path ensembles; binned Crooks identity')
save(fig,'Q19_Q20_Crooks_path_ensembles',['Q19','Q20','Q21'])
fig,axs=plt.subplots(1,3,figsize=(12.5,4.1));xx=np.arange(4)
for j,key in enumerate(('df_F','df_R','df_BAR')):
 ss=[thermal(g,a)[key+'_MAE'] for g,a in conds]
 ci(axs[0],xx+(j-1)*.14,ss,fmt=['o','s','^'][j],label=['F Jarzynski','R Jarzynski','BAR'][j])
axs[0].set(ylabel='Mean absolute ΔF error / E*',title='Free energy estimation');axs[0].legend(fontsize=7)
for j,key in enumerate(('ift_F','ift_R')):
 ci(axs[1],xx+(j-.5)*.15,[thermal(g,a)[key] for g,a in conds],fmt=['o','s'][j],label=['F','R'][j])
axs[1].axhline(1,c='gray',ls=':');axs[1].set(ylabel='Mean normalized weight',title='Finite-sample IFT');axs[1].legend()
for j,key in enumerate(('ESS_F_fraction','ESS_R_fraction')):
 ci(axs[2],xx+(j-.5)*.15,[thermal(g,a)[key] for g,a in conds],fmt=['o','s'][j],label=['F','R'][j])
axs[2].set(ylabel='Mean per-root ESS fraction',title='Rare-event overlap');axs[2].legend()
for a in axs:a.set(xticks=xx,xticklabels=lab);a.tick_params(axis='x',rotation=12)
save(fig,'Q19_Q23_free_energy_estimator_audit',['Q19','Q23'])
fig,axs=plt.subplots(1,2,figsize=(10.7,4.3))
for i,(g,a) in enumerate(conds):
 q=[x for x in r['convergence'] if (x['generation'],x['alpha'])==(g,a)]
 for ax,key in zip(axs,('F','R')):
  xx=np.array([x['n'] for x in q]);y=np.array([x[key]['mean'] for x in q]);c=np.array([x[key]['ci95'] for x in q]);ax.plot(xx,y,'o-',c=colors[i],ms=3,label=lab[i]);ax.fill_between(xx,c[:,0],c[:,1],color=colors[i],alpha=.075)
for a,k in zip(axs,('Forward','Reverse')):a.axhline(1,c='gray',ls=':');a.set(xscale='log',xlabel='Paths per root',ylabel='Normalized exponential average',title=k+' convergence');a.legend(fontsize=8)
save(fig,'Q22_Jarzynski_convergence',['Q19','Q22'])
fig,axs=plt.subplots(2,2,figsize=(10.8,7.3))
for i,(g,a) in enumerate(conds):
 x=next(x for x in r['evolution'] if (x['generation'],x['alpha'])==(g,a));ax=axs.ravel()[i];t=np.arange(9)
 for key in ('W','Q','deltaF'):
  ax.plot(t,x[key]['mean'],'o-',ms=3,label=key)
 ax.plot(t,np.array(x['W']['mean'])+x['Q']['mean'],'k--',lw=1,label='ΔE = W+Q')
 ax.set(xlabel='Protocol stage',ylabel='Model energy / E*',title=lab[i]);ax.legend(fontsize=8)
fig.suptitle('Exact ensemble work, heat and free energy; frozen native-score energy schedules')
save(fig,'Q01_Q03_Q04_thermal_evolution',['Q01','Q03','Q04','L06','L33'])
fig,axs=plt.subplots(1,2,figsize=(10.6,4.3))
for j,key in enumerate(('mean_dissipation_exact','mean_actual_EP_exact','final_KL_exact')):
 ci(axs[0],np.arange(4)+(j-1)*.15,[thermal(g,a)[key] for g,a in conds],fmt=['o','s','^'][j],label=['β(W−ΔF)','Actual mean EP','Final KL'][j])
axs[0].set(xticks=np.arange(4),xticklabels=lab,ylabel='Nats',title='Endpoint reference vs actual marginal');axs[0].legend(fontsize=8)
for i,(g,a) in enumerate(conds):
 x=next(x for x in r['evolution'] if (x['generation'],x['alpha'])==(g,a));axs[1].plot(range(9),x['S']['mean'],'o-',ms=3,c=colors[i],label=lab[i])
axs[1].set(xlabel='Protocol stage',ylabel='Shannon entropy (nats)',title='Actual propagated model ensemble');axs[1].legend(fontsize=8)
save(fig,'Q05_Q38_entropy_production',['Q05','Q38','Q01'])
# Actual individual thermal paths, deterministic selected protocol and prefix.
ds=read(S/'THERMAL_SCHEDULES.json')
fig,axs=plt.subplots(1,3,figsize=(12.5,4.2))
for i,(g,a) in enumerate([(0,.25),(3,.25)]):
 p=next(x for x in tr['results'] if x['generation']==g and x['alpha']==a and x['core']==0)
 d=next(x for x in ds if x['generation']==g and x['core']==0);E=np.array(d['energy']);j=np.arange(1,9)[None,:]
 with np.load(S/'data/thermal'/p['file']) as z:
  seq=z['forward'][:20];W=np.c_[np.zeros(len(seq)),np.cumsum(E[j,seq[:,:-1]]-E[j-1,seq[:,:-1]],1)]
  for line in W:axs[i].plot(range(9),line,alpha=.45,lw=.8)
  axs[i].set(xlabel='Protocol stage',ylabel='Cumulative W / E*',title=f'g{g}, root 0: 20 actual paths')
  axs[2].scatter(z['W'][::64],z['logPF'][::64],s=7,alpha=.35,label=f'g{g}')
axs[2].set(xlabel='Work / E*',ylabel='Forward log path probability',title='Actual normalized path weights');axs[2].legend(fontsize=8)
save(fig,'Q20_Q21_actual_work_paths',['Q20','Q21'])
fig,axs=plt.subplots(1,3,figsize=(12.5,4.3))
for i,g in enumerate((0,3)):
 q=series('erasure',g);k=[x['steps'] for x in q];m=[x['metrics'] for x in q]
 axs[0].plot(k,[x['beta_work_exact']['mean'] for x in m],'o-',c=colors[i*2],label=f'g{g} exact')
 axs[0].plot(k,[BETA*x['mean_work_sample']['mean'] for x in m],'x--',c=colors[i*2],label=f'g{g} samples')
 axs[0].axhline(m[0]['bound']['mean'],ls=':',c=colors[i*2],label=f'g{g} H bound')
 axs[1].plot(k,[x['mean_EP_exact']['mean'] for x in m],'o-',c=colors[i*2],label=f'g{g}')
 ci(axs[2],np.array(k)*(1 if i==0 else 1.06),[x['final_error_rate'] for x in m],fmt=['o','s'][i],c=colors[i*2],label=f'g{g}')
axs[2].axhline(.05,c='gray',ls=':');axs[0].set(ylabel='β mean work',title='Finite-error erasure');axs[1].set(ylabel='Mean entropy production (nats)',title='Excess above reversible bound',yscale='log');axs[2].set(ylabel='Final error probability',title='Independent reset result')
for a in axs:a.set(xlabel='Driving stages',xscale='log',xticks=[1,4,16,64],xticklabels=[1,4,16,64]);a.legend(fontsize=7)
fig.suptitle('Binary record memory model: native pilot pair distribution, prescribed reset')
save(fig,'L32_L08_record_erasure_work',['L32','L08','Q38'])
fig,axs=plt.subplots(1,3,figsize=(12.5,4.2))
for i,g in enumerate((0,3)):
 q=series('feedback',g);k=[x['steps'] for x in q];m=[x['metrics'] for x in q]
 axs[0].plot(k,[-BETA*x['mean_work_exact']['mean'] for x in m],'o-',c=colors[i*2],label=f'g{g} extracted')
 axs[0].axhline(m[0]['information']['mean'],c=colors[i*2],ls=':',label=f'g{g} I bound')
 axs[1].plot(k,[x['efficiency_exact']['mean'] for x in m],'o-',c=colors[i*2],label=f'g{g} exact')
 ci(axs[2],np.array(k)*(1 if i==0 else 1.06),[x['ift_F'] for x in m],fmt=['o','s'][i],c=colors[i*2],label=f'g{g}')
axs[1].axhline(1,c='gray',ls=':');axs[2].axhline(1,c='gray',ls=':');axs[0].set(ylabel='−β mean work (nats)',title='Information-to-work');axs[1].set(ylabel='Mean per-root efficiency',title='Record-conditioned extraction');axs[2].set(ylabel='Mean exp(−βW−i)',title='Feedback IFT')
for a in axs:a.set(xlabel='Annealing stages',xscale='log',xticks=[1,4,16,64],xticklabels=[1,4,16,64]);a.legend(fontsize=7)
fig.suptitle('Defined feedback model (detector accuracy 0.8); measurement/reset cost excluded')
save(fig,'L08_L32_information_feedback',['L08','L32','Q23'])
fig,axs=plt.subplots(1,2,figsize=(10.8,4.2))
for i,g in enumerate((0,3)):
 q=[x for x in r['equilibrium'] if x['generation']==g];x=[v['beta'] for v in q]
 for a,key in zip(axs,('entropy','heat_capacity_over_kB')):
  y=[v[key]['mean'] for v in q];c=np.array([v[key]['ci95'] for v in q]);a.plot(x,y,'o-',c=colors[i*2],label=f'g{g}');a.fill_between(x,c[:,0],c[:,1],color=colors[i*2],alpha=.12)
axs[0].set(xlabel='βE*',ylabel='Equilibrium entropy (nats)',title='Fixed endpoint energy spectrum');axs[1].set(xlabel='βE*',ylabel='Heat capacity / kB',title='Model equilibrium fluctuation')
for a in axs:a.legend()
save(fig,'Q38_model_equilibrium_entropy_capacity',['Q38'])
# Animated actual finite-sample convergence and exact model stage trajectory.
def animate(kind):
 fig,axs=plt.subplots(1,2,figsize=(10.6,4.4),dpi=95);lines=[]
 if kind=='Q22':
  frames=9
  for i,(g,a) in enumerate(conds):
   q=[x for x in r['convergence'] if (x['generation'],x['alpha'])==(g,a)];t=np.array([x['n'] for x in q])
   for ax,key in zip(axs,('F','R')):
    l,=ax.plot([],[],'o-',ms=3,c=colors[i],label=lab[i]);lines.append((l,t,[x[key]['mean'] for x in q]))
  for ax,key in zip(axs,('Forward','Reverse')):
   ys=[v for l,t,y in lines for v in y];ax.set(xscale='log',xlim=(32,8192),ylim=(min(ys+[1])*.96,max(ys+[1])*1.04),xlabel='Paths per root',ylabel='Normalized exponential mean',title=key);ax.axhline(1,c='gray',ls=':');ax.legend(fontsize=8)
  name='Q22_jarzynski_actual_convergence.gif'
 else:
  frames=9;t=np.arange(9)
  for i,(g,a) in enumerate(conds):
   q=next(x for x in r['evolution'] if (x['generation'],x['alpha'])==(g,a))
   for key,ls in [('W','-'),('Q','--')]:
    l,=axs[0].plot([],[],ls,c=colors[i],label=lab[i]+'/'+key);lines.append((l,t,q[key]['mean']))
   l,=axs[1].plot([],[],'-',c=colors[i],label=lab[i]);lines.append((l,t,q['S']['mean']))
  ys=[v for _,_,y in lines[:0] for v in y] # axis-specific bounds from actual series
  a0=[v for q in r['evolution'] for key in ('W','Q') for v in q[key]['mean']];a1=[v for q in r['evolution'] for v in q['S']['mean']]
  axs[0].set(xlim=(0,8),ylim=(min(a0)-.05,max(a0)+.05),xlabel='Protocol stage',ylabel='Model energy / E*',title='Work and heat');axs[1].set(xlim=(0,8),ylim=(min(a1)-.05,max(a1)+.05),xlabel='Protocol stage',ylabel='Nats',title='Ensemble entropy')
  axs[0].legend(fontsize=6,ncol=2);axs[1].legend(fontsize=8);name='Q04_thermal_actual_schedule.gif'
 title=fig.suptitle('');fig.tight_layout()
 def upd(k):
  for l,t,y in lines:l.set_data(t[:k+1],y[:k+1])
  title.set_text('Auxiliary finite thermal model | '+('paths per root = '+str(lines[0][1][k]) if kind=='Q22' else 'protocol stage = '+str(k)))
  return [l for l,_,_ in lines]+[title]
 p=S/'plots'/name;FuncAnimation(fig,upd,frames=frames,interval=240).save(p,writer=PillowWriter(fps=4));plt.close(fig)
 with Image.open(p) as im:
  assert im.n_frames==frames
  for k in range(frames):im.seek(k);im.load()
 animations.append(dict(name=name,ids=[kind],frames=frames,sha256=sha(p),bytes=p.stat().st_size))
animate('Q22');animate('Q04')
dump(S/'PLOT_MANIFEST66.json',dict(figures=manifest,animations=animations,sources={p:sha(S/p) for p in ['PHYSICS_RESULTS.json','THERMAL66.json','ERASURE66.json','FEEDBACK66.json','NATIVE66.npz']},scope='native and auxiliary models distinguished on panels; no measured physical CPU heat'))
imgs=[]
for x in manifest:
 with Image.open(S/'plots'/(x['figure']+'.png')) as im:
  j=im.convert('RGB');j.thumbnail((640,440));imgs.append(j.copy())
sheet=Image.new('RGB',(1280,440*((len(imgs)+1)//2)),'white')
for i,im in enumerate(imgs):sheet.paste(im,(640*(i%2),440*(i//2)))
sheet.save(S/'logs/plot_qa.jpg',quality=85)
print('PLOTS',len(manifest),'ANIMATIONS',len(animations))
