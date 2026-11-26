from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from PIL import Image
r=read(S/'PHYSICS_RESULTS.json');g0=read(S/'GENERATOR65.json');cal=read(S/'CALIBRATION65.json')
with np.load(S/'PHASE65.npz') as d:z={k:d[k] for k in d.files}
m=z['meta'];W=z['waves'];wf=list(z['wave_fields']);arms=list(z['arms']);colors=['#677A93','#A78255','#8271A0','#25866E','#A65564','#676767']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.dpi':120})
manifest=[];animations=[]
def save(fig,name,ids):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=165);fig.savefig(str(p)+'.pdf');plt.close(fig)
 with Image.open(str(p)+'.png') as im:im.verify()
 assert Path(str(p)+'.pdf').read_bytes().startswith(b'%PDF')
 manifest.append(dict(figure=name,ids=ids,png_sha=sha(Path(str(p)+'.png')),pdf_sha=sha(Path(str(p)+'.pdf'))))
def ci(ax,xx,ys,cc,**kw):
 ys=np.asarray(ys);cc=np.asarray(cc);ax.errorbar(xx,ys,yerr=np.maximum(0,np.vstack([ys-cc[:,0],cc[:,1]-ys])),capsize=3,**kw)
def wave(g,arm):
 return W[(m[:,0]==g)&(m[:,3]==arms.index(arm))]
t=np.arange(32)*4
fig,axs=plt.subplots(2,3,figsize=(12.7,7.4))
for i,g in enumerate((0,3)):
 for arm in ('fresh64','reset_plus','persist_plus','persist_minus','persist_dephased'):
  v=wave(g,arm);j=arms.index(arm)
  for a,key in zip(axs[i],('Y','entropy','purity')):
   a.plot(t,v[:,:,wf.index(key)].mean(0),c=colors[j],label=arm)
 for a,key in zip(axs[i],('Mean Y','Mean S_vN (nat)','Mean purity')):a.set(xlabel='Completed backups before wave',ylabel=key,title=f'g{g} | {key}')
axs[0,0].legend(ncol=2,fontsize=7)
fig.suptitle('Persistent selective states versus repeated preparation (actual native histories)')
save(fig,'L09_Q24_Q42_persistent_density',['L09','Q24','Q42','L20'])
fig,axs=plt.subplots(1,3,figsize=(12.6,4.4))
for j,g in enumerate((0,3)):
 q=[x for x in r['local'] if x['generation']==g];xx=np.arange(len(q))+(j-.5)*.15
 for a,key in zip(axs,('abs_memory_delta','measurement_TV','non_iid_wave_fraction')):
  ci(a,xx,[x['metrics'][key]['mean'] for x in q],[x['metrics'][key]['ci95'] for x in q],fmt=['o','s'][j],label=f'g{g}',c=colors[0 if j==0 else 3])
for a,key in zip(axs,('Absolute q_a − reset q_a','Selective trace distance','Non-iid wave fraction')):
 a.set(xticks=np.arange(4),xticklabels=['reset','plus','minus','deph.'],ylabel=key,title=key);a.legend(fontsize=8)
fig.suptitle('State persistence, measurement backaction and within-wave feedback')
save(fig,'L40_L43_Q42_state_memory',['L40','L43','Q42'])
fig,axs=plt.subplots(2,2,figsize=(11,7.2))
for i,g in enumerate((0,3)):
 for j,arm in enumerate(arms):
  q=[x for x in r['observables'] if x['generation']==g and x['arm']==arm];xx=[x['k'] for x in q]
  for a,key in zip(axs[i],('Na','mean_mark')):a.plot(xx,[x['metrics'][key]['mean'] for x in q],marker='o',ms=3,c=colors[j],label=arm)
 for a,key in zip(axs[i],('Source a count','Mean backup mark')):
  a.set(xscale='log',xlabel='Completed backups',ylabel=key,title=f'g{g} | {key}',xticks=[4,8,16,32,64,128],xticklabels=[4,8,16,32,64,128])
axs[0,0].legend(ncol=2,fontsize=7)
save(fig,'Q45_persistent_native_response',['Q45','Q43','L16'])
fig,axs=plt.subplots(2,2,figsize=(11.7,7.4))
for i,g in enumerate((0,3)):
 q=[x for x in r['utility'] if x['generation']==g]
 for a,key in zip(axs[i],('top5','regret')):
  p=[x['scores'][key] if key=='top5' else x['scores'][key]['16'] for x in q]
  ci(a,np.arange(6),[x['mean'] for x in p],[x['ci95'] for x in p],fmt='o',c=colors[3])
  a.set(xticks=np.arange(6),xticklabels=['base','fresh','reset','plus','minus','deph.'],ylabel='Top-5 match' if key=='top5' else 'Reference Q regret',title=f'g{g} | {key}')
fig.suptitle('Independent 4×1024-backup g3 reference, not true game regret')
save(fig,'Q45_independent_reference_utility',['Q45','L49'])
fig,axs=plt.subplots(1,2,figsize=(11,4.5))
keys=['persist_minus','persist_dephased','reset_plus','fresh64','baseline']
for j,g in enumerate((0,3)):
 q=[next(x for x in r['utility_contrasts'] if x['generation']==g and x['left']=='persist_plus' and x['right']==k) for k in keys]
 for a,key in zip(axs,('top5_diff','regret_diff')):
  v=[x['metrics'][key] if key=='top5_diff' else x['metrics'][key]['16'] for x in q]
  ci(a,np.arange(len(keys))+(j-.5)*.14,[x['mean'] for x in v],[x['ci95'] for x in v],fmt=['o','s'][j],label=f'g{g}',c=colors[0 if j==0 else 3])
for a,key in zip(axs,('Top-5 match difference','Reference Q regret difference')):
 a.axhline(0,c='gray',ls=':');a.set(xticks=np.arange(len(keys)),xticklabels=['minus','deph.','reset','fresh','base'],ylabel=key,title='Persistent plus − comparator');a.legend()
save(fig,'Q45_paired_utility_contrasts',['Q45','L48'])
fig,axs=plt.subplots(1,3,figsize=(12.5,4.3))
q=g0['rows'];xx=np.arange(4);lab=[f"g{x['generation']}/"+('+' if x['arm']=='persist_plus' else '−') for x in q]
ci(axs[0],xx,[x['difference']['mean'] for x in q],[x['difference']['ci95'] for x in q],fmt='o')
axs[0].axhline(0,c='gray',ls=':');axs[0].set(xticks=xx,xticklabels=lab,ylabel='Mixture − iid NLL (nat)',title='Actual count prediction')
ci(axs[1],xx,[x['extra_variance_a']['mean'] for x in q],[x['extra_variance_a']['ci95'] for x in q],fmt='o')
axs[1].set(xticks=xx,xticklabels=lab,ylabel='Extra count variance',title='Exact latent-state contribution')
for j,key in enumerate(('covariance_residual','iid_covariance_residual')):
 v=[x[key] for x in q];ci(axs[2],xx+(j-.5)*.15,[x['mean'][0][0] for x in v],[[x['ci95'][0][0][0],x['ci95'][1][0][0]] for x in v],fmt=['o','s'][j],label=['exact mixture','iid'][j])
axs[2].axhline(0,c='gray',ls=':');axs[2].set(xticks=xx,xticklabels=lab,ylabel='Empirical covariance residual aa',title='Conditional moment check');axs[2].legend(fontsize=8)
fig.suptitle('49,152 persistent-wave observations: exact two-component count generator')
save(fig,'L16_L41_L43_persistent_wave_generator',['L16','L41','L43','Q43'])
# Original Q44 and Q27 animations, use actual prepared wave-averaged density,
# not assigned hbar time series or invented system-size labels.
def series(g,arm):
 v=wave(g,arm);x=v[:,:,wf.index('X')];y=v[:,:,wf.index('Y')];co=np.hypot(x,y)
 assert co.max()<=1+1e-12
 C=np.minimum(co,1.)
 # Unit-action absolute visibility proxy; no clipping of singular cases into finite values.
 proxy=np.full_like(C,np.nan);mask=C<1-1e-12;proxy[mask]=1/np.arccos(C[mask])
 return dict(C=np.mean(C,axis=0),purity=v[:,:,wf.index('purity')].mean(0),proxy=np.nanmedian(proxy,axis=0),invalid=int((~mask).sum()))
series0={(g,arm):series(g,arm) for g in (0,3) for arm in ('reset_plus','persist_plus','persist_minus','persist_dephased')}
fig,axs=plt.subplots(1,2,figsize=(10.6,4.4))
for g in (0,3):
 for arm in ('reset_plus','persist_plus'):
  a=series0[g,arm];axs[0].plot(t,a['C'],label=f'g{g}/{arm}',ls='-' if g==3 else '--',c=colors[arms.index(arm)])
  axs[1].plot(t,a['proxy'],label=f'g{g}/{arm}',ls='-' if g==3 else '--',c=colors[arms.index(arm)])
axs[0].set(xlabel='Completed backups',ylabel='Mean transverse coherence',title='Actual wave-averaged register')
axs[1].axhline(2/np.pi,c='gray',ls=':',label='2/π floor');axs[1].set(xlabel='Completed backups',ylabel='Median unit-action proxy',title='Not physical ħ');axs[0].legend(fontsize=8);axs[1].legend(fontsize=8)
save(fig,'Q44_visibility_match_actual_clock',['Q44','Q41'])
def anim(name,keys,labels,ids):
 fig,axs=plt.subplots(1,2,figsize=(10,4.5),dpi=100);lines=[]
 for i,(a,key,label) in enumerate(zip(axs,keys,labels)):
  ys=[v[key] for v in series0.values()];a.set(xlim=(0,124),ylim=(min(0,float(np.nanmin(ys))*.95),float(np.nanmax(ys))*1.08+.01),xlabel='Completed backups before wave',ylabel=label,title=label)
  if key=='proxy':a.axhline(2/np.pi,c='gray',ls=':',label='2/π floor')
  for (g,arm),v in series0.items():
   l,=a.plot([],[],c=colors[arms.index(arm)],ls='-' if g==3 else '--',label=f'g{g}/{arm}');lines.append((l,v[key]))
 axs[0].legend(fontsize=6,ncol=2);title=fig.suptitle('')
 fig.tight_layout()
 def update(k):
  for l,v in lines:l.set_data(t[:k+1],v[:k+1])
  title.set_text(f'Actual auxiliary register | completed backups = {t[k]} | not physical quantum data')
  return [l for l,v in lines]+[title]
 p=S/'plots'/name;FuncAnimation(fig,update,frames=32,interval=125,repeat=True).save(p,writer=PillowWriter(fps=8));plt.close(fig)
 with Image.open(p) as im:
  assert im.n_frames==32
  for i in range(im.n_frames):im.seek(i);im.load()
 animations.append(dict(name=name,ids=ids,frames=32,sha256=sha(p),bytes=p.stat().st_size))
anim('Q44_hbar_proxy_evolution.gif',('C','proxy'),('Mean transverse coherence','Median visibility-match proxy'),['Q44'])
anim('Q27_coherence_purity_evolution.gif',('C','purity'),('Mean transverse coherence','Mean selective purity'),['Q27'])
# Dedicated static counterpart for Q27 to keep paired inventory clear.
fig,axs=plt.subplots(1,2,figsize=(10.6,4.4))
for (g,arm),v in series0.items():
 for a,key in zip(axs,('C','purity')):a.plot(t,v[key],label=f'g{g}/{arm}',ls='-' if g==3 else '--',c=colors[arms.index(arm)])
for a,key in zip(axs,('Mean transverse coherence','Mean selective purity')):a.set(xlabel='Completed backups',ylabel=key,title=key)
axs[0].legend(fontsize=7,ncol=2)
fig.suptitle('Actual persistent-register coherence and purity, no entanglement claim')
save(fig,'Q27_coherence_purity_static',['Q27','L09','Q24'])
dump(S/'PLOT_MANIFEST65.json',dict(figures=manifest,animations=animations,sources={p:sha(S/p) for p in ['PHYSICS_RESULTS.json','PHASE65.npz','REFERENCE65.npz','GENERATOR65.json','CALIBRATION65.json']},proxy_scope='wave-averaged conditional density transverse coherence; per-path median proxy; singular omitted not clipped',proxy_singular_counts={f'g{g}_{arm}':v['invalid'] for (g,arm),v in series0.items()}))
imgs=[]
for x in manifest:
 with Image.open(S/'plots'/(x['figure']+'.png')) as im:
  j=im.convert('RGB');j.thumbnail((650,440));imgs.append(j.copy())
sheet=Image.new('RGB',(1300,440*((len(imgs)+1)//2)),'white')
for j,im in enumerate(imgs):sheet.paste(im,(650*(j%2),440*(j//2)))
sheet.save(S/'logs/plot_qa.jpg',quality=88)
print('PLOT_PAIRS',len(manifest),'ANIMATIONS',animations)
