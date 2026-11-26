from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
r=read(S/'PHYSICS_RESULTS.json');k=read(S/'KERNEL_RESULTS63.json');v=read(S/'VERIFICATION63.json');s=read(S/'RAPFI_TRANSFER63.json')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.dpi':120})
manifest=[]
def save(fig,name,ids):
 fig.tight_layout();p=S/'plots'/name
 fig.savefig(str(p)+'.png',dpi=165);fig.savefig(str(p)+'.pdf');plt.close(fig)
 with Image.open(str(p)+'.png') as im:im.verify()
 assert Path(str(p)+'.pdf').read_bytes().startswith(b'%PDF')
 manifest.append(dict(figure=name,ids=ids,png_sha=sha(Path(str(p)+'.png')),pdf_sha=sha(Path(str(p)+'.pdf'))))
def ci(ax,xx,y,c,fmt='o',label=None):
 y=np.asarray(y);c=np.asarray(c);ax.errorbar(xx,y,yerr=np.maximum(0,np.vstack([y-c[:,0],c[:,1]-y])),fmt=fmt,capsize=3,label=label)
fig,ax=plt.subplots(1,2,figsize=(11,4.5))
z=[r['fisher'][a] for a in ('old','new','combined_secondary')];xx=np.arange(3)
for j,key in enumerate(['normal_approx95','bonferroni95','block_simultaneous95']):
 ci(ax[0],xx+(j-1)*.13,[a['residual']['mean'][0][0] for a in z],[np.asarray(a[key])[:,0,0] for a in z],fmt=['o','s','^'][j],label=key.replace('_',' '))
ax[0].axhline(0,c='gray',ls=':');ax[0].set(xticks=xx,xticklabels=['R62 (12)','R63 (64)','Combined (76)'],ylabel='Fisher residual aa',title='Fixed geometry; fresh-seed test');ax[0].legend(fontsize=8)
for j,a in enumerate(z[:2]):ax[1].hist(np.asarray(a['root_standardized'])[:,0,0],bins=np.linspace(-4,4,17),alpha=.55,label=['R62','R63'][j],density=True)
xx=np.linspace(-4,4,100);ax[1].plot(xx,np.exp(-xx*xx/2)/np.sqrt(2*np.pi),'k--',label='Normal reference');ax[1].set(xlabel='Root residual / sqrt bracket',ylabel='Density',title='Descriptive standardization');ax[1].legend(fontsize=8)
save(fig,'Q32_G07_martingale_fisher_residual',['Q32','G07'])
fig,ax=plt.subplots(1,2,figsize=(11,4.3))
l=r['legacy'];old=l['R62'];labels=['direct FD','score','raw reweight','centered reweight']
for j,(src,keys) in enumerate([(old,['fd','score','rw','rw_centered']),(l,['fd','score','reweight','centered'])]):
 q=[src[a] for a in keys];ci(ax[0],np.arange(4)+(j-.5)*.17,[a['mean'] for a in q],[a['ci95'] for a in q],fmt=['o','s'][j],label=['R62 · 16 seeds','R63 · 32 fresh'][j])
ax[0].set(xticks=np.arange(4),xticklabels=labels,ylabel='Source count response',title='Same raw-frame roots; h=±0.35');ax[0].tick_params(axis='x',rotation=17);ax[0].legend()
for j,key in enumerate(['fd_minus_score','fd_minus_rw','fd_minus_center']):
 q=l[key];ci(ax[1],[j],[q['mean']],[q['ci95']])
ax[1].axhline(0,c='gray',ls=':');ax[1].set(xticks=range(3),xticklabels=['FD−score','FD−raw','FD−center'],ylabel='Paired contrast',title='Independent R63 block intervals')
save(fig,'G07_Q32_legacy_replication',['G07','Q32'])
fig,axs=plt.subplots(2,2,figsize=(12,7.7))
colors=['#a98558','#4f7499','#24876f','#b4778e']
for i,g in enumerate((0,3)):
 rows=[q for q in k['results'] if q['generation']==g];horizons=[1,2,4,8]
 means=np.array([np.mean([np.array(q['scores']['mean'])[:4,0] for q in rows if q['horizon']==h],0) for h in horizons])
 for j,name in enumerate(['occupancy frozen','full p frozen','p state composed','count composed']):axs[i,0].plot(horizons,means[:,j],'o-',c=colors[j],label=name)
 axs[i,0].set(xticks=horizons,xlabel='Future wave horizon',ylabel='Heldout NLL (nat)',title=f'g{g}: mean across four starts');axs[i,0].legend(fontsize=8)
 a=axs[i,1];D=np.array([[next(q['p_minus_full_p']['mean'] for q in rows if q['start_wave']==t and q['horizon']==h) for h in horizons] for t in (4,8,16,24)])
 im=a.imshow(D,cmap='RdBu_r',vmin=-.2,vmax=.2);a.grid(False);a.set(xticks=range(4),xticklabels=horizons,yticks=range(4),yticklabels=[16,32,64,96],xlabel='Future waves',ylabel='Starting backups',title='p composed − frozen full p NLL')
 for x in range(4):
  for y in range(4):a.text(y,x,f'{D[x,y]:+.3f}',ha='center',va='center',fontsize=9)
fig.suptitle('New 64 geometry blocks; models fitted only on prior R62',fontsize=14)
save(fig,'L40_L43_multiscale_predictive_kernel',['L40','L43','L48','L49'])
fig,axs=plt.subplots(1,3,figsize=(12.5,4.2))
for g,c in [(0,colors[1]),(3,colors[2])]:
 a=next(x for x in k['oracle_calibration'] if x['generation']==g);y=np.asarray(a['pit_histogram']['mean']);C=np.asarray(a['pit_histogram']['ci95']).T
 ci(axs[0],np.arange(10)+(g/3-.5)*.15,y,C,fmt='o',label=f'g{g}')
 rows=[x for x in k['results'] if x['generation']==g and x['horizon']==1];xs=[4*x['start_wave'] for x in rows];z=np.array([x['scores']['mean'] for x in rows])
 axs[1].plot(xs,z[:,1,0],'-o',c=c,label=f'g{g} exact');axs[1].plot(xs,z[:,4,0],'--s',c=c,label=f'g{g} source Q + bath')
 yy=np.array([a['mean_residual']['mean'][j] for j in range(3)]);cc=np.asarray(a['mean_residual']['ci95']).T
 ci(axs[2],np.arange(3)+(g/3-.5)*.15,yy,cc,fmt='o',label=f'g{g}')
axs[0].axhline(.1,c='gray',ls=':');axs[0].set(xlabel='Randomized PIT decile',ylabel='Frequency',title='Exact local count calibration');axs[0].legend()
axs[1].set(xlabel='Starting backups',ylabel='One-wave NLL',title='Reduced rest normalizer');axs[1].legend(fontsize=7)
axs[2].axhline(0,c='gray',ls=':');axs[2].set(xticks=range(3),xticklabels=['a','b','rest'],ylabel='Count − conditional mean',title='Heldout innovation mean');axs[2].legend()
save(fig,'L41_L49_conditional_multinomial_calibration',['L41','L49','L40'])
fig,axs=plt.subplots(1,2,figsize=(10.5,4.3))
for g,c in [(0,colors[1]),(3,colors[2])]:
 q=[x for x in v['curves'] if x['generation']==g];j=[x['J'] for x in q]
 axs[0].plot(j,[x['mean_exact_action'] for x in q],'-',c=c,label=f'g{g} exact KL');axs[0].plot(j,[x['mean_quadratic_action'] for x in q],'--',c=c,label=f'g{g} quadratic')
 axs[1].plot(j,[x['max_abs_local_quadratic_error'] for x in q],'-o',ms=3,c=c,label=f'g{g}')
axs[0].set(xlabel='Tilt Ja, Jb=−Ja/2',ylabel='Mean conditional conjugate',title='Native local multinomial action');axs[0].legend(fontsize=8)
axs[1].set(xlabel='Tilt Ja',ylabel='Max absolute quadratic error',title='Finite local approximation error');axs[1].legend()
save(fig,'L16_conditional_action_curvature',['L16','L38'])
fig,axs=plt.subplots(1,2,figsize=(10.5,4.3))
q=s['results'];L=[x['L'] for x in q];axs[0].plot(L,[x['mean_nll'] for x in q],'o-',label='Exact recorded p');axs[0].plot(L,[x['occupancy_nll'] for x in q],'s--',label='Smoothed occupancy');axs[0].set(xticks=L,xlabel='Board side',ylabel='NLL per actual phase (nat)',title='R58 Rapfi transfer, B=32');axs[0].legend()
for j,n in enumerate(['a','b','rest']):axs[1].plot(L,[x['mean_residual'][j] for x in q],'o-',label=n)
axs[1].axhline(0,c='gray',ls=':');axs[1].set(xticks=L,xlabel='Board side',ylabel='Conditional count residual',title='Separate evaluator / beta design');axs[1].legend()
save(fig,'L38_L41_Rapfi_size_transfer',['L38','L41'])
fig,axs=plt.subplots(1,2,figsize=(10.5,4.3))
for g,c in [(0,colors[1]),(3,colors[2])]:
 for group,ls in [(0,'-'),(1,'--')]:
  q=[x for x in v['innovation_lags'] if x['generation']==g and x['group']==group];lag=[x['lag'] for x in q]
  axs[0].plot(lag,[x['raw_correlation'] for x in q],ls,c=c,marker='o',ms=3,label=f'g{g}/{"ab"[group]}')
  a=axs[1];ys=np.array([x['conditional_innovation_product']['mean'] for x in q]);cc=np.array([x['conditional_innovation_product']['ci95'] for x in q]);a.plot(lag,ys,ls,c=c,marker='o',ms=3,label=f'g{g}/{"ab"[group]}');a.fill_between(lag,cc[:,0],cc[:,1],color=c,alpha=.08)
axs[0].set(xlabel='Wave lag',ylabel='Pooled raw correlation',title='Raw counts');axs[0].legend(fontsize=8)
axs[1].axhline(0,c='gray',ls=':');axs[1].set(xlabel='Wave lag',ylabel='Mean innovation product',title='Causal conditional residual');axs[1].legend(fontsize=8)
save(fig,'L39_L42_conditional_innovation_lags',['L39','L42','L44'])
fig,axs=plt.subplots(1,3,figsize=(12.5,4.3))
for i,g in enumerate((0,3)):
 with np.load(S/f'KERNEL_MODEL_g{g}.npz') as a:T=a['KP'][8].sum(0)
 axs[i].imshow(T,vmin=0,vmax=1,cmap='Blues');axs[i].grid(False);axs[i].set(xlabel='Next probability bin',ylabel='Current probability bin',title=f'g{g}: fitted wave 8 kernel',xticks=[0,4,8,12,15],yticks=[0,4,8,12,15])
q=[x for x in k['tilt_composition'] if x['generation']==0 and x['start_wave']==4 and x['J']==[.1,-.05,0.]]
h=[x['horizon'] for x in q];axs[2].plot(h,[x['empirical_W'] for x in q],'ko-',label='Actual heldout');axs[2].plot(h,[x['pred_W'][1] for x in q],'s--',label='Frozen p');axs[2].plot(h,[x['pred_W'][2] for x in q],'^-',label='Composed p kernel');axs[2].set(xticks=h,xlabel='Future waves',ylabel='Log mean MGF',title='Local generating operator');axs[2].legend(fontsize=8)
save(fig,'L43_L16_joint_transition_generator',['L43','L16','L40'])
dump(S/'PLOT_MANIFEST63.json',dict(figures=manifest,sources={p:sha(S/p) for p in ['PHYSICS_RESULTS.json','KERNEL_RESULTS63.json','VERIFICATION63.json','RAPFI_TRANSFER63.json','KERNEL_MODEL_g0.npz','KERNEL_MODEL_g3.npz']}))
imgs=[]
for x in manifest:
 with Image.open(S/'plots'/(x['figure']+'.png')) as im:
  j=im.convert('RGB');j.thumbnail((650,460));imgs.append(j.copy())
sheet=Image.new('RGB',(1300,460*((len(imgs)+1)//2)),'white')
for i,im in enumerate(imgs):sheet.paste(im,(650*(i%2),460*(i//2)))
sheet.save(S/'logs/plot_qa.jpg',quality=85)
print('PLOT_PAIRS',len(manifest))
