from arena68 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from PIL import Image
r=read(S/'PHYSICS_RESULTS.json');t=read(S/'TACTICAL68.json');a=read(S/'AUDIT68.json')
arms=['clean','single','recovery'];colors=['#537898','#AC626C','#25866E'];manifest=[];animations=[]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.dpi':120})
def save(fig,name,ids):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=165);fig.savefig(str(p)+'.pdf');plt.close(fig)
 with Image.open(str(p)+'.png') as im:im.verify()
 assert Path(str(p)+'.pdf').read_bytes().startswith(b'%PDF')
 manifest.append(dict(figure=name,ids=ids,png_sha=sha(Path(str(p)+'.png')),pdf_sha=sha(Path(str(p)+'.pdf'))))
def ci(ax,xx,ss,**kw):
 y=np.array([x['mean'] for x in ss]);c=np.array([x['ci95'] for x in ss]);ax.errorbar(xx,y,yerr=np.maximum(0,np.vstack([y-c[:,0],c[:,1]-y])),capsize=3,**kw)
match={x['match']:x for x in r['results']}
fig,axs=plt.subplots(1,2,figsize=(11.4,4.5))
for ax,ms,labels in [(axs[0],['ext_clean','ext_single','ext_recovery'],['Clean','Single','Recovery']),(axs[1],['internal_recovery_single','internal_recovery_clean'],['vs single','vs clean'])]:
 q=[match[k] for k in ms];bottom=np.zeros(len(q))
 for key,label,col in [('wins','Win','#25866e'),('draws','Draw','#ccb784'),('losses','Loss','#ac626c'),('failed','Incomplete','#adadad')]:
  n=np.array([v[key] for v in q]);ax.bar(np.arange(len(q)),n,bottom=bottom,label=label,color=col);bottom+=n
 ax.set(xticks=np.arange(len(q)),xticklabels=labels,ylim=(0,16),ylabel='Registered games',title='Independent Rapfi search' if len(q)==3 else 'Native recovery focal side');ax.legend(fontsize=8)
fig.suptitle('15×15 complete-game arena, paired openings and colors')
save(fig,'L12_arena_outcomes',['L12'])
fig,axs=plt.subplots(1,2,figsize=(11.2,4.4))
q=r['contrasts'];labels=['Ext R−single','Ext R−clean','Ext single−clean','Internal R−single','Internal R−clean'];ci(axs[0],np.arange(5),[x['complete_opening_contrast'] for x in q],fmt='o');axs[0].axhline(0,c='gray',ls=':');axs[0].set(xticks=np.arange(5),xticklabels=labels,ylabel='Opening-paired score difference',title='Descriptive 95% block intervals');axs[0].tick_params(axis='x',rotation=27)
for i,x in enumerate(r['results']):
 if x['match'].startswith('ext_'):
  lo,hi=x['all_registered_score_bounds'];axs[1].plot([i,i],[lo,hi],'o-',lw=2)
axs[1].set(xticks=[0,1,2],xticklabels=['Clean','Recovery','Single'],ylim=(-.03,.33),ylabel='Score bounds over all 16 games',title='Incomplete outcomes bounded, not draws')
save(fig,'L12_paired_score_bounds',['L12'])
fig,axs=plt.subplots(1,3,figsize=(12.5,4.3))
for i,c in enumerate(('external','internal')):
 q=[x for x in r['native'] if x['cohort']==c];xx=np.arange(3)+(i-.5)*.14
 for ax,key in zip(axs,('sign_error_rate','value_MAE','native_seconds')):
  ci(ax,xx,[x[key] for x in q],fmt=['o','s'][i],label=c)
for ax,y in zip(axs,('Residual sign error','Value MAE','Wall seconds / native search')):
 ax.set(xticks=np.arange(3),xticklabels=arms,ylabel=y,title=y);ax.legend(fontsize=8)
save(fig,'L23_15x15_native_recovery',['L23','L12'])
fig,axs=plt.subplots(2,3,figsize=(13,7.3))
for i,co in enumerate(('external','internal')):
 for j,arm in enumerate(arms):
  q=[x for x in r['cuts'] if x['cohort']==co and x['arm']==arm];k=[x['k'] for x in q]
  for ax,key in zip(axs[i],('H','KL','variance')):
   yy=np.array([x['metrics'][key]['mean'] for x in q]);cc=np.array([x['metrics'][key]['ci95'] for x in q]);ax.plot(k,yy,'o-',ms=3,c=colors[j],label=arm);ax.fill_between(k,cc[:,0],cc[:,1],color=colors[j],alpha=.1)
 for ax,key in zip(axs[i],('Visit entropy (nats)','KL(visit || NNUE prior)','Value variance')):
  ax.set(xscale='log',xlabel='Completed backups',ylabel=key,title=f'{co} | {key}',xticks=[4,8,16,32,64],xticklabels=[4,8,16,32,64]);ax.legend(fontsize=7)
fig.suptitle('Actual 15×15 finite-budget search statistics; not size scaling')
save(fig,'Q36_actual_budget_entropy',['Q36','Q39'])
fig,axs=plt.subplots(1,3,figsize=(12.5,4.2))
for i,co in enumerate(('external','internal')):
 q=[x for x in r['cuts'] if x['cohort']==co and x['k']==64];xx=np.arange(3)+(i-.5)*.15
 ci(axs[0],xx,[x['metrics']['conditional_H'] for x in q],fmt=['o','s'][i],label=co)
 ci(axs[1],xx,[x['metrics']['surprisal_residual'] for x in q],fmt=['o','s'][i],label=co)
 z=[x for x in r['information'] if x['cohort']==co];ci(axs[2],xx,[x['MI'] for x in z],fmt=['o','s'][i],label=co)
for ax,y in zip(axs,('Conditional H (nats/draw)','Observed surprise − H','Wave-index mutual information (nats)')):
 ax.set(xticks=np.arange(3),xticklabels=arms,ylabel=y,title=y);ax.legend(fontsize=8)
axs[1].axhline(0,c='gray',ls=':');fig.suptitle('Classical conditional root laws and empirical wave-index mixture')
save(fig,'Q37_actual_information_flow',['Q37'])
fig,axs=plt.subplots(2,2,figsize=(11.5,7.2))
for i,co in enumerate(('external','internal')):
 for j,arm in enumerate(arms):
  q=[x for x in r['temporal'] if x['cohort']==co and x['arm']==arm];xx=[x['ply_lo'] for x in q]
  for ax,key in zip(axs[i],('H','variance')):
   ss=[x['metrics'][key] for x in q];y=[x['mean'] for x in ss];ax.plot(xx,y,'o-',c=colors[j],ms=3,label=arm)
 for ax,key in zip(axs[i],('Visit entropy','Value variance')):ax.set(xlabel='Game ply bin start',ylabel=key,title=f'{co} | {key}');ax.legend(fontsize=8)
fig.suptitle('Actual game-time evolution; changing at-risk set, not balanced longitudinal sample')
save(fig,'Q39_actual_game_entropy',['Q39','Q36'])
fig,axs=plt.subplots(2,2,figsize=(11.5,7.2))
for i,co in enumerate(('external','internal')):
 for j,arm in enumerate(arms):
  q=[x for x in r['spatial'] if x['cohort']==co and x['arm']==arm]
  for ax,k in zip(axs[i],(16,64)):
   z=next(v for v in q if v['k']==k);y=np.array([x['mean'] for x in z['metrics']]);cc=np.array([x['ci95'] for x in z['metrics']]);xx=np.arange(1,13);ax.plot(xx,y,'o-',ms=3,c=colors[j],label=arm);ax.fill_between(xx,cc[:,0],cc[:,1],alpha=.09,color=colors[j]);ax.axhline(0,c='gray',ls=':',lw=.6);ax.set(xlabel='Manhattan distance',ylabel='Connected visit-field correlation',title=f'{co}, B={k}');ax.legend(fontsize=8)
fig.suptitle('Actual fixed-size spatial correlation, centered on legal actions')
save(fig,'Q33_actual_spatial_correlation',['Q33'])
fig,axs=plt.subplots(1,2,figsize=(10.7,4.3))
for i,co in enumerate(('external','internal')):
 q=[x for x in t['summary'] if x['cohort']==co];x=np.arange(3)+(i-.5)*.13
 axs[0].plot(x,[z['mean_top5_prior_mass'] for z in q],'o',label=f'{co} prior')
 axs[0].plot(x,[z['mean_top5_initial_law_mass'] for z in q],'x',label=f'{co} initial root law')
 axs[1].plot(x,[z['final_top5_visit_fraction'] for z in q],'o-',label=f'{co} visits')
 axs[1].plot(x,[z['recommendation_top5_fraction'] for z in q],'s--',label=f'{co} recommended')
for ax,k in zip(axs,('NNUE top-5 initial mass','Final top-5 mass / recommendation')):
 ax.set(xticks=np.arange(3),xticklabels=arms,ylabel='Fraction',title=k);ax.legend(fontsize=7)
fig.suptitle('Post-outcome diagnostic: zero initial PUCT produces uniform root law')
save(fig,'L12_native_prior_cold_start',['L12'])
# animations actual finite cuts (correlations) and game-ply at-risk curves (entropy)
def animate_spatial():
 fig,ax=plt.subplots(figsize=(7.8,4.5),dpi=100);ls=[];K=[4,8,16,32,64]
 q=[x for x in r['spatial'] if x['cohort']=='external']
 ys=[v['mean'] for x in q for v in x['metrics']]
 for j,arm in enumerate(arms):l,=ax.plot([],[],'o-',ms=3,c=colors[j],label=arm);ls.append(l)
 ax.set(xlim=(1,12),ylim=(min(ys)-.01,max(ys)+.01),xlabel='Manhattan distance',ylabel='Connected visit-field correlation');ax.axhline(0,c='gray',ls=':');ax.legend()
 def upd(i):
  for arm,l in zip(arms,ls):
   z=next(x for x in q if x['arm']==arm and x['k']==K[i]);l.set_data(np.arange(1,13),[v['mean'] for v in z['metrics']])
  ax.set_title(f'Actual 15×15 spatial correlation | B={K[i]}');return ls
 p=S/'plots/Q33_actual_spatial_evolution.gif';FuncAnimation(fig,upd,frames=len(K),interval=700).save(p,writer=PillowWriter(fps=2));plt.close(fig);return p,len(K),'Q33'
def animate_entropy():
 fig,ax=plt.subplots(figsize=(7.8,4.5),dpi=100);ls=[];T=sorted({x['ply_lo'] for x in r['temporal'] if x['cohort']=='external'});q=[x for x in r['temporal'] if x['cohort']=='external']
 for j,arm in enumerate(arms):l,=ax.plot([],[],'o-',ms=3,c=colors[j],label=arm);ls.append(l)
 ax.set(xlim=(min(T),max(T)+1),ylim=(0,4.4),xlabel='Game ply bin start',ylabel='Mean visit entropy (nats)');ax.legend()
 def upd(i):
  for arm,l in zip(arms,ls):
   z=[x for x in q if x['arm']==arm and x['ply_lo']<=T[i]];l.set_data([x['ply_lo'] for x in z],[x['metrics']['H']['mean'] for x in z])
  n=sum(x['n'] for x in q if x['ply_lo']==T[i]);ax.set_title(f'Actual game entropy | ply {T[i]}–{T[i]+7}, n={n} native searches');return ls
 p=S/'plots/Q39_actual_game_entropy.gif';FuncAnimation(fig,upd,frames=len(T),interval=700).save(p,writer=PillowWriter(fps=2));plt.close(fig);return p,len(T),'Q39'
for p,n,k in (animate_spatial(),animate_entropy()):
 with Image.open(p) as im:
  assert im.n_frames==n
  for i in range(n):im.seek(i);im.load()
 animations.append(dict(name=p.name,ids=[k],frames=n,sha256=sha(p),bytes=p.stat().st_size))
dump(S/'PLOT_MANIFEST68.json',dict(figures=manifest,animations=animations,sources={k:sha(S/k) for k in ('PHYSICS_RESULTS.json','TACTICAL68.json','SPATIAL68.json','INFORMATION68.json','AUDIT68.json')}))
imgs=[]
for v in manifest:
 with Image.open(S/'plots'/(v['figure']+'.png')) as im:
  z=im.convert('RGB');z.thumbnail((650,440));imgs.append(z.copy())
sheet=Image.new('RGB',(1300,440*((len(imgs)+1)//2)),'white')
for i,im in enumerate(imgs):sheet.paste(im,(650*(i%2),440*(i//2)))
sheet.save(S/'logs/plot_qa.jpg',quality=87)
print('PLOTS',len(manifest),'ANIMATIONS',[(v['name'],v['frames']) for v in animations])
