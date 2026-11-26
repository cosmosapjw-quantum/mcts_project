from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
A=read(S/'INSTRUMENT73.json');D=read(S/'DECISION_RESULTS73.json');V=read(S/'INDEPENDENT73.json');U=read(S/'AUDIT73.json')
rows=A['contexts'];gs=[(0,'persist_plus'),(0,'persist_minus'),(3,'persist_plus'),(3,'persist_minus')];names=['g0+','g0−','g3+','g3−'];figs=[]
def save(fig,name,ids,scope):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=160);fig.savefig(str(p)+'.pdf');plt.close(fig)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 assert b.read_bytes()[:4]==b'%PDF'
 figs.append(dict(figure=name,ids=ids,scope=scope,png_sha=sha(a),pdf_sha=sha(b)))
def select(g):return [r for r in rows if (r['generation'],r['arm'])==g]
def arr(g,mode,k):return np.array([r[mode][k] for r in select(g)])
fig,ax=plt.subplots(1,2,figsize=(11,4))
for i,g in enumerate(gs):
 a=arr(g,'adaptive','record_neg');ax[0].scatter(np.full(16,i),a,s=16,alpha=.7);ax[0].scatter(i,a.mean(),marker='D',color='k')
 ax[1].bar(i,a.mean(),.6,label=None)
ax[0].set(title='Adaptive coherent-control model',ylabel='record-pair negativity');ax[1].set(title='Mean; frozen and measured A both zero',ylabel='negativity')
for a in ax:a.set_xticks(range(4),names);a.axhline(0,color='black',lw=.6)
save(fig,'record_pair_negativity',['Q35','Q40'],'Defined coherent dilation only; frozen and measured controls zero.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for j,key in enumerate(['expected_conditional_TV','adaptive_vs_frozen_classical_TV']):
 for i,g in enumerate(gs):
  v=np.array([r[key] for r in select(g)]);ax[j].scatter(np.full(len(v),i),v,s=16);ax[j].scatter(i,v.mean(),marker='D',color='k')
 ax[j].set_xticks(range(4),names);ax[j].set(ylabel='TV distance',title='Full fine conditional law' if j==0 else 'Coarse joint diagonal')
save(fig,'adaptive_vs_frozen_law',['Q24','Q37','Q28'],'Actual branch-dependent conditional laws vs frozen observed branch.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for j,k in enumerate(['first_record_dephasing_TD','pair_dephasing_TD']):
 for i,g in enumerate(gs):
  v=arr(g,'adaptive',k);ax[j].scatter(np.full(len(v),i),v,s=16);ax[j].scatter(i,v.mean(),marker='D',color='k')
 ax[j].set_xticks(range(4),names);ax[j].set(ylabel='trace distance',title='S + two records' if j==0 else 'Two records')
save(fig,'coherent_record_dephasing',['Q24','Q26','Q28'],'Trace distance to measured-first-record control.')
r=max(rows,key=lambda x:x['adaptive']['record_neg']);z=np.array(r['adaptive']['record_density_real'])+1j*np.array(r['adaptive']['record_density_imag']);f=np.array(r['frozen']['record_density_real'])+1j*np.array(r['frozen']['record_density_imag'])
fig,ax=plt.subplots(2,2,figsize=(9,8))
for a,m,title in zip(ax.flat,[z.real,z.imag,(z-f).real,(z-f).imag],['adaptive Re','adaptive Im','adaptive − frozen Re','adaptive − frozen Im']):
 lim=max(abs(m).max(),1e-15);im=a.imshow(m,vmin=-lim,vmax=lim,cmap='coolwarm');fig.colorbar(im,ax=a,shrink=.75);a.set_title(title);a.set_xlabel('record basis');a.set_ylabel('record basis')
fig.suptitle(f"Actual branch context g{r['generation']} {r['arm']} core {r['core']}")
save(fig,'complex_record_matrix',['Q35','Q40'],'Full complex reduced two-record density, largest model negativity.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for j,key in enumerate(['record_MI','record_classical_MI']):
 for i,g in enumerate(gs):
  for k,mode in enumerate(['adaptive','frozen']):ax[j].bar(i+(k-.5)*.32,arr(g,mode,key).mean(),.3,label=mode if i==0 else None)
 ax[j].set_xticks(range(4),names);ax[j].set(ylabel='nat',title=key);ax[j].legend()
save(fig,'record_information',['Q25','Q37','Q40'],'Von Neumann versus diagonal classical MI, not final-decision information.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for j,mode in enumerate(['adaptive','frozen']):
 for i,g in enumerate(gs):
  v=arr(g,mode,'system_entropy');ax[j].scatter(np.full(16,i),v,s=14);ax[j].scatter(i,v.mean(),marker='D',color='black')
 ax[j].set_xticks(range(4),names);ax[j].set(title=mode,ylabel='system von Neumann entropy (nat)')
save(fig,'system_entropy',['Q24','Q40'],'System entropy after two records, branch-averaged model.')
fresh=[r for r in D['results'] if r['dataset']=='fresh_R73'];lab=lambda r:('Rapfi' if r['generation']==4 else f"g{3*r['generation']}"+['+','−','dephased'][r['arm']])
fig,ax=plt.subplots(2,4,figsize=(14,7),sharex=True);xx=np.arange(1,8)
for a,r in zip(ax.flat,fresh):
 v=np.array(r['gain'])[1:];a.errorbar(xx,v[:,0],yerr=[v[:,0]-v[:,1],v[:,2]-v[:,0]],fmt='o-');a.axhline(0,color='black',lw=.6);a.set_title(lab(r));a.set_xticks(xx,['F1','F2','F3','F4','early','late','all'],rotation=40);a.set_ylabel('log-loss gain (nat)')
ax.flat[-1].axis('off')
save(fig,'decision_fragment_gain',['L15','L30','Q25','Q37'],'Fresh roots, four disjoint 32-event fragments and two halves; predictive gains, not exact MI.')
fig,ax=plt.subplots(1,2,figsize=(12,4))
for i,r in enumerate(fresh):
 v=np.array(r['gain'])[7];q=r['shuffle_gain'][7]
 ax[0].errorbar(i-.11,v[0],yerr=[[v[0]-v[1]],[v[2]-v[0]]],fmt='o',label='actual' if i==0 else None)
 ax[0].errorbar(i+.11,q[0],yerr=[[q[0]-q[1]],[q[2]-q[0]]],fmt='s',label='within-root shuffle' if i==0 else None)
 ax[1].bar(i,v[0]-q[0])
for a in ax:a.set_xticks(range(len(fresh)),[lab(r) for r in fresh],rotation=35);a.axhline(0,color='black',lw=.6)
ax[0].legend();ax[0].set(ylabel='all-record log-loss gain (nat)',title='Actual vs shuffled');ax[1].set(ylabel='actual minus shuffle mean gain',title='Seed-specific predictive component')
save(fig,'decision_shuffle',['Q25','Q37','L30'],'Within-root permutation negative control; geometry effects remain in shuffled predictors.')
fig,ax=plt.subplots(1,2,figsize=(12,4))
for i,r in enumerate(fresh):
 v=np.array(r['decision_counts'])/r['paths'];b=0
 for k,z0 in enumerate(v):ax[0].bar(i,z0,bottom=b,label=['source a','source b','other'][k] if i==0 else None);b+=z0
 ax[1].bar(i,r['empirical_root_entropy'])
for a in ax:a.set_xticks(range(len(fresh)),[lab(r) for r in fresh],rotation=35)
ax[0].legend();ax[0].set(ylabel='fraction',title='Most-visited recommendation class');ax[1].set(ylabel='empirical entropy (nat)',title='Within-root plugin entropy (biased)')
save(fig,'decision_class_distribution',['Q26','Q37'],'Actual final recommendation D is distinct from qubit pointer Z; no redundancy certification.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].bar(['native paths','forced replay','conditional\ncontexts','certified physical\nBell tests'],[U['paths'],A['branch_paths'],64,0]);ax[0].set(ylabel='count',title='Typed evidence counts (not additive)');ax[0].tick_params(axis='x',rotation=25)
keys=['max_adaptive_matrix_error','max_frozen_matrix_error','max_negativity_error','max_diagonal_error','max_trace_error'];ax[1].bar(range(len(keys)),[V[k] for k in keys]);ax[1].set_yscale('log');ax[1].set_xticks(range(len(keys)),['adaptive','frozen','negativity','diagonal','trace'],rotation=25);ax[1].set(title='Independent numerical audit',ylabel='max absolute error')
save(fig,'measurement_quality',['L05','L34','Q40'],'Scoped measurement-quality report with explicit categories and verification errors; not physical temperature or critical-point count.')
assert len(figs)==10 and len({q['png_sha'] for q in figs})==10
dump(S/'PLOT_MANIFEST73.json',dict(figures=figs,animations=[],sources={f:sha(S/f) for f in ['INSTRUMENT73.json','DECISION_RESULTS73.json','INDEPENDENT73.json','AUDIT73.json','plot73.py']}))
print('PLOTS',len(figs),flush=True)
