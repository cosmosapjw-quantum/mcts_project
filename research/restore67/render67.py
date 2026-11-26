from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
r=read(S/'PHYSICS_RESULTS.json');q=read(S/'QEC67.json');sr=read(S/'STATE_RECOVERY67.json')
arms=read(S/'REGISTRATION.json')['arms'];labels=['clean','single iid','rep iid','single mix','rep mix','rep common']
colors=['#637f99','#287d6b','#b05b69']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'figure.dpi':120})
man=[]
def save(fig,name,ids):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=160);fig.savefig(str(p)+'.pdf');plt.close(fig)
 with Image.open(str(p)+'.png') as im:im.verify()
 assert Path(str(p)+'.pdf').read_bytes().startswith(b'%PDF')
 man.append(dict(figure=name,ids=ids,png_sha=sha(Path(str(p)+'.png')),pdf_sha=sha(Path(str(p)+'.pdf'))))
def ci(a,x,v,**kw):
 y=np.array([i['mean'] for i in v]);c=np.array([i['ci95'] for i in v]);a.errorbar(x,y,yerr=np.maximum(0,np.vstack([y-c[:,0],c[:,1]-y])),capsize=3,**kw)
def rows(g):return [x for x in r['rows'] if x['generation']==g]
fig,axs=plt.subplots(1,2,figsize=(10.8,4.3))
for i,mu in enumerate((0.,.5,1.)):
 a=[x for x in q['stochastic'] if x['mu']==mu];p=[x['p'] for x in a]
 axs[0].plot(p,[x['pL'] for x in a],c=colors[i],label=f'μ={mu:g}')
 axs[1].plot(p,[x['Fe'] for x in a],c=colors[i],label=f'μ={mu:g}')
axs[0].plot([0,.5],[0,.5],'k:',label='uncoded')
axs[0].set(xlabel='Physical bit flip p',ylabel='Logical bit flip',title='Ideal syndrome recovery');axs[1].set(xlabel='Physical bit flip p',ylabel='Entanglement fidelity',title='Logical channel fidelity')
for a in axs:a.legend()
fig.suptitle('Auxiliary three-qubit code, restricted X-error distance')
save(fig,'L23_exact_logical_noise',['L23','L21'])
fig,axs=plt.subplots(1,2,figsize=(10.8,4.3));a=q['coherent'];angle=[x['angle'] for x in a]
axs[0].plot(angle,[1-x['Fe'] for x in a],label='Logical infidelity')
axs[0].plot(angle,[abs(x['logical_coherent_coefficient']) for x in a],label='|Coherent coefficient|')
axs[0].set(xlabel='Physical rotation half-angle (rad)',ylabel='Magnitude',title='Non-Pauli residual');axs[0].legend()
for i,n in enumerate((1,10,100)):
 axs[1].plot(angle,[x['repeated_coherent'][i][1] for x in a],c=colors[i],label=f'{n} cycles exact')
 axs[1].plot(angle,[x['repeated_pauli'][i][1] for x in a],c=colors[i],ls='--',label=f'{n} cycles Pauli')
axs[1].set(xlabel='Physical rotation half-angle (rad)',ylabel='Entanglement fidelity',title='Repeated coherent vs Pauli');axs[1].legend(fontsize=8,ncol=2)
save(fig,'L23_coherent_cycles',['L23','L21'])
fig,axs=plt.subplots(2,2,figsize=(11.4,7.4))
for i,g in enumerate((0,3)):
 s=[x for x in sr['summary'] if x['generation']==g];lab=['iid X','mixed X','common X','coherent X','single Z'];xx=np.arange(5)
 for a,key in zip(axs[i],('trace_distance','Y_error')):
  a.bar(xx-.18,[x['uncoded_'+key] for x in s],.34,label='Uncoded',color=colors[0]);a.bar(xx+.18,[x['recovered_'+key] for x in s],.34,label='Recovered',color=colors[1]);a.set(xticks=xx,xticklabels=lab,ylabel=key.replace('_',' '),title=f'g{g} | {key.replace("_"," ")}');a.legend(fontsize=8)
fig.suptitle('768 actual R65 conditional states, 3840 auxiliary channel applications')
save(fig,'L23_actual_register_recovery',['L23','L21'])
fig,axs=plt.subplots(1,3,figsize=(13.2,4.4))
for j,g in enumerate((0,3)):
 s=rows(g);xx=np.arange(6)+(j-.5)*.15
 for a,key in zip(axs,('residual_rate','value_MAE','syndrome_rate')):
  ci(a,xx,[x['fault'][key] for x in s],fmt=['o','s'][j],c=colors[j],label=f'g{g}')
for a,key in zip(axs,('Residual sign error','Value MAE','Nonzero syndrome rate')):
 a.set(xticks=np.arange(6),xticklabels=labels,ylabel=key,title=key);a.tick_params(axis='x',rotation=27);a.legend(fontsize=8)
axs[0].plot(np.arange(6),[0,.12,.039744,.12,.079872,.12],'kx',label='Model expectation')
axs[0].legend(fontsize=7);fig.suptitle('Actual native injected faults before leaf backup; root/terminal values excluded')
save(fig,'L23_native_value_errors',['L23'])
fig,axs=plt.subplots(1,2,figsize=(10.8,4.4))
for j,g in enumerate((0,3)):
 s=rows(g);xx=np.arange(6)+(j-.5)*.14
 a=[(i,x['fault']['success_given_syndrome']) for i,x in enumerate(s) if x['fault']['success_given_syndrome']['mean'] is not None]
 ci(axs[0],np.array([i for i,_ in a])+(j-.5)*.14,[v for _,v in a],fmt=['o','s'][j],c=colors[j],label=f'g{g}')
 ci(axs[1],xx,[x['fault']['triple_rate'] for x in s],fmt=['o','s'][j],c=colors[j],label=f'g{g}')
axs[0].set(ylabel='Correct sign given syndrome',title='Conditional correction success');axs[1].set(ylabel='Triple-fault rate',title='Invisible common-mode faults')
for a in axs:a.set(xticks=np.arange(6),xticklabels=labels);a.tick_params(axis='x',rotation=25);a.legend()
fig.suptitle('No syndrome in clean/common-mode arms; undefined conditional success omitted')
save(fig,'L23_syndrome_detection',['L23'])
fig,axs=plt.subplots(2,2,figsize=(11.7,7.4))
for i,g in enumerate((0,3)):
 s=rows(g)
 for a,key in zip(axs[i],('top5','regret')):
  v=[x['utility'][key] if key=='top5' else x['utility'][key]['16'] for x in s];ci(a,np.arange(6),v,fmt='o',c=colors[i])
  a.set(xticks=np.arange(6),xticklabels=labels,ylabel='Top-5 match' if key=='top5' else 'Reference Q regret',title=f'g{g}: {key}');a.tick_params(axis='x',rotation=22)
fig.suptitle('Independent g3 high-budget proxy, 4×1024 per root')
save(fig,'L12_native_reference_utility',['L12','L23'])
fig,axs=plt.subplots(1,3,figsize=(12.6,4.2))
for j,g in enumerate((0,3)):
 s=[x for x in r['contrasts'] if x['generation']==g and x['right'].startswith('single')];xx=np.arange(2)+(j-.5)*.15
 for a,key in zip(axs,('top5_diff','regret_diff','rapfi9_diff')):
  v=[x['metrics'][key] if key!='regret_diff' else x['metrics'][key]['16'] for x in s];ci(a,xx,v,fmt=['o','s'][j],label=f'g{g}',c=colors[j])
for a,key in zip(axs,('Reference top-5 difference','Reference Q regret difference','Rapfi9 agreement difference')):
 a.axhline(0,c='gray',ls=':');a.set(xticks=np.arange(2),xticklabels=['iid','mixed'],ylabel=key,title=key);a.legend()
fig.suptitle('Paired recovery − matched-cost single-carrier comparator')
save(fig,'L12_L23_paired_recovery_utility',['L12','L23'])
fig,axs=plt.subplots(1,3,figsize=(13.2,4.3))
for j,g in enumerate((0,3)):
 s=rows(g);xx=np.arange(6)+(j-.5)*.16
 for a,key in zip(axs[:2],('rapfi9','rapfi15_embedded')):
  ci(a,xx,[x['utility'][key] for x in s],fmt=['o','s'][j],c=colors[j],label=f'g{g}')
for a,y in zip(axs[:2],('Same-board top-1 agreement','Embedded-board agreement')):
 a.set(xticks=np.arange(6),xticklabels=labels,ylabel=y);a.tick_params(axis='x',rotation=28);a.legend(fontsize=8)
axs[0].set_title('Rapfi 9×9: fallback, no NNUE');axs[1].set_title('Rapfi 15×15: NNUE, 30 in-window roots')
keys=['top1','top5_overlap'];st=r['reference'];v=[st[k] for k in keys]+[r['external']['same_board_time_agreement'],r['external']['external_vs_g3_top5']]
ci(axs[2],np.arange(4),v,fmt='o');axs[2].set(xticks=np.arange(4),xticklabels=['g3 top1 split','g3 top5 overlap','Rapfi time match','Rapfi in g3 top5'],ylabel='Agreement fraction',title='Independent reference disagreement');axs[2].tick_params(axis='x',rotation=25)
save(fig,'L12_Rapfi_reference_comparison',['L12'])
dump(S/'PLOT_MANIFEST67.json',dict(figures=man,animations=[],sources={k:sha(S/k) for k in ['PHYSICS_RESULTS.json','QEC67.json','STATE_RECOVERY67.json','RAPFI67.json','NATIVE67.npz','REFERENCE67.npz']}))
imgs=[]
for x in man:
 with Image.open(S/'plots'/(x['figure']+'.png')) as im:
  j=im.convert('RGB');j.thumbnail((650,420));imgs.append(j.copy())
sheet=Image.new('RGB',(1300,420*((len(imgs)+1)//2)),'white')
for i,im in enumerate(imgs):sheet.paste(im,(650*(i%2),420*(i//2)))
sheet.save(S/'logs/plot_qa.jpg',quality=86);print('PLOTS',len(man))
