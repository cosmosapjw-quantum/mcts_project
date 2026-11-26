"""Original L46/L47 export and plots; matched native observables from audited traces."""
import json,hashlib,math,re,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
S=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n')
T=[read(S/f'TRAINING{i}.json') for i in (1,2,3)]
G=[read(S/f'SELF{i}_VERIFIED.json') for i in (1,2,3)]
A=read(S/'AUDIT60.json');R=read(S/'MATCHED_ROOTS60.json')['rows'];arms=[read(S/f'{tag.upper()}_ARENA.json') for tag in ['a10b32','a10b128','a21b32','a21b128','a32b32','a32b128','a30b32','a30b128']]
assert len(R)==384 and A['total']['games']==1592
offset=[0,300,600];ends=[300,600,1400];gens=np.arange(1,4)
def walltime(i):
 s=(S/f'logs/self{i}-time.log').read_text()
 v=re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)',s).group(1)
 return sum(float(x)*60**k for k,x in enumerate(reversed(v.split(':'))))
stats=[]
for i,(t,g) in enumerate(zip(T,G),1):
 a=g['games'];w=collections.Counter(q['winner'] for q in a);stats.append(dict(generation=i,games=len(a),train_positions=t['train_positions'],val_positions=t['val_positions'],mean_plies=float(np.mean([x['plies'] for x in a])),mean_visit_entropy=float(np.mean([x['entropy'] for x in a])),black_wins=w[1],white_wins=w[-1],draws=w[0],generation_wall_seconds=walltime(i),games_per_second=len(a)/walltime(i),backups=g['totals']['backups'],evaluator_calls=g['totals']['evaluator_calls']))
series=collections.defaultdict(list)
for i,t in enumerate(T):
 for l in t['losses']:
  for k in ['policy_loss','value_loss','total_loss','learning_rate']:series[k].append([offset[i]+l['step'],l[k]])
 for c in t['cuts']:
  for split in ['train','validation']:
   for k,v in c[split].items():series[f'custom_{split}_{k}'].append([offset[i]+c['step'],v])
 a=next(x for x in arms if x['new']==i+1 and x['old']==i and x['budget']==128)
 series['win_rate'].append([ends[i],a['score']])
 series['policy_entropy'].append([ends[i],stats[i]['mean_visit_entropy']])
 series['avg_game_length'].append([ends[i],stats[i]['mean_plies']])
 series['games_per_second'].append([ends[i],stats[i]['games_per_second']])
 series['custom_train_positions'].append([ends[i],t['train_positions']])
 series['custom_validation_positions'].append([ends[i],t['val_positions']])
dump(S/'TRAINING_METRICS_EXPORT60.json',dict(schema='original export_for_plotting metric->list of [iteration,value], wrapped with explicit scope',iteration='cumulative new optimizer updates; validation curves have boundaries, do not connect across new datasets',metrics=dict(series),metric_definitions=dict(policy_entropy='mean per-game Shannon entropy of actual root visit distribution N/B; equal game weights',win_rate='new checkpoint vs immediately preceding checkpoint, B128, 48 paired openings (96 games); no best-player gate',games_per_second='observed native game command walltime including JSON output; local finite benchmark',loss='per-batch policy cross entropy, value MSE, unregularized total; AdamW weight decay separate'),unavailable=['elo_rating','elo_change','gpu_memory_mb','physical_thermodynamic_targets']))
# Same openings, same budget; descriptive paired block comparison.
fields=['entropy_visit','entropy_prior','KL_visit_prior','score_energy','mark_variance','within','between','effective_actions','top_share']
by={}
for B in (32,128):
 for g in range(4):
  z=[x for x in R if x['budget']==B and x['generation']==g]
  assert len(z)==48
  by[f'{B}/{g}']={k:dict(mean=float(np.mean([q[k] for q in z])),sd=float(np.std([q[k] for q in z],ddof=1))) for k in fields}
diffs={};rng=np.random.default_rng(600611);boot=rng.integers(0,48,(10000,48))
for B in (32,128):
 z0={tuple(x['opening']):x for x in R if x['budget']==B and x['generation']==0};z3={tuple(x['opening']):x for x in R if x['budget']==B and x['generation']==3};assert z0.keys()==z3.keys()
 for k in fields:
  d=np.array([z3[q][k]-z0[q][k] for q in sorted(z0)]);diffs[f'{B}/{k}']=dict(mean=float(d.mean()),bootstrap_95=np.quantile(d[boot].mean(1),[.025,.975]).tolist())
# Paired budget dependence: same opening blocks, estimate B128 minus B32.
arena_diffs={}
for new,old in [(1,0),(2,1),(3,2),(3,0)]:
 a,b=[next(x for x in arms if (x['new'],x['old'],x['budget'])==(new,old,B)) for B in (32,128)]
 assert [x['opening'] for x in a['games']]==[x['opening'] for x in b['games']]
 d=np.array(b['pair_scores'])-np.array(a['pair_scores']);arena_diffs[f'{new}/{old}']=dict(mean=float(d.mean()),bootstrap_95=np.quantile(d[boot].mean(1),[.025,.975]).tolist())
summary=dict(stats=stats,matched_root_means=by,matched_g3_minus_g0=diffs,arena_budget_differences=arena_diffs,scope='exploratory paired summaries of native dimensionless scores, occupancy and actual backed-up returns; no temperature/quantum/critical exponent',sources={f.name:sha(f) for f in [S/'AUDIT60.json',S/'MATCHED_ROOTS60.json']+[S/f'TRAINING{i}.json' for i in (1,2,3)]+[S/f'SELF{i}_VERIFIED.json' for i in (1,2,3)]+list(S.glob('A*B*_ARENA.json'))})
dump(S/'PHYSICS_RESULTS.json',summary)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,'axes.grid':True,'grid.alpha':.2})
manifest=[];colors=['#4979a4','#39876c','#c18543']
def save(fig,name,ids):
 fig.tight_layout();base=S/'plots'/name
 fig.savefig(base.with_suffix('.png'),dpi=170);fig.savefig(base.with_suffix('.pdf'));plt.close(fig)
 with Image.open(base.with_suffix('.png')) as im:im.verify()
 assert base.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
 manifest.append(dict(figure=name,ids=ids,png_sha=sha(base.with_suffix('.png')),pdf_sha=sha(base.with_suffix('.pdf'))))
# Dedicated upgraded original combined training curve; no fabricated Elo subplot.
fig,ax=plt.subplots(2,3,figsize=(13.5,8))
for i,t in enumerate(T):
 x=np.array([q['step'] for q in t['losses']])+offset[i];v=np.array([q['total_loss'] for q in t['losses']]);ax[0,0].plot(x,v,alpha=.2,c=colors[i]);w=20
 ax[0,0].plot(x[w-1:],np.convolve(v,np.ones(w)/w,'valid'),c=colors[i],label=f'g{i+1} (20-step mean)')
 for k,ls in [('policy_loss','-'),('value_loss','--')]:
  q=t['cuts'];ax[0,1].plot([z['step']/q[-1]['step'] for z in q],[z['validation'][k] for z in q],ls=ls,marker='o',c=colors[i],label=f'g{i+1} {k.split("_")[0]}')
for a in ax[0,:2]:a.legend(fontsize=8)
ax[0,0].set(xlabel='Cumulative optimizer step',ylabel='Minibatch total loss',title='Actual training')
ax[0,1].set(xlabel='Fraction of iteration updates',ylabel='Held-out loss',title='Within-iteration validation')
for B,style in [(32,'o-'),(128,'s--')]:
 z=[next(q for q in arms if q['new']==i and q['old']==i-1 and q['budget']==B) for i in gens];y=np.array([x['score'] for x in z]);ci=np.array([x['paired_bootstrap_95'] for x in z])
 ax[0,2].errorbar(gens,y,yerr=np.vstack([y-ci[:,0],ci[:,1]-y]),fmt=style,capsize=3,label=f'B={B}')
ax[0,2].axhline(.5,c='gray',ls=':');ax[0,2].set(ylim=(.35,.85),xticks=gens,xlabel='New generation',ylabel='Paired arena score',title='Against preceding checkpoint');ax[0,2].legend()
ax[1,0].step([0,300,600,1400],[.001,.0007,.0007,.0007],where='post');ax[1,0].set(yscale='log',xlabel='Optimizer step',ylabel='Learning rate',title='Registered schedule')
ax[1,1].bar(gens,[x['mean_visit_entropy'] for x in stats],color=colors);ax[1,1].set(xticks=gens,xlabel='Generation',ylabel='Visit entropy (nat)',title='Actual self-play entropy')
ax[1,2].bar(gens,[x['mean_plies'] for x in stats],color=colors);ax[1,2].set(xticks=gens,xlabel='Generation',ylabel='Mean full game plies',title='Self-play game length')
fig.suptitle('RESTORE60: actual iterative learning (9x9, single-orientation evaluator)',fontsize=15)
save(fig,'L47_training_curves',['L46','L47'])
fig,ax=plt.subplots(1,3,figsize=(12,4))
for i,t in enumerate(T):
 q=t['cuts'];x=[z['step'] for z in q]
 for a,key in zip(ax,['policy_loss','value_loss','total_loss']):
  a.plot(x,[z['train'][key] for z in q],c=colors[i],ls='--',marker='.',label=f'g{i+1} train')
  a.plot(x,[z['validation'][key] for z in q],c=colors[i],ls='-',marker='o',label=f'g{i+1} held-out')
for a,t in zip(ax,['Policy CE','Terminal value MSE','Total loss']):a.set(xlabel='Iteration-local optimizer step',ylabel=t,title=t)
ax[2].legend(fontsize=7,ncol=2)
save(fig,'L46_loss_calibration',['L46','L47'])
fig,ax=plt.subplots(1,2,figsize=(11,4.2));names=['g1/g0','g2/g1','g3/g2','g3/g0'];x=np.arange(4)
for j,B in enumerate([32,128]):
 z=[next(q for q in arms if (q['new'],q['old'],q['budget'])==(n,o,B)) for n,o in [(1,0),(2,1),(3,2),(3,0)]]
 y=np.array([q['score'] for q in z]);ci=np.array([q['paired_bootstrap_95'] for q in z]);ax[0].errorbar(x+(j-.5)*.16,y,yerr=np.vstack([y-ci[:,0],ci[:,1]-y]),fmt=['o','s'][j],capsize=3,label=f'B={B}')
 ax[1].bar(x+(j-.5)*.35,[sum(q['pair_scores'][i]==1 for i in range(48)) for q in z],width=.35,label=f'B={B}')
ax[0].axhline(.5,c='gray',ls=':');ax[0].set(xticks=x,xticklabels=names,ylim=(.35,.9),ylabel='Score (48 paired blocks)',title='New checkpoint vs reference');ax[0].legend()
ax[1].set(xticks=x,xticklabels=names,ylabel='Pairs winning both colors',title='Opening-block outcomes');ax[1].legend()
save(fig,'L47_independent_arena',['L46','L47'])
fig,ax=plt.subplots(1,3,figsize=(11,4))
ax[0].bar(gens,[x['games'] for x in stats],color=colors);ax[0].set(xticks=gens,yscale='log',xlabel='Generation',ylabel='Completed self-play games',title='Scale')
ax[1].bar(gens,[x['black_wins'] for x in stats],label='Black');ax[1].bar(gens,[x['white_wins'] for x in stats],bottom=[x['black_wins'] for x in stats],label='White');ax[1].set(xticks=gens,xlabel='Generation',ylabel='Wins',title='Actual game outcomes');ax[1].legend()
ax[2].bar(gens,[x['games_per_second'] for x in stats],color=colors);ax[2].set(xticks=gens,xlabel='Generation',ylabel='Games per second',title='Native command incl. I/O')
save(fig,'L46_selfplay_resources',['L46','L47'])
fig,ax=plt.subplots(2,3,figsize=(13,7));keys=['entropy_visit','entropy_prior','KL_visit_prior','score_energy','mark_variance','effective_actions'];titles=['Visit Shannon entropy (nat)','Prior Shannon entropy (nat)','KL(visit || prior) (nat)','Negative mean root mark','Mark population variance','Inverse Simpson actions']
for a,key,title in zip(ax.ravel(),keys,titles):
 for B,style in [(32,'o-'),(128,'s--')]:
  a.plot(range(4),[by[f'{B}/{g}'][key]['mean'] for g in range(4)],style,label=f'B={B}')
 a.set(xlabel='Checkpoint generation',xticks=range(4),title=title);a.legend(fontsize=8)
fig.suptitle('384 deduplicated matched initial roots; native dimensionless observables',fontsize=14)
save(fig,'L47_matched_native_observables',['L47'])
dump(S/'PLOT_MANIFEST60.json',dict(figures=manifest,sources={k:sha(S/k) for k in ['TRAINING_METRICS_EXPORT60.json','PHYSICS_RESULTS.json','AUDIT60.json','MATCHED_ROOTS60.json']}))
# Contact sheet solely for visual QA, excluded from plot count.
imgs=[]
for e in manifest:
 with Image.open(S/'plots'/f"{e['figure']}.png") as im:
  im=im.convert('RGB');im.thumbnail((650,440));imgs.append(im.copy())
sheet=Image.new('RGB',(1300,1320),'white')
for i,im in enumerate(imgs):sheet.paste(im,(i%2*650,i//2*440))
sheet.save(S/'logs/plot_qa.jpg',quality=85)
print('PLOT_PAIRS',len(manifest),'STATS',stats,'MATCHED_G3_G0',diffs,'ARENA_BUDGET',arena_diffs)
