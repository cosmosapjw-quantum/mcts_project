from base61 import *
from itertools import product
from qec89 import post, masks, SYN, synd

O67=B/'mqft_restore67_20261126';O68=B/'mqft_restore68_20261126'
D=read(S/'NATIVE89.json');Q=read(S/'QEC89.json');P67=read(O67/'PHYSICS_RESULTS.json');P68=read(O68/'PHYSICS_RESULTS.json')
def aggregate(z,arm,groups,seed):
 f=np.array([x['fault'] for x in z]);m=np.array([x['mask'] for x in z]);g=np.asarray(groups);s=f.sum(0);assert s[0]>0 and abs(m[:,:,0].sum()-s[0])==0
 p=0. if arm in ('clean','reference') else .12;mu=1. if arm=='rep_common' else .5 if arm.endswith('_mix') else 0.
 ref=post(p,mu);sr=[];ma=m.sum(0);isrep=arm.startswith('rep') or arm=='recovery'
 for r in ref['syndromes']:
  sx=tuple(r['syndrome']);ix=[i for i,a in enumerate(masks) if synd(a)==sx];a=ma[ix].sum(0);bad=sum(ma[i,0] for i in ix if sum(masks[i])>0)
  pred=r['majority_residual'] if isrep else (sum((1-mu)*p**sum(masks[i])*(1-p)**(3-sum(masks[i]))*masks[i][0]+mu*(p if sum(masks[i])==3 else 0.) for i in ix)/r['prob'] if r['prob'] else None)
  sr.append(dict(syndrome=list(sx),n=int(a[0]),fault_posterior=float(bad/a[0]) if a[0] else None,residual=float(a[1]/a[0]) if a[0] else None,model_fault_posterior=r['fault_posterior'],model_residual=pred))
 rr=np.array([f[g==k].sum(0) for k in np.unique(g)]);rng=np.random.default_rng(seed);boot=rr[rng.integers(0,len(rr),(10000,len(rr)))].sum(1)
 metrics={}
 for name,num,den in [('sign_error',4,0),('value_MAE',5,0),('syndrome_rate',3,0),('success_given_syndrome',6,3),('triple_rate',7,0)]:
  good=boot[:,den]>0;metrics[name]=dict(mean=float(s[num]/s[den]) if s[den] else None,ci95=np.quantile(boot[good,num]/boot[good,den],[.025,.975]).tolist() if good.any() else None,n=int(s[den]))
 return dict(arm=arm,paths=len(z),blocks=len(rr),n=int(s[0]),metrics=metrics,syndromes=sr,mask_counts=ma[:,0].astype(int).tolist(),model_error=ref['majority_error'] if isrep else p,carrier_rate=float(s[1]/(3*s[0])),first_rate=float(s[2]/s[0]),root_fault_sums=rr.tolist(),per_search_value_MAE=(f[:,5]/f[:,0]).tolist())
roots=[];games=[]
for g in (0,3):
 for arm in ('clean','single_iid','rep_iid','single_mix','rep_mix','rep_common'):
  z=[x for x in D['root_records'] if x['generation']==g and x['arm']==arm];assert len(z)==256
  r=aggregate(z,arm,[x['core'] for x in z],890001);r['generation']=g
  prev=next(x for x in P67['rows'] if x['generation']==g and x['arm']==arm)['fault']
  for k,n in [('sign_error','residual_rate'),('value_MAE','value_MAE'),('syndrome_rate','syndrome_rate'),('triple_rate','triple_rate'),('success_given_syndrome','success_given_syndrome')]:
   a,b=r['metrics'][k]['mean'],prev[n]['mean'];assert (a is None and b is None) or abs(a-b)<1e-12
  roots.append(r)
for co in ('external','internal'):
 for arm in ('clean','single','recovery'):
  z=[];ids=[]
  for x in D['game_records']:
   if (x['match'].startswith('ext_'))!=(co=='external'):continue
   for r in x['records']:
    if r['arm']==arm:z.append(r);ids.append(x['core'])
  r=aggregate(z,arm,ids,890002);r['cohort']=co
  prev=next(x for x in P68['native'] if x['cohort']==co and x['arm']==arm)
  for k,n in [('sign_error','sign_error_rate'),('value_MAE','value_MAE'),('syndrome_rate','syndrome_rate')]:assert abs(r['metrics'][k]['mean']-prev[n]['mean'])<1e-12
  games.append(r)
# independently reconstruct recorded high-budget and same-board proxy outcome from native recommendation.
reference={}
for x in D['root_records']:
 if x['arm']=='reference':reference.setdefault(x['core'],[]).append(x)
with np.load(O67/'REFERENCE67.npz') as f:
 N=f['counts'].reshape(32,4,81).sum(1);W=f['marks'].reshape(32,4,81).sum(1)
top=np.argsort(-N,axis=1,kind='stable')[:,:5];external=read(O67/'RAPFI67.json');proxy={x['core']:x['result']['action'] for x in external['rows'] if x['L']==9 and x['ms']==350};assert len(proxy)==32
utility=[]
for r in roots:
 z=[x for x in D['root_records'] if x['generation']==r['generation'] and x['arm']==r['arm']];a=np.array([np.mean([x['recommendation'] in top[k] for x in z if x['core']==k]) for k in range(32)]);b=np.array([np.mean([x['recommendation']==proxy[k] for x in z if x['core']==k]) for k in range(32)])
 old=next(x for x in P67['rows'] if x['generation']==r['generation'] and x['arm']==r['arm'])['utility'];assert abs(a.mean()-old['top5']['mean'])<1e-12 and abs(b.mean()-old['rapfi9']['mean'])<1e-12
 utility.append(dict(generation=r['generation'],arm=r['arm'],top5=a.tolist(),external9=b.tolist()))
# 80 registered games, preserve incomplete bounds; exact paired block and sign-flip test.
by={}
for x in D['game_records']:
 by.setdefault(x['match'],np.full((8,2),np.nan))
 if x['status']=='complete':by[x['match']][x['core'],0 if x['color']==1 else 1]=.5 if x['winner']==0 else float(x['winner']==x['color'])
results=[]
for k,A in by.items():
 a=next(x for x in P68['results'] if x['match']==k);assert np.array_equal(np.nan_to_num(A,nan=-1),np.array([[v if v is not None else -1 for v in r] for r in a['matrix']]))
 bounds=[np.nansum(A)/16,(np.nansum(A)+np.isnan(A).sum())/16];assert bounds==a['all_registered_score_bounds']
 results.append(dict(match=k,complete=int(np.isfinite(A).sum()),bounds=bounds,paired_scores=[float(v.mean()) for v in A if np.isfinite(v).all()]))
contrasts=[]
for r in P68['contrasts']:
 A=by[r['left']];B1=by[r['right']] if r['right'] in by else np.full_like(A,.5);ix=np.isfinite(A).all(1)&np.isfinite(B1).all(1);v=(A[ix]-B1[ix]).mean(1);p=np.mean([abs(np.mean(v*np.array(s)))>=abs(v.mean())-1e-14 for s in product([-1,1],repeat=len(v))])
 assert abs(v.mean()-r['complete_opening_contrast']['mean'])<1e-12 and abs(p-r['sign_flip_p'])<1e-12
 contrasts.append(dict(left=r['left'],right=r['right'],block_differences=v.tolist(),prior=r))
dump(S/'SUMMARY89.json',dict(root_rows=roots,game_rows=games,utility=utility,prior_R67_contrasts=P67['contrasts'],game_results=results,game_contrasts=contrasts,raw_paths=3200,registered_games=80,complete_games=78,eval_records=579026,root_backups=524288,game_backups=54592,new_fixed_roots=0,cumulative_fixed_roots=93336,scope='Recomputed observed fault rates and conditional syndrome calibration, actual game score matrix and prior proxy agreement. Root/opening-block bootstrap, not iid evaluation intervals.'))
print('SUMMARY',len(roots),len(games),len(utility),len(results),len(contrasts),flush=True)
