"""Independent second pass over retained receipts, sampled actual evaluator states,
common initial root physics and optimizer determinism. No fitting/tuning on arena."""
from train60 import *
from collections import defaultdict
def compare_states():
 reg=read(S/'REGISTRATION.json');scale=read(S/'SCALE_REGISTRATION.json')
 assert reg['source_sha']==sha(S/'train60.py') and scale['script_sha']==sha(S/'scale60.py') and scale['train_script_sha']==sha(S/'train60.py')
 assert reg['binary_sha']==scale['binary_sha']==sha(S/'mqft_train')
 assert sha(S/'REGISTRATION.json')==scale['parent_registration_sha']
 planmap={**reg['plans'],**scale['plans']}
 assert all(sha(S/'plans'/p)==h for p,h in planmap.items())
 arms={x['tag']:(x['new'],x['old']) for x in scale['arms']}
 models={f'self{i}':(i-1,i-1) for i in (1,2,3)}|{f'arena{i}':(i,i-1) for i in (1,2)}|arms
 pts=[S/f'checkpoints/g{i}.pt' for i in range(4)];bins=[p.with_suffix('.bin') for p in pts]
 nets=[load(p)[0] for p in pts]
 lineage=[]
 for i,(p,b) in enumerate(zip(pts,bins)):
  _,c=load(p);r=read(S/f'INFERENCE{i}.json');assert sha(p)==r['checkpoint_sha'] and sha(b)==r['weights_sha']
  want=np.concatenate([c['model'][k].detach().numpy().ravel() for k in KEYS]).astype('<f4').tobytes()
  assert b.read_bytes()==want
  if i:
   assert c['parent_sha256']==sha(pts[i-1]) and c['dataset_sha256']==sha(S/f'SELF{i}_VERIFIED.json')
   assert c['registration_sha256']==sha(S/('REGISTRATION.json' if i<3 else 'SCALE_REGISTRATION.json'))
  lineage.append(dict(generation=i,checkpoint_sha=sha(p),weights_sha=sha(b),parent_sha=c['parent_sha256']))
 rng=np.random.default_rng(600601);checks=defaultdict(list);roots={};counts={};total=dict(games=0,searched=0,backups=0,evaluator_calls=0,bytes=0)
 for tag,pair in sorted(models.items()):
  plan=S/f'plans/{tag}.json';jobs=read(plan);z=read(S/f'{tag.upper()}_VERIFIED.json');assert z['plan_sha']==sha(plan) and z['n']==len(jobs)==len(z['games'])
  assert [x['id'] for x in z['games']]==[j['id'] for j in jobs]
  selected=set(rng.choice(len(jobs),min(8,len(jobs)),replace=False).tolist())
  for i,(j,g) in enumerate(zip(jobs,z['games'])):
   p=S/f'data/{tag}/{j["id"]}.json';assert p.stat().st_size==g['bytes'] and sha(p)==g['sha256']
   if i not in selected and not (tag in arms and not j['swap']):continue
   d=read(p);rows=d['rows']
   if tag in arms and not j['swap']:
    # Unswapped black is new checkpoint. Add also swapped black below.
    pass
   if i in selected:
    ix=rng.choice(len(rows),min(2,len(rows)),replace=False)
    for k in ix:
     r=rows[k];b=np.asarray(r['board'],np.int8);side=r['side'];gen=pair[r['model']]
     checks[gen].append((planes(b,side),np.asarray(r['prior']),r['evaluations'][0]['value'],'root'))
     cand=[e for e in r['events'] if e['eval_id'] is not None]
     for e in [cand[i] for i in rng.choice(len(cand),min(4,len(cand)),replace=False)]:
      bb=b.copy();s=side
      for a in e['actions']:assert bb[a]==0;bb[a]=1 if s==1 else 2;s=-s
      checks[gen].append((planes(bb,s),None,e['leaf_value'],'leaf'))
  # Deduplicate common first root with same model, budget, opening, seed; both colors.
  if tag in arms:
   arm=next(a for a in scale['arms'] if a['tag']==tag)
   for j in jobs:
    d=read(S/f'data/{tag}/{j["id"]}.json');q=d['rows'][0];gen=pair[q['model']];key=(arm['budget'],tuple(j['opening']),gen)
    vals={k:q[k] for k in ('counts','prior','w','s2','search_seed','state_hash')}
    if key in roots:
     assert vals==roots[key],key
    else:roots[key]=vals
  counts[tag]={'games':z['n'],**z['totals']}
  total['games']+=z['n']
  for k,v in z['totals'].items():total[k]+=v
  print('HASH_AUDIT',tag,len(jobs),flush=True)
 assert total['games']==1592 and total['backups']==1132064 and len(roots)==4*2*48
 errs={}
 with torch.no_grad():
  for gen,rows in checks.items():
   X=np.stack([x[0] for x in rows]);l,v=nets[gen](torch.from_numpy(X));p=torch.softmax(l,1).numpy();v=v.numpy()
   pe=max(float(np.max(np.abs(p[i]-z[1]))) for i,z in enumerate(rows) if z[1] is not None)
   ve=float(np.max(np.abs(v-np.asarray([z[2] for z in rows]))))
   assert pe<3e-6 and ve<3e-5
   errs[str(gen)]=dict(n=len(rows),roots=sum(x[3]=='root' for x in rows),leaves=sum(x[3]=='leaf' for x in rows),max_policy_error=pe,max_value_error=ve)
 d4={}
 with torch.no_grad():
  fx=read(S/'plans/inference.json');X=np.stack([planes(board(m),1 if len(m)%2==0 else -1) for m in fx]);a=X.reshape(-1,2,N)
  for i,net in enumerate(nets):
   l,v=net(torch.from_numpy(X));p0=torch.softmax(l,1).numpy();v0=v.numpy();ps=[];vs=[]
   for perm in PERM:
    l,v=net(torch.from_numpy(a[:,:,perm].reshape(-1,2,L,L).copy()));ps.append(torch.softmax(l,1).numpy()[:,np.argsort(perm)]);vs.append(v.numpy())
   pbar=np.mean(ps,axis=0);vbar=np.mean(vs,axis=0)
   d4[str(i)]=dict(mean_policy_TV=float(np.mean(np.sum(abs(p0-pbar),axis=1)/2)),max_policy_TV=float(np.max(np.sum(abs(p0-pbar),axis=1)/2)),mean_abs_value=float(np.mean(abs(v0-vbar))),max_abs_value=float(np.max(abs(v0-vbar))))
 rr=[]
 for (budget,opening,gen),a in sorted(roots.items()):
  n=np.array(a['counts'],float);p=np.array(a['prior']);w=np.array(a['w']);s2=np.array(a['s2']);pi=n/budget;use=n>0;q=np.divide(w,n,out=np.zeros_like(w),where=use)
  mean=w.sum()/budget;var=s2.sum()/budget-mean**2
  within=(s2[use]-n[use]*q[use]**2).sum()/budget
  between=(pi*(q-mean)**2).sum();assert abs(var-within-between)<1e-10 and min(var,within,between)>-1e-10
  assert abs(pi@q-mean)<1e-12
  entropy=lambda x:float(-np.sum(x[x>0]*np.log(x[x>0])))
  rr.append(dict(budget=budget,opening=opening,generation=gen,root_hash=a['state_hash'],seed=a['search_seed'],entropy_visit=entropy(pi),entropy_prior=entropy(p),KL_visit_prior=float(np.sum(pi[use]*np.log(pi[use]/p[use]))),score_energy=float(-mean),mark_variance=float(var),within=float(within),between=float(between),effective_actions=float(1/(pi@pi)),top_share=float(pi.max())))
 dump(S/'MATCHED_ROOTS60.json',dict(scope='48 common initial openings x four checkpoint identities x two budgets; duplicate identical deterministic initial roots removed; no extra searches',rows=rr))
 # Validate cross-generation orbit overlap against previous current-generation training data; no posthoc exclusions.
 prev=set();overlap={}
 for i in (1,2,3):
  a=np.load(S/f'TRAIN_DATA{i}.npz');tr={orbit(x) for x in a['x']};vv=[orbit(x) for x in a['val_x']];assert not tr&set(vv)
  overlap[str(i)]=dict(train_overlap_previous=sum(orbit(x) in prev for x in a['x']),val_overlap_previous=sum(x in prev for x in vv),val_overlap_current=0)
  prev|=tr
 # Recompute exact policy and value loss for stored cuts, independent direct numpy
 # using final torch probabilities. Cross-check after each generation.
 metric_errors={}
 with torch.no_grad():
  for i in (1,2,3):
   a=np.load(S/f'TRAIN_DATA{i}.npz');t=read(S/f'TRAINING{i}.json');err=[]
   for name,prefix in [('train',''),('validation','val_')]:
    X=a[prefix+'x'];pi=a[prefix+'pi'];z=a[prefix+'z'];g=a['game' if not prefix else 'val_game'];v=[];ce=[]
    for k in range(0,len(X),256):
     logits,u=nets[i](torch.from_numpy(X[k:k+256]));p=logits.numpy().astype(np.float64);p-=p.max(1,keepdims=True)
     logz=np.log(np.exp(p).sum(1));ce.extend((logz-(pi[k:k+256]*p).sum(1)).tolist());v.extend(((u.numpy().astype(np.float64)-z[k:k+256])**2).tolist())
    a1=float(np.mean([np.mean(np.asarray(ce)[g==j]) for j in np.unique(g)]));a2=float(np.mean([np.mean(np.asarray(v)[g==j]) for j in np.unique(g)]))
    target=t['cuts'][-1][name];err.extend([abs(a1-target['policy_loss']),abs(a2-target['value_loss'])])
   metric_errors[str(i)]=max(err);assert max(err)<1e-6
 out=dict(total=total,cohorts=counts,lineage=lineage,actual_state_inference=errs,d4_single_vs_ensemble=d4,cross_generation_orbit_overlap=overlap,metric_crosscheck_max_error=metric_errors,matched_initial_roots=len(roots),native_agent='NOT_RUN',scope='all raw SHA/receipts plus stratified actual state inference; previous audit checked every game and every path, not full independent evaluator recomputation of all 871025 calls')
 dump(S/'AUDIT60.json',out);print('AUDIT',json.dumps({k:out[k] for k in ('total','actual_state_inference','d4_single_vs_ensemble','cross_generation_orbit_overlap','metric_crosscheck_max_error','matched_initial_roots')}),flush=True)
def replay():
 # independent exact rerun g3 update path, same training data, not new evidence.
 r=read(S/'SCALE_REGISTRATION.json');a=np.load(S/'TRAIN_DATA3.npz');m,c=load(S/'checkpoints/g2.pt');target=load(S/'checkpoints/g3.pt')[1]
 opt=torch.optim.AdamW(m.parameters(),lr=r['learning_rate'],weight_decay=r['weight_decay']);opt.load_state_dict(c['optimizer']);opt.param_groups[0]['lr']=r['learning_rate']
 x,p,z,g=(a[k] for k in ['x','pi','z','game']);groups=[np.where(g==j)[0] for j in np.unique(g)];rng=np.random.default_rng(600506);torch.manual_seed(600507)
 for step in range(r['optimizer_steps']):
  ix=np.asarray([rng.choice(groups[k]) for k in rng.integers(0,len(groups),size=64)]);pr=PERM[rng.integers(0,8,len(ix))];xx=np.stack([x[j].reshape(2,N)[:,q].reshape(2,L,L) for j,q in zip(ix,pr)]);yy=np.stack([p[j,q] for j,q in zip(ix,pr)])
  m.train();l,v=m(torch.from_numpy(xx));loss=-(torch.from_numpy(yy)*F.log_softmax(l,1)).sum(1).mean()+F.mse_loss(v,torch.from_numpy(z[ix]))
  opt.zero_grad();loss.backward();opt.step()
 def same(a,b):
  if torch.is_tensor(a):return torch.equal(a,b)
  if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
  if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
  return a==b
 ok=same(m.state_dict(),target['model']);op=same(opt.state_dict(),target['optimizer']);assert ok and op
 dump(S/'OPTIMIZER_REPLAY60.json',dict(generation=3,replayed_updates=800,model_bit_exact=ok,optimizer_bit_exact=op,scope='replay not counted as new learning',source_dataset_sha=sha(S/'TRAIN_DATA3.npz'),target_checkpoint_sha=sha(S/'checkpoints/g3.pt')))
 print('REPLAY EXACT model and optimizer',flush=True)
if __name__=='__main__':compare_states() if sys.argv[1]=='audit' else replay()
