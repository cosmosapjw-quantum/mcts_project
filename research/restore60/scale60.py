"""Separate prospective extension registered after initial two-generation pilot.
Fixed g2 input, deeper B128 data; no tuning on pilot arena. Cross-budget paired arena.
"""
from train60 import *
import concurrent.futures as cf
import multiprocessing as mp
def setup():
 r=read(S/'REGISTRATION.json')
 assert not (S/'SCALE_REGISTRATION.json').exists()
 # Correct metadata typo in original preregistration, not data/weights identity.
 actual=R/'checkpoints/board_2301_120.pt'
 assert sha(actual)==r['initial_checkpoint_sha']==sha(S/'checkpoints/g0.pt')
 assert sha(R/'checkpoints/board_2301_60.pt')==load(actual)[1]['parent_sha256']
 dump(S/'REGISTRATION_CORRECTION.json',dict(original_registration_sha=sha(S/'REGISTRATION.json'),field='initial_source',recorded=r['initial_source'],actual=str(actual.relative_to(B)),identity_verified=r['initial_checkpoint_sha'],effect='path variable overwritten by plan loop; exact checkpoint bytes and ancestry already validated; no change to training inputs or results'))
 seen=set()
 for k in ('self1','self2','arena1','arena2'):
  for j in read(S/f'plans/{k}.json'):seen.add(orbit(planes(board(j['opening']),1)))
 rng=np.random.default_rng(600505)
 def opening():
  while True:
   # Matched four-ply central opening distribution, independent orbits.
   mv=rng.choice([y*9+x for y in range(2,7) for x in range(2,7)],4,replace=False).tolist()
   k=orbit(planes(board(mv),1))
   if k not in seen:seen.add(k);return mv
 jobs=[]
 for typ,n in [('train',512),('val',128)]:
  for i in range(n):
   jobs.append(dict(id=f'g3_{typ}_{i:04}',opening=opening(),seed=int(rng.integers(0,2**63)),budget=128,training=True,swap=False))
 plan=S/'plans/self3.json';dump(plan,jobs);plans={plan.name:sha(plan)}
 ops=[(opening(),int(rng.integers(0,2**63))) for i in range(48)]
 arms=[]
 for new,old in [(1,0),(2,1),(3,2),(3,0)]:
  for budget in (32,128):
   tag=f'a{new}{old}b{budget}'
   jobs=[dict(id=f'{tag}_{i:03}_{int(swap)}',opening=op,seed=sd,budget=budget,training=False,swap=swap) for i,(op,sd) in enumerate(ops) for swap in (False,True)]
   f=S/f'plans/{tag}.json';dump(f,jobs);plans[f.name]=sha(f);arms.append(dict(tag=tag,new=new,old=old,budget=budget))
 reg=dict(scope='separately prospective large-scale extension after two-round pilot, no tuning on arena',parent_registration_sha=sha(S/'REGISTRATION.json'),input_checkpoint_sha=sha(S/'checkpoints/g2.pt'),input_weights_sha=sha(S/'checkpoints/g2.bin'),binary_sha=sha(S/'mqft_train'),script_sha=sha(S/'scale60.py'),train_script_sha=sha(S/'train60.py'),plans=plans,arms=arms,train_games=512,val_games=128,budget=128,arena_openings=48,workers=14,audit_workers=10,optimizer_steps=800,batch=64,learning_rate=.0007,weight_decay=1e-4,cut_steps=[0,100,200,400,800],seed=600505,noise=r['noise'],data_scope='same 4-ply opening design, disjoint orbits, new seeds; current generation only; unchanged architecture and optimizer type',arena_scope='same independent 48 opening blocks for all four checkpoint pairs and two budgets; color swapped; no training or retuning')
 dump(S/'SCALE_REGISTRATION.json',reg);print('SCALE_REGISTERED',sha(S/'SCALE_REGISTRATION.json'),flush=True)
def check_worker(arg):
 path,j=arg
 return verify_game(path,j)
def audit(tag):
 r=read(S/'SCALE_REGISTRATION.json');p=S/f'plans/{tag}.json';assert sha(p)==r['plans'][p.name]
 js=read(p);folder=S/f'data/{tag}';assert len(list(folder.glob('*.json')))==len(js)
 # No Torch inference in worker: independent board/rule/trace verification.
 with cf.ProcessPoolExecutor(max_workers=r['audit_workers'],mp_context=mp.get_context('fork')) as pool:
  rows=list(pool.map(check_worker,[(folder/(j['id']+'.json'),j) for j in js],chunksize=4))
 ss=[q['summary'] for q in rows]
 dump(S/f'{tag.upper()}_VERIFIED.json',dict(tag=tag,plan_sha=sha(p),games=ss,totals={k:sum(x[k] for x in ss) for k in ['searched','backups','evaluator_calls','bytes']},n=len(ss)))
 print('AUDITED',tag,len(ss),sum(x['searched'] for x in ss),sum(x['backups'] for x in ss),flush=True)
 return rows
def fit3():
 r=read(S/'SCALE_REGISTRATION.json');assert sha(S/'checkpoints/g2.pt')==r['input_checkpoint_sha'] and sha(S/'checkpoints/g2.bin')==r['input_weights_sha']
 rows=audit('self3');train=[q for q in rows if '_train_' in q['summary']['id']];val=[q for q in rows if '_val_' in q['summary']['id']]
 assert len(train)==512 and len(val)==128
 tr=tensors(train);orbits={orbit(x) for x in tr[0]};va=tensors(val,orbits);assert not orbits&{orbit(x) for x in va[0]}
 m,c=load(S/'checkpoints/g2.pt');opt=torch.optim.AdamW(m.parameters(),lr=r['learning_rate'],weight_decay=r['weight_decay']);opt.load_state_dict(c['optimizer']);opt.param_groups[0]['lr']=r['learning_rate']
 x,p,z,g,_=tr;groups=[np.flatnonzero(g==i) for i in np.unique(g)];rng=np.random.default_rng(600506);torch.manual_seed(600507);losses=[];cuts=[];start=time.perf_counter()
 for t in range(r['optimizer_steps']+1):
  if t in r['cut_steps']:
   cuts.append(dict(step=t,train=metrics(m,tr),validation=metrics(m,va)));print('CUT3',cuts[-1],flush=True)
  if t==r['optimizer_steps']:break
  ix=np.array([rng.choice(groups[j]) for j in rng.integers(0,len(groups),size=r['batch'])]);pp=PERM[rng.integers(0,8,len(ix))]
  xx=np.stack([x[k].reshape(2,N)[:,a].reshape(2,L,L) for k,a in zip(ix,pp)]);yy=np.stack([p[k,a] for k,a in zip(ix,pp)])
  m.train();l,v=m(torch.from_numpy(xx));lp=-(torch.from_numpy(yy)*F.log_softmax(l,1)).sum(1).mean();lv=F.mse_loss(v,torch.from_numpy(z[ix]));loss=lp+lv
  opt.zero_grad();loss.backward();opt.step();losses.append(dict(step=t+1,policy_loss=float(lp.detach()),value_loss=float(lv.detach()),total_loss=float(loss.detach()),learning_rate=opt.param_groups[0]['lr']))
 ck=S/'checkpoints/g3.pt';prev=S/'checkpoints/g2.pt';assert not ck.exists()
 torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),generation=3,steps=r['optimizer_steps'],parent_sha256=sha(prev),registration_sha256=sha(S/'SCALE_REGISTRATION.json'),dataset_sha256=sha(S/'SELF3_VERIFIED.json'),rng_state_json=json.dumps(rng.bit_generator.state)),ck)
 bp=export(ck);re,c2=load(ck);assert all(torch.equal(v,c2['model'][k]) for k,v in m.state_dict().items()) and c2['parent_sha256']==sha(prev)
 np.savez_compressed(S/'TRAIN_DATA3.npz',x=x,pi=p,z=z,game=g,val_x=va[0],val_pi=va[1],val_z=va[2],val_game=va[3])
 out=dict(generation=3,train_positions=len(x),val_positions=len(va[0]),val_excluded=va[4],train_unique_orbits=len(orbits),train_outcomes=dict(collections.Counter(str(q['summary']['winner']) for q in train)),val_outcomes=dict(collections.Counter(str(q['summary']['winner']) for q in val)),train_value_targets=dict(collections.Counter(str(int(v)) for v in z)),validation_games_retained=len(set(va[3])),losses=losses,cuts=cuts,checkpoint_sha=sha(ck),weights_sha=sha(bp),parent_sha=sha(prev),seconds=time.perf_counter()-start)
 dump(S/'TRAINING3.json',out);print('TRAINED3',len(x),len(va[0]),cuts[0],cuts[-1],flush=True)
 infer_check(3)
def arena_audit(tag):
 r=read(S/'SCALE_REGISTRATION.json');arm=next(x for x in r['arms'] if x['tag']==tag);rows=audit(tag);js=read(S/f'plans/{tag}.json');out=[]
 for j,q in zip(js,rows):
  s=q['summary'];newside=-1 if j['swap'] else 1;win=s['winner'];score=.5 if win==0 else float(win==newside)
  out.append(dict(id=j['id'],opening=j['opening'],swap=j['swap'],winner=win,new_score=score,plies=s['plies']))
 pairs=np.array([np.mean([out[k]['new_score'],out[k+1]['new_score']]) for k in range(0,len(out),2)]);rng=np.random.default_rng(600508);bs=pairs[rng.integers(0,len(pairs),(10000,len(pairs)))].mean(1)
 result=dict(**arm,n=len(out),games=out,W=sum(x['new_score']==1 for x in out),D=sum(x['new_score']==.5 for x in out),L=sum(x['new_score']==0 for x in out),score=float(pairs.mean()),pair_scores=pairs.tolist(),paired_bootstrap_95=np.quantile(bs,[.025,.975]).tolist(),scope='conditional paired opening resampling, shared blocks across arms; no Elo')
 dump(S/f'{tag.upper()}_ARENA.json',result);print('ARENA',tag,{k:result[k] for k in ['W','D','L','score','paired_bootstrap_95']},flush=True)
if __name__=='__main__':
 op=sys.argv[1]
 if op=='init':setup()
 elif op=='fit3':fit3()
 else:arena_audit(sys.argv[2])
