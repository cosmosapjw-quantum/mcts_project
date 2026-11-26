"""R60 registered actual two-iteration native selfplay feedback.
Requires existing PyTorch CPU 2.10 and exact canonical feature build.
Never uses arena or validation labels in optimizer."""
from pathlib import Path
import os,sys,math,json,hashlib,subprocess,time,shutil,gzip,collections
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
S=Path(__file__).resolve().parent;B=S.parent;R=B/'mqft_restore23_20261125'
N=81;L=9;PERM=np.array([np.rot90(g,k).ravel() for g in (np.arange(N).reshape(L,L),np.fliplr(np.arange(N).reshape(L,L))) for k in range(4)])
KEYS=['body.0.weight','body.0.bias','body.2.weight','body.2.bias','body.4.weight','body.4.bias','policy.weight','policy.bias','value.1.weight','value.1.bias','value.3.weight','value.3.bias']
SHAPES=[(16,2,3,3),(16,),(16,16,3,3),(16,),(16,16,3,3),(16,),(1,16,1,1),(1,),(24,1296),(24,),(1,24),(1,)]
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,d):
 p=Path(p);q=p.with_name(p.name+'.part')
 with q.open('w') as f:json.dump(d,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 q.replace(p)
class BoardNet(nn.Module):
 def __init__(self):
  super().__init__();self.body=nn.Sequential(nn.Conv2d(2,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.ReLU());self.policy=nn.Conv2d(16,1,1);self.value=nn.Sequential(nn.Flatten(),nn.Linear(1296,24),nn.ReLU(),nn.Linear(24,1),nn.Tanh())
 def forward(self,x):
  h=self.body(x);l=self.policy(h).flatten(1);legal=~(x[:,0]+x[:,1]).flatten(1).bool()
  return l.masked_fill(~legal,-1e9),self.value(h).flatten()
def load(p):
 c=torch.load(p,weights_only=True,map_location='cpu');m=BoardNet()
 assert [(k,tuple(v.shape)) for k,v in c['model'].items()]==list(zip(KEYS,SHAPES));m.load_state_dict(c['model']);m.eval();return m,c
def export(p):
 m,c=load(p);out=Path(p).with_suffix('.bin');data=np.concatenate([c['model'][k].detach().numpy().reshape(-1) for k in KEYS]).astype('<f4');assert len(data)==36114 and np.isfinite(data).all()
 with out.open('wb') as f:f.write(data.tobytes())
 return out
def board(moves):
 a=np.zeros(N,dtype=np.int8)
 for t,m in enumerate(moves):
  assert 0<=m<N and a[m]==0
  a[m]=1+t%2
  if t<len(moves)-1:assert not won_last(a,m)
 return a
def won_last(b,a):
 p=int(b[a]);y,x=divmod(a,L)
 for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
  n=1
  for s in (-1,1):
   yy=y+s*dy;xx=x+s*dx
   while 0<=yy<L and 0<=xx<L and b[yy*L+xx]==p:n+=1;yy+=s*dy;xx+=s*dx
  if n>=5:return True
 return False
def planes(b,side):
 p=1 if side==1 else 2
 return np.stack([b==p,b==3-p]).astype(np.float32).reshape(2,L,L)
def orbit(x):return min(x.reshape(2,N)[:,p].tobytes() for p in PERM)
def init():
 assert not (S/'REGISTRATION.json').exists()
 p=R/'checkpoints/board_2301_120.pt';m,c=load(p)
 assert c['registration_sha256']==sha(R/'registration.json') and c['seed']==2301 and c['step']==120
 assert c['parent_sha256']==sha(R/'checkpoints/board_2301_60.pt')
 cp=S/'checkpoints/g0.pt';shutil.copyfile(p,cp);assert sha(cp)==sha(p);bp=export(cp)
 # disjoint D4 opening geometry across all cohorts; all four random central moves
 rng=np.random.default_rng(600101);seen=set()
 def opening():
  while True:
   mv=rng.choice([y*9+x for y in range(2,7) for x in range(2,7)],4,replace=False).tolist()
   k=orbit(planes(board(mv),1))
   if k not in seen:seen.add(k);return mv
 plans={}
 for it in (1,2):
  jobs=[]
  for typ,n in [('train',48),('val',12)]:
   for i in range(n):
    jobs.append(dict(id=f'g{it}_{typ}_{i:03}',opening=opening(),seed=int(rng.integers(0,2**63)),budget=32,training=True,swap=False))
  p=S/f'plans/self{it}.json';dump(p,jobs);plans[p.name]=sha(p)
  jobs=[]
  for i in range(16):
   op=opening();seed=int(rng.integers(0,2**63))
   for swap in (False,True):jobs.append(dict(id=f'g{it}_arena_{i:03}_{int(swap)}',opening=op,seed=seed,budget=32,training=False,swap=swap))
  p=S/f'plans/arena{it}.json';dump(p,jobs);plans[p.name]=sha(p)
 # fixture legal states across both signs, asymmetric boundaries and known win proximity; no training/arena labels
 fixtures=[[],[0],[0,80],[0,80,1,79],[0,9,1,10,2,11,3,12],[40,0,41,80,42],[80,0,71,1,62,2],[4,40,76,31,36,9,44]]
 for j in range(64):
  mv=[];b=np.zeros(N,dtype=np.int8)
  for t in range(int(rng.integers(1,35))):
   a=int(rng.choice(np.flatnonzero(b==0)));b[a]=1+t%2
   if won_last(b,a) or t==80:break
   mv.append(a)
  fixtures.append(mv)
 dump(S/'plans/inference.json',fixtures)
 reg=dict(parent_commit='7bd48b692fbfcb3ed89bbe1dee052b5a5445b3cf',initial_checkpoint_sha=sha(cp),initial_export_sha=sha(bp),initial_source=str(p.relative_to(B)),initial_training_registration_sha=sha(R/'registration.json'),binary_sha=sha(S/'mqft_train'),source_sha=sha(S/'train60.py'),plans=plans,fixture_sha=sha(S/'plans/inference.json'),rounds=2,train_games_per_round=48,val_games_per_round=12,arena_pairs_per_round=16,workers=10,board=9,rules='freestyle>=5',budget=32,c_puct=1.4,noise=dict(alpha=.3,epsilon=.25,first_sampled_plies=12),optimizer='AdamW; fresh optimizer at g0, continued optimizer after g1',lr=[.001,.0007],steps_per_round=300,batch=64,weight_decay=1e-4,cut_steps=[0,50,100,200,300],train_weight='uniform game then uniform ply; D4 augmentation; current iteration only',value='winner sign times side, draw0',test='validation game-disjoint and D4 orbit purge against train same iteration; arena paired and not trained',device='CPU',torch=torch.__version__,numpy=np.__version__,seed=600101)
 dump(S/'REGISTRATION.json',reg);print('REGISTERED',sha(S/'REGISTRATION.json'),flush=True)
@torch.no_grad()
def infer_check(it):
 p=S/f'checkpoints/g{it}.pt';m,c=load(p);b=p.with_suffix('.bin');assert b.is_file()
 r=read(S/'REGISTRATION.json');assert sha(S/'mqft_train')==r['binary_sha']
 f=S/'plans/inference.json';assert sha(f)==r['fixture_sha'];moves=read(f)
 out=json.loads(subprocess.check_output([str(S/'mqft_train'),'infer',str(b),str(f)],text=True))
 x=np.stack([planes(board(m),1 if len(m)%2==0 else -1) for m in moves]);l,v=m(torch.from_numpy(x));p1=torch.softmax(l,1).numpy();v=v.numpy();p2=np.array([q['p'] for q in out]);v2=np.array([q['v'] for q in out])
 assert [q['moves'] for q in out]==moves and all(q['side']==(1 if len(m)%2==0 else -1) for q,m in zip(out,moves))
 pe=float(abs(p1-p2).max());ve=float(abs(v-v2).max());norm=float(abs(p2.sum(1)-1).max());illegal=float(abs(p2[x.sum(1).reshape(-1,N)>0]).max(initial=0))
 assert pe<2e-6 and ve<2e-5 and norm<2e-6 and illegal==0 and np.isfinite(p2).all()
 # deterministic repeated invocation
 assert out==json.loads(subprocess.check_output([str(S/'mqft_train'),'infer',str(b),str(f)],text=True))
 q=dict(generation=it,n=len(moves),checkpoint_sha=sha(p),weights_sha=sha(b),max_policy_error=pe,max_value_error=ve,max_norm_error=norm,illegal_mass=illegal,repeat_exact=True);dump(S/f'INFERENCE{it}.json',q);print('INFER',q,flush=True)
def verify_game(path,j):
 q=read(path);assert q['schema']=='mqft.train9.v1' and q['job']==j and q['terminal'] is True and q['moves'][:len(j['opening'])]==j['opening'];mv=q['moves'];b=board(mv);last=mv[-1];wins=won_last(b,last);assert wins or len(mv)==N
 win=(1 if (len(mv)-1)%2==0 else -1) if wins else 0
 assert win==q['winner'];assert len(q['rows'])==len(mv)-len(j['opening'])
 b=board(j['opening']);x=[];pi=[];z=[];entropy=[];total_evals=0;roots=[];max_w=max_s2=0.
 for i,r in enumerate(q['rows']):
  t=i+len(j['opening']);side=1 if t%2==0 else -1
  assert r['ply']==t and r['side']==side and r['model']==(int(side==-1)^int(j['swap'])) and r['board']==b.tolist() and mv[t]==r['action'] and b[mv[t]]==0
  ns=np.asarray(r['counts']);W=np.asarray(r['w']);S2=np.asarray(r['s2']);p=np.asarray(r['prior']);legal=b==0
  assert ns.sum()==j['budget'] and np.all(ns[~legal]==0) and np.all(p[~legal]==0) and abs(p.sum()-1)<2e-6 and np.all(p[legal]>0)
  assert ns[mv[t]]>0 and (r['sampled']==(j['training'] and i<12))
  if not r['sampled']:assert mv[t]==int(np.argmax(ns))
  rec=r['evaluations'];assert rec[0]['id']==0 and rec[0]['state_hash']==r['state_hash'] and rec[0]['side']==side
  assert [e['id'] for e in rec]==list(range(len(rec)));total_evals+=len(rec)
  n=np.zeros(N,dtype=int);w=np.zeros(N);s2=np.zeros(N);ids=[]
  assert len(r['events'])==j['budget']
  for k,e in enumerate(r['events']):
   a=e['actions'];assert a and e['phase']==k+1 and e['complete_seq']==k and e['parent_hashes'][0]==r['state_hash']
   bb=b.copy();s=side;term=False
   for l,act in enumerate(a):
    assert 0<=act<N and bb[act]==0 and not term
    bb[act]=1 if s==1 else 2;term=won_last(bb,act) or np.all(bb!=0);s=-s
    if term:assert l==len(a)-1
   leaf=float(e['leaf_value']);root=float(e['root_value']);assert abs(leaf)<=1 and root==((-1)**len(a))*leaf and e['terminal']==bool(term)
   if term:
    assert e['eval_id'] is None and leaf==(-1 if won_last(bb,a[-1]) else 0)
   else:
    ei=e['eval_id'];assert ei is not None and rec[ei]['state_hash']==e['leaf_hash'] and rec[ei]['side']==s and rec[ei]['value']==leaf;ids.append(ei)
   n[a[0]]+=1;w[a[0]]+=root;s2[a[0]]+=root*root
  assert sorted(ids)==list(range(1,len(rec))) and np.array_equal(n,ns)
  ew=float(abs(w-W).max());es=float(abs(s2-S2).max());assert ew<1e-8 and es<1e-8;max_w=max(max_w,ew);max_s2=max(max_s2,es)
  target=ns.astype(np.float32)/j['budget'];x.append(planes(b,side));pi.append(target);z.append(win*side);entropy.append(float(-np.sum(target[target>0]*np.log(target[target>0]))))
  roots.append(r['state_hash']);b[mv[t]]=1 if side==1 else 2
 return dict(x=np.array(x),pi=np.array(pi),z=np.array(z,dtype=np.float32),summary=dict(id=j['id'],winner=win,plies=len(mv),searched=len(x),backups=len(x)*j['budget'],evaluator_calls=total_evals,entropy=float(np.mean(entropy)),elapsed_sec=q['elapsed_sec'],max_w_error=max_w,max_s2_error=max_s2,roots=roots,sha256=sha(path),bytes=path.stat().st_size))
def collect(it,kind):
 reg=read(S/'REGISTRATION.json');plan=S/f'plans/{kind}{it}.json';assert sha(plan)==reg['plans'][plan.name]
 jobs=read(plan);folder=S/f'data/{kind}{it}';assert len(list(folder.glob('*.json')))==len(jobs)
 rows=[];ss=[]
 for j in jobs:
  q=verify_game(folder/(j['id']+'.json'),j);rows.append(q);ss.append(q['summary'])
 dump(S/f'{kind.upper()}{it}_VERIFIED.json',dict(kind=kind,generation=it,plan_sha=sha(plan),games=ss,totals={k:sum(x[k] for x in ss) for k in ['searched','backups','evaluator_calls','bytes']},n=len(ss)))
 return rows
def tensors(rows,train_orbits=None):
 xs=[];ps=[];zs=[];gs=[];ex=0
 for i,r in enumerate(rows):
  for x,p,z in zip(r['x'],r['pi'],r['z']):
   if train_orbits is not None and orbit(x) in train_orbits:ex+=1;continue
   xs.append(x);ps.append(p);zs.append(z);gs.append(i)
 assert xs
 return np.stack(xs),np.stack(ps),np.array(zs,np.float32),np.array(gs),ex
@torch.no_grad()
def metrics(m,ds):
 x,p,z,g,_=ds;m.eval();vals=[]
 for a in range(0,len(x),256):
  l,v=m(torch.from_numpy(x[a:a+256]));q=torch.from_numpy(p[a:a+256]);t=torch.from_numpy(z[a:a+256]);lp=-(q*F.log_softmax(l,1)).sum(1);lv=(v-t)**2;vals.extend(zip(lp.tolist(),lv.tolist()))
 y=np.asarray(vals);w=np.array([1/(len(set(g))*np.count_nonzero(g==i)) for i in g]);ce,vm=w@y;return dict(policy_loss=float(ce),value_loss=float(vm),total_loss=float(ce+vm))
def fit(it):
 reg=read(S/'REGISTRATION.json');rows=collect(it,'self');train=[r for r in rows if '_train_' in r['summary']['id']];val=[r for r in rows if '_val_' in r['summary']['id']]
 assert len(train)==48 and len(val)==12
 tr=tensors(train);orb={orbit(x) for x in tr[0]};va=tensors(val,orb)
 assert not orb&{orbit(x) for x in va[0]}
 m,c=load(S/f'checkpoints/g{it-1}.pt');opt=torch.optim.AdamW(m.parameters(),lr=reg['lr'][it-1],weight_decay=reg['weight_decay'])
 if it>1:
  opt.load_state_dict(c['optimizer']);opt.param_groups[0]['lr']=reg['lr'][it-1]
 x,p,z,g,_=tr;groups=[np.flatnonzero(g==i) for i in np.unique(g)];rng=np.random.default_rng(600200+it);torch.manual_seed(600300+it);losses=[];cuts=[];start=time.perf_counter()
 for t in range(reg['steps_per_round']+1):
  if t in reg['cut_steps']:
   cuts.append(dict(step=t,train=metrics(m,tr),validation=metrics(m,va)))
  if t==reg['steps_per_round']:break
  ix=np.array([rng.choice(groups[j]) for j in rng.integers(0,len(groups),size=reg['batch'])])
  pp=PERM[rng.integers(0,8,len(ix))]
  xx=np.stack([x[k].reshape(2,N)[:,a].reshape(2,L,L) for k,a in zip(ix,pp)]);yy=np.stack([p[k,a] for k,a in zip(ix,pp)])
  m.train();l,v=m(torch.from_numpy(xx));lp=-(torch.from_numpy(yy)*F.log_softmax(l,1)).sum(1).mean();lv=F.mse_loss(v,torch.from_numpy(z[ix]));loss=lp+lv
  opt.zero_grad();loss.backward();opt.step()
  losses.append(dict(step=t+1,policy_loss=float(lp.detach()),value_loss=float(lv.detach()),total_loss=float(loss.detach()),learning_rate=opt.param_groups[0]['lr']))
 ck=S/f'checkpoints/g{it}.pt';prev=S/f'checkpoints/g{it-1}.pt'
 assert not ck.exists();torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),generation=it,steps=reg['steps_per_round'],parent_sha256=sha(prev),registration_sha256=sha(S/'REGISTRATION.json'),dataset_sha256=sha(S/f'SELF{it}_VERIFIED.json'),rng_state_json=json.dumps(rng.bit_generator.state)),ck)
 bp=export(ck);re,c2=load(ck);assert all(torch.equal(v,c2['model'][k]) for k,v in m.state_dict().items());assert c2['parent_sha256']==sha(prev)
 np.savez_compressed(S/f'TRAIN_DATA{it}.npz',x=x,pi=p,z=z,game=g,val_x=va[0],val_pi=va[1],val_z=va[2],val_game=va[3])
 result=dict(generation=it,train_positions=len(x),val_positions=len(va[0]),val_excluded=va[4],train_unique_orbits=len(orb),train_outcomes=dict(collections.Counter(str(r['summary']['winner']) for r in train)),val_outcomes=dict(collections.Counter(str(r['summary']['winner']) for r in val)),train_value_targets=dict(collections.Counter(str(int(v)) for v in z)),validation_games_retained=len(set(va[3])),losses=losses,cuts=cuts,checkpoint_sha=sha(ck),weights_sha=sha(bp),parent_sha=sha(prev),seconds=time.perf_counter()-start)
 dump(S/f'TRAINING{it}.json',result);print('TRAINED',it,len(x),len(va[0]),cuts[0],cuts[-1],flush=True)
def arena(it):
 rows=collect(it,'arena');plan=read(S/f'plans/arena{it}.json');r=[]
 for j,q in zip(plan,rows):
  s=q['summary'];newside=-1 if j['swap'] else 1;w=s['winner'];pay=0.5 if not w else float(w==newside)
  r.append(dict(id=j['id'],opening=j['opening'],swap=j['swap'],winner=w,new_score=pay,plies=s['plies']))
 pairs=np.array([np.mean([r[k]['new_score'],r[k+1]['new_score']]) for k in range(0,len(r),2)]);rng=np.random.default_rng(600400+it);bs=pairs[rng.integers(0,len(pairs),(10000,len(pairs)))].mean(1)
 out=dict(generation=it,games=r,n=len(r),W=sum(x['new_score']==1 for x in r),D=sum(x['new_score']==.5 for x in r),L=sum(x['new_score']==0 for x in r),score=float(pairs.mean()),pair_scores=pairs.tolist(),paired_bootstrap_95=np.quantile(bs,[.025,.975]).tolist(),scope='conditional paired opening resampling; no population Elo')
 dump(S/f'ARENA{it}.json',out);print('ARENA',it,{k:out[k] for k in ['W','D','L','score','paired_bootstrap_95']},flush=True)
if __name__=='__main__':
 op=sys.argv[1]
 if op=='init':init()
 else:{'infer':infer_check,'fit':fit,'arena':arena}[op](int(sys.argv[2]))
