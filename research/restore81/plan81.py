from base61 import *
import subprocess,sys
R=B/'mqft_restore74_20261126';P=B/'mqft_restore39_20261125/repo';sys.path.insert(0,str(P/'python'))
from mqft.rapfi import verify
r=read(R/'REGISTRATION.json');assert sha(S/'mqft_train')==r['binary_sha'];assert verify(P/'.external/rapfi')==r['external']
assert all(sha(B/x['path'])==x['sha256'] for x in r['weights'].values())
seen=set((B/'mqft_restore61_20261126/R60_ORBITS.txt').read_text().splitlines());used=set()
for f in B.glob('mqft_restore*/REGISTRATION.json'):
 try:
  q=read(f)
  for k in ('cores','holdout_cores'):
   seen.update(c['orbit'] for c in q.get(k,[]) if isinstance(c,dict) and 'orbit' in c)
 except (ValueError,TypeError,KeyError):pass
for f in B.glob('mqft_restore*/plans/*.json'):
 try:
  a=read(f)
  if isinstance(a,list):used.update(j['seed'] for j in a if isinstance(j,dict) and 'seed' in j)
 except (ValueError,TypeError):pass
rng=np.random.default_rng(810001);cores=[]
def seed():
 while True:
  sd=int(rng.integers(0,2**63))
  if sd not in used:used.add(sd);return sd
for i in range(24):
 ply=4+4*(i%4)
 while True:
  m=rng.choice(N,ply,replace=False).tolist()
  if any(win(board(m[:j+1]),a) for j,a in enumerate(m)):continue
  k=orbit(board(m)).hex();opts=[tuple(p[m]) for p in PERM]
  if k in seen or len(set(opts))!=8:continue
  m=list(map(int,min(opts)));seen.add(k);break
 cores.append(dict(id=i,moves=m,orbit=k,seeds=[seed() for _ in range(8)],rapfi_seeds=[seed() for _ in range(8)],split='test' if i%16<4 else 'train'))
p=S/'plans/roots.json';dump(p,[c['moves'] for c in cores])
v=json.loads(subprocess.check_output([str(S/'mqft_train'),'infer',str(B/r['weights']['0']['path']),str(p)],text=True));dump(S/'ROOT_PREPARATION.json',v)
for c,x in zip(cores,v):
 assert c['moves']==x['moves'];b=board(c['moves']);pr=np.asarray(x['p']);c['sources']=list(map(int,sorted(np.flatnonzero(b==0),key=lambda a:(-pr[a],int(a)))[:2]))
plans={}
arms=('plus81','minus81','eraseplus81','eraseminus81','swap81','base81')
for g in (0,3):
 jobs=[dict(id=f'g{g}_c{c["id"]:02d}_r{i:02d}_{a}',moves=c['moves'],sources=c['sources'],seed=sd,budget=128,width=4,beta=10/3,generation=g,core=c['id'],rep=i,arm=a) for c in cores for i,sd in enumerate(c['seeds']) for a in arms]
 tag=f'schedule{g}';p=S/'plans'/f'{tag}.json';dump(p,jobs);plans[tag]=dict(n=len(jobs),sha256=sha(p))
 # pilot 128 paths followed by chunks of at most 1024. Pilot included, not duplicated.
 parts=[jobs[:128]]+[jobs[i:i+1024] for i in range(128,len(jobs),1024)]
 for i,q in enumerate(parts):dump(S/'plans'/f'{tag}_shard{i}.json',q)
jobs=[dict(id=f'rapfi_c{c["id"]:02d}_r{r:02d}_{a}',core=c['id'],rep=r,moves=[(x//9+3)*15+(x%9+3) for x in c['moves']],sources=[(x//9+3)*15+(x%9+3) for x in c['sources']],seed=sd,budget=128,arm=a,phase81=a) for c in cores for r,sd in enumerate(c['rapfi_seeds']) for a in arms]
p=S/'plans/rapfi.json';dump(p,jobs);plans['rapfi']=dict(n=len(jobs),sha256=sha(p))
r.update(parent='9168e93ef9669ea6ea8c05d63a20c749b419f466',pulse_binary_sha=sha(S/'mqft_pulse74'),pulse_patch_sha=sha(S/'pulse74.patch'),seed=810001,phase_binary_sha=sha(S/'mqft_phase81'),phase_patch_sha=sha(S/'phase81.patch'),arms=list(arms),cores=cores,plans=plans,workers=12,scope='R81 24 fresh geometries x8 seeds x6 matched arms x3 evaluators; B128/w4; phase gates at t32/36/40; fixed preregistered pair.')
dump(S/'REGISTRATION.json',r);print(plans,flush=True)
