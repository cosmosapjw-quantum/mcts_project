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
rng=np.random.default_rng(780001);cores=[]
def seed():
 while True:
  sd=int(rng.integers(0,2**63))
  if sd not in used:used.add(sd);return sd
for i in range(16):
 ply=4+4*(i%4)
 while True:
  m=rng.choice(N,ply,replace=False).tolist()
  if any(win(board(m[:j+1]),a) for j,a in enumerate(m)):continue
  k=orbit(board(m)).hex();opts=[tuple(p[m]) for p in PERM]
  if k in seen or len({board(list(q)).tobytes() for q in opts})!=8:continue
  m=list(map(int,min(opts)));seen.add(k);break
 cores.append(dict(id=i,moves=m,orbit=k,seeds=[seed() for _ in range(4)],rapfi_seeds=[seed() for _ in range(4)],split='test' if i%16<4 else 'train'))
p=S/'plans/roots.json';dump(p,[c['moves'] for c in cores])
v=json.loads(subprocess.check_output([str(S/'mqft_train'),'infer',str(B/r['weights']['0']['path']),str(p)],text=True));dump(S/'ROOT_PREPARATION.json',v)
for c,x in zip(cores,v):
 assert c['moves']==x['moves'];b=board(c['moves']);pr=np.asarray(x['p']);c['sources']=list(map(int,sorted(np.flatnonzero(b==0),key=lambda a:(-pr[a],int(a)))[:2]))
plans={}
for g in (0,3):
 jobs=[dict(id=f'g{g}_c{c["id"]:02d}_o{o}_r{i:02d}_zero74',moves=list(map(int,p0[c['moves']])),sources=list(map(int,p0[c['sources']])),seed=sd,budget=128,width=4,beta=10/3,generation=g,core=c['id'],rep=i,arm='zero74') for c in cores for o,p0 in enumerate(PERM) for i,sd in enumerate(c['seeds'])]
 tag=f'schedule{g}';p=S/'plans'/f'{tag}.json';dump(p,jobs);plans[tag]=dict(n=len(jobs),sha256=sha(p))
 for i in range(0,len(jobs),128):dump(S/'plans'/f'{tag}_shard{i//128}.json',jobs[i:i+128])
jobs=[dict(id=f'rapfi_c{c["id"]:02d}_o{o}_r{i:02d}_zero74',core=c['id'],rep=i,moves=[(int(x)//9+3)*15+(int(x)%9+3) for x in p0[c['moves']]],sources=[(int(x)//9+3)*15+(int(x)%9+3) for x in p0[c['sources']]],seed=sd,budget=128,arm='zero74',pulse74=0.) for c in cores for o,p0 in enumerate(PERM) for i,sd in enumerate(c['rapfi_seeds'])]
p=S/'plans/rapfi.json';dump(p,jobs);plans['rapfi']=dict(n=len(jobs),sha256=sha(p))
r.update(parent='888da8f4bbeb0b4aa07b3c4b6124bfec8caaa2ff',pulse_binary_sha=sha(S/'mqft_pulse74'),pulse_patch_sha=sha(S/'pulse74.patch'),seed=780001,cores=cores,plans=plans,workers=12,scope='R78 16 fresh board orbits, 8 raw orientations, 4 common seeds, 3 evaluators B128/w4. Gauge contexts fixed orientation0/rep0 at H32.')
r['serialization_fix']='remove redundant orientation field, encoded unchanged in job id; no raw registered data produced before fix'
dump(S/'REGISTRATION.json',r);print(plans,flush=True)
