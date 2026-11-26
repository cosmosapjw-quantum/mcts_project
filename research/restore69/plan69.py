from base61 import *
import subprocess,sys
P=B/'mqft_restore39_20261125/repo';sys.path.insert(0,str(P/'python'))
from mqft.rapfi import verify
def main():
 R=B/'mqft_restore65_20261126';old=read(R/'REGISTRATION.json');rng=np.random.default_rng(690001);seen=set((B/'mqft_restore61_20261126/R60_ORBITS.txt').read_text().splitlines());used=set()
 for i in range(61,68):
  s=B/f'mqft_restore{i}_20261126';q=read(s/'REGISTRATION.json');seen.update(c['orbit'] for c in q.get('cores',q.get('holdout_cores',[])))
  for p in (s/'plans').glob('*.json'):
   v=read(p)
   if isinstance(v,list):used.update(j['seed'] for j in v if isinstance(j,dict) and 'seed' in j)
 def seed():
  while True:
   s=int(rng.integers(0,2**63))
   if s not in used:used.add(s);return s
 cores=[]
 for i in range(16):
  ply=4+4*(i//4)
  while True:
   m=rng.choice(N,ply,replace=False).tolist()
   if any(win(board(m[:j+1]),a) for j,a in enumerate(m)):continue
   k=orbit(board(m)).hex();opts=[tuple(p[m]) for p in PERM]
   if k in seen or len(set(opts))!=8:continue
   m=list(map(int,min(opts)));seen.add(k);break
  cores.append(dict(id=i,moves=m,orbit=k,seeds=[seed() for _ in range(4)],rapfi_seeds=[seed() for _ in range(2)]))
 p=S/'plans/roots.json';dump(p,[c['moves'] for c in cores])
 w=old['weights'];v=json.loads(subprocess.check_output([str(S/'mqft_train'),'infer',str(B/w['0']['path']),str(p)],text=True))
 for c,x in zip(cores,v):
  assert c['moves']==x['moves'];b=board(c['moves']);prior=np.asarray(x['p']);c['sources']=list(map(int,sorted(np.flatnonzero(b==0),key=lambda a:(-prior[a],int(a)))[:2]))
 dump(S/'ROOT_PREPARATION.json',v)
 plans={}
 for g in (0,3):
  js=[dict(id=f'g{g}_c{c["id"]:02d}_r{r}_{a}',moves=c['moves'],sources=c['sources'],seed=sd,budget=128,width=4,beta=10/3,generation=g,core=c['id'],rep=r,arm=a) for c in cores for r,sd in enumerate(c['seeds']) for a in ('persist_plus','persist_minus')]
  p=S/'plans'/f'phase{g}.json';dump(p,js);plans[f'phase{g}']=dict(n=len(js),sha256=sha(p))
 # Same centered board geometry, separate 15x15 game and actual MixNet weights. No direct size effect inferred.
 jobs=[dict(id=f'rapfi_c{c["id"]:02d}_r{r}',core=c['id'],rep=r,moves=[(a//9+3)*15+(a%9+3) for a in c['moves']],sources=[(a//9+3)*15+(a%9+3) for a in c['sources']],seed=sd,budget=128,arm='clean') for c in cores for r,sd in enumerate(c['rapfi_seeds'])]
 dump(S/'plans/rapfi.json',jobs);plans['rapfi']=dict(n=len(jobs),sha256=sha(S/'plans/rapfi.json'))
 dump(S/'REGISTRATION.json',dict(parent='9de636fc56fd34057553d78c0aad63fcc3832c08',binary_sha=sha(S/'mqft_train'),weights=w,external=verify(P/'.external/rapfi'),old_generator_sha=sha(R/'GENERATOR65.npz'),old_generator_receipt_sha=sha(R/'GENERATOR65.json'),cores=cores,plans=plans,seed=690001,workers=12,scope='fresh real R65 phase instrument on g0/g3 plus actual 15x15 direct Rapfi NNUE frozen base-law records, no game strength. R65 all prior waves reused separately.'))
 print('REGISTER',plans,flush=True)
if __name__=='__main__':main()
