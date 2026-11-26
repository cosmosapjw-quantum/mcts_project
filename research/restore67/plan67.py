from base61 import *
import sys
P=B/'mqft_restore39_20261125/repo';sys.path.insert(0,str(P/'python'))
from mqft.rapfi import verify
R=[B/f'mqft_restore{i}_20261126' for i in range(61,67)]
def main():
 old=[read(x/'REGISTRATION.json') for x in R];rng=np.random.default_rng(670001)
 seen=set((R[0]/'R60_ORBITS.txt').read_text().splitlines())
 for q in old:seen.update(c['orbit'] for c in q.get('cores',q.get('holdout_cores',[])))
 used=set()
 for s in R:
  for p in (s/'plans').glob('*.json'):
   v=read(p)
   if isinstance(v,list):used.update(j['seed'] for j in v if isinstance(j,dict) and 'seed'in j)
 def seed():
  while True:
   sd=int(rng.integers(0,2**63))
   if sd not in used:used.add(sd);return sd
 cores=[]
 for i in range(32):
  ply=(4,8,12,16)[i//8]
  while True:
   m=rng.choice(N,ply,replace=False).tolist()
   if any(win(board(m[:t+1]),a) for t,a in enumerate(m)):continue
   key=orbit(board(m)).hex()
   if key in seen:continue
   opts=[tuple(p[m]) for p in PERM]
   if len(set(opts))!=8:continue
   m=list(map(int,min(opts)));seen.add(key);break
  legal=np.flatnonzero(board(m)==0);a=sorted(legal,key=lambda a:((a//9-4)**2+(a%9-4)**2,int(a)))[:2]
  cores.append(dict(id=i,moves=m,orbit=key,sources=list(map(int,a)),seeds=[seed() for _ in range(8)],reference_seeds=[seed() for _ in range(4)]))
 arms=['clean','single_iid','rep_iid','single_mix','rep_mix','rep_common'];plans={}
 def job(g,c,r,sd,arm,B):return dict(id=f'g{g}_c{c["id"]:02d}_r{r}_{arm}',moves=c['moves'],sources=c['sources'],seed=sd,budget=B,width=4,beta=10/3,generation=g,core=c['id'],rep=r,arm=arm)
 for tag,g,js in [(f'fault{g}',g,[job(g,c,r,sd,a,128) for c in cores for r,sd in enumerate(c['seeds']) for a in arms]) for g in (0,3)]+[('reference3',3,[job(3,c,r,sd,'reference',1024) for c in cores for r,sd in enumerate(c['reference_seeds'])])]:
  p=S/'plans'/f'{tag}.json';dump(p,js);plans[tag]=dict(n=len(js),backups=sum(j['budget'] for j in js),generation=g,sha256=sha(p))
 reg=dict(parent='30f231e9a81f0ca3ba8624b5d75a9e9f0f60d475',prior_registration_sha=sha(R[-1]/'REGISTRATION.json'),weights=old[-1]['weights'],cores=cores,arms=arms,plans=plans,binary_sha=sha(S/'mqft_train'),plan_code_sha=sha(S/'plan67.py'),external=verify(P/'.external/rapfi'),seed=670001,noise_p=.12,mixed_common=.5,workers=16,audit_workers=12,reference=dict(budget=1024,seeds=4,checkpoint=3,pooled_min_visits=[8,16,32],primary=16),scope='classical native sign-bit protection, exact auxiliary quantum channel separate; no hardware noise or physical quantum advantage; Rapfi finite-time proxy')
 dump(S/'REGISTRATION.json',reg);print('REGISTER',sha(S/'REGISTRATION.json'),plans,flush=True)
if __name__=='__main__':main()
