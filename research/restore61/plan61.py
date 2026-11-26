from base61 import *
def main():
 old=read(O/'AUDIT60.json');assert len(old['lineage'])==4
 weights={str(i):dict(path=str((O/f'checkpoints/g{i}.bin').relative_to(B)),sha256=sha(O/f'checkpoints/g{i}.bin')) for i in (0,3)}
 assert all(weights[str(i)]['sha256']==old['lineage'][i]['weights_sha'] for i in (0,3))
 rng=np.random.default_rng(610001);seen=set((S/'R60_ORBITS.txt').read_text().splitlines());cores=[]
 # Independent D4 orbits, four depths balanced. Reject terminal prefixes.
 for i in range(32):
  ply=(4,8,12,16)[i//8]
  while True:
   b=np.zeros(N,np.int8);moves=[];legal=True
   for t,a in enumerate(rng.choice(N,ply,replace=False).tolist()):
    b[a]=1+t%2
    if win(b,a):legal=False;break
    moves.append(a)
   if not legal:continue
   key=orbit(b).hex()
   if key in seen:continue
   seen.add(key)
   cand=[a for a in range(N) if b[a]==0 and max(abs(a//9-4),abs(a%9-4))<=2]
   src=int(rng.choice(cand))
   seeds=[int(rng.integers(0,2**63)) for _ in range(8)]
   cores.append(dict(id=i,moves=moves,source=src,seeds=seeds,orbit=key));break
 plans={}
 for gen in (0,3):
  jobs=[]
  for c in cores:
   for width in (1,4):
    for hi,h in enumerate((-.35,0.,.35)):
     for rep,seed in enumerate(c['seeds']):
      jobs.append(dict(id=f'r_g{gen}_c{c["id"]:02d}_w{width}_h{hi}_r{rep}',moves=c['moves'],source=c['source'],seed=seed,budget=128,width=width,beta=10/3,h=h,d4=False,generation=gen,core=c['id'],orientation=0,rep=rep,cohort='response'))
  p=S/f'plans/response{gen}.json';dump(p,jobs);plans[p.name]=sha(p)
  jobs=[]
  for c in [cores[i] for i in (0,1,8,9,16,17,24,25)]:
   for d4 in (False,True):
    for g,perm in enumerate(PERM):
     for rep,seed in enumerate(c['seeds'][:2]):
      jobs.append(dict(id=f's_g{gen}_c{c["id"]:02d}_d{int(d4)}_o{g}_r{rep}',moves=perm[c['moves']].tolist(),source=int(perm[c['source']]),seed=seed,budget=64,width=4,beta=10/3,h=0.,d4=d4,generation=gen,core=c['id'],orientation=g,rep=rep,cohort='symmetry'))
  p=S/f'plans/symmetry{gen}.json';dump(p,jobs);plans[p.name]=sha(p)
 fixtures=[PERM[g][c['moves']].tolist() for c in cores for g in range(8)]
 dump(S/'plans/d4_fixture.json',fixtures)
 # 4 independent symmetry-related fixtures across orientations, compare all 32x8
 p=S/'REGISTRATION.json';assert not p.exists()
 dump(p,dict(parent='be9fc78107d78d7e9c2d8d9806de7467a6f7a40b',binary_sha=sha(S/'mqft_train'),source_sha=sha(S/'plan61.py'),old_reduction_sha=sha(S/'R60_REDUCTION61.json'),weights=weights,plans=plans,fixture_sha=sha(S/'plans/d4_fixture.json'),cores=cores,permutations=PERM.tolist(),response=dict(n=3072,budget=128,width=[1,4],h=[-.35,0,.35],beta=10/3,cores=32,reps=8),symmetry=dict(n=512,budget=64,width=4,cores=8,orientations=8,reps=2),workers=16,audit_workers=12,scope='paired finite native stochastic root sampler, actual trained leaf value; source only root logit; deterministic within-wave evaluation, parallel independent jobs; no physical temperature',seed=610001))
 print('REGISTER',sha(p),{k:len(read(S/'plans'/k)) for k in plans},flush=True)
if __name__=='__main__':main()
