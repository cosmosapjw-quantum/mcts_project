from base61 import *
R=B/'mqft_restore61_20261126'
def frame(m,s,legacy=False):
 if legacy:return list(m),list(s),[0]
 keys=[(tuple(p[m]),tuple(p[s])) for p in PERM];key=min(keys)
 return list(key[0]),list(key[1]),[g for g,v in enumerate(keys) if v==key]
def main():
 old=read(R/'REGISTRATION.json');rng=np.random.default_rng(620001)
 seen=set((R/'R60_ORBITS.txt').read_text().splitlines())|{c['orbit'] for c in old['cores']}
 cores=[]
 for i in range(64):
  ply=(4,8,12,16)[i//16]
  while True:
   m=rng.choice(N,ply,replace=False).tolist();b=board(m)
   if any(win(board(m[:t+1]),a) for t,a in enumerate(m)):continue
   k=orbit(b).hex()
   if k in seen:continue
   pool=[a for a in range(N) if b[a]==0 and max(abs(a//9-4),abs(a%9-4))<=2]
   s=rng.choice(pool,2,replace=False).tolist()
   if len(frame(m,s)[2])!=1:continue
   seen.add(k);cores.append(dict(id=i,moves=m,sources=s,orbit=k,seeds=[int(rng.integers(0,2**63)) for _ in range(12)],split='train' if i%16<8 else 'test'));break
 plans={};arms={'z':[0.,0.],'am2':[-.2,0.],'ap2':[.2,0.],'bm2':[0.,-.2],'bp2':[0.,.2],'am1':[-.1,0.],'ap1':[.1,0.],'am35':[-.35,0.],'ap35':[.35,0.]}
 def job(id,m,src,hs,seed,budget,w,g,c,o,r,cohort,rank,arm,d4=False):return dict(id=id,moves=m,sources=src,field=hs,seed=seed,budget=budget,width=w,beta=10/3,d4=d4,generation=g,core=c,orientation=o,rep=r,cohort=cohort,frame_rank=rank,arm=arm)
 def save(tag,j):
  p=S/'plans'/f'{tag}.json';dump(p,j);plans[tag]=dict(sha256=sha(p),n=len(j),backups=sum(x['budget'] for x in j))
 for g in (0,3):
  for w in (1,4):
   js=[]
   for c in cores:
    for arm,h in arms.items():
     if (g,w)!=(0,1) and arm not in ('z','am2','ap2','bm2','bp2'):continue
     for r,seed in enumerate(c['seeds']):
      js.append(job(f'g{g}w{w}c{c["id"]:02d}r{r:02d}_{arm}',c['moves'],c['sources'],h,seed,128,w,g,c['id'],0,r,'response',0,arm))
   save(f'response{g}w{w}',js)
 # Direct replicate previous g0/w1 law on same 32 R61 geometries, fresh independent seeds.
 js=[]
 for c in old['cores']:
  for rep in range(16):
   seed=int(rng.integers(0,2**63))
   for arm,h in [('am35',-.35),('z',0.),('ap35',.35)]:
    js.append(job(f'leg_c{c["id"]:02d}r{rep:02d}_{arm}',c['moves'],[c['source']],[h],seed,128,1,0,c['id'],0,rep,'legacy',0,arm))
 save('legacy0',js)
 # All R61 first-divergence witness contexts (old eight cores), plus stabilizers.
 sym=[]
 for i in (0,1,8,9,16,17,24,25):
  c=old['cores'][i];sym.append(dict(id=i,moves=c['moves'],sources=[c['source']],seeds=c['seeds'][:2],old=True))
 for i,(m,s) in enumerate([([],[40]),([40],[30]),([],[40,30]),([40,31],[49])],100):
  sym.append(dict(id=i,moves=m,sources=s,seeds=[int(rng.integers(0,2**63))],old=False))
 for g in (0,3):
  js=[]
  for c in sym:
   for o,p in enumerate(PERM):
    m=p[c['moves']].tolist();s=p[c['sources']].tolist();_,_,fs=frame(m,s)
    for rank in range(len(fs)):
     for rep,seed in enumerate(c['seeds']):
      js.append(job(f'sg{g}c{c["id"]:03d}o{o}f{rank}r{rep}',m,s,[0.]*len(s),seed,64,4,g,c['id'],o,rep,'symmetry',rank,'z',True))
  save(f'symmetry{g}',js)
 reg=dict(parent='16f292a5baf4849d14265735058a44e3f84cb112',binary_sha=sha(S/'mqft_train'),plan_code_sha=sha(S/'plan62.py'),prior_registration_sha=sha(R/'REGISTRATION.json'),weights=old['weights'],plans=plans,cores=cores,symmetry_cores=sym,arms=arms,permutations=PERM.tolist(),workers=16,geometry_blocks=64,seeds_per_block=12,scope='canonical history+labelled sources; single-orientation evaluation in shared frame; D4 ensemble for symmetry witness suite; legacy noncanonical scalar replication; fresh root orbits disjoint R60/R61; explicit enumeration of stabilizer frames',closure='fit 32 geometry blocks, hold out 32 (stratified), fixed ridge lambda .02; predict normalized increments at 32→64 and 64→128 using count-only vs mark features; exact last-wave Rao-Blackwell, seed-crossfit baseline',seed=620001)
 dump(S/'REGISTRATION.json',reg);print('REGISTER',sha(S/'REGISTRATION.json'),plans,'TOTAL',sum(x['n'] for x in plans.values()),sum(x['backups'] for x in plans.values()),flush=True)
if __name__=='__main__':main()
