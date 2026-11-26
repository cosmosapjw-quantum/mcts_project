from base61 import *
from plan62 import frame
R=B/'mqft_restore62_20261126';R1=B/'mqft_restore61_20261126'
def main():
 old=read(R/'REGISTRATION.json');r1=read(R1/'REGISTRATION.json');rng=np.random.default_rng(630001)
 seen=set((R1/'R60_ORBITS.txt').read_text().splitlines())|{c['orbit'] for c in r1['cores']+old['cores']}
 used={j['seed'] for q in (R,R1) for path in (q/'plans').glob('*.json') if path.name.startswith(('response','legacy','symmetry')) and not any(s in path.name for s in ('smoke','rest')) for j in read(path)}
 def seed():
  while True:
   x=int(rng.integers(0,2**63))
   if x not in used:used.add(x);return x
 cores=[]
 for i in range(64):
  ply=(4,8,12,16)[i//16]
  while True:
   m=rng.choice(N,ply,replace=False).tolist();b=board(m)
   if any(win(board(m[:t+1]),a) for t,a in enumerate(m)):continue
   k=orbit(b).hex()
   if k in seen:continue
   pool=[a for a in range(N) if b[a]==0 and max(abs(a//9-4),abs(a%9-4))<=2];s=rng.choice(pool,2,replace=False).tolist()
   if len(frame(m,s)[2])!=1:continue
   seen.add(k);cores.append(dict(id=i,moves=m,sources=s,orbit=k,seeds=[seed() for _ in range(16)]));break
 plans={}
 def job(tag,c,r,sd,g,w,arm,h,cohort):
  return dict(id=f'{tag}_c{c["id"]:02d}_r{r:02d}_{arm}',moves=c['moves'],sources=c['sources'],field=h,seed=sd,budget=128,width=w,beta=10/3,d4=False,generation=g,core=c['id'],orientation=0,rep=r,cohort=cohort,frame_rank=0,arm=arm)
 def save(tag,js):
  p=S/'plans'/f'{tag}.json';dump(p,js);plans[tag]=dict(sha256=sha(p),n=len(js),backups=sum(x['budget'] for x in js))
 js=[]
 for c in old['cores']:
  for r in range(64):js.append(job('fisher',c,r,seed(),0,4,'z',[0.,0.],'response'))
 save('fisher0',js)
 js=[]
 for c0 in r1['cores']:
  c=dict(c0,sources=[c0['source']])
  for r in range(32):
   sd=seed()
   for arm,h in [('am35',-.35),('z',0.),('ap35',.35)]:js.append(job('legacy',c,r,sd,0,1,arm,[h],'legacy'))
 save('legacy0',js)
 for g in (0,3):
  js=[job(f'hold{g}',c,r,sd,g,4,'z',[0.,0.],'response') for c in cores for r,sd in enumerate(c['seeds'])]
  save(f'hold{g}',js)
 r=dict(parent='0c597cc0520322cd305ff27c25a36c25088a31cf',binary_sha=sha(S/'mqft_train'),old_registration_sha=sha(R/'REGISTRATION.json'),old_audit_sha=sha(R/'AUDIT62.json'),old_array_sha=sha(R/'CONTROL62.npz'),prior_legacy_registration_sha=sha(R1/'REGISTRATION.json'),weights=old['weights'],plans=plans,holdout_cores=cores,workers=16,audit_workers=12,seed=630001,primary='g0/w4 same 64 roots 64 fresh seeds, scalar legacy same 32 roots 32 fresh seeds; matrix residual family 6 unique symmetric entries',closure=dict(train='R62 g0/g3 w4 z, 64 roots x12 seeds each',test='R63 hold0/3: 64 new geometries x16 seeds each',bins_p=[.01,.03,.10],bins_count=[.01,.03,.10],smoothing=.5,transition='time-dependent joint increment and reduced-state successor; shrink emission to mean p multinomial, transition to unconditional successor within state',horizons=[1,2,4,8],starts=[4,8,16,24],width=4))
 assert r['binary_sha']==old['binary_sha']
 dump(S/'REGISTRATION.json',r);print('REGISTER',sha(S/'REGISTRATION.json'),plans,flush=True)
if __name__=='__main__':main()
