from base61 import *
import subprocess
R=[B/f'mqft_restore{i}_20261126' for i in (61,62,63,64)]
def main():
 old=[read(x/'REGISTRATION.json') for x in R];rng=np.random.default_rng(650001)
 seen=set((R[0]/'R60_ORBITS.txt').read_text().splitlines())
 for q in old:seen.update(c['orbit'] for c in q.get('cores',q.get('holdout_cores',[])))
 used=set()
 for s in R:
  for p in (s/'plans').glob('*.json'):
   if p.name.startswith(('response','legacy','symmetry','fisher','hold','phase')) and not any(x in p.name for x in ('smoke','rest')):
    used.update(j['seed'] for j in read(p))
 def seed():
  while True:
   sd=int(rng.integers(0,2**63))
   if sd not in used:used.add(sd);return sd
 cores=[]
 for i in range(48):
  ply=(4,8,12,16)[i//12]
  while True:
   m=rng.choice(N,ply,replace=False).tolist()
   if any(win(board(m[:t+1]),a) for t,a in enumerate(m)):continue
   key=orbit(board(m)).hex()
   if key in seen:continue
   opts=[tuple(p[m]) for p in PERM]
   if len(set(opts))!=8:continue
   m=list(map(int,min(opts)));seen.add(key);break
  cores.append(dict(id=i,moves=m,orbit=key,seeds=[seed() for _ in range(8)],reference_seeds=[seed() for _ in range(4)]))
 f=S/'plans/roots.json';dump(f,[c['moves'] for c in cores])
 weights=old[-1]['weights'];g0=B/weights['0']['path'];assert sha(g0)==weights['0']['sha256']
 p=json.loads(subprocess.check_output([str(S/'mqft_train'),'infer',str(g0),str(f)],text=True))
 for c,x in zip(cores,p):
  assert c['moves']==x['moves'];b=board(c['moves']);p0=np.asarray(x['p']);a=sorted(np.flatnonzero(b==0),key=lambda a:(-p0[a],int(a)))[:2]
  c['sources']=list(map(int,a));c['g0_prior_pair']=p0[a].tolist()
 dump(S/'ROOT_PREPARATION.json',p)
 arms=['baseline','fresh64','reset_plus','persist_plus','persist_minus','persist_dephased'];plans={}
 def job(g,c,r,seed,arm,bu):return dict(id=f'g{g}_c{c["id"]:02d}_r{r}_{arm}',moves=c['moves'],sources=c['sources'],seed=seed,budget=bu,width=4,beta=10/3,generation=g,core=c['id'],rep=r,arm=arm)
 for tag,g,js in [(f'phase{g}',g,[job(g,c,r,sd,a,128) for c in cores for r,sd in enumerate(c['seeds']) for a in arms]) for g in (0,3)]+[('reference3',3,[job(3,c,r,sd,'reference',1024) for c in cores for r,sd in enumerate(c['reference_seeds'])])]:
  p=S/'plans'/f'{tag}.json';dump(p,js);plans[tag]=dict(n=len(js),backups=sum(j['budget'] for j in js),generation=g,sha256=sha(p))
 r=dict(parent='b77047387f09693466487193d195e80b3e2e7519',binary_sha=sha(S/'mqft_train'),prior_registration_sha=sha(R[-1]/'REGISTRATION.json'),prior_calibration_sha=sha(R[-1]/'CALIBRATION64.json'),weights=weights,cores=cores,arms=arms,plans=plans,root_preparation_sha=sha(S/'ROOT_PREPARATION.json'),plan_code_sha=sha(S/'plan65.py'),workers=16,audit_workers=12,seed=650001,lambda_=0.65,alpha=.35,eta0=.8,gamma=1.2,kappa=6,source_rule='top two legal g0 prior in canonical coordinates before search; same pair all arms and models',reference=dict(checkpoint=3,budget=1024,seeds=4,pooled_min_visits=[8,16,32],primary=16,definition='empirical high-budget Q regret conditional on eligible action; reference policy TV and top5; no true minimax or game win claim'),scope='persistent qubit on selection clock, free propagation at completed-wave boundary, full-support weak selective instrument; exact classical Bloch state realization, no physical hardware')
 dump(S/'REGISTRATION.json',r);print('REGISTER',sha(S/'REGISTRATION.json'),plans,flush=True)
if __name__=='__main__':main()
