from base61 import *
import subprocess
R=B/'mqft_restore63_20261126';R62=B/'mqft_restore62_20261126';R61=B/'mqft_restore61_20261126'
def main():
 r63=read(R/'REGISTRATION.json');r62=read(R62/'REGISTRATION.json');r61=read(R61/'REGISTRATION.json')
 rng=np.random.default_rng(640001)
 seen=set((R61/'R60_ORBITS.txt').read_text().splitlines())|{c['orbit'] for c in r61['cores']+r62['cores']+r63['holdout_cores']}
 used={j['seed'] for st in (R,R62,R61) for p in (st/'plans').glob('*.json') if p.name.startswith(('response','legacy','symmetry','fisher','hold')) and not any(x in p.name for x in ('smoke','rest')) for j in read(p)}
 cores=[]
 for i in range(64):
  ply=(4,8,12,16)[i//16]
  while True:
   m=rng.choice(N,ply,replace=False).tolist()
   if any(win(board(m[:t+1]),a) for t,a in enumerate(m)):continue
   key=orbit(board(m)).hex()
   if key in seen:continue
   opts=[tuple(p[m]) for p in PERM]
   if len(set(opts))!=8:continue
   m=list(map(int,min(opts)));seen.add(key)
   seeds=[]
   for _ in range(8):
    while True:
     sd=int(rng.integers(0,2**63))
     if sd not in used:used.add(sd);seeds.append(sd);break
   cores.append(dict(id=i,moves=m,orbit=key,seeds=seeds));break
 # Independent pre-search root inference, not outcome-based source selection.
 f=S/'plans/roots.json';dump(f,[c['moves'] for c in cores])
 weights=r63['weights'];g0=B/weights['0']['path'];assert sha(g0)==weights['0']['sha256']
 result=json.loads(subprocess.check_output([str(S/'mqft_train'),'infer',str(g0),str(f)],text=True))
 assert len(result)==64
 for c,x in zip(cores,result):
  assert x['moves']==c['moves']
  b=board(c['moves']);p=np.asarray(x['p']);assert abs(p.sum()-1)<2e-6
  a=sorted(np.flatnonzero(b==0),key=lambda a:(-p[a],int(a)))[:2]
  c['sources']=list(map(int,a));c['g0_prior_pair']=p[a].tolist()
 dump(S/'ROOT_PREPARATION.json',result)
 arms=['baseline','dephased','real','plus','minus','rotate0'];plans={}
 for g in (0,3):
  jobs=[]
  for c in cores:
   for rep,seed in enumerate(c['seeds']):
    for arm in arms:
     jobs.append(dict(id=f'g{g}_c{c["id"]:02d}_r{rep}_{arm}',moves=c['moves'],sources=c['sources'],seed=seed,budget=128,width=4,beta=10/3,generation=g,core=c['id'],rep=rep,arm=arm))
  tag=f'phase{g}';p=S/'plans'/f'{tag}.json';dump(p,jobs);plans[tag]=dict(n=len(jobs),backups=len(jobs)*128,sha256=sha(p))
 reg=dict(parent='a236944c943c8119a7e4f4a6dec66542136ffb96',binary_sha=sha(S/'mqft_train'),prior_registration_sha=sha(R/'REGISTRATION.json'),weights=weights,cores=cores,arms=arms,plans=plans,roots_sha=sha(f),root_preparation_sha=sha(S/'ROOT_PREPARATION.json'),plan_code_sha=sha(S/'plan64.py'),workers=16,audit_workers=12,seed=640001,source_rule='top two legal g0 prior before any native search, same pair across g0/g3 and arms; canonical ordered prefix',eta0=.8,gamma=1.2,kappa=6.,mix_alpha=.35,clock='tau=completed backups / 128; exposure sum of previous completed wave pre-selection score gap * width/128',readout='full support pair replacement; same diagonal real/dephased controls; plus and minus separated pi at fixed history; native search feedback',calibration=dict(train_seed=640101,test_seed=640102,train_states=768,test_states=384,shots_per_quadrature=1024,fit='nonlinear least squares normalized quadratures; independently prepared known p, theta0, exposure, tau; no native outcome fit'),scope='auxiliary software CPTP instrument with actual native selection, not physically identified hbar or quantum hardware')
 dump(S/'REGISTRATION.json',reg);print('REGISTER',sha(S/'REGISTRATION.json'),plans,flush=True)
if __name__=='__main__':main()
