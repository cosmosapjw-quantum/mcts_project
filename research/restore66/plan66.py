from base61 import *
import gzip
R=B/'mqft_restore65_20261126'
def main():
 old=read(R/'REGISTRATION.json');audit=read(R/'AUDIT65.json');rng=np.random.default_rng(660001)
 used=set()
 for i in range(61,66):
  for p in (B/f'mqft_restore{i}_20261126'/'plans').glob('*.json'):
   try:
    v=read(p);used.update(j['seed'] for j in v if isinstance(j,dict) and 'seed' in j)
   except (TypeError,AttributeError):pass
 def seed():
  while True:
   s=int(rng.integers(0,2**63))
   if s not in used:used.add(s);return s
 receipts={x['job']['id']:x for x in audit['jobs']}
 schedules=[];plans={}
 for g in (0,3):
  jobs=[];pilot=[j for j in read(R/f'plans/phase{g}.json') if j['rep']==0 and j['arm']=='baseline']
  assert len(pilot)==48
  for j in pilot:
   p=R/'data'/f'phase{g}'/(j['id']+'.json.gz');r=receipts[j['id']];assert sha(p)==r['sha256']
   with gzip.open(p,'rt') as f:d=json.load(f)
   assert d['input']==j
   es=d['trace']['events'];legal=sorted(es[0]['root_selection']['law']['actions']);a=j['sources'][0];U=[]
   for t in (0,4,8,12,16,20,24,28,31):
    l=es[4*t]['root_selection']['law'];s=dict(zip(l['actions'],l['scores']));assert sorted(s)==legal;U.append([s[x] for x in legal])
   h=np.linspace(-.45,.45,9);E=-np.asarray(U);E[:,legal.index(a)]-=h/(10/3)
   schedules.append(dict(generation=g,core=j['core'],pilot_id=j['id'],pilot_sha256=r['sha256'],legal=legal,source=a,score=U,energy=E.tolist(),source_logit=h.tolist(),knots=[0,4,8,12,16,20,24,28,31]))
  for c in old['cores']:
   for arm in ('forward','counter'):
    for rep in range(16):
     jobs.append(dict(id=f'g{g}_c{c["id"]:02d}_{arm}_r{rep:02d}',moves=c['moves'],sources=c['sources'],seed=seed(),budget=128,width=4,beta=10/3,generation=g,core=c['id'],rep=rep,arm=arm))
  p=S/f'plans/schedule{g}.json';dump(p,jobs);plans[f'schedule{g}']=dict(n=len(jobs),backups=len(jobs)*128,generation=g,sha256=sha(p))
 dump(S/'THERMAL_SCHEDULES.json',schedules)
 reg=dict(parent='6340af6613074737daa30f95115648ce1d51be80',prior_registration_sha=sha(R/'REGISTRATION.json'),prior_audit_sha=sha(R/'AUDIT65.json'),weights=old['weights'],cores=old['cores'],plans=plans,binary_sha=sha(S/'mqft_train'),schedule_sha=sha(S/'THERMAL_SCHEDULES.json'),plan_code_sha=sha(S/'plan66.py'),seed=660001,native_workers=16,audit_workers=12,thermal=dict(paths_per_direction=8192,alphas=[.25,1.],beta=10/3,seed=660101,energy_unit='declared E_*; no physical calibration'),scope='R65 matched roots, fresh independent seeds for both native directions; auxiliary schedules frozen from separately generated R65 baseline; same-order native counter schedule is not time reversal')
 dump(S/'REGISTRATION.json',reg);print('REGISTER',sha(S/'REGISTRATION.json'),plans,'THERMAL',len(schedules),flush=True)
if __name__=='__main__':main()
