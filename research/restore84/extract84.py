"""Fresh native audit and pre-search evaluator identity checks."""
from base61 import *
from audit_native74 import one as audit9
from audit_rapfi74 import audit_search
from population84 import features,OFF
import gzip,subprocess
TAGS=['schedule0','schedule3','rapfi']
def one(arg):
 g,tag,j,L=arg;f=S/'data'/tag/(j['id']+'.json.gz')
 with gzip.open(f,'rt') as h:d=json.load(h)
 assert d['input']==j;tr=d['trace'];es=tr['events'];cs={c['k']:c for c in tr['cuts']}
 au=audit_search(tr,j['moves'],'clean',j['seed'],128,j['sources'][0],0.) if g==2 else audit9((tag,j))
 err={k:au[k] for k in ('max_score','max_probability','max_log')};assert len(es)==128
 p=np.array(cs[4]['prior']);r=tr['evaluations'][0];val=r['clean' if g==2 else 'value']
 assert L['moves']==j['moves'] and L['pair']==j['sources'] and L['hash']==tr['state_hash']
 e=float(abs(p-np.asarray(L['p'])).max());v=abs(val-L['v']);assert e<2e-7 and v<1e-7,(e,v)
 ch={q['a']:q for q in L['children']};acts=np.array([q['actions'][0] for q in es]);z=np.array([q['root_value'] for q in es]);n0=0;ce=0.
 for q in es:
  if len(q['actions'])!=1:continue
  a=q['actions'][0];s=ch[a];assert s['terminal']==q['terminal']
  if not q['terminal']:
   ev=tr['evaluations'][q['eval_id']];assert ev['state_hash']==s['hash'] and ev['side']==s['side']
  er=abs(q['root_value']-s['root_v']);assert er<1e-7,(g,j['id'],a,er);ce=max(ce,er);n0+=1
 x=[float(np.isin(acts,j['sources']).mean()),float(z.mean())]
 u=[q['root_selection']['law']['u'] for q in es]
 rec=dict(path=str(f.relative_to(B)),sha256=sha(f),bytes=f.stat().st_size,family=g,core=j['core'],rep=j['rep'],seed=j['seed'],errors=err,root_prior_error=e,root_value_error=v,depth1_error=ce,depth1_events=n0,eval_calls=len(tr['evaluations']),elapsed_ns=tr['elapsed_ns'])
 return rec,x,u
def main():
 r=read(S/'REGISTRATION.json');m=read(S/'MODEL84.json');a=read(S/'LANDSCAPE84.json');assert m['landscape_sha']==sha(S/'LANDSCAPE84.json') and a['status']=='COMPLETE'
 ls={(x['family'],x['core']):x for x in a['rows'] if x['group']==3};assert len(ls)==144
 args=[]
 for g,tag in enumerate(TAGS):
  p=S/'plans'/f'{tag}.json';assert sha(p)==r['plans'][tag]['sha256'];jobs=read(p);assert len(jobs)==768 and {j['id']+'.json.gz' for j in jobs}=={f.name for f in (S/'data'/tag).glob('*.gz')}
  args.extend((g,tag,j,ls[g,j['core']]) for j in jobs)
 out=[]
 with cf.ProcessPoolExecutor(max_workers=8) as pool:
  for i,x in enumerate(pool.map(one,args,chunksize=8)):
   out.append(x)
   if (i+1)%384==0:print('AUDIT',i+1,flush=True)
 rec=[x[0] for x in out];X=np.array([x[1] for x in out]);M=np.array([[3,r['family'],r['core'],84,r['rep']] for r in rec]);assert len(set(map(tuple,M)))==2304
 streams={}
 for x,_,u in out:
  sd=x['seed']
  if sd in streams:assert streams[sd]==u
  streams[sd]=u
 assert len(streams)==1536
 dump(S/'RNG_INPUT84.json',[dict(seed=sd,u=u) for sd,u in sorted(streams.items())])
 p=B/'mqft_restore61_20261126/mqft_trained_rng61';q=subprocess.run([str(p),str(S/'RNG_INPUT84.json')],capture_output=True,text=True,timeout=30,check=True);u=json.loads(q.stdout);assert u['streams']==1536 and u['independent_uniforms']==196608 and u['max_error']<1e-12;u.update(binary_sha=sha(p),input_sha=sha(S/'RNG_INPUT84.json'));dump(S/'RNG_AUDIT84.json',u)
 np.savez_compressed(S/'FRESH84.npz',meta=M,X=X)
 old=read(B/'mqft_restore82_20261127/AUDIT82.json');assert len(old['raw_receipts'])==4032
 prior=[dict(path=x['path'],sha256=x['sha256'],role='development_R82') for x in old['raw_receipts']]
 # One authenticated development path per root/family, check direct landscape equality and observed depth1 marks.
 dev=0;dn=0;err=0.;rp=0.
 for x in old['raw_receipts']:
  if x['rep']!=0 or x['stage']==79:continue
  f=B/x['path'];assert sha(f)==x['sha256']
  with gzip.open(f,'rt') as h:d=json.load(h)
  tr=d['trace'];g=x['family'];L0=next(z for z in a['rows'] if [z['family'],z['group'],z['core']]==[g,x['group'],x['core']])
  p0=np.array(tr['cuts'][0]['prior']);e=float(abs(p0-L0['p']).max());rp=max(rp,e);assert e<2e-7 and tr['state_hash']==L0['hash']
  v0=tr['evaluations'][0]['clean' if g==2 else 'value'];assert abs(v0-L0['v'])<1e-7
  ch={v['a']:v for v in L0['children']}
  for q in tr['events']:
   if len(q['actions'])!=1:continue
   z=ch[q['actions'][0]];assert q['terminal']==z['terminal']
   if not q['terminal']:assert tr['evaluations'][q['eval_id']]['state_hash']==z['hash']
   er=abs(q['root_value']-z['root_v']);assert er<1e-7;err=max(err,er);dn+=1
  dev+=1
 assert dev==312
 q=dict(paths=2304,backups=294912,prior_paths=4032,prior_backups=516096,raw_bytes=sum(r['bytes'] for r in rec),errors={k:max(r['errors'][k] for r in rec) for k in rec[0]['errors']},max_root_prior_error=max(r['root_prior_error'] for r in rec),max_root_value_error=max(r['root_value_error'] for r in rec),max_child_mark_error=max(r['depth1_error'] for r in rec),fresh_depth1_events=sum(r['depth1_events'] for r in rec),dev_representative_paths=dev,dev_depth1_events=dn,dev_root_prior_error=rp,dev_child_mark_error=err,rng=u,array_sha=sha(S/'FRESH84.npz'),raw_receipts=rec,prior_raw_receipts=prior,scope='Full fresh path law replay, direct root and observed depth1 sign/identity checks. R82 authenticated development array; representative raw replay per development root; all source bytes rehashed at publication. No feature leakage.')
 dump(S/'AUDIT84.json',q);print('AUDIT',{k:v for k,v in q.items() if k not in ('raw_receipts','prior_raw_receipts')},flush=True)
if __name__=='__main__':main()
