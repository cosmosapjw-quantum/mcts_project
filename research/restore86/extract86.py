"""Reauthenticate actual fixed-root histories; calculate quiescent wave data."""
from base61 import *
from audit_native74 import one as audit9
from audit_rapfi74 import audit_search
import gzip,subprocess
TAGS=['schedule0','schedule3','rapfi']
def one(arg):
 st,group,g,core,rep,rec0=arg
 f=B/rec0['path'];h=sha(f)
 if 'sha256' in rec0:assert h==rec0['sha256']
 with gzip.open(f,'rt') as q:d=json.load(q)
 j=d['input'];tr=d['trace'];es=tr['events'];cs={c['k']:c for c in tr['cuts']}
 assert j['core']==core and j['rep']==rep and len(es)==128 and set(cs)=={4,8,16,32,64,128}
 if st==86:
  au=audit_search(tr,j['moves'],'clean',j['seed'],128,j['sources'][0],0.) if g==2 else audit9((TAGS[g],j))
 else:au={}
 prior=np.array(cs[4]['prior'],float);N=len(prior);legal=np.flatnonzero(prior>0);aa,bb=j['sources'];assert aa in legal and bb in legal and aa!=bb
 assert set(legal)==set(range(N))-set(j['moves'])
 n=np.zeros(N,np.int64);w=np.zeros(N);s2=np.zeros(N);seen=collections.Counter();lp=0.;err=collections.defaultdict(float)
 z=np.array([e['root_value'] for e in es]);act=np.array([e['actions'][0] for e in es]);hit=np.isin(act,[aa,bb])
 C=[];W=[];P=[];NH=[];WH=[];TH=[];U=[]
 def key(v):return hashlib.sha256(v).hexdigest()
 for start in range(0,128,4):
  Q=np.divide(w.astype(np.float32),n.astype(np.float32),out=np.zeros(N,np.float32),where=n>0)
  score=Q+np.float32(1.4)*prior.astype(np.float32)*np.sqrt(np.float32(start))/(np.float32(1)+n.astype(np.float32))
  NH.append(key(n.tobytes()));WH.append(key(n.tobytes()+w.tobytes()));TH.append(key(repr(sorted(seen.items())).encode()))
  p0=None
  for k in range(start,start+4):
   e=es[k];l=e['root_selection']['law'];a=np.asarray(l['actions']);sc=np.asarray(l['scores']);p=np.asarray(l['p']);assert np.array_equal(np.sort(a),legal) and all(l['active']) and l['beta_p']==10/3 and l['logit_h']==0 and l['t']==k and e['complete_seq']==k
   if st==81:assert j['arm']=='base81' and l['phase65']['lambda81']==0
   else:assert j['arm']=='zero74' and l['source']==aa
   err['score']=max(err['score'],float(abs(sc-score[a]).max()));zz=sc*(10/3);q=np.exp(zz-zz.max());q/=q.sum()
   err['probability']=max(err['probability'],float(abs(q-p).max()));assert np.all(p>0) and abs(p.sum()-1)<1e-12
   p1=p[np.argsort(a)]
   if p0 is None:p0=p1
   else:err['wave']=max(err['wave'],float(abs(p1-p0).max()))
   pick=int(l['picked']);u=l['u'];assert int(np.searchsorted(np.cumsum(p),u,side='right'))==pick and a[pick]==act[k]
   lp+=math.log(p[pick]);err['log_path']=max(err['log_path'],abs(lp-l['log_p_path']))
   assert z[k]==(-1)**len(e['actions'])*e['leaf_value'];U.append(u)
  C.append(int(hit[start:start+4].sum()));W.append(float(z[start:start+4].sum()));P.append(float(p0[np.isin(legal,[aa,bb])].sum()))
  for k in range(start,start+4):
   a=act[k];n[a]+=1;w[a]+=z[k];s2[a]+=z[k]**2
   for d0 in range(1,len(es[k]['actions'])+1):seen[tuple(es[k]['actions'][:d0])]+=1
  if start+4 in cs:
   c=cs[start+4];assert np.array_equal(n,c['n']) and abs(w-c['w']).max()<1e-9 and abs(s2-c['s2']).max()<1e-9 and np.array_equal(prior,c['prior'])
 assert err['score']<2e-6 and err['probability']<2e-12 and err['log_path']<1e-9 and err['wave']<2e-12
 receipt=dict(stage=st,group=group,family=g,core=core,rep=rep,seed=j['seed'],path=str(f.relative_to(B)),sha256=h,bytes=f.stat().st_size,errors=dict(err),full_fresh_audit={k:au[k] for k in ('max_score','max_probability','max_log')} if st==86 else None)
 return receipt,C,W,P,NH,WH,TH,U
def main():
 O82=B/'mqft_restore82_20261127';O84=B/'mqft_restore84_20261127';r=read(S/'REGISTRATION.json');r0=read(O84/'REGISTRATION.json')
 assert r['weights']==r0['weights'] and r['external']==r0['external'] and sha(S/'mqft_pulse74')==r['pulse_binary_sha']==r0['pulse_binary_sha']
 groups=[set(x['orbit'] for x in read(B/f'mqft_restore{st}_20261127/REGISTRATION.json')['cores']) for st in (80,81,82,84,86)]
 assert list(map(len,groups))==[48,24,32,48,24] and all(not a&b for i,a in enumerate(groups) for b in groups[i+1:])
 args=[]
 for x in read(O82/'AUDIT82.json')['raw_receipts']:
  args.append((x['stage'],0,x['family'],[0,48,72][x['group']]+x['core'],x['rep'],x)) if False else None
  # source job retains local core; canonical core mapping applied after reduction
  args.append((x['stage'],x['group'],x['family'],x['core'],x['rep'],x))
 for x in read(O84/'AUDIT84.json')['raw_receipts']:args.append((84,3,x['family'],x['core'],x['rep'],x))
 rap={x['job']['id']:x for x in read(S/'RAPFI_RECEIPTS86.json')}
 for g,tag in enumerate(TAGS):
  p=S/'plans'/f'{tag}.json';assert sha(p)==r['plans'][tag]['sha256']
  for j in read(p):
   f=S/'data'/tag/(j['id']+'.json.gz');q=dict(path=str(f.relative_to(B)))
   if g==2:q['sha256']=rap[j['id']]['sha256']
   args.append((86,4,g,j['core'],j['rep'],q))
 assert len(args)==7488
 out=[]
 with cf.ProcessPoolExecutor(max_workers=8) as pool:
  for i,x in enumerate(pool.map(one,args,chunksize=8)):
   out.append(x)
   if (i+1)%768==0:print('AUDIT',i+1,flush=True)
 rec=[x[0] for x in out];M=np.array([[x[k] for k in ('group','family','core','stage','rep')] for x in rec]);assert len(set(map(tuple,M)))==7488
 C,W,P,NH,WH,TH,U=map(np.stack,zip(*(x[1:] for x in out)))
 # Match immutable R82/R84 endpoint reductions exactly within float tolerance.
 for st,f0 in [(82,O82/'POPULATION82.npz'),(84,O84/'FRESH84.npz')]:
  with np.load(f0) as a:
   m0=a['meta'];x0=a['X']
  lookup={tuple(m):i for i,m in enumerate(M)}
  i=[lookup[tuple(m)] for m in m0]
  assert abs(C[i].sum(1)/128-x0[:,0]).max()<1e-14 and abs(W[i].sum(1)/128-x0[:,1]).max()<1e-14
 # Check all seed identity scopes and fresh RNG stream replay.
 seen={};streams={}
 for i,x in enumerate(rec):
  key=(x['group'],x['core'],x['stage'],x['rep'],x['family']==2);sd=x['seed']
  if sd in seen:assert seen[sd]==key and np.array_equal(streams[sd],U[i])
  seen[sd]=key;streams[sd]=U[i]
 fresh={x['seed'] for x in rec if x['group']==4};assert len(fresh)==768
 dump(S/'RNG_INPUT86.json',[dict(seed=k,u=streams[k].tolist()) for k in sorted(fresh)])
 p=B/'mqft_restore61_20261126/mqft_trained_rng61';q=subprocess.run([str(p),str(S/'RNG_INPUT86.json')],capture_output=True,text=True,timeout=40,check=True);u=json.loads(q.stdout)
 assert u['streams']==768 and u['independent_uniforms']==98304 and u['max_error']<1e-12
 u.update(binary_sha=sha(p),input_sha=sha(S/'RNG_INPUT86.json'));dump(S/'RNG_AUDIT86.json',u)
 np.savez_compressed(S/'WAVES86.npz',meta=M,count=C,mark=W,prob=P,counts_hash=NH,root_nw_hash=WH,prefix_multiset_hash=TH)
 a=dict(paths=1152,backups=147456,prior_paths=6336,prior_backups=811008,raw_bytes=sum(x['bytes'] for x in rec if x['group']==4),array_sha=sha(S/'WAVES86.npz'),raw_receipts=rec,errors={k:max(x['errors'][k] for x in rec) for k in ('score','probability','log_path','wave')},rng=u,scope='All 7488 clean histories reauthenticated; complete fresh native downstream audit, all recorded conditional probabilities, cuts, marks and seed identity. Hashes of exact root N,W and prefix multiplicity at wave starts.')
 dump(S/'AUDIT86.json',a);print('AUDIT COMPLETE',len(rec),a['errors'],a['raw_bytes'],flush=True)
if __name__=='__main__':main()
