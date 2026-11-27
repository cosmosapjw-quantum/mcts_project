"""Authenticated R74 forward path and explicit inverse bookkeeping, not a reverse MCTS."""
from base61 import *
import gzip
from scipy.stats import t as student
O=B/'mqft_restore74_20261126';ARMS=['zero74','plus74','minus74'];TAGS=['schedule0','schedule3','rapfi'];FAMS=['g0','g3','Rapfi'];H=.35;b=10/3;NS=2048
def ci(x):
 x=np.asarray(x);m=x.mean(0);h=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return [m.tolist(),(m-h).tolist(),(m+h).tolist()]
def readgz(p):
 with gzip.open(p,'rt') as h:return json.load(h)
def one(arg):
 g,tag,j,r=arg;p=O/'data'/tag/(j['id']+'.json.gz');assert sha(p)==r['sha256']
 d=readgz(p);assert d['input']==j;tr=d['trace'];es=tr['events'];cuts={x['k']:x for x in tr['cuts']};assert len(es)==128 and set(cuts)=={4,8,16,32,64,128}
 n=np.array(cuts[128]['n'],int);w=np.array(cuts[128]['w'],float);s2=np.array(cuts[128]['s2'],float);N=len(n);src=j['sources'][0];legal=sorted(set(range(N))-set(j['moves']));ref=collections.Counter(tuple(e['actions'][:i]) for e in es for i in range(1,len(e['actions'])+1))
 # Preserve full ordered event records, pop last event; reversible ledger, not native MCTS memory.
 maxerr=0.;maxp=0.;maxlog=0.;lp=0.;p0=None;act=[];prefix=[]
 for k,e in enumerate(es):
  l=e['root_selection']['law'];a=np.asarray(l['actions']);sc=np.asarray(l['scores']);pr=np.asarray(l['p']);assert sorted(a.tolist())==legal and l['t']==k and l['beta_p']==b and all(l['active'])
  z=b*sc;z-=z.max();q=np.exp(z);q/=q.sum();ix=int(np.flatnonzero(a==src)[0]);h=([0,H,-H][ARMS.index(j['arm'])] if 32<=k<36 else 0.)
  assert l['logit_h']==h
  want=q.copy();want[ix]*=np.exp(h);want/=want.sum();maxp=max(maxp,float(abs(want-pr).max()))
  assert maxp<1e-12 and np.all(pr>0) and abs(pr.sum()-1)<1e-12
  pick=l['picked'];assert int(np.searchsorted(np.cumsum(pr),l['u'],side='right'))==pick and a[pick]==e['actions'][0]
  lp+=math.log(pr[pick]);maxlog=max(maxlog,abs(lp-l['log_p_path']),abs(math.log(pr[pick])-l['log_p']));assert maxlog<1e-9
  if k<32:prefix.append((e['actions'],e['root_value'],e['leaf_hash'],l['u']))
  if 32<=k<36:
   order=np.argsort(a);v=q[order];u=sc[order]
   if p0 is None:p0=v;sc0=u
   else:assert np.array_equal(p0,v) and np.array_equal(sc0,u)
   act.append(int(np.searchsorted(legal,e['actions'][0])))
 # Reverse ledger audit: every previously recorded cut and total prefix reference multiplicities.
 for k in range(128,-1,-1):
  assert n.sum()==k and np.all(n>=0)
  if k in cuts:
   c=cuts[k];assert np.array_equal(n,c['n'])
   maxerr=max(maxerr,float(abs(w-c['w']).max()),float(abs(s2-c['s2']).max()))
   if 'visited_prefix_edges' in c:assert len(ref)==c['visited_prefix_edges']
  if k==0:break
  e=es[k-1];a=e['actions'][0];z=e['root_value'];assert z==(-1)**len(e['actions'])*e['leaf_value']
  n[a]-=1;w[a]-=z;s2[a]-=z*z
  for i in range(1,len(e['actions'])+1):
   key=tuple(e['actions'][:i]);assert ref[key]>0;ref[key]-=1
   if ref[key]==0:del ref[key]
 assert not ref and not n.any();maxerr=max(maxerr,float(abs(w).max()),float(abs(s2).max()));assert maxerr<1e-9
 ai=int(np.searchsorted(legal,src));hit=sum(a==ai for a in act);logs=[h*hit-4*np.log1p(p0[ai]*np.expm1(h)) for h in (H,-H)]
 return dict(family=g,core=j['core'],rep=j['rep'],arm=ARMS.index(j['arm']),seed=j['seed'],path=r['path'],sha256=r['sha256'],bytes=r['bytes'],ledger_error=maxerr,probability_error=maxp,log_error=maxlog,source_index=ai,legal=legal,score=sc0.tolist(),p=p0.tolist(),actions=act,hits=hit,log_Lplus_minus=logs,prefix_sha=hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest())
def aux(q,seed):
 p=np.array(q['p']);a=q['source_index'];n=len(p);p1=p.copy();p1[a]*=np.exp(H);p1/=p1.sum();x=p[a];y=p1[a]
 h=np.arange(5);bf=np.array([math.comb(4,int(k))*y**k*(1-y)**(4-k) for k in h]);br=np.array([math.comb(4,int(k))*x**k*(1-x)**(4-k) for k in h]);sf=bf[1:].sum();sr=br[1:].sum();assert sf>0 and sr>0
 # conditional forward on positive hit count; explicit conditional fine histories.
 rng=np.random.default_rng(seed);count=rng.choice(np.arange(1,5),size=NS,p=bf[1:]/sf)
 non=p.copy();non[a]=0;non/=non.sum()
 f=rng.choice(n,size=(NS,4),p=non);order=np.argsort(rng.random((NS,4)),axis=1);mask=order<count[:,None];f[mask]=a;assert np.array_equal((f==a).sum(1),count)
 # actual independent reverse simulation on endpoint counts, then delete proportionally.
 rngr=np.random.default_rng(seed+1);endpoint=rngr.multinomial(4,p,size=NS);c=endpoint.copy();r=np.zeros((NS,4),np.int16);logr=np.zeros(NS)
 # Reverse deletes last to first.
 for t in range(4,0,-1):
  u=rngr.random(NS)*t;idx=(np.cumsum(c,1)<=u[:,None]).sum(1);assert np.all(c[np.arange(NS),idx]>0)
  logr+=np.log(c[np.arange(NS),idx]/t);c[np.arange(NS),idx]-=1;r[:,t-1]=idx
 assert not c.any()
 # endpoint log multinomial = log4! - sum log(n_i!)+ n_i log p_i.
 fac=np.log(np.array([math.factorial(k) for k in range(5)]));le=fac[4]-fac[endpoint].sum(1)+endpoint@np.log(p)
 er=float(abs(le+logr-np.log(p[r]).sum(1)).max());assert er<1e-12
 rc=(r==a).sum(1);sig=np.log(p1[f]).sum(1)-np.log(sf)-np.log(p[f]).sum(1)
 exact_sig=h[1:]*np.log(y/x)+(4-h[1:])*np.log((1-y)/(1-x))-np.log(sf)
 assert abs(sig-exact_sig[count-1]).max()<1e-12
 wt=np.exp(-sig);exact_ift=float((bf[1:]/sf)@np.exp(-exact_sig));assert abs(exact_ift-sr)<1e-12
 mean=float((bf[1:]/sf)@exact_sig);bound=-np.log(sr);assert mean>=bound-1e-12
 r0=dict(family=q['family'],core=q['core'],rep=q['rep'],seed=seed,p0=x,pplus=y,s_forward=sf,s_reverse=sr,lambda_singular=1-sr,exact_ift=exact_ift,exact_mean_sigma=mean,jensen_lower=bound,mc_ift=float(wt.mean()),mc_ift_se=float(wt.std(ddof=1)/np.sqrt(NS)),mc_reverse_support=float(np.mean(rc>0)),mc_reverse_support_se=float(np.sqrt(np.mean(rc>0)*(1-np.mean(rc>0))/NS)),mc_mean_sigma=float(sig.mean()),ess=float(wt.sum()**2/(wt@wt)),reverse_log_error=er)
 return r0,f.astype(np.int16),r
def main():
 r=read(O/'REGISTRATION.json');a=read(O/'AUDIT74.json');by={x['path']:x for x in a['raw_receipts']};args=[]
 for g,tag in enumerate(TAGS):
  p=O/'plans'/f'{tag}.json';assert sha(p)==r['plans'][tag]['sha256']
  for j in read(p):
   f=O/'data'/tag/(j['id']+'.json.gz');args.append((g,tag,j,by[str(f.relative_to(B))]))
 assert len(args)==2304
 out=[]
 with cf.ProcessPoolExecutor(max_workers=8) as pool:
  for i,q in enumerate(pool.map(one,args,chunksize=8)):
   out.append(q)
   if (i+1)%384==0:print('NATIVE',i+1,flush=True)
 ix={(x['family'],x['core'],x['rep'],x['arm']):x for x in out};assert len(ix)==2304
 contexts=[];F=[];R=[];res=[]
 for g in range(3):
  for c in range(16):
   for i in range(16):
    z=[ix[g,c,i,k] for k in range(3)];assert len({q['prefix_sha'] for q in z})==1 and z[0]['p']==z[1]['p']==z[2]['p']
    v,f,r0=aux(z[0],85000000+2*len(contexts));contexts.append(v);F.append(f);R.append(r0)
  rows=[x for x in contexts if x['family']==g];assert len(rows)==256
  fields=['p0','s_reverse','lambda_singular','exact_mean_sigma','jensen_lower','mc_ift','mc_reverse_support','ess']
  q=np.array([[v[k] for k in fields] for v in rows]).reshape(16,16,-1)
  rr=np.array([[v['mc_ift']-v['exact_ift'],v['mc_reverse_support']-v['s_reverse']] for v in rows]).reshape(16,16,2)
  # Actual R74 full history log RN on common coordinates: reference generation + deterministic ledger deletion.
  plus=np.array([[ix[g,c,i,1]['log_Lplus_minus'][0] for i in range(16)] for c in range(16)])
  zero=np.array([[ix[g,c,i,0]['log_Lplus_minus'][0] for i in range(16)] for c in range(16)])
  res.append(dict(family=FAMS[g],fields=fields,mean_ci=ci(q.mean(1)),sample_minus_exact_ci=ci(rr.mean(1)),actual_plus_IFT_ci=ci(np.exp(-plus).mean(1)),actual_zero_inverse_IFT_ci=ci(np.exp(zero).mean(1)),actual_mean_logRN_ci=ci(plus.mean(1))))
 np.savez_compressed(S/'WAVE_SAMPLES85.npz',forward=np.array(F),reverse=np.array(R))
 dump(S/'NATIVE85.json',dict(paths=2304,backups=294912,fresh_native_paths=0,contexts=contexts,results=res,samples_per_direction=NS,auxiliary_samples=len(contexts)*NS*2,samples_sha=sha(S/'WAVE_SAMPLES85.npz'),max_ledger_error=max(x['ledger_error'] for x in out),max_probability_error=max(x['probability_error'] for x in out),max_log_error=max(x['log_error'] for x in out),max_reverse_log_error=max(x['reverse_log_error'] for x in contexts),raw_receipts=out,scope='Inverse ordered stored ledger and conditional one-wave reverse multinomial endpoint deletion; native plus/zero same-order law not genuine inverse growing engine. Positive-hit conditioned forward is distinct from unconditional native.'))
 print('NATIVE COMPLETE',len(out),len(contexts),len(contexts)*NS*2,flush=True)
if __name__=='__main__':main()
