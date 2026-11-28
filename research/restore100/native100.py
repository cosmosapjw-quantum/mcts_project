from base61 import *
from scipy.special import logsumexp
from scipy.stats import t as student
import gzip,time,sys,platform
O=B/'mqft_restore99_20261129';R87=B/'mqft_restore87_20261128';LS=[13,15,17,19,21];HS=[16,32,64]
FIELDS=['Sigma','conditional_KL','centered','conditional_var','logP','logQ','score_switch','score_action_change','score_delta']
def ck(e,k,a,b,tol=1e-9):
 d=float(np.max(abs(np.asarray(a)-np.asarray(b))));e[k]=max(e.get(k,0.),d);assert d<tol,(k,d)
def ci(x):
 x=np.asarray(x);m=x.mean(0);w=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return np.stack([m,m-w,m+w])
def one(arg):
 L,r=arg;f=B/r['path'];assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256']
 with gzip.open(f,'rt') as z:hd,row,end=map(json.loads,z)
 tr=row['trace'];sp=tr['spec'];k=r['budget'];beta=sp['root_law']['beta_p'][0];assert end['backups']==k==sp['budget'] and sp['seed']==r['seed'];assert beta==[0.,10/3,8.,16.,math.log(L*L)][r['beta_index']]
 es=sorted(tr['replicas'][0]['events'],key=lambda e:e['root_selection']['clock']);assert len(es)==k
 legal=np.flatnonzero(np.array(tr['root']['board'])==0);n=len(legal);u=np.ones(n)/n;prev=u;prvscore=None;prvact=None;first=None;err={};data=[]
 for j,e in enumerate(es):
  law=e['root_selection']['law'];ac=np.array(law['actions']);o=np.argsort(ac);p=np.array(law['p'])[o];s=np.array(law['scores'])[o];assert np.array_equal(ac[o],legal) and law['t']==j and all(law['active']) and law['logit_h']==0 and ac[law['picked']]==e['actions'][0]
  idx=int(np.searchsorted(legal,e['actions'][0]));assert legal[idx]==e['actions'][0] and p.min()>0
  lp=np.log(p);z=beta*s;ck(err,'raw_softmax',lp,z-logsumexp(z),1e-9)
  en=-s;first=en[idx] if first is None else first
  w=0 if prvscore is None else en[prvact]-prvscore[prvact];qscore=0 if prvscore is None else en[idx]-en[prvact];d=en[idx]-first
  zz=[]
  for q in [u,.5*u+.5*prev]:
   l=lp-np.log(q);m=float(p@l);v=float(p@((l-m)**2));ck(err,'conditional_IFT',p@np.exp(-l),1);ck(err,'conditional_KL_direct',m,float(np.sum(p*np.log(p/q))));assert m>=-1e-10
   zz.append([l[idx],m,l[idx]-m,v,lp[idx],np.log(q[idx]),w,qscore,d])
  data.append(zz);prev=p;prvscore=en;prvact=idx
 a=np.asarray(data);v=np.cumsum(a,axis=0);v[:,:,8]=a[:,:,8];ck(err,'score_ledger',v[:,:,6]+v[:,:,7],v[:,:,8]);ck(err,'likelihood_sum',v[:,:,0],v[:,:,4]-v[:,:,5]);ck(err,'compensator',v[:,:,2],v[:,:,0]-v[:,:,1])
 if beta==0:ck(err,'uniform_control',v[:,:,:3],0)
 return [r[x] for x in ['cohort','core','arm','beta_index','rep','budget']],v,err
def main():
 t=time.monotonic();m=read(O/'RUN_MANIFEST99.json');raw=m['prior_raw_receipts'];assert len(raw)==4320
 groups=[];arr={};errors={};sources={};draws=0;prefix=0;count=0
 for L in LS:
  p=R87/f'AUDIT_L{L}.json';a=read(p)['raw_receipts'];assert len(a)==864
  # authenticate audit via R98 manifest
  r98=read(B/'mqft_restore98_20261128'/'RUN_MANIFEST98.json');k=str(p.relative_to(B));assert sha(p)==r98['inputs'][k];sources[k]=sha(p)
  assert all(raw[x['path']]==x['sha256'] for x in a)
  with cf.ProcessPoolExecutor(max_workers=10) as pool:res=list(pool.map(one,[(L,x) for x in a],chunksize=4))
  M=[];V=[];T=np.full((len(res),64,2,len(FIELDS)),np.nan);meta=[]
  for i,(mm,v,e) in enumerate(res):
   T[i,:len(v)]=v;meta.append(mm);draws+=len(v)
   for h in HS:
    if len(v)>=h:M.append(mm+[h]);V.append(v[h-1]);prefix+=1
   for k,x in e.items():errors[k]=max(errors.get(k,0.),x)
  M=np.array(M);V=np.array(V);assert len(M)==2080
  arr.update({f'M{L}':M,f'V{L}':V,f'T{L}':T,f'P{L}':np.array(meta)})
  for co,arm,bi,bud,h in sorted(set(tuple(x[[0,2,3,5,6]]) for x in M)):
   ix=np.all(M[:,[0,2,3,5,6]]==[co,arm,bi,bud,h],axis=1);root=np.stack([V[ix&(M[:,1]==c)].mean(0) for c in range(8)]);z=V[ix];logw=-z[:,:,0];lw=logsumexp(logw,axis=0);ess=np.exp(2*lw-logsumexp(2*logw,axis=0))
   groups.append(dict(L=L,cohort=int(co),arm=int(arm),bi=int(bi),beta=[0.,10/3,8.,16.,math.log(L*L)][bi],budget=int(bud),h=int(h),roots=8,reps=int(ix.sum()/8),ci=ci(root).tolist(),log_sample_IFT=(lw-np.log(len(z))).tolist(),ess=ess.tolist(),samples=len(z)))
  print('NATIVE L',L,len(res),draws,flush=True)
 assert draws==215040 and prefix==10400 and len(groups)==400
 np.savez_compressed(S/'NATIVE100.npz',**arr)
 dump(S/'NATIVE100.json',dict(status='PASS_SCOPED',paths=4320,draws=draws,prefix_rows=prefix,groups=groups,fields=FIELDS,comparators=['uniform','half_uniform_half_previous'],checks=errors,raw_receipts=raw,source_hashes=sources,array_sha=sha(S/'NATIVE100.npz'),elapsed_s=time.monotonic()-t,scope='Recorded root-law likelihood martingale versus predictable comparator; finite sampled IFT not required to equal 1; score ledger not physical work/heat or actual reverse.'))
 print('NATIVE DONE',errors,flush=True)
if __name__=='__main__':main()
