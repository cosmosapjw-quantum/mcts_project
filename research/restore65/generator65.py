"""Exact four-draw instrument generating function vs frozen multinomial.
Lueders effects commute within a frozen score wave: two-component multinomial mixture."""
from audit65 import *
from scipy.special import gammaln
YGRID=np.array([(i,j,4-i-j) for i in range(5) for j in range(5-i)],int)
def scan(arg):
 tag,j=arg
 with gzip.open(S/'data'/tag/(j['id']+'.json.gz'),'rt') as f:d=json.load(f)
 assert d['input']==j;es=d['trace']['events'];a,b=j['sources'];out=[]
 for t in range(32):
  v=es[4*t:4*t+4];ph=v[0]['root_selection']['law']['phase65'];assert ph['wave']==t
  count=np.zeros(3,int)
  for e in v:
   ac=e['actions'][0];count[0 if ac==a else 1 if ac==b else 2]+=1
  out.append([*ph['prepared'],ph['pair_mass'],ph['population_a'],*count])
 return out
def mass(p):
 return np.exp(gammaln(5)-gammaln(YGRID+1).sum(1)+np.log(p)@YGRID.T)
def main():
 r=read(S/'REGISTRATION.json');a=read(S/'AUDIT65.json')
 js=[(tag,j) for tag in ('phase0','phase3') for j in read(S/f'plans/{tag}.json') if j['arm'] in ('persist_plus','persist_minus')]
 assert len(js)==1536
 with cf.ProcessPoolExecutor(max_workers=12) as pool:raw=np.asarray(list(pool.map(scan,js,chunksize=8)))
 assert raw.shape==(1536,32,8)
 v=raw[:,:,:3];m=raw[:,:,3];r0=raw[:,:,4];y=raw[:,:,5:].astype(int)
 z=v[:,:,1]*np.sin(.7)+v[:,:,2]*np.cos(.7);d=(1+z)/2
 e0=.35*r0+.65;e1=.35*r0
 P0=np.stack([m*e0,m*(1-e0),1-m],-1);P1=np.stack([m*e1,m*(1-e1),1-m],-1);P=d[:,:,None]*P0+(1-d[:,:,None])*P1
 M=d[:,:,None]*mass(P0)+(1-d[:,:,None])*mass(P1);F=mass(P)
 assert abs(M.sum(-1)-1).max()<1e-12 and abs(F.sum(-1)-1).max()<1e-12
 mu=4*P;delta=P0-P1
 cov=4*(np.eye(3)[None,None,:,:]*P[:,:,:,None]-P[:,:,:,None]*P[:,:,None,:])+12*d[:,:,None,None]*(1-d[:,:,None,None])*(delta[:,:,:,None]*delta[:,:,None,:])
 c=YGRID[None,None,:,:]-mu[:,:,None,:]
 cv=np.einsum('nty,ntyi,ntyj->ntij',M,c,c)
 errmean=float(abs(M@YGRID-mu).max());errcov=float(abs(cv-cov).max())
 assert errmean<1e-11 and errcov<1e-11
 idx=np.argmax(np.all(y[:,:,None,:]==YGRID,axis=-1),axis=-1);assert np.array_equal(YGRID[idx],y)
 pr=np.take_along_axis(M,idx[:,:,None],-1)[:,:,0];pf=np.take_along_axis(F,idx[:,:,None],-1)[:,:,0]
 assert np.all(pr>0) and np.all(pf>0)
 nll=-np.log(pr);nll0=-np.log(pf)
 # Matrix superoperator independently verifies mixture on a fixed deterministic subset of wave states.
 err=0.;matchecks=0
 for i in range(0,len(js),8):
  for t in (0,7,15,31):
   vv=v[i,t];mm=m[i,t];rr=r0[i,t];_,A,Bm=effects(rr);K=[np.sqrt(mm)*A,np.sqrt(mm)*Bm,np.sqrt(1-mm)*I]
   for J in (np.array([-.2,.1,0.]),np.zeros(3),np.array([.2,-.1,0.])):
    state=rho(vv)
    for _ in range(4):state=sum(np.exp(j)*k@state@dagger(k) for j,k in zip(J,K))
    zz=float(np.trace(state).real);want=float(M[i,t]@np.exp(YGRID@J));err=max(err,abs(zz-want));matchecks+=1
 assert err<2e-12
 meta=np.array([[j['generation'],j['core'],j['rep'],int(j['arm']=='persist_minus')] for _,j in js])
 rng=np.random.default_rng(650401);boot=rng.integers(0,48,(10000,48))
 def stat(v):
  v=np.asarray(v);assert len(v)==48
  return dict(mean=v.mean(0).tolist(),ci95=np.quantile(v[boot].mean(1),[.025,.975],axis=0).tolist())
 out=[]
 for g in (0,3):
  for arm in (0,1):
   mask=(meta[:,0]==g)&(meta[:,3]==arm);ix=np.flatnonzero(mask);ix=ix[np.lexsort((meta[ix,2],meta[ix,1]))]
   assert len(ix)==384
   agg=lambda q:stat(q[ix].reshape((48,8,32)+q.shape[2:]).mean((1,2)))
   res=y-mu;C0=4*(np.eye(3)[None,None,:,:]*P[:,:,:,None]-P[:,:,:,None]*P[:,:,None,:])
   out.append(dict(generation=g,arm=['persist_plus','persist_minus'][arm],n_waves=len(ix)*32,nll=agg(nll),iid_nll=agg(nll0),difference=agg(nll-nll0),mean_residual=agg(res),covariance_residual=agg(res[:,:,:,None]*res[:,:,None,:]-cov),iid_covariance_residual=agg(res[:,:,:,None]*res[:,:,None,:]-C0),extra_variance_a=agg((cov-C0)[:,:,0,0]),mean_variance_a=agg(cov[:,:,0,0])))
 np.savez_compressed(S/'GENERATOR65.npz',meta=meta,first_bloch=v,p0=P0,p1=P1,weight=d,counts=y,pred=M,iid=F,grid=YGRID)
 report=dict(scope='exact frozen wave for persistent plus/minus only, commuting within-wave effects; conditional mixture, not autonomous cross-wave closure',n_waves=int(len(js)*32),matrix_checks=matchecks,max_matrix_mgf_error=err,max_mean_error=errmean,max_covariance_error=errcov,rows=out,array_sha=sha(S/'GENERATOR65.npz'))
 dump(S/'GENERATOR65.json',report);print(report)
if __name__=='__main__':main()
