"""Independent array-level verification: raw artifact hashes, direct energy/path ledger,
Gibbs CDF RNG replay for both directions, and independent exact tilted matrix checks."""
from base61 import *
from scipy.special import logsumexp
BETA=10/3
def independent_pi(E):
 p=np.exp(-BETA*(E-E.min(axis=1,keepdims=True)));return p/p.sum(1,keepdims=True)
def independent_sample(pi,n,seed,alpha,reverse):
 rng=np.random.default_rng(seed);K=len(pi)-1
 def choose(p):
  c=np.cumsum(p);c/=c[-1];return np.searchsorted(c,rng.random(n),side='right')
 x=choose(pi[-1] if reverse else pi[0]);seq=np.empty((n,K+1),np.int16);seq[:,K if reverse else 0]=x
 for j in (range(K,0,-1) if reverse else range(1,K+1)):
  refresh=rng.random(n)<alpha;candidate=choose(pi[j]);x=np.where(refresh,candidate,x)
  seq[:,j-1 if reverse else j]=x
 return seq
def ledger(E,a,seq):
 pi=independent_pi(E);lF=np.log(pi[0,seq[:,0]]);lR=np.log(pi[-1,seq[:,-1]]);w=np.zeros(len(seq));q=w.copy()
 for j in range(1,len(E)):
  x=seq[:,j-1];y=seq[:,j];same=(x==y)*(1-a)
  lF+=np.log(same+a*pi[j,y]);lR+=np.log(same+a*pi[j,x])
  w+=E[j,x]-E[j-1,x];q+=E[j,y]-E[j,x]
 return w,q,lF,lR
def one(arg):
 kind,i,d,r=arg;path=S/'data'/kind/r['file'];assert sha(path)==r['sha256'] and path.stat().st_size==r['bytes']
 E=np.asarray(d['energy']);p=independent_pi(E);a=d['legal'].index(d['source']);b=d['legal'].index(d['source_b']);pair=p[4,[a,b]];pair/=pair.sum();n=r['n']
 with np.load(path) as z:
  f=z['forward'];v=z['reverse_as_forward'];assert len(f)==len(v)==n
  if kind=='thermal':
   a=r['alpha'];pi=independent_pi(E)
   seed=660101+4*i+2*int(a==1)
   assert np.array_equal(f,independent_sample(pi,n,seed,a,False))
   assert np.array_equal(v,independent_sample(pi,n,seed+1,a,True))
   W,Q,lf,lr=ledger(E,a,f);WR,QR,lfr,lrr=ledger(E,a,v)
   pairs=[(W,z['W']),(Q,z['Q']),(-WR,z['W_reverse']),(-QR,z['Q_reverse']),(lf,z['logPF']),(lr,z['logPR']),(lfr,z['logPF_on_reverse']),(lrr,z['logPR_on_reverse']),(lf-lr,z['sigma']),(lrr-lfr,z['sigma_reverse'])]
  else:
   K=r['steps']
   def interp(q,t):
    u=np.linspace(np.log(q[0]/q[1]),np.log(t[0]/t[1]),K+1)
    pp=np.stack([1/(1+np.exp(-u)),1/(1+np.exp(u))],-1);return -np.log(pp)/BETA
   if kind=='erasure':
    T=interp(pair,np.array([.05,.95]));pi=independent_pi(T);seed=6700000+i*16+2*int(np.log2(K))
    assert np.array_equal(f,independent_sample(pi,n,seed,1.,False));assert np.array_equal(v,independent_sample(pi,n,seed+1,1.,True))
    W,Q,lf,lr=ledger(T,1.,f);WR,QR,lfr,lrr=ledger(T,1.,v)
    W+=T[0,f[:,0]]-T[-1,f[:,-1]];WR+=T[0,v[:,0]]-T[-1,v[:,-1]]
    pairs=[(W,z['W']),(-WR,z['W_reverse']),(W+Q,np.zeros(n)),(WR+QR,np.zeros(n))]
   else:
    D=np.array([[.8,.2],[.2,.8]]);pm=pair@D;post=(pair[:,None]*D/pm).T
    seed=6800000+i*16+2*int(np.log2(K));pairs=[]
    for reverse,arr,mkey,wkey,ikey,skey,sd in [(False,f,'m_forward','W','information','sigma',seed),(True,v,'m_reverse','W_reverse','information_reverse_as_forward','sigma_reverse',seed+1)]:
     m=z[mkey];rng=np.random.default_rng(sd);assert np.array_equal(m,np.searchsorted(np.cumsum(pm)/pm.sum(),rng.random(n),side='right'))
     for j in (0,1):
      ix=np.flatnonzero(m==j);T=interp(post[j],pair);pi=independent_pi(T);s=arr[ix]
      assert np.array_equal(s,independent_sample(pi,len(ix),sd+100000+100000*j,1.,reverse))
      W,Q,lf,lr=ledger(T,1.,s);I=np.log(post[j,s[:,0]]/pair[s[:,0]]);W-=I/BETA
      sign=-1 if reverse else 1
      pairs += [(sign*W,z[wkey][ix]),(I,z[ikey][ix]),(sign*(lf-lr),z[skey][ix]),(lf-lr-BETA*W-I,np.zeros(len(ix)))]
  err=max(float(np.max(abs(x-y))) for x,y in pairs);assert err<1e-10,(kind,r['file'],err)
 return dict(kind=kind,file=r['file'],max_reconstruction_error=err,paths=n*2,transitions=n*2*(f.shape[1]-1),sha256=r['sha256'])
def main():
 reg=read(S/'REGISTRATION.json');ds=read(S/'THERMAL_SCHEDULES.json');mp={}
 for i,d in enumerate(ds):d['source_b']=reg['cores'][d['core']]['sources'][1];mp[d['generation'],d['core']]=(i,d)
 jobs=[]
 for kind in ('thermal','erasure','feedback'):
  for r in read(S/(kind.upper()+'66.json'))['results']:
   i,d=mp[r['generation'],r['core']];jobs.append((kind,i,d,r))
 with cf.ProcessPoolExecutor(max_workers=12) as pool:out=list(pool.map(one,jobs,chunksize=2))
 report=dict(scope='all stored auxiliary trajectories independently reconstructed from fixed seeds via explicit inverse CDF; all energy/likelihood arrays independently recomputed',files=len(out),paths=sum(x['paths'] for x in out),transitions=sum(x['transitions'] for x in out),max_reconstruction_error=max(x['max_reconstruction_error'] for x in out),records=out)
 dump(S/'VERIFY_AUX66.json',report);print({k:v for k,v in report.items() if k!='records'})
if __name__=='__main__':main()
