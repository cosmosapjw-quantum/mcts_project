"""Explicit finite measurement-feedback thermodynamics on a pilot-native binary pair."""
from thermal66 import *
def setup(d,K):
 E0=np.asarray(d['energy']);pi,_=gibbs(E0);a=d['legal'].index(d['source']);b=d['legal'].index(d['source_b'])
 p=pi[4,[a,b]];p=p/p.sum()
 D=np.array([[.8,.2],[.2,.8]]) # x,m
 pm=p@D;post=(p[:,None]*D/pm).T # m,x
 E=-np.log(p)/BETA
 schedules=[]
 for q in post:
  u=np.linspace(np.log(q[0]/q[1]),np.log(p[0]/p[1]),K+1)
  z=np.stack([1/(1+np.exp(-u)),1/(1+np.exp(u))],1)
  schedules.append(-np.log(z)/BETA)
 return p,pm,post,E,np.array(schedules)
def draw(d,K,n,seed,rev=False):
 p,pm,post,E,schedules=setup(d,K)
 rng=np.random.default_rng(seed);m=rng.choice(2,n,p=pm);seq=np.empty((n,K+1),np.int8);W=np.empty(n);I=np.empty(n);sig=np.empty(n);err=0.
 for a in (0,1):
  ix=np.flatnonzero(m==a)
  # disjoint deterministic random streams for the two record values
  x=sample(schedules[a],1.,len(ix),seed+100000+100000*a,rev)
  s,w,q,lf,lr,e,b=x;ini=s[:,0]
  i=np.log(post[a,ini]/p[ini])
  wt=w-i/BETA
  # Includes p(m) in both path measures (cancels in ratio).
  assert abs(lf-lr-BETA*wt-i).max()<1e-11
  seq[ix]=s;W[ix]=wt;I[ix]=i;sig[ix]=lf-lr;err=max(err,e)
 return m,seq,W,I,sig,err
def job(arg):
 i,d,K,n=arg;p,pm,post,E,s=setup(d,K);info=float(np.sum(pm[:,None]*post*np.log(post/p)))
 diss=0.
 for a in (0,1):diss+=pm[a]*exact(s[a],1.)['meanW']*BETA
 assert info>0 and diss>=-1e-12
 mean=(-info+diss)/BETA
 sd=6800000+i*16+2*int(np.log2(K))
 mf,f,W,I,SF,err=draw(d,K,n,sd);mr,r,WR,IR,SR,er=draw(d,K,n,sd+1,True)
 name=f'g{d["generation"]}_c{d["core"]:02d}_k{K}.npz';path=S/'data/feedback'/name
 np.savez_compressed(path,m_forward=mf,m_reverse=mr,forward=f,reverse_as_forward=r,W=W,W_reverse=-WR,information=I,information_reverse_as_forward=IR,sigma=SF,sigma_reverse=-SR)
 return dict(generation=d['generation'],core=d['core'],steps=K,n=n,information=info,mean_work_exact=mean,mean_work_sample=float(W.mean()),mean_information_sample=float(I.mean()),mean_EP_exact=float(diss),efficiency_exact=float(-BETA*mean/info),efficiency_sample=float(-BETA*W.mean()/I.mean()),ift=weights_summary(-SF),reverse_ift=weights_summary(SR),max_ratio_error=max(err,er),file=name,sha256=sha(path),bytes=path.stat().st_size)
def main():
 ds=read(S/'THERMAL_SCHEDULES.json');reg=read(S/'REGISTRATION.json')
 for d in ds:d['source_b']=reg['cores'][d['core']]['sources'][1]
 jobs=[(i,d,K,2048) for i,d in enumerate(ds) for K in (1,4,16,64)];(S/'data/feedback').mkdir()
 with cf.ProcessPoolExecutor(max_workers=12) as pool:out=list(pool.map(job,jobs,chunksize=2))
 result=dict(scope='explicit measured-record feedback model with detector error .2, independent conditional annealing F/R, no physical device cycle costs',contract_sha=sha(S/'FEEDBACK_CONTRACT66.md'),paths=len(jobs)*4096,transitions=sum(K*4096 for _,_,K,_ in jobs),results=out)
 dump(S/'FEEDBACK66.json',result);print('FEEDBACK',result['paths'],result['transitions'],max(x['max_ratio_error'] for x in out))
if __name__=='__main__':main()
