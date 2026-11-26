"""Frozen native pilot energies, actual independent forward/reverse finite thermal model.
Units E_* = declared unit, beta*E_* = 10/3; no physical calorimetry."""
from base61 import *
from scipy.special import logsumexp
from scipy.optimize import brentq
BETA=10/3
def gibbs(E):
 z=logsumexp(-BETA*E,axis=-1);return np.exp(-BETA*E-z[...,None]),-z/BETA
def exact(E,alpha):
 pi,F=gibbs(E);mu=pi[0].copy();meanW=meanQ=0.;v=mu.copy();w=np.zeros(len(mu));w2=w.copy()
 for j in range(1,len(E)):
  de=E[j]-E[j-1];meanW+=mu@de
  nxt=(1-alpha)*mu+alpha*pi[j];meanQ+=(nxt-mu)@E[j];mu=nxt
  z=w2+2*w*de+v*de**2;w=w+v*de;w2=z
  v=(1-alpha)*v+alpha*v.sum()*pi[j];w=(1-alpha)*w+alpha*w.sum()*pi[j];w2=(1-alpha)*w2+alpha*w2.sum()*pi[j]
 assert abs(w.sum()-meanW)<1e-12
 dF=F[-1]-F[0];de=mu@E[-1]-pi[0]@E[0];assert abs(de-meanW-meanQ)<1e-12
 kl=float(mu@np.log(mu/pi[-1]))
 mgf=[]
 for s in (-BETA,-BETA/2,0,BETA/2,BETA):
  x=pi[0].copy();mat=x.copy()
  for j in range(1,len(E)):
   e=np.exp(-s*(E[j]-E[j-1]));x*=e;x=(1-alpha)*x+alpha*x.sum()*pi[j]
   K=(1-alpha)*np.eye(len(x))+alpha*np.broadcast_to(pi[j],(len(x),len(x)));mat=(mat*e)@K
  assert abs(x.sum()-mat.sum())<1e-11*max(1,mat.sum())
  mgf.append(float(x.sum()))
 assert abs(mgf[-1]*np.exp(BETA*dF)-1)<1e-12
 return dict(dF=float(dF),meanW=float(meanW),meanQ=float(meanQ),mean_deltaE=float(de),varW=float(w2.sum()-meanW**2),mean_dissipation=float(BETA*(meanW-dF)),mean_actual_EP=float(BETA*(meanW-dF)-kl),final_KL=kl,mgf=mgf,final=mu.tolist())
def pathprob(E,alpha,seq):
 pi,F=gibbs(E);x=seq[:,:-1];y=seq[:,1:];j=np.arange(1,len(E))[None,:]
 a=alpha*pi[j,y]+(1-alpha)*(x==y);b=alpha*pi[j,x]+(1-alpha)*(x==y)
 lf=np.log(pi[0,seq[:,0]])+np.log(a).sum(1);lr=np.log(pi[-1,seq[:,-1]])+np.log(b).sum(1)
 W=(E[j,x]-E[j-1,x]).sum(1);Q=(E[j,y]-E[j,x]).sum(1);dE=E[-1,seq[:,-1]]-E[0,seq[:,0]]
 dF=F[-1]-F[0]
 err=float(abs(lf-lr-BETA*(W-dF)).max());bal=float(abs(dE-W-Q).max())
 assert err<1e-11 and bal<1e-11
 return W,Q,lf,lr,err,bal
def sample(E,alpha,n,seed,reverse=False):
 rng=np.random.default_rng(seed);pi,F=gibbs(E);K=len(E)-1;seq=np.zeros((n,K+1),np.int16)
 x=rng.choice(E.shape[1],n,p=pi[-1] if reverse else pi[0]);seq[:,K if reverse else 0]=x
 for j in (range(K,0,-1) if reverse else range(1,K+1)):
  refresh=rng.random(n)<alpha;u=rng.choice(E.shape[1],n,p=pi[j]);x=np.where(refresh,u,x)
  seq[:,j-1 if reverse else j]=x
 W,Q,lf,lr,err,bal=pathprob(E,alpha,seq)
 # reverse array kept oriented as forward sequence, reverse W=-W etc.
 return seq,W,Q,lf,lr,err,bal
def weights_summary(x):
 z=np.exp(x);return dict(mean=float(z.mean()),se=float(z.std(ddof=1)/np.sqrt(len(z))),ess=float(z.sum()**2/(z@z)),min=float(z.min()),max=float(z.max()))
def job(arg):
 i,d,alpha,n=arg;E=np.array(d['energy']);pi,F=gibbs(E);df=float(F[-1]-F[0]);e=exact(E,alpha)
 # LDB all positive and normalized, including diagonal.
 errdb=0.
 for j in range(1,len(E)):
  K=(1-alpha)*np.eye(len(pi[j]))+alpha*pi[j][None,:];errdb=max(errdb,float(abs(pi[j][:,None]*K-(pi[j][:,None]*K).T).max()));assert abs(K.sum(1)-1).max()<1e-12 and K.min()>0
 assert errdb<1e-13
 fs=sample(E,alpha,n,660101+4*i+int(alpha==1)*2);rs=sample(E,alpha,n,660102+4*i+int(alpha==1)*2,True)
 f,W,Q,lf,lr,ep,eb=fs;rv,WR,QR,lfr,lrr,erp,erb=rs;sig=lf-lr;sigr=lfr-lrr
 mu=np.asarray(e['final']);ep_actual=np.log(pi[0,f[:,0]]/mu[f[:,-1]])-BETA*Q
 assert abs(ep_actual-(sig+np.log(pi[-1,f[:,-1]]/mu[f[:,-1]]))).max()<1e-11
 # Independent Bennett acceptance ratio with equal sample numbers.
 bfun=lambda x:np.mean(1/(1+np.exp(np.clip(BETA*(W-x),-600,600))))-np.mean(1/(1+np.exp(np.clip(BETA*(x-WR),-600,600))))
 bar=brentq(bfun,min(W.min(),WR.min())-10,max(W.max(),WR.max())+10)
 p=S/'data'/'thermal'/f'g{d["generation"]}_c{d["core"]:02d}_a{int(alpha*100)}.npz'
 np.savez_compressed(p,forward=f,reverse_as_forward=rv,W=W,Q=Q,W_reverse=-WR,Q_reverse=-QR,sigma=sig,sigma_reverse=-sigr,logPF=lf,logPR=lr,logPF_on_reverse=lfr,logPR_on_reverse=lrr,actual_EP=ep_actual)
 return dict(generation=d['generation'],core=d['core'],alpha=alpha,n=n,exact=e,meanW=float(W.mean()),meanQ=float(Q.mean()),mean_reverseW=float((-WR).mean()),mean_sigma=float(sig.mean()),mean_actual_EP=float(ep_actual.mean()),jarzynski=weights_summary(-sig),reverse_jarzynski=weights_summary(sigr),df_F=float(-logsumexp(-BETA*W)/BETA+np.log(n)/BETA),df_R=float(logsumexp(BETA*WR)/BETA-np.log(n)/BETA),df_BAR=float(bar),max_path_ratio_error=max(ep,erp),max_first_law_error=max(eb,erb),max_db_error=errdb,file=p.name,sha256=sha(p),bytes=p.stat().st_size)
def small_exact():
 # explicit enumeration all 3^4 paths, independent transfer reference
 import itertools
 E=np.array([[0,.2,-.1],[.4,-.3,.2],[-.2,.1,.5],[.1,.2,-.4]])
 seq=np.array(list(itertools.product(range(3),repeat=4)));q=[]
 for a in (.25,1.):
  W,Q,lf,lr,err,bal=pathprob(E,a,seq);p=np.exp(lf);r=np.exp(lr);ex=exact(E,a)
  assert abs(p.sum()-1)<1e-14 and abs(r.sum()-1)<1e-14 and abs(p@W-ex['meanW'])<1e-14
  assert abs(p@np.exp(lr-lf)-1)<1e-14
  q.append(dict(alpha=a,paths=len(seq),normalization=[float(p.sum()),float(r.sum())],ratio_error=err,first_law=bal))
 return q
def main():
 reg=read(S/'REGISTRATION.json');assert sha(S/'THERMAL_SCHEDULES.json')==reg['schedule_sha']
 ds=read(S/'THERMAL_SCHEDULES.json');n=reg['thermal']['paths_per_direction'];(S/'data/thermal').mkdir()
 jobs=[(i,d,a,n) for i,d in enumerate(ds) for a in reg['thermal']['alphas']]
 out=[]
 with cf.ProcessPoolExecutor(max_workers=12) as pool:
  for j,x in enumerate(pool.map(job,jobs,chunksize=2)):
   out.append(x)
   if (j+1)%48==0:print('THERMAL',j+1,flush=True)
 report=dict(scope='independent sampled auxiliary reversible chain on fixed legal states; energy schedule from immutable R65 pilot, not physical native heat',paths=len(jobs)*n*2,transitions=len(jobs)*n*2*8,exact_enumeration=small_exact(),results=out)
 dump(S/'THERMAL66.json',report);print('THERMAL COMPLETE',report['paths'],report['transitions'],max(x['max_path_ratio_error'] for x in out),max(x['max_first_law_error'] for x in out),flush=True)
if __name__=='__main__':main()
