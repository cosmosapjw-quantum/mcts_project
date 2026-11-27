"""Exact branch-driven finite feedback cycle and independent F/R Monte Carlo."""
from base61 import *
from scipy.special import logsumexp
from scipy.stats import t as student
BETA=10/3;EPS=(0.,.05,.2,1.);ALPHAS=(.25,1.);NS=2048
def gibbs(e):return np.exp(-BETA*e-logsumexp(-BETA*e,axis=-1,keepdims=True))
def sample(p,rng,n):
 return np.searchsorted(np.cumsum(p),rng.random(n),side='right')
def eval_case(c,eps,alpha,seed):
 e=np.array(c['e0']);em=np.array(c['em']);g=np.array(c['group']);n=len(e);m=len(em);assert em.shape==(m,n)
 p=gibbs(e);pi=gibbs(em);M=np.full((n,m),eps/m);M[np.arange(n),g]+=1-eps;pm=p@M;assert min(pm)>0 and abs(pm.sum()-1)<1e-12
 mass=massr=lam=ift=work=heat=info=sigma=uninfo=wr2=0.;logerr=ldb=balance=0.;end=np.zeros(n);mom2=0.;maxw=0.;worst=0.;ntrip=0;singular=0
 for k in range(m):
  P=(1-alpha)*np.eye(n)+alpha*np.broadcast_to(pi[k],(n,n));assert abs(P.sum(1)-1).max()<1e-12
  Q=e[k]*0 if False else em[k][None,:]-em[k][:,None];delta=e[None,:]-e[:,None]
  d=em[k]-e;W=d[:,None]-d[None,:]
  ldb=max(ldb,float(abs(np.log(P/P.T)+BETA*Q).max()));balance=max(balance,float(abs(W+Q-delta).max()))
  f=(p*M[:,k])[:,None]*P;r=pm[k]*p[None,:]*P.T;mask=f>0
  assert np.all(r>0);mass+=f.sum();massr+=r.sum();lam+=r[~mask].sum();singular+=int((~mask).sum());ntrip+=f.size
  logi=np.zeros(n);pos=M[:,k]>0;logi[pos]=np.log(M[pos,k]/pm[k]);s=BETA*W+logi[:,None];logerr=max(logerr,float(abs(np.log(f[mask]/r[mask])-s[mask]).max()))
  assert ldb<1e-10 and balance<1e-12 and logerr<1e-10
  t=np.exp(-s[mask]);ift+=float(f[mask]@t);mom2+=float(f[mask]@(t*t));maxw=max(maxw,float(t.max()))
  work+=float((f*W).sum());heat+=float((f*Q).sum());info+=float((p*M[:,k])@logi);sigma+=float(f[mask]@s[mask]);uninfo+=float((f*np.exp(-BETA*W)).sum());wr2+=float((f*W*W).sum());end+=f.sum(0)
 support=1-lam;assert abs(mass-1)<1e-11 and abs(massr-1)<1e-11 and abs(ift-support)<1e-11 and abs(sigma-BETA*work-info)<1e-11
 assert abs(end@e-p@e-work-heat)<1e-11 and sigma>=-np.log(support)-1e-10
 if eps>0:assert lam==0 and singular==0
 else:assert lam>0 and singular>0
 if eps==1:assert abs(info)<1e-12 and work>=-1e-12
 # Independent draws forward and reverse. State reset by declared heat-bath kernel.
 rng=np.random.default_rng(seed);i=sample(p,rng,NS);m0=g[i].copy();bad=rng.random(NS)<eps;m0[bad]=rng.integers(m,size=int(bad.sum()));j=i.copy();u=rng.random(NS)<alpha
 for k in range(m):
  sel=(m0==k)&u;j[sel]=sample(pi[k],rng,int(sel.sum()))
 assert np.all(M[i,m0]>0)
 wf=em[m0,i]-e[i]+e[j]-em[m0,j];qf=em[m0,j]-em[m0,i];inf=np.log(M[i,m0]/pm[m0]);sf=BETA*wf+inf;weight=np.exp(-sf)
 rng=np.random.default_rng(seed+1);mr=sample(pm,rng,NS);jr=sample(p,rng,NS);ir=jr.copy();u=rng.random(NS)<alpha
 for k in range(m):
  sel=(mr==k)&u;ir[sel]=sample(pi[k],rng,int(sel.sum()))
 rev= M[ir,mr]>0
 # reverse work and heat are opposite on the matched orientation (i,m,j).
 wr=em[mr,jr]-e[jr]+e[ir]-em[mr,ir];qr=em[mr,ir]-em[mr,jr];assert abs(wr+qr-(e[ir]-e[jr])).max()<1e-12
 irat=np.log(p[i])+np.log(M[i,m0])+np.log((1-alpha)*(i==j)+alpha*pi[m0,j])-np.log(pm[m0])-np.log(p[j])-np.log((1-alpha)*(i==j)+alpha*pi[m0,i]);assert abs(irat-sf).max()<1e-10
 mean=lambda x:float(np.mean(x))
 se=lambda x:float(np.std(x,ddof=1)/np.sqrt(NS))
 row=dict(protocol=c['protocol'],family=c['family'],core=c['core'],eps=eps,alpha=alpha,seed=seed,n=n,m=m,triples=ntrip,zero_forward=singular,forward_mass=mass,reverse_mass=massr,lambda_singular=lam,exact_ift=ift,mean_sigma=sigma,jensen_lower=-np.log(support),mean_work=work,mean_heat=heat,mean_delta_E=float(end@e-p@e),mean_info=info,uncorrected_ift=uninfo,work_sd=float(np.sqrt(max(0,wr2-work**2))),exact_weight_ESS_fraction=float(ift**2/mom2),max_weight=maxw,ldb_error=ldb,energy_error=balance,logratio_error=logerr,mean_final_KL=float(end@np.log(end/p)),mc_ift=mean(weight),mc_ift_se=se(weight),mc_reverse_support=mean(rev),mc_reverse_support_se=se(rev.astype(float)),mc_work=mean(wf),mc_work_se=se(wf),mc_sigma=mean(sf),mc_heat=mean(qf),mc_info=mean(inf),mc_ess=float(weight.sum()**2/(weight@weight)),mc_reverse_work=mean(wr),mc_reverse_heat=mean(qr))
 return row,np.stack([i,m0,j],1).astype(np.int16),np.stack([ir,mr,jr],1).astype(np.int16)
def main():
 r=read(S/'BRANCH_ENERGY85.json');rows=[];F=[];R=[]
 for c in r['contexts']:
  for eps in EPS:
   for a in ALPHAS:
    q,f,v=eval_case(c,eps,a,85100000+2*len(rows));rows.append(q);F.append(f);R.append(v)
  if len(rows)%128==0:print('THERMAL',len(rows),flush=True)
 assert len(rows)==832
 np.savez_compressed(S/'THERMAL_SAMPLES85.npz',forward=np.array(F),reverse=np.array(R))
 dump(S/'THERMAL85.json',dict(contexts=104,cases=rows,samples_per_direction=NS,auxiliary_samples=len(rows)*2*NS,raw_array_sha=sha(S/'THERMAL_SAMPLES85.npz'),energy_sha=sha(S/'BRANCH_ENERGY85.json'),beta=BETA,scope='Explicit cyclic finite-state feedback heat-bath reference, actual branch score energies but no native heat or physical reservoir. Exact support-aware FT and independent Monte Carlo, record/control cost excluded.'))
 print('THERMAL COMPLETE',len(rows),len(rows)*2*NS,flush=True)
if __name__=='__main__':main()
