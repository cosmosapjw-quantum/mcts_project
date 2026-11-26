"""Autonomous time-inhomogeneous empirical marked transition closure.
No heldout intermediate state in forward generation. Exact Feynman-Kac recursion
and raw moment propagation; separate teacher-forced proper scores."""
from base61 import *
from scipy.special import logsumexp
from scipy.stats import t as student
from math import comb
import sys
DIRECTIONS=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,-1,.5]],float)
AMPLITUDES=np.array([-8,-4,-2,-1,0,1,2,4,8],float)
JS=(DIRECTIONS[:,None,:]*AMPLITUDES[None,:,None]).reshape(-1,3)
LABELS=['g0 +','g0 -','g3 +','g3 -','Rapfi NNUE']
MODELS=['clock + root','state','state + lag']
NS=[1,20,80]
ALPHA=8.
def base(p):
 m=p[...,:2].sum(-1);r=p[...,0]/m
 return np.digitize(m,[.015,.04,.10,.25])*4+np.digitize(r,[.25,.5,.75])
def markbin(z):return np.digitize(z,[-.5,0,.5])
def norm(w):
 assert np.all(w>=0) and w.sum()>0
 return w/w.sum()
def backoff(mask,parent,alpha):
 n=mask.sum();return (mask.astype(float)+alpha*parent)/(n+alpha)
def weights(v,s,model,vb):
 n=len(v);uniform=np.ones(n)/n;root=backoff(v==vb,uniform,16.)
 if model==0:return root[None,:]
 sb=s if model==1 else s//4
 basew=np.array([backoff((v==vb)&(sb==i),root,ALPHA) for i in range(20)])
 if model==1:return basew
 return np.array([backoff((v==vb)&(s==i),basew[i//4],ALPHA) for i in range(80)])
def cumulants(raw):
 mu=raw[...,1];v=raw[...,2]-mu**2
 c3=raw[...,3]-3*mu*raw[...,2]+2*mu**3
 c4=raw[...,4]-4*mu*raw[...,3]+6*mu**2*raw[...,2]-3*mu**4-3*v**2
 return np.stack([mu,v,c3,c4],-1)
def ci(x):
 x=np.asarray(x);m=x.mean(0);h=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x))
 return [m.tolist(),(m-h).tolist(),(m+h).tolist()]
def family(g):
 A=np.load(S/'JOINT71.npz');m=A['meta'];ix=m[:,0]==g;m=m[ix]
 Y=A['counts'][ix];Z=A['marks'][ix].sum(-1);P=A['prob'][ix];F=A['conditional_pmf'][ix];V=A['initial_value'][ix]
 X=np.concatenate([Y[:,:,:2],Z[...,None]],-1)/128.
 b=base(P);lag=np.concatenate([np.zeros((len(m),1)),Z[:,:-1]/4],1);s=b*4+markbin(lag)
 vb=markbin(V);train=m[:,1]%16>=4;test=~train
 # direct observations at every registered heldout root; matched equal root weights.
 grid=A['grid'];yi=np.argmax(np.all(Y[:,:,None,:]==grid[None,None,:,:],axis=-1),axis=-1)
 assert np.array_equal(grid[yi],Y);o=yi*4+markbin(Z/4)
 assert np.all(X[:,:,:2]>=0) and np.all(X[:,:,:2].sum(-1)<=4/128)
 cores=np.unique(m[test,1]);assert len(cores)==16
 yall=X.cumsum(1);xend=yall[:,-1];dirend=xend@DIRECTIONS.T
 empirical=[];results=[];rng=np.random.default_rng(710301+g)
 for c in cores:
  ids=np.flatnonzero(m[:,1]==c);assert np.all(test[ids]);assert np.ptp(V[ids])<1e-7
  assert np.all(b[ids,0]==b[ids[0],0]) and np.all(s[ids,0]==s[ids[0],0])
  raw=np.stack([np.mean(dirend[ids]**k,axis=0) for k in range(5)],axis=-1)
  mgf=np.mean(np.exp(xend[ids]@JS.T),axis=0)
  ess=np.sum(np.exp(xend[ids]@JS.T),0)**2/np.sum(np.exp(2*xend[ids]@JS.T),0)
  empirical.append(dict(core=int(c),raw=raw.tolist(),mgf=mgf.tolist(),ess=ess.tolist(),prefix=yall[ids].mean(0).tolist(),root_v=float(V[ids[0]]),ply=int(m[ids[0],3]),exact_count_nll=float(-np.log(np.take_along_axis(F[ids],yi[ids,...,None],-1)[...,0]).mean())))
 for model in range(3):
  Sdim=NS[model];ss=np.zeros_like(b) if model==0 else b if model==1 else s
  pred=[];mc_checks=[];max_norm=0.;max_mass=0.
  for ply in (4,8,12,16):
   it=np.flatnonzero(train&(m[:,3]==ply));assert len(np.unique(m[it,1]))==12
   for vbin in range(4):
    ic=[int(c) for c in cores if (m[m[:,1]==c,3][0]==ply and vb[m[:,1]==c][0]==vbin)]
    if not ic:continue
    W=[];Dest=[];Rewards=[];Mark=[];Count=[];OC=[];BC=[];ZB=[]
    for t in range(32):
     w=weights(vb[it],ss[it,t],model,vbin);max_norm=max(max_norm,float(np.max(abs(w.sum(1)-1))));assert np.all(w>=0) and np.max(abs(w.sum(1)-1))<1e-12
     W.append(w);dest=ss[it,min(t+1,31)] if t<31 else np.zeros(len(it),int);Dest.append(dest);Rewards.append(X[it,t]);Mark.append(Z[it,t]/4);Count.append(yi[it,t]);OC.append(o[it,t]);BC.append(b[it,min(t+1,31)])
     ZB.append(markbin(Z[it,t]/4))
     if model==2 and t<31:assert np.array_equal(dest%4,ZB[-1])
    # Exact backward propagation of marked partition and raw moments up to fourth.
    U=np.ones((Sdim,len(JS)));raw=np.zeros((Sdim,len(DIRECTIONS),5));raw[:,:,0]=1
    for t in range(31,-1,-1):
     w,d,x=W[t],Dest[t],Rewards[t];U=w@(np.exp(x@JS.T)*U[d])
     r=x@DIRECTIONS.T;z=raw[d];temp=np.zeros_like(z)
     for k in range(5):
      for l in range(k+1):temp[:,:,k]+=comb(k,l)*r**(k-l)*z[:,:,l]
     raw=(w@temp.reshape(len(it),-1)).reshape(Sdim,len(DIRECTIONS),5)
    assert np.max(abs(U[:,np.flatnonzero(np.all(JS==0,axis=1))]-1))<1e-11
    assert np.max(abs(raw[:,:,0]-1))<1e-11
    for c in ic:
     ids=np.flatnonzero(m[:,1]==c);initial=int(ss[ids[0],0]);assert np.all(ss[ids,0]==initial)
     p=np.eye(Sdim)[initial];pref=np.zeros((32,3));running=np.zeros(3);one_scores=[];supports=[];next_scores=[]
     for t in range(32):
      w,d,x=W[t],Dest[t],Rewards[t];q=p@w;running+=q@x;pref[t]=running
      p=np.bincount(d,weights=q,minlength=Sdim);max_mass=max(max_mass,abs(p.sum()-1));assert abs(p.sum()-1)<1e-11
      # Diagnostic one-step prediction uses observed current state only; no use for recursion.
      hist=w@np.eye(60)[OC[t]];pr=hist[ss[ids,t]];actual=o[ids,t]
      one_scores.append(float(np.mean(np.sum(pr**2,axis=1)-2*pr[np.arange(len(ids)),actual]+1)))
      supports.append(float(np.mean(pr[np.arange(len(ids)),actual]==0)))
      if t<31:
       h=w@np.eye(20)[BC[t]];pr=h[ss[ids,t]];actual=b[ids,t+1];next_scores.append(float(np.mean(np.sum(pr**2,axis=1)-2*pr[np.arange(len(ids)),actual]+1)))
     assert np.max(abs(pref[-1]-raw[initial,:3,1]))<1e-11
     pred.append(dict(core=c,mgf=U[initial].tolist(),raw=raw[initial].tolist(),prefix=pref.tolist(),joint_brier=float(np.mean(one_scores)),zero_joint_support=float(np.mean(supports)),next_state_brier=float(np.mean(next_scores))))
     # Independently simulate every step from the same kernel on one root per family/model.
     if not mc_checks:
      nmc=8192;cur=np.full(nmc,initial);total=np.zeros((nmc,3))
      for t in range(32):
       w,d,x=W[t],Dest[t],Rewards[t];nxt=np.empty_like(cur)
       for st in np.unique(cur):
        jj=np.flatnonzero(cur==st);chosen=rng.choice(len(it),len(jj),p=w[st]);total[jj]+=x[chosen];nxt[jj]=d[chosen]
       cur=nxt
      a=np.exp(total@JS.T);obs=a.mean(0);se=a.std(0,ddof=1)/np.sqrt(nmc);z=(obs-U[initial])/np.maximum(se,1e-12)
      # exact zero source checked separately. No hard stochastic z acceptance on many correlated sources.
      mc_checks.append(dict(core=c,n=nmc,max_abs_z=float(np.max(abs(z))),max_mean_abs=float(np.max(abs(total.mean(0)-raw[initial,:3,1]))),mean_z=z.tolist(),source_mgf=obs.tolist(),exact_mgf=U[initial].tolist()))
  pred=sorted(pred,key=lambda x:x['core']);assert [x['core'] for x in pred]==[x['core'] for x in empirical]
  results.append(dict(model=MODELS[model],pred=pred,mc_checks=mc_checks,max_norm_error=max_norm,max_forward_mass_error=max_mass))
 # Root-block bootstrap of difference: conditional on fitted kernel, not training uncertainty.
 boot=rng.integers(0,16,(2000,16));em=np.array([x['mgf'] for x in empirical]);er=np.array([x['raw'] for x in empirical]);ep=np.array([x['prefix'] for x in empirical])
 W0=np.log(em.mean(0));c0=cumulants(er.mean(0))
 for r in results:
  pp=r['pred'];pm=np.array([x['mgf'] for x in pp]);pr=np.array([x['raw'] for x in pp]);pf=np.array([x['prefix'] for x in pp]);W=np.log(pm.mean(0));c=cumulants(pr.mean(0))
  diff=np.log(pm[boot].mean(1))-np.log(em[boot].mean(1))
  cboot=cumulants(pr[boot].mean(1))-cumulants(er[boot].mean(1))
  r.update(W=W.tolist(),W_delta=(W-W0).tolist(),W_delta_ci95=np.quantile(diff,[.025,.975],axis=0).tolist(),W_rmse=float(np.sqrt(np.mean((W-W0)**2))),W_rmse_moderate=float(np.sqrt(np.mean((W-W0)[np.abs(JS).max(1)<=4]**2))),cumulants=c.tolist(),cumulant_delta=(c-c0).tolist(),cumulant_delta_ci95=np.quantile(cboot,[.025,.975],axis=0).tolist(),prefix=pf.mean(0).tolist(),prefix_rmse=float(np.sqrt(np.mean((pf.mean(0)-ep.mean(0))**2))),root_prefix_rmse=float(np.sqrt(np.mean((pf-ep)**2))),joint_brier=ci([x['joint_brier'] for x in pp]),next_state_brier=ci([x['next_state_brier'] for x in pp]),zero_joint_support=ci([x['zero_joint_support'] for x in pp]))
 return dict(family=g,label=LABELS[g],train_paths=int(train.sum()),test_paths=int(test.sum()),test_roots=len(cores),empirical=empirical,W_direct=W0.tolist(),cumulants_direct=c0.tolist(),prefix_direct=ep.mean(0).tolist(),results=results)
def main():
 assert sha(S/'JOINT71.npz')==read(S/'AUDIT71.json')['array_sha256']
 with cf.ProcessPoolExecutor(max_workers=5) as pool:
  R=list(pool.map(family,range(5)))
 for r in R:print(r['label'],[(x['model'],x['W_rmse'],x['W_rmse_moderate'],x['root_prefix_rmse'],x['joint_brier'][0],x['next_state_brier'][0],x['mc_checks'][0]['max_abs_z']) for x in r['results']],flush=True)
 dump(S/'PHYSICS_RESULTS.json',dict(scope='new geometry holdout, autonomous empirical marked transition; exact within fitted model, not exact true law; separate teacher-forced proper scores',directions=DIRECTIONS.tolist(),amplitudes=AMPLITUDES.tolist(),sources=JS.tolist(),families=R,geometry_blocks=64,train_blocks=48,test_blocks=16,alpha=ALPHA,state_bins={'mass':[.015,.04,.10,.25],'ratio':[.25,.5,.75],'mark':[-.5,0,.5],'root_value':[-.5,0,.5]},array_sha256=sha(S/'JOINT71.npz')))
if __name__=='__main__':main()
