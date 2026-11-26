"""Autonomous count/marked state reductions, fixed-point and heldout CK diagnostics."""
from base61 import *
from scipy.special import logsumexp
from itertools import combinations
LS=[13,15,17,19,21];BETAS=[0.,10/3,16.];ALPHA=8.;NAMES=['clock','count','count_mark','count_mark_prob','count_mark_prob_lag'];H=14
JGRID=np.array([[0.,0.]]+[list(np.array(v)*a) for v in [(1,0),(0,1),(1,1),(1,-1)] for a in [-4,-2,-1,1,2,4]])
def bins(d):
 c=d['counts'];m=d['marks'];p=d['prob']
 count=np.digitize(np.cumsum(c[:,:,0],axis=1)[:,:15]/(4*np.arange(1,16)),[.1,.4])
 mark=np.digitize(np.cumsum(m,axis=1)[:,:15]/np.arange(1,16),[-.2,.2])
 pr=np.digitize(p[:,1:16,0],[.1,.4]);lag=np.digitize(m[:,:15],[-.2,.2])
 # Each state precedes the emissions at waves 1..14; t=4..60.
 n=count;states=[np.zeros_like(n),n,n*3+mark,(n*3+mark)*3+pr,((n*3+mark)*3+pr)*3+lag]
 y=np.stack([c[:,1:15,0]/4,m[:,1:15]],-1)
 return states,n,y
class Donor:
 def __init__(self,s,sn,y,nc,N):
  self.s=s.ravel();self.sn=sn.ravel();self.y=y.reshape(-1,2);self.nc=nc.ravel();self.N=N;self.n=len(self.s)
  self.support=np.bincount(self.s,minlength=N);self.den=self.support+ALPHA
  self.K=self.transfer(np.ones(self.n));self.E=self.reg(self.y)
  self.C=self.reg(np.eye(3)[self.nc]);assert abs(self.K.sum(1)-1).max()<1e-12 and abs(self.C.sum(1)-1).max()<1e-12
 def reg(self,a):
  a=np.asarray(a);v=np.zeros((self.N,a.shape[1]));np.add.at(v,self.s,a)
  return (v+ALPHA*a.mean(0))/self.den[:,None]
 def transfer(self,w):
  a=np.bincount(self.s*self.N+self.sn,weights=w,minlength=self.N*self.N).reshape(self.N,self.N)
  g=np.bincount(self.sn,weights=w,minlength=self.N)/self.n
  return (a+ALPHA*g[None,:])/self.den[:,None]
 def tilt(self,J):
  return self.transfer(np.exp(self.y@J/H))
def stationary(K):
 support=np.flatnonzero(K.sum(0)>0);A=K[np.ix_(support,support)]
 assert np.all(A>0) and abs(A.sum(1)-1).max()<1e-12
 n=len(A);pi=np.linalg.lstsq(np.vstack([A.T-np.eye(n),np.ones((1,n))]),np.r_[np.zeros(n),1.],rcond=None)[0]
 assert min(pi)>-1e-12 and abs(pi.sum()-1)<1e-12
 P=np.tile(pi,(n,1));ev=np.linalg.eigvals(A);i=np.argmin(abs(ev-1));sub=np.delete(ev,i)
 delta=float(.5*np.max(abs(A[:,None,:]-A[None,:,:]).sum(-1)))
 assert delta<1+1e-12
 errors=[];Q=A.copy()
 for _ in range(7):
  errors.append(float(.5*np.max(abs(Q-P).sum(1))));Q=Q@Q
 # Explicit independent direct stationary iteration.
 v=np.ones(n)/n
 for _ in range(4096):
  nxt=v@A
  if abs(nxt-v).sum()<1e-14:v=nxt;break
  v=nxt
 err=float(abs(v-pi).max());assert err<1e-9
 return dict(support=int(n),pi=pi.tolist(),states=support.tolist(),residual=float(abs(pi@A-pi).max()),projector_residual=float(abs(P@P-P).max()),eigen_subdominant=float(max(abs(sub),default=0.)),dobrushin=delta,dyadic_TV=errors,stationary_crosscheck=err,stationary_family_tangent=n-1,transverse_zero_directions=(n-1)**2)
def root_ci(x,roots):
 a=np.stack([x[roots==r].mean(0) for r in range(8)])
 from scipy.stats import t
 mu=a.mean(0);e=t.ppf(.975,7)*a.std(0,ddof=1)/np.sqrt(8)
 return [mu.tolist(),(mu-e).tolist(),(mu+e).tolist()]
def fit_models(train,test,roots):
 ts,tn,ty=bins(train);vs,vn,vy=bins(test);out=[];cross=[]
 for m,name in enumerate(NAMES):
  n=3**m;s=ts[m];v=vs[m];models=[]
  for kind in ('homogeneous','clock_conditioned'):
   if kind=='homogeneous':D=Donor(s[:,:H],s[:,1:H+1],ty,tn[:,1:H+1],n);ds=[D]*H
   else:ds=[Donor(s[:,t],s[:,t+1],ty[:,t],tn[:,t+1],n) for t in range(H)]
   K=[d.K for d in ds];E=[d.E for d in ds];C=[d.C for d in ds];truth=np.eye(3)[vn[:,1:H+1]]
   bp=np.stack([C[t][v[:,t]] for t in range(H)],1);ep=np.stack([E[t][v[:,t]] for t in range(H)],1)
   one=np.stack([((bp-truth)**2).sum(-1).mean(1),((ep-vy)**2).mean((1,2)),np.mean(np.stack([ds[t].support[v[:,t]]==0 for t in range(H)],1),1)],1)
   # All block starts, same heldout paths, not treated as independent roots.
   for h in (1,2,4,8):
    b=[];e=[];bd=[];ed=[];ck=[];ckstate=[]
    for t0 in range(H+1-h):
     P=np.eye(n);Y=np.zeros((n,2))
     for t in range(t0,t0+h):
      Y+=P@E[t];Cend=P@C[t];P=P@K[t]
     if kind=='homogeneous':
      starts=np.concatenate([s[:,t] for t in range(H+1-h)])
      ends=np.concatenate([s[:,t+h] for t in range(H+1-h)])
      yn=np.concatenate([ty[:,t:t+h].sum(1) for t in range(H+1-h)])
      nc=np.concatenate([tn[:,t+h] for t in range(H+1-h)])
      direct=Donor(starts,ends,yn,nc,n)
     else:direct=Donor(s[:,t0],s[:,t0+h],ty[:,t0:t0+h].sum(1),tn[:,t0+h],n)
     if h==1:assert abs(Cend-direct.C).max()<1e-12 and abs(P-direct.K).max()<1e-12
     vv=v[:,t0];tt=np.eye(3)[vn[:,t0+h]];yy=vy[:,t0:t0+h].sum(1)/h
     b.append(((Cend[vv]-tt)**2).sum(1));e.append(((Y[vv]/h-yy)**2).mean(1))
     bd.append(((direct.C[vv]-tt)**2).sum(1));ed.append(((direct.E[vv]/h-yy)**2).mean(1))
     ck.append(.5*abs(Cend[vv]-direct.C[vv]).sum(1));ckstate.append(.5*abs(P[vv]-direct.K[vv]).sum(1))
    metrics=np.stack([np.mean(x,0) for x in (b,e,bd,ed,ck,ckstate)],1)
    cross.append(dict(model=name,kind=kind,h=h,ci=root_ci(metrics,roots),fields=['composed_count_brier','composed_emission_mse','direct_count_brier','direct_emission_mse','count_CK_TV','state_CK_TV']))
   # exact tilted finite model, autonomous from initial measured state only.
   W=[];Wactual=[];Wr=[];Wa=[];maxerr=0.
   for J in JGRID:
    mat=[d.tilt(J) for d in ds]
    F=np.ones(n)
    for T in reversed(mat):F=T@F
    pred=F[v[:,0]];actual=np.exp(vy.sum(1)@J/H)
    W.append(float(np.log(pred.mean())));Wactual.append(float(np.log(actual.mean())))
    Wr.append([float(np.log(pred[roots==r].mean())) for r in range(8)])
    Wa.append([float(np.log(actual[roots==r].mean())) for r in range(8)])
    # independent forward root distribution contraction vs backward column recursion.
    initial=np.bincount(v[:,0],minlength=n)/len(v)
    for T in mat:initial=initial@T
    maxerr=max(maxerr,abs(np.log(initial.sum())-W[-1]))
   assert maxerr<1e-10 and abs(W[0])<1e-12
   wr=np.array(Wr);wa=np.array(Wa);err=np.asarray(W)-Wactual
   models.append(dict(kind=kind,one_step_ci=root_ci(one,roots),W=W,Wactual=Wactual,W_rmse=float(np.sqrt(np.mean(err[1:]**2))),W_root_rmse_ci=root_ci(np.sqrt(np.mean((wr[1:]-wa[1:])**2,0)),np.arange(8)),tilted_check=maxerr,fixed_point=stationary(K[0]) if kind=='homogeneous' else None,states=n))
  out.append(dict(model=name,models=models))
 return out,cross
def collisions(d,meta):
 s,_,_=bins(d);C=np.cumsum(d['counts'],axis=1)[:,:15];p=d['prob'][:,1:16];result=[]
 for cohort in (0,1):
  for m,name in [(-1,'exact_three_sector_count')]+list(enumerate(NAMES[1:],1)):
   tv=[];witness=[];n=0
   for core in range(8):
    ix=np.flatnonzero((meta[:,0]==cohort)&(meta[:,1]==core))
    assert len(ix)==4
    for t in range(15):
     for i,j in combinations(ix,2):
      equal=np.array_equal(C[i,t],C[j,t]) if m==-1 else s[m][i,t]==s[m][j,t]
      if equal:
       n+=1;v=float(.5*abs(p[i,t]-p[j,t]).sum());tv.append(v)
       if v>1e-10:witness.append(dict(core=core,t=4*(t+1),rep1=int(meta[i,4]),rep2=int(meta[j,4]),tv=v,counts1=C[i,t].tolist(),counts2=C[j,t].tolist(),p1=p[i,t].tolist(),p2=p[j,t].tolist()))
   result.append(dict(cohort=cohort,model=name,pairs=n,nonzero=len(witness),mean_TV=float(np.mean(tv)) if n else None,max_TV=max(tv,default=0.),example=max(witness,key=lambda z:z['tv']) if witness else None))
 return result
def main():
 out=[];col=[];inputs={}
 for L in LS:
  a=read(S/f'AUDIT_L{L}.json');f=S/f'PATHS_L{L}.npz';assert sha(f)==a['array_sha256'];inputs[str(f.relative_to(B))]=sha(f)
  with np.load(f) as z:d={k:z[k] for k in z.files}
  for arm in (0,1):
   for bi,beta in enumerate(BETAS):
    ix=(d['meta'][:,2]==arm)&(d['meta'][:,3]==bi);z={k:v[ix] for k,v in d.items()}
    m=z['meta'];tr=m[:,0]==0;te=~tr;assert tr.sum()==te.sum()==32
    train={k:v[tr] for k,v in z.items()};test={k:v[te] for k,v in z.items()}
    x,c=fit_models(train,test,m[te,1]);out.append(dict(L=L,arm=arm,beta=beta,models=x,CK=c))
    col.append(dict(L=L,arm=arm,beta=beta,results=collisions(z,m)))
  print('CLOSURE',L,flush=True)
 dump(S/'CLOSURE77.json',dict(scope='R76 trained, new R77 fixed-root holdout; independent size/density/beta strata; state approximations not exact native RG',strata=out,collisions=col,J=JGRID.tolist(),alpha=ALPHA,inputs=inputs))
if __name__=='__main__':main()
