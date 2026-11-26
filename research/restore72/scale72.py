"""Exact nested spatial pushforward, fixed-clock nonlumpability and finite-size alternatives."""
from base61 import *
from scipy.special import logsumexp
from scipy.stats import t as student
from itertools import combinations
LS=[13,15,17,19,21];CUTS=[4,8,16,32];SCALES=[3,7,13,15,17,19,21]
def crop(L,m):
 assert m<=L and (L-m)%2==0
 y,x=np.indices((L,L));h=(L-m)//2;z=np.full((L,L),m*m,int);mask=(abs(y-L//2)<=m//2)&(abs(x-L//2)<=m//2);z[mask]=(y[mask]-h)*m+x[mask]-h
 return np.r_[z.ravel(),m*m]
def push(a,mp,n):
 out=np.zeros(a.shape[:-1]+(n,))
 for j,i in enumerate(mp):out[...,i]+=a[...,j]
 return out
def ring(L):
 y,x=np.indices((L,L));r=np.maximum(abs(y-L//2),abs(x-L//2))
 return np.digitize(r.ravel(),[2,4,7,8,9,10])
def kl(p,q):
 ix=p>0;assert np.all(q[ix]>0);return float(np.sum(p[ix]*np.log(p[ix]/q[ix])))
def H(p):
 q=p[p>0];return float(-q@np.log(q))
def cgf(x,j):
 return float(logsumexp(x@j)-np.log(len(x)))
def ci(x):
 x=np.asarray(x);a=x.mean(0);b=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return [a.tolist(),(a-b).tolist(),(a+b).tolist()]
def main():
 audits=[read(S/f'AUDIT_L{L}.json') for L in LS]
 regs=[read(S/f'REGISTRATION_L{L}.json') for L in LS]
 assert all(x['cores']==regs[0]['cores'] for x in regs)
 assert all(x['model']['files']==regs[0]['model']['files'] for x in regs)
 # New exact D4-distinct cohorts relative to R49 and R50/R58 registrations where identifiable.
 def orbit(m,L):
  z=np.zeros((L,L),np.int8)
  for i,a in enumerate(m):z.ravel()[a]=1+i%2
  return min(q.tobytes() for k in range(4) for q in (np.rot90(z,k),np.fliplr(np.rot90(z,k))))
 for L,r in zip(LS,regs):
  old=[]
  for k in (49,50,58):
   d=read(B/f'mqft_restore{k}_20261126'/f'REGISTRATION_L{L}.json')
   old.extend(d.get('anchors',[])+d.get('positions',[]))
  assert not ({orbit(q['moves'],L) for q in r['positions']}&{orbit(q['moves'],L) for q in old})
 # Cross-size dense openings are literally nested restrictions, including color parity.
 for i,L in enumerate(LS[:-1]):
  for a,b in zip(regs[i]['positions'],regs[i+1]['positions']):
   z=crop(LS[i+1],L);assert [int(z[x]) for x in b['moves'] if z[x]<L*L]==a['moves']
 errs=collections.defaultdict(float);groups=[];witnesses=[];composition=0;statepairs=0;cross=[];means={};raws={}
 for L,au in zip(LS,audits):
  f=S/f'SCALE_L{L}.npz';assert sha(f)==au['array_sha256'];A={k:v for k,v in np.load(f).items()};raws[L]=A
  n=L*L;P=np.pad(A['p'],((0,0),(0,0),(0,1)));R=np.pad(A['prior'],((0,0),(0,1)));act=A['actions'];mark=A['marks'];meta=A['meta'];board=A['board']
  mp={m:crop(L,m) for m in SCALES if m<=L};m3=ring(L);# common 3,7,13 cumulative groups
  for k in mp:
   for m in mp:
    if m<=k:assert np.array_equal(crop(k,m)[mp[k]],mp[m]);composition+=1
  for i in range(len(P)):
   for t in (0,7,15,31):
    p=P[i,t];q=R[i];base=kl(p,q);ent=H(p)
    for m,z in mp.items():
     pc=push(p,z,m*m+1);qc=push(q,z,m*m+1);cl=kl(pc,qc);assert cl<=base+1e-12
     ix=p>0;cond=float(np.sum(p[ix]*np.log((p[ix]/pc[z[ix]])/(q[ix]/qc[z[ix]]))))
     ch=-float(np.sum(p[ix]*np.log(p[ix]/pc[z[ix]])))
     errs['KL_chain']=max(errs['KL_chain'],abs(base-cl-cond));errs['H_chain']=max(errs['H_chain'],abs(ent-H(pc)-ch))
     errs['mass']=max(errs['mass'],abs(pc.sum()-1))
  # Root-conditioned statistics, four independent replicate paths per geometry/arm.
  yy,xx=np.indices((L,L));modes=np.array([np.cos(np.pi*(xx+.5)/L),np.cos(np.pi*(yy+.5)/L)]).reshape(2,n)
  for core in range(8):
   for arm in (0,1):
    ids=np.flatnonzero((meta[:,0]==core)&(meta[:,1]==arm));assert len(ids)==4
    aa=act[ids];zz=mark[ids];pp=P[ids];rg=np.digitize(np.maximum(abs(np.arange(n)//L-L//2),abs(np.arange(n)%L-L//2)),[2,4,7])
    proj=np.eye(4)[rg];pr=pp[:,:,:n]@proj;assert np.max(abs(pr.sum(-1)-1))<1e-11
    for t in (4,8,12,16,20,24,28):
     ns=np.stack([np.bincount(rg[a[:t]],minlength=4) for a in aa])
     for i,j in combinations(range(4),2):
      if np.array_equal(ns[i],ns[j]):
       tv=float(.5*abs(pr[i,t]-pr[j,t]).sum());statepairs+=1
       if tv>1e-9:witnesses.append(dict(L=L,core=core,arm=arm,t=t,rep1=int(meta[ids[i],2]),rep2=int(meta[ids[j],2]),counts=ns[i].tolist(),p1=pr[i,t].tolist(),p2=pr[j,t].tolist(),tv=tv,mark_history_1=float(zz[i,:t].sum()),mark_history_2=float(zz[j,:t].sum())))
    for k in CUTS:
     X=np.array([np.bincount(a[:k],minlength=n)/k for a in aa]);M=X.mean(0);U=X-M
     C=U.T@U/(len(ids)-1);tr=float(np.trace(C));assert abs(C.sum())<1e-10
     # Predictable martingale bracket in same selection filtration, no IID assumption.
     Y=np.eye(n)[aa[:,:k]];p=pp[:,:k,:n];A0=p.sum(1)/k;D0=(Y-p).sum(1)/k
     def cv(z):z=z-z.mean(0);return z.T@z/(len(z)-1)
     Cd,Ca=cv(D0),cv(A0);cross0=C-Cd-Ca
     br=float(np.mean(np.sum(1-np.sum(p*p,-1),axis=1))/k**2)
     Cg=proj.T@C@proj;direct=cv(X@proj);errs['covariance_push']=max(errs['covariance_push'],float(abs(Cg-direct).max()))
     P7=push(np.pad(X,((0,0),(0,1))),mp[7],50);P13=push(np.pad(X,((0,0),(0,1))),mp[13],170)
     # Three-way exact action-source pushforward, sample CGF incl. signed mark.
     jv=np.linspace(-.5,.5,50);J=jv[mp[7]][:n];v=np.c_[P7,zz[:,:k].mean(1)]
     wc=cgf(v,np.r_[jv,.3]);wf=cgf(np.c_[X,zz[:,:k].mean(1)],np.r_[J,.3]);errs['CGF_lift']=max(errs['CGF_lift'],abs(wc-wf))
     # DCT open-grid modes (dimensionless, not a physical correlation length).
     low=float(np.trace(modes@C@modes.T)/2)
     mass=np.stack([X[:,rg==r].sum(1) for r in range(4)],1)
     binder=None;z=mass[:,0]-mass[:,0].mean();v2=np.mean(z*z)
     if v2>0:binder=float(1-np.mean(z**4)/(3*v2*v2))
     groups.append(dict(L=L,core=core,arm=arm,k=k,entropy=float(np.mean([H(q) for q in X])),cov_trace=tr,low_mode=low,bracket=br,drift_trace=float(np.trace(Ca)),martingale_trace=float(np.trace(Cd)),cross_trace=float(np.trace(cross0)),mass=mass.mean(0).tolist(),central_binder=binder,mark=float(zz[:,:k].mean()),W=wc,coarse13=P13.mean(0).tolist(),coarse7=P7.mean(0).tolist(),coarse_cov_trace=float(np.trace(Cg))))
     means[L,core,arm,k]=P13.mean(0)
 for i,L in enumerate(LS[:-1]):
  for core in range(8):
   for arm in (0,1):
    for k in CUTS:
     a,b=means[L,core,arm,k],means[LS[i+1],core,arm,k];cross.append(dict(L=L,Lnext=LS[i+1],core=core,arm=arm,k=k,tv=float(.5*abs(a-b).sum()),outside_delta=float(b[-1]-a[-1])))
 # Finite-size alternatives on matched root-block means at B32, not critical exponent inference.
 fits=[]
 for arm in (0,1):
  for key in ('cov_trace','low_mode'):
   V=np.array([[next(q[key] for q in groups if q['L']==L and q['core']==c and q['arm']==arm and q['k']==32) for L in LS] for c in range(8)])
   y=V.mean(0);x=np.array(LS,float);rows=[]
   for name in ('constant','inverse_area_offset','power'):
    pred=[]
    for i in range(5):
     keep=np.arange(5)!=i
     if name=='constant':v=y[keep].mean()
     elif name=='inverse_area_offset':v=np.r_[1,x[i]**-2]@np.linalg.lstsq(np.c_[np.ones(4),x[keep]**-2],y[keep],rcond=None)[0]
     else:
      assert np.all(y>0);b,a=np.polyfit(np.log(x[keep]),np.log(y[keep]),1);v=np.exp(a+b*np.log(x[i]))
     pred.append(float(v))
    rows.append(dict(model=name,rmse=float(np.sqrt(np.mean((np.array(pred)-y)**2))),pred=pred))
   fits.append(dict(arm=arm,observable=key,mean=y.tolist(),ci95=ci(V),models=rows))
 # R58 historical 3-beta independent cohort (only 2 core geometries, not pooled).
 old=[]
 for L in LS:
  p=B/'mqft_restore58_20261126'/f'PHYSICS_L{L}.json';r=read(p)
  for a in r['spatial']:
   old.append({k:a[k] for k in ['L','B','beta','position','trace','S_low','central_mass','central_binder','density']})
  errs['historic_S0']=max(errs['historic_S0'],max(abs(q['S0']) for q in r['spatial']))
 assert max(errs.values())<1e-10
 witnesses.sort(key=lambda q:-q['tv'])
 out=dict(status='scoped_exact_pushforward_and_nonclosed_count_projection',groups=groups,cross_scale=cross,witnesses=witnesses,collision_pairs=statepairs,composition_checks=composition,errors=dict(errs),fits=fits,prior_R58_spatial=old,prior_R58_source_hashes={str((B/'mqft_restore58_20261126'/f'PHYSICS_L{L}.json').relative_to(B)):sha(B/'mqft_restore58_20261126'/f'PHYSICS_L{L}.json') for L in LS},fresh_paths=sum(q['paths'] for q in audits),fresh_backups=sum(q['backups'] for q in audits),raw_bytes=sum(q['raw_bytes'] for q in audits),candidate_samples=sum(q['candidate_samples'] for q in audits),scope='exact spatial map within same trace; independent-size native ensembles not asserted equal; finite B prefix not separate budget; no thermodynamic critical limit')
 dump(S/'SCALE_RESULTS72.json',out)
 print('SCALE', {k:out[k] for k in ['fresh_paths','fresh_backups','raw_bytes','candidate_samples','collision_pairs','composition_checks','errors']},'witnesses',len(witnesses),'max',witnesses[0] if witnesses else None,flush=True)
 for q in fits:print(q['arm'],q['observable'],q['mean'],[(x['model'],x['rmse']) for x in q['models']],flush=True)
if __name__=='__main__':main()
