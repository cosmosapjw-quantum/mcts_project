from base61 import *
from scipy.special import xlogy
from scipy.stats import t as student
import gzip,time,zipfile,csv,sys,platform
O=B/'mqft_restore97_20261128'; R87=B/'mqft_restore87_20261128'; R92=B/'mqft_restore92_20261128'; R77=B/'mqft_restore77_20261127'
LS=[13,15,17,19,21]; BS=[1,2,4,8,16,32]
FIELDS=['KL_A','KL_X','KL_conditional_mean','Fisher','KL_step_loss','Fisher_step_loss','beta_field_rms','beta_field_mean_u','theta_x','theta_y','rank','projection_TV','push_projection_TV','theta_next_error','iterated_TV','frozen_logit_derivative_rms','frozen_logit_curvature_rms','nblocks','max_block_mass','entropy_A','entropy_X','KL_A_vs_frozen_mean_gap','beta_theta_x','beta_theta_y']
def kl(q,u): return np.sum(xlogy(q,q/u),axis=-1)
def ent(q): return -np.sum(xlogy(q,q),axis=-1)
def tv(a,b): return .5*np.sum(abs(a-b),axis=-1)
def sm(s,b):
 z=b*s;z-=z.max(axis=-1,keepdims=True);p=np.exp(z);return p/p.sum(axis=-1,keepdims=True)
def ck(err,k,a,b,tol=1e-9):
 e=float(np.max(abs(np.asarray(a)-np.asarray(b))));err[k]=max(err.get(k,0.),e);assert e<tol,(k,e)
def agg(a,inv,n):
 a=np.asarray(a);ix=np.argsort(inv,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(inv[ix]))+1];assert len(starts)==n
 return np.add.reduceat(a[...,ix],starts,axis=-1)
def labels(L,legal,b):
 y,x=np.divmod(legal,L);nb=(L+b-1)//b;k=(y//b)*nb+x//b
 return np.unique(k,return_inverse=True)
def geometry(L,legal):
 y,x=np.divmod(legal,L);F=np.stack([np.cos(np.pi*(x+.5)/L),np.cos(np.pi*(y+.5)/L)],axis=1);n=len(legal);out=[]
 for b in BS:
  ks,inv=labels(L,legal,b);m=len(ks);u=np.bincount(inv,minlength=m)/n
  f=agg(F.T,inv,m).T/(n*u[:,None]);f-=u@f
  A=np.sqrt(u)[:,None]*f;rank=int(np.linalg.matrix_rank(A,tol=1e-10));pinv=np.linalg.pinv(A,rcond=1e-10) if rank else np.zeros((2,m))
  out.append(dict(b=b,ks=ks,inv=inv,u=u,f=f,pinv=pinv,rank=rank))
 for i in range(5):
  a,b=out[i:i+2];idx=np.empty(len(a['u']),int)
  for j in range(len(idx)):
   v=np.unique(b['inv'][a['inv']==j]);assert len(v)==1;idx[j]=v[0]
  a['next']=idx
 return out
def project(q,g):
 u=g['u'];h=np.log(q/u);th=g['pinv']@(np.sqrt(u)*h);z=g['f']@th;z-=z.max();p=u*np.exp(z);p/=p.sum();return th,p
def one(arg):
 L,r=arg;f=B/r['path'];assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256']
 with gzip.open(f,'rt') as z:hd,row,end=map(json.loads,z)
 tr=row['trace'];sp=tr['spec'];k=r['budget'];beta=sp['root_law']['beta_p'][0];rep=tr['replicas'][0];assert end['backups']==k==sp['budget'] and sp['seed']==r['seed'];assert beta==[0.,10/3,8.,16.,math.log(L*L)][r['beta_index']]
 es=sorted(rep['events'],key=lambda e:e['root_selection']['clock']);assert len(es)==k
 legal=np.flatnonzero(np.array(tr['root']['board'])==0);n=len(legal);p=[];s=[];actions=[];er={}
 for j,e in enumerate(es):
  law=e['root_selection']['law'];ac=np.array(law['actions']);o=np.argsort(ac);q=np.array(law['p'])[o];score=np.array(law['scores'])[o];assert np.array_equal(ac[o],legal) and law['t']==j and all(law['active']) and law['logit_h']==0 and ac[law['picked']]==e['actions'][0]
  assert q.min()>0;ck(er,'raw_probability',q,sm(score,beta));ck(er,'normalization',q.sum(),1)
  p.append(q);s.append(score);actions.append(e['actions'][0])
 p=np.array(p);s=np.array(s);a=np.searchsorted(legal,actions);assert np.array_equal(legal[a],actions)
 G=geometry(L,legal);mu=(p*s).sum(-1);var=(p*(s-mu[:,None])**2).sum(-1);seq=[]
 for g in G:
  q=agg(p,g['inv'],len(g['u']));m1=agg(p*s,g['inv'],len(g['u']))/q;m2=agg(p*s*s,g['inv'],len(g['u']))/q
  d=m1-mu[:,None];v=m2-m1*m1;assert v.min()>-1e-9
  F=(q*d*d).sum(-1);cv=var[:,None]-v
  assert np.all(F>=-1e-12) and np.all(F<=var+1e-9)
  ck(er,'fisher_total_variance',F+(q*v).sum(-1),var)
  seq.append(dict(q=q,m1=m1,v=v,F=F,d=d,d2=-cv,K=kl(q,g['u'])))
 for i in range(5):
  g,h=G[i:i+2];a0,b0=seq[i:i+2];u=g['u'];w=h['u'];q=a0['q'];qp=b0['q'];idx=g['next'];ck(er,'push_composition',agg(q,idx,len(w)),qp)
  ck(er,'reference_push',agg(u,idx,len(w)),w)
  loss=(q*np.log((q/qp[:,idx])/(u/w[idx]))).sum(-1);ck(er,'KL_chain_selection',a0['K']-b0['K'],loss);assert loss.min()>-1e-9
  lossF=(q*(a0['m1']-b0['m1'][:,idx])**2).sum(-1);ck(er,'Fisher_chain',a0['F']-b0['F'],lossF)
  gc=np.log(q/u);gp=np.log(qp/w);check=agg(u*np.exp(gc),idx,len(w))/w;ck(er,'logsumexp_push',np.log(check),gp)
 # t32 frozen score derivative and second derivative at every map; actual beta not changed in history
 fd=0
 if r['cohort']==2 and r['rep']==0 and k==64:
  t=32;score=s[t].astype(np.longdouble);delta=np.longdouble(1e-3)/max(1.,float(np.ptp(score)))
  for g,v in zip(G,seq):
   # Independent long-double logsumexp and fourth-order finite differences.
   def gg(b):
    z=b*score;z-=z.max();w=np.exp(z);q=agg(w,g['inv'],len(g['u']))/w.sum()
    return np.log(q/g['u'].astype(np.longdouble))
   mm,lo,mid,hi,pp=[gg(np.longdouble(beta)+i*delta) for i in [-2,-1,0,1,2]]
   ck(er,'frozen_derivative_fd',(mm-8*lo+8*hi-pp)/(12*delta),v['d'][t],1e-5);ck(er,'frozen_hessian_fd',(-mm+16*lo-30*mid+16*hi-pp)/(12*delta**2),v['d2'][t],1e-5);fd+=1
 out=[];meta=[];vec=[];sample=None
 for h in [16,32,64]:
  if h>k:continue
  X=np.bincount(a[:h],minlength=n)/h;A=p[:h].mean(0);vec.append((X,A));meta.append([r[x] for x in ['cohort','core','arm','beta_index','rep','budget']]+[h])
  z=[];qa=[];qx=[];th=[];proj=[];its=[];fullq=[]
  for g,v in zip(G,seq):
   m=len(g['u']);u=g['u'];q=v['q'][:h].mean(0);x=agg(X,g['inv'],m)
   ck(er,'average_push',q,agg(A,g['inv'],m));ck(er,'independent_bincount',q,np.bincount(g['inv'],weights=A,minlength=m));ck(er,'empirical_push',x,np.bincount(g['inv'][a[:h]],minlength=m)/h)
   th0,ph=project(q,g);qa.append(q);qx.append(x);th.append(th0);proj.append(ph);fullq.append(q[g['inv']]/u[g['inv']])
  it=proj[0];its.append(it)
  for i in range(5):
   it0=agg(it,G[i]['next'],len(G[i+1]['u']));_,it=project(it0,G[i+1]);its.append(it)
  for i,(g,v,q,x,t0,ph,it0) in enumerate(zip(G,seq,qa,qx,th,proj,its)):
   u=g['u'];K=float(kl(q,u));Kx=float(kl(x,u));Kn=float(v['K'][:h].mean());F=float(v['F'][:h].mean());assert Kn>=K-1e-9
   dx=dy=loss=floss=rms=mean=cp=ce=0.
   if i<5:
    nxt=G[i+1];j=g['next'];w=nxt['u'];qp=qa[i+1];p0=agg(ph,j,len(w));cp=float(tv(p0,qp));tp,_=project(p0,nxt);ce=float(np.linalg.norm(tp-th[i+1]))
    ck(er,'prefix_composition',agg(q,j,len(w)),qp);ck(er,'prefix_X_composition',agg(x,j,len(w)),qx[i+1])
    f0=np.log(q/u);f1=np.log(qp/w)[j];beta_field=(f1-f0)/np.log(2)
    loss=K-float(kl(qp,w));loss2=float(np.sum(q*np.log((q/qp[j])/(u/w[j]))));ck(er,'KL_chain_prefix',loss,loss2)
    assert loss>=-1e-9
    floss=F-float(seq[i+1]['F'][:h].mean());assert floss>=-1e-9
    rms=float(np.sqrt(u@(beta_field**2)));mean=float(u@beta_field);dx,dy=(th[i+1]-t0)/np.log(2)
   z.append([K,Kx,Kn,F,loss,floss,rms,mean,*t0,g['rank'],tv(q,ph),cp,ce,tv(q,it0),float(np.sqrt(np.mean(np.sum(v['q'][:h]*v['d'][:h]**2,axis=-1)))),float(np.sqrt(np.mean(np.sum(v['q'][:h]*v['d2'][:h]**2,axis=-1)))),len(u),q.max(),ent(q),ent(x),Kn-K,dx,dy])
   if beta==0:
    ck(er,'beta0_uniform',q,u);ck(er,'beta0_projected',ph,u);ck(er,'beta0_beta_field',rms,0)
  assert abs(z[-1][0])<1e-9 and abs(z[-1][3])<1e-9
  out.append(z)
  if (L,r['cohort'],r['arm'],r['beta_index'],r['core'],r['rep'],k,h)==(21,2,1,3,0,0,64,64):
   sample=dict(L=L,legal=legal.tolist(),q_ratio= [v.tolist() for v in fullq],projected_ratio=[(v['u']**-1*q0)[v['inv']].tolist() for v,q0 in zip(G,proj)],metrics=z,meta=meta[-1])
 return dict(meta=meta,out=out,vec=vec,legal=legal,errors=er,fd=fd,draws=k,sample=sample)
def ci(a):
 a=np.asarray(a);m=a.mean(0);w=student.ppf(.975,len(a)-1)*a.std(0,ddof=1)/np.sqrt(len(a));return np.stack([m,m-w,m+w])
def controls():
 G=geometry(4,np.arange(16));u=np.ones(16)/16;z=np.zeros((4,4));z[:2,:2]=[[1.,-1.],[-1.,1.]];z=z.ravel()
 p=u*np.exp(z);p/=p.sum();a=G[0];b=G[1];t0,q0=project(u,a);t1,q1=project(p,a);t2,p2=project(agg(p,a['next'],len(b['u'])),b)
 exact=np.cosh(1)/(3+np.cosh(1));assert np.max(abs(t1-t0))<1e-12 and abs(agg(p,a['next'],len(b['u']))[0]-exact)<1e-12 and np.linalg.norm(t2)>0.01
 # Lifted reference-equilibration projector P, fixed subspace and linear spectrum.
 P=np.eye(len(b['u']))[a['next']]@((np.eye(len(b['u']))[a['next']]*a['u'][:,None]/b['u'][None,:]).T)
 assert np.max(abs(P@P-P))<1e-12 and np.linalg.matrix_rank(P)==4
 return dict(same_fine_theta=t1.tolist(),uniform_theta=t0.tolist(),coarse_theta=t2.tolist(),block_mass=exact,uniform_mass=.25,coarse_TV=float(tv(agg(p,a['next'],4),b['u'])),rank=4,linear_eigenvalues=np.linalg.eigvals(P).real.tolist())
def main():
 start=time.monotonic();m=read(R92/'RUN_MANIFEST92.json');src={};raw={};allg=[];allerrors={};samples=[];totalfd=draws=0;data={}
 for L in LS:
  for f in [f'AUDIT_L{L}.json',f'OBS_L{L}.npz']:
   k=str((R87/f).relative_to(B));assert sha(B/k)==m['inputs'][k];src[k]=m['inputs'][k]
  rec=read(R87/f'AUDIT_L{L}.json')['raw_receipts'];assert len(rec)==864
  for r in rec:assert r['path'] not in raw and m['prior_raw_receipts'][r['path']]==r['sha256'];raw[r['path']]=r['sha256']
  with cf.ProcessPoolExecutor(max_workers=10) as pool:rows=list(pool.map(one,[(L,r) for r in rec],chunksize=3))
  M=np.concatenate([r['meta'] for r in rows]);V=np.concatenate([r['out'] for r in rows]);assert len(M)==2080 and V.shape==(2080,6,len(FIELDS))
  old=np.load(R87/f'OBS_L{L}.npz');assert np.array_equal(M,old['meta']);j=0
  for r in rows:
   for x,a in r['vec']:
    z=r['legal'];ck(allerrors,'R87_X',x,old['X'][j,z]);ck(allerrors,'R87_A',a,old['A'][j,z]);assert abs(old['X'][j].sum()-x.sum())<1e-12;j+=1
   for k,v in r['errors'].items():allerrors[k]=max(v,allerrors.get(k,0.))
   if r['sample'] is not None:samples.append(r['sample'])
   totalfd+=r['fd'];draws+=r['draws']
  data[f'M{L}']=M;data[f'V{L}']=V
  for co,arm,bi,bud,h in sorted(set(tuple(x[[0,2,3,5,6]]) for x in M)):
   ix=np.all(M[:,[0,2,3,5,6]]==[co,arm,bi,bud,h],axis=1);rr=np.stack([V[ix&(M[:,1]==c)].mean(0) for c in range(8)]);assert ix.sum()==8*(2 if co==2 else 4)
   allg.append(dict(L=L,cohort=int(co),arm=int(arm),bi=int(bi),beta=[0.,10/3,8.,16.,math.log(L*L)][bi],budget=int(bud),h=int(h),roots=8,reps=int(ix.sum()/8),ci=ci(rr).tolist()))
  print('L',L,'rows',len(M),'draws',draws,flush=True)
 assert len(raw)==4320 and raw==m['prior_raw_receipts'] and draws==215040 and len(allg)==400 and len(samples)==1 and totalfd==2400
 for n,f in [(77,'ORIGINAL_RG_AUDIT77.json'),(71,'RESULTS_KO.md'),(72,'THEORY.md'),(86,'RESULTS_KO.md'),(87,'ERRATUM86.json'),(87,'CRITICAL87.json'),(92,'SUMMARY92.json')]:
  P=next(B.glob(f'mqft_restore{n}_20*'));r=read(P/f'RUN_MANIFEST{n}.json');k=str((P/f).relative_to(B));assert sha(B/k)==r['files'][f],(P,f);src[k]=r['files'][f]
 np.savez_compressed(S/'RG98.npz',**data)
 dump(S/'SUMMARY98.json',dict(groups=allg,fields=FIELDS,scales=BS,representative=samples[0],controls=controls(),scope='Root-block t intervals, not simultaneous; actual conditional categorical probability, frozen score derivatives; clipped open dyadic maps; independent size/budget strata, no physical critical point'))
 dump(S/'INDEPENDENT98.json',dict(status='PASS_SCOPED',checks=allerrors,raw_receipts=raw,source_hashes=src,software=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),independence='Direct bincount vs nested reduceat; exact chain/variance identities; R87 independent prior reductions, finite difference frozen derivative; same-agent analytic control'))
 dump(S/'RG98.json',dict(status='PASS_SCOPED',paths=len(raw),draws=draws,prefix_rows=10400,scale_rows=62400,fd_cases=totalfd,selection_scale_evaluations=draws*6,strata=len(allg),array_sha=sha(S/'RG98.npz'),elapsed_s=time.monotonic()-start,scope='Finite exact spatial pushforward and two-coordinate truncation, not full native dynamics'))
 print('RESULT',read(S/'RG98.json'),allerrors,flush=True)
if __name__=='__main__':main()
