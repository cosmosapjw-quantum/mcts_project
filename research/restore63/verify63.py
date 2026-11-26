from kernel63 import *
def main():
 lock=read(S/'MODEL_LOCK63.json');assert lock['code_sha']==sha(S/'kernel63.py')
 r=read(S/'KERNEL_RESULTS63.json');assert len(r['results'])==32 and len(r['tilt_composition'])==96
 with np.load(S/'WAVES63.npz') as d:w={k:d[k] for k in d.files}
 tags=list(w['tags']);meta=w['meta'];ops=[];local=[];acf=[];curves=[]
 for g in (0,3):
  p=S/f'KERNEL_MODEL_g{g}.npz';assert sha(p)==lock['models'][str(g)]['kernel_sha']
  with np.load(p) as d:k={n:d[n] for n in d.files}
  # Independent tilted-transfer product vs coefficient DP, all registered horizons.
  err=0.;errc=0.;nchecks=0
  for t in (4,8,16,24):
   for h in (1,2,4,8):
    d=propagate_p(k['KP'],t,h);grid=np.indices(d.shape[1:]).transpose(1,2,0)
    for J in (np.array([-.1,.05,0]),np.zeros(3),np.array([.1,-.05,0])):
     T=np.eye(16)
     for s in range(h):T=T@np.einsum('y,yij->ij',np.exp(Y@J),k['KP'][t+s])
     want=T.sum(1);actual=np.sum(d*np.exp(grid@J[:2])[None,:,:],axis=(1,2))
     err=max(err,float(abs(want-actual).max()));nchecks+=len(want)
    if h==2:
     # Explicit two-wave brute enumeration for a finite set of initial actual counts.
     starts=np.array([[0,0],[1,2],[2,0]],int);q=propagate_count(k['emC'],t,2,starts)
     for i,c in enumerate(starts):
      ref=np.zeros((9,9))
      for j,a in enumerate(Y):
       st=int(states(c/(4*t)));st2=int(states((c+a[:2])/(4*(t+1))))
       for l,b in enumerate(Y):ref[a[0]+b[0],a[1]+b[1]]+=k['emC'][t,st,j]*k['emC'][t+1,st2,l]
      errc=max(errc,float(abs(q[i]-ref).max()))
  assert err<1e-10 and errc<1e-12
  ops.append(dict(generation=g,checks=nchecks,tilted_operator_dp_error=err,count_two_wave_enumeration_error=errc))
  ix=np.flatnonzero(meta[:,1]==tags.index(f'hold{g}'));ix=ix[np.lexsort((meta[ix,4],meta[ix,3]))]
  P=w['wave_p'][ix,:32];C=w['wave_c'][ix];YY=np.diff(C,axis=1);assert len(ix)==1024
  mass=masses(P);maxerr=0.;mean_err=0.;quad_err=0.
  for j in np.linspace(-.3,.3,25):
   J=np.array([j,-j/2,0]);logz=logsumexp(np.log(P)+J,axis=-1);Q=np.exp(np.log(P)+J-logz[:,:,None]);m=4*Q
   Wc=4*logz;Wsum=np.log(np.sum(mass*np.exp(Y@J),axis=-1));maxerr=max(maxerr,float(abs(Wc-Wsum).max()))
   gam=4*np.sum(Q*np.log(Q/P),axis=-1);ident=np.einsum('nti,i->nt',m,J)-Wc;mean_err=max(mean_err,float(abs(ident-gam).max()))
   dm=m[:,:,:2]-4*P[:,:,:2];quad=(np.sum(dm*dm/P[:,:,:2],axis=-1)+dm.sum(-1)**2/P[:,:,2])/8
   curves.append(dict(generation=g,J=j,mean_exact_action=float(gam.mean()),mean_quadratic_action=float(quad.mean()),mean_logmgf=float(Wc.mean()),max_abs_local_quadratic_error=float(abs(quad-gam).max())))
  assert maxerr<1e-11 and mean_err<1e-11
  local.append(dict(generation=g,states=P.shape[0]*P.shape[1],tilts=25,mgf_error=maxerr,legendre_error=mean_err))
  # Conditional innovations: no same-sample phase mean removal. Root-block product means.
  X=YY-4*P
  for lag in range(1,9):
   for group in (0,1):
    a=YY[:,:-lag,group].ravel();b=YY[:,lag:,group].ravel()
    raw=float(np.corrcoef(a,b)[0,1])
    a=X[:,:-lag,group];b=X[:,lag:,group]
    prod=(a*b).mean(1).reshape(64,16).mean(1);z=summary(prod)
    acf.append(dict(generation=g,group=group,lag=lag,raw_correlation=raw,conditional_innovation_product=z,normalized_product=float(prod.mean()/np.sqrt(np.mean(a*a)*np.mean(b*b)))))
 dump(S/'VERIFICATION63.json',dict(operator=ops,conditional_action=local,curves=curves,innovation_lags=acf,scope='operator algebra and conditional exact action verified on heldout states; lag products are descriptive, not physical relaxation times'))
 print('VERIFY',ops,local,flush=True)
if __name__=='__main__':main()
