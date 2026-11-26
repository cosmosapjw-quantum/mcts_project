"""Finite empirical Legendre in J3=0 gauge; curvature only for sampled measure."""
from base61 import *
def main():
 with np.load(S/'CONTROL62.npz') as a:
  m=a['meta'];C=a['C'];z=list(a['arm_names']).index('z')
 r=read(S/'REGISTRATION.json');tr=np.array([c['split']=='train' for c in r['cores']])
 def calc(x,j,mixture):
  a=x if not mixture else x.reshape(1,-1,2)
  z=a@j;mx=z.max(1);p=np.exp(z-mx[:,None]);den=p.sum(1);p/=den[:,None]
  mu=np.einsum('nr,nri->ni',p,a);d=a-mu[:,None,:];H=np.einsum('nr,nri,nrj->nij',p,d,d)
  W=(mx+np.log(den/a.shape[1])).mean()
  return float(W),mu.mean(0),H.mean(0),float(np.mean(1/np.sum(p*p,1)))
 def finite(x,j,mix):
  eps=1e-4;f=lambda v:calc(x,v,mix)[0];u=np.eye(2)*eps;H=np.zeros((2,2))
  for i in range(2):
   H[i,i]=(f(j+u[i])-2*f(j)+f(j-u[i]))/eps**2
  H[0,1]=H[1,0]=(f(j+u[0]+u[1])-f(j+u[0]-u[1])-f(j-u[0]+u[1])+f(j-u[0]-u[1]))/(4*eps**2)
  return H
 out=[]
 for g in (0,3):
  for w in (1,4):
   mask=(m[:,1]==g)&(m[:,2]==w)&(m[:,5]==128)&(m[:,6]==1)&(m[:,8]==z)
   ix=np.flatnonzero(mask);ix=ix[np.lexsort((m[ix,4],m[ix,3]))];x=C[ix,:2].reshape(64,12,2)
   for mix in (False,True):
    _,mu,H,_=calc(x,np.zeros(2),mix);assert np.linalg.eigvalsh(H).min()>0;inv=np.linalg.inv(H);assert abs(inv@H-np.eye(2)).max()<1e-12
    # Independent finite-difference Hessian at two interior points.
    err=max(float(abs(finite(x,j,mix)-calc(x,j,mix)[2]).max()) for j in (np.zeros(2),np.array([.05,-.04])))
    assert err<1e-4
    curve=[]
    for s in np.linspace(-.1,.1,21):
     j=np.array([s,-.75*s]);W,mm,hh,ess=calc(x,j,mix);target=mm.copy();u=np.zeros(2)
     # Invert gradient for exact local empirical conjugate.
     for k in range(80):
      WW,mm2,hh2,_=calc(x,u,mix);delta=mm2-target
      if np.linalg.norm(delta)<1e-9:break
      step=np.linalg.solve(hh2,delta);obj=WW-u@target;decrease=delta@step
      scale=1.
      for line in range(40):
       trial=u-scale*step;wv=calc(x,trial,mix)[0]-trial@target
       if wv<=obj-1e-4*scale*decrease+1e-14:break
       scale*=.5
      else:raise RuntimeError('dual line search failed')
      u=trial
     assert np.linalg.norm(u-j)<1e-7
     Gamma=float(j@mm-W);quad=float((mm-mu)@inv@(mm-mu)/2)
     train=calc(x[tr],j,mix);test=calc(x[~tr],j,mix)
     curve.append(dict(J=j.tolist(),mean=mm.tolist(),W=W,Gamma=Gamma,quadratic=quad,ess=ess,train_W=train[0],test_W=test[0],train_mean=train[1].tolist(),test_mean=test[1].tolist()))
    out.append(dict(generation=g,width=w,mixture=mix,mean=mu.tolist(),hessian=H.tolist(),legendre_curvature=inv.tolist(),hessian_fd_error=err,curve=curve))
 dump(S/'LEGENDRE62.json',dict(scope='finite empirical measure, J3=0 gauge; mean per-root W vs mixture W; local exact convex conjugate, not physical loop or continuum 1PI; train/test root split diagnostic only',rows=out))
 print('LEGENDRE',[(x['generation'],x['width'],x['mixture'],x['hessian'],x['legendre_curvature'],x['hessian_fd_error']) for x in out])
if __name__=='__main__':main()
