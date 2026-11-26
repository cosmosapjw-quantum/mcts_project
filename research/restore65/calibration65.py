"""Additional independent synthetic calibration for unsharp readout, not native outcome fitting."""
from audit65 import *
from scipy.optimize import least_squares
def make(seed,n):
 rng=np.random.default_rng(seed)
 v=rng.normal(size=(n,3));v/=np.linalg.norm(v,axis=1)[:,None];v*=rng.uniform(.1,.9,n)[:,None]
 r=rng.uniform(.1,.9,n);s=1024;mu=[];check=0.;cp=0.
 for vv,rr in zip(v,r):
  x=rho(vv);e,A,B=effects(rr)
  ma=A@x@dagger(A);mb=B@x@dagger(B);p=float(np.trace(ma).real);q=float(np.trace(mb).real)
  assert abs(p+q-1)<1e-14
  # independent unconditioned map transverse factor in rotated frame.
  u=RX@x@dagger(RX);g=np.sqrt(e.prod())+np.sqrt((1-e).prod())
  ref=u.copy();ref[0,1]*=g;ref[1,0]*=g
  check=max(check,float(abs(ma+mb-dagger(RX)@ref@RX).max()))
  for o in (ma,mb):cp=min(cp,float(np.linalg.eigvalsh(o).min()))
  mu.append(p)
 p=np.array(mu);y=rng.binomial(s,p);return dict(v=v,r=r,p=p,y=y),check,cp
def predict(d,pars):
 lam,al=pars;x,y,z=d['v'].T;return (1-lam)*d['r']+lam*(1+y*np.sin(2*al)+z*np.cos(2*al))/2
def main():
 tr,e1,p1=make(650300,512);te,e2,p2=make(650301,256);s=1024
 fit=least_squares(lambda z:predict(tr,z)-tr['y']/s,[.5,.2],bounds=([0.,0.],[.999,np.pi/2]),ftol=1e-12,xtol=1e-12,gtol=1e-12)
 assert fit.success and abs(fit.x-[.65,.35]).max()<.02 and max(e1,e2)<1e-13 and min(p1,p2)>-1e-12
 def score(d,z):return float(np.sqrt(np.mean((predict(d,z)-d['y']/s)**2)))
 np.savez_compressed(S/'CALIBRATION65.npz',**{f'train_{k}':v for k,v in tr.items()},**{f'test_{k}':v for k,v in te.items()})
 out=dict(scope='independent software-only calibration, parameters registered prior to native generation; not changed by fit; no physical data',truth=[.65,.35],fit=fit.x.tolist(),train=score(tr,fit.x),test=score(te,fit.x),test_null=score(te,[.65,0.]),test_no_measurement=score(te,[0.,0.]),shots=(512+256)*s,nonselective_matrix_error=max(e1,e2),minimum_selective_eigenvalue=min(p1,p2),prior_R64_calibration_sha=read(S/'REGISTRATION.json')['prior_calibration_sha'],array_sha=sha(S/'CALIBRATION65.npz'))
 dump(S/'CALIBRATION65.json',out);print(out)
if __name__=='__main__':main()
