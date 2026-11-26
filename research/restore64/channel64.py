"""Independent complex128 reference and software tomography.
Calibration shots are model-generated, never physical measurements."""
from base61 import *
from scipy.optimize import least_squares
I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1.,-1.]).astype(complex)
PROJ=np.array([(I+X)/2,(I+Y)/2])
def dagger(a):return a.swapaxes(-1,-2).conj()
def rho_channel(r,theta,eta):
 r=np.asarray(r,float);theta=np.asarray(theta,float);eta=np.asarray(eta,float)
 psi=np.stack([np.sqrt(r),np.sqrt(1-r)],-1).astype(complex)
 U=np.zeros(psi.shape[:-1]+(2,2),complex)
 U[...,0,0]=np.exp(-.5j*theta);U[...,1,1]=np.exp(.5j*theta)
 pure=(psi[...,None]*psi[...,None,:].conj());rot=U@pure@dagger(U)
 # explicit dephasing Kraus matrices, not set offdiagonal by hand
 K0=np.sqrt((1+eta)/2)[...,None,None]*I
 K1=np.sqrt((1-eta)/2)[...,None,None]*Z
 rho=K0@rot@dagger(K0)+K1@rot@dagger(K1)
 return rho
def readout(rho,alpha=.35):
 U=np.cos(alpha)*I-1j*np.sin(alpha)*X
 out=U@rho@dagger(U)
 xy=np.stack([np.trace(rho@X,axis1=-2,axis2=-1).real,np.trace(rho@Y,axis1=-2,axis2=-1).real],-1)
 return out[...,0,0].real,xy
def channel_checks():
 errs=[]
 for theta in np.linspace(-4,4,17):
  for eta in np.linspace(0,1,11):
   U=np.diag([np.exp(-.5j*theta),np.exp(.5j*theta)])
   K=np.stack([np.sqrt((1+eta)/2)*U,np.sqrt((1-eta)/2)*Z@U])
   tp=float(abs(np.sum(dagger(K)@K,axis=0)-I).max());vec=K.reshape(2,4);choi=vec.T@vec.conj();eig=float(np.linalg.eigvalsh(choi).min())
   assert tp<2e-15 and eig>-2e-15
   errs.append((tp,eig))
 # exact gauge control and independent explicit ODE using scipy expm
 from scipy.linalg import expm
 from scipy.integrate import solve_ivp
 rng=np.random.default_rng(640110);ge=oe=0.
 for k in range(64):
  r=float(rng.uniform(.05,.95));t=float(rng.uniform(0,1));gap=float(rng.uniform(-1,1));theta=float(rng.uniform(-np.pi,np.pi))
  rho0=rho_channel(r,theta,.8);H=3*gap*Z;g=1.2
  def rhs(t,y):
   x=y.reshape(2,2);return (-1j*(H@x-x@H)+g/2*(Z@x@Z-x)).ravel()
  x=solve_ivp(rhs,(0,t),rho0.ravel(),rtol=1e-11,atol=1e-13).y[:,-1].reshape(2,2)
  rho=rho_channel(r,theta+6*gap*t,.8*np.exp(-g*t));oe=max(oe,float(abs(x-rho).max()))
  D=np.diag(np.exp(1j*rng.uniform(-np.pi,np.pi,2)));U=expm(-.35j*X);P=I.copy();P[1,1]=0
  q=np.trace(P@U@rho@dagger(U));Q=D@P@dagger(D);V=D@U@dagger(D);rr=D@rho@dagger(D);ge=max(ge,float(abs(q-np.trace(Q@V@rr@dagger(V)))))
 assert oe<2e-11 and ge<2e-15
 return dict(kraus_cases=len(errs),max_tp_error=max(x[0] for x in errs),min_choi_eigenvalue=min(x[1] for x in errs),ode_cases=64,max_ode_error=oe,max_gauge_error=ge)
def synth(seed,n,shots,pars):
 rng=np.random.default_rng(seed);r=rng.uniform(.12,.88,n);theta0=rng.uniform(-np.pi,np.pi,n);tau=rng.uniform(0,1,n);exposure=rng.uniform(-.85,.85,n)
 eta0,gamma,kappa=pars;theta=theta0+kappa*exposure;eta=eta0*np.exp(-gamma*tau)
 rho=rho_channel(r,theta,eta)
 pr=np.stack([np.trace(rho@P,axis1=-2,axis2=-1).real for P in PROJ],-1)
 assert np.all((pr>=0)&(pr<=1));y=rng.binomial(shots,pr)
 return dict(r=r,theta0=theta0,tau=tau,exposure=exposure,y=y,pr=pr)
def pred(x,z):
 eta0,gamma,kappa=z;theta=x['theta0']+kappa*x['exposure'];eta=eta0*np.exp(-gamma*x['tau']);amp=2*np.sqrt(x['r']*(1-x['r']))*eta
 return np.stack([amp*np.cos(theta),amp*np.sin(theta)],-1)
def calibration():
 reg=read(S/'REGISTRATION.json');c=reg['calibration'];pars=[reg['eta0'],reg['gamma'],reg['kappa']];n=c['shots_per_quadrature']
 tr=synth(c['train_seed'],c['train_states'],n,pars);te=synth(c['test_seed'],c['test_states'],n,pars)
 # Fixed initial guess and bounded optimization, not fit to native search outcomes.
 f=lambda z:(pred(tr,z)-(2*tr['y']/n-1)).ravel()
 fit=least_squares(f,[.6,.8,4.],bounds=([0.,0.,0.],[1.,5.,12.]),ftol=1e-12,xtol=1e-12,gtol=1e-12)
 assert fit.success and np.max(abs(fit.x-pars))<.10
 def score(x,z):
  e=pred(x,z)-(2*x['y']/n-1);p=np.clip((pred(x,z)+1)/2,1e-12,1-1e-12)
  nll=-np.mean(x['y']/n*np.log(p)+(1-x['y']/n)*np.log1p(-p))
  return dict(rmse=float(np.sqrt(np.mean(e*e))),nll=float(nll))
 control=te.copy()
 # Quadrature null: phase stays theta0, eta fitted decay, and full dephasing.
 np.savez_compressed(S/'CALIBRATION64.npz',**{f'train_{k}':v for k,v in tr.items()},**{f'test_{k}':v for k,v in te.items()})
 result=dict(scope='software-generated independent calibration/heldout shots, known model controls; not physical data',truth=pars,fitted=fit.x.tolist(),train=score(tr,fit.x),heldout=score(te,fit.x),heldout_true=score(te,pars),heldout_no_rotation=score(te,[*fit.x[:2],0.]),heldout_dephased=score(te,[0.,0.,0.]),shots=n*2*(len(tr['r'])+len(te['r'])),checks=channel_checks(),array_sha=sha(S/'CALIBRATION64.npz'))
 dump(S/'CALIBRATION64.json',result);print('CALIBRATION',result,flush=True)
if __name__=='__main__':calibration()
