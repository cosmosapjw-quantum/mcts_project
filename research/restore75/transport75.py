"""Actual score-graph coherent reference: stationary open leads and monitored finite transport."""
from base61 import *
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
from scipy.special import logsumexp
J=.35;ENERGIES=[.05,.15,.35];HORIZON=64
def hamiltonian(g,path=False,phi=0.,leads=False):
 vertices=g['path'] if path else g['component'];index={v:i for i,v in enumerate(vertices)};pot=dict(zip(g['component'],g['potential']));V=np.array([pot[v] for v in vertices]);H=np.diag(V).astype(complex)
 edges=list(zip(vertices[:-1],vertices[1:])) if path else g['edges'];edge=tuple(g['plaquette'][:2]) if g['plaquette'] else None
 for a,b in edges:
  i,j=index[a],index[b];z=np.exp(1j*phi) if not path and (a,b)==edge else np.exp(-1j*phi) if not path and (b,a)==edge else 1.
  H[i,i]+=J;H[j,j]+=J;H[i,j]-=J*z;H[j,i]-=J*np.conj(z)
 s,t=index[g['source']],index[g['target']]
 if leads:H[s,s]+=J;H[t,t]+=J
 assert np.max(abs(H-H.conj().T))<1e-14
 return H,s,t,V
def scatter(H,s,t,E,eta=1e-11,dense=False,J0=J):
 k=np.arccos(1-E/(2*J0));gamma=2*J0*np.sin(k);sig=-J0*np.exp(1j*k);A=(E+1j*eta)*np.eye(len(H))-H
 A[s,s]-=sig;A[t,t]-=sig;b=np.zeros(len(H),complex);b[s]=1
 v=solve(A,b,assume_a='gen',check_finite=False) if dense else spsolve(csc_matrix(A),b)
 T=float(abs(gamma*v[t])**2);R=float(abs(1-1j*gamma*v[s])**2);loss=float(2*eta*gamma*np.vdot(v,v).real);res=float(np.linalg.norm(A@v-b));balance=abs(T+R+loss-1)
 assert res<1e-6 and balance<1e-6 and T<=1+1e-6 and R<=1+1e-6
 return dict(T=T,R=R,regulator_loss=loss,flux_error=balance,residual=res)
def transfer(V,E,J0=J):
 k=np.arccos(1-E/(2*J0));C=np.array([[np.exp(1j*k),np.exp(-1j*k)],[1,1]],complex);A=np.eye(2)
 for v in V:A=np.array([[2+(v-E)/J0,-1.],[1.,0.]])@A
 D=np.linalg.solve(C,A@C)
 return float(1/abs(D[1,1])**2)
def actions(V,E,J0=J):
 x=1+(V-E)/(2*J0);s=float(np.arccosh(np.maximum(abs(x),1)).sum());sc=float(np.sqrt(np.maximum((V-E)/J0,0)).sum())
 return s,sc,int(np.sum(x>1)),int(np.sum(x<-1))
def detector(H,s,t):
 val,V=np.linalg.eigh(H);U=(V*np.exp(-1j*val)[None,:])@V.conj().T
 err=float(abs(U.conj().T@U-np.eye(len(H))).max());assert err<1e-11
 B=abs(U)**2;assert abs(B.sum(0)-1).max()<1e-11 and abs(B.sum(1)-1).max()<1e-11
 psi=np.zeros(len(H),complex);psi[s]=1;p=np.zeros(len(H));p[s]=1
 fc=[];fd=[];sc=[1.];sd=[1.]
 for k in range(HORIZON):
  psi=U@psi;p=B@p
  fc.append(float(abs(psi[t])**2));fd.append(float(p[t]));psi[t]=0;p[t]=0
  sc.append(float(np.vdot(psi,psi).real));sd.append(float(p.sum()))
  assert abs(sum(fc)+sc[-1]-1)<2e-10 and abs(sum(fd)+sd[-1]-1)<2e-10
 return dict(coherent_f=fc,dephased_f=fd,coherent_survival=sc,dephased_survival=sd,coherent_hit=sum(fc),dephased_hit=sum(fd),coherent_RMST=sum(sc[:-1]),dephased_RMST=sum(sd[:-1]),max_unitarity=err)
def run(g):
 H,s,t,V=hamiltonian(g,leads=True);Hp,sp,tp,Vp=hamiltonian(g,path=True,leads=True);has=bool(g['plaquette']);Hphi,_,_,_=hamiltonian(g,phi=np.pi/2,leads=True)
 out=[]
 for E in ENERGIES:
  full=scatter(H,s,t,E);reg=scatter(H,s,t,E,eta=1e-9);pa=scatter(Hp,sp,tp,E);tx=transfer(Vp,E)
  assert abs(tx-pa['T'])<1e-7
  a,c,lo,hi=actions(Vp,E);ph=scatter(Hphi,s,t,E);pr=scatter(Hphi.conj(),s,t,E)
  # Site gauge with lead couplings transformed only up to unobservable phase at contacts.
  z=np.exp(1j*.371*np.arange(len(H)));Hg=z[:,None]*Hphi*z.conj()[None,:];gauge=scatter(Hg,s,t,E)
  assert abs(ph['T']-pr['T'])<1e-7 and abs(ph['T']-gauge['T'])<1e-7
  out.append(dict(E=E,full=full,regularized=reg,path=pa,transfer_T=tx,transfer_error=abs(tx-pa['T']),action_discrete=a,action_continuum=c,lower_forbidden_sites=lo,upper_forbidden_sites=hi,phase_T=ph['T'],reverse_phase_T=pr['T'],gauge_T=gauge['T'],phase_reciprocity_error=abs(ph['T']-pr['T']),gauge_error=abs(ph['T']-gauge['T']),regulator_sensitivity=abs(full['T']-reg['T'])))
 h0,s0,t0,_=hamiltonian(g);h1,_,_,_=hamiltonian(g,phi=np.pi/2);d0=detector(h0,s0,t0);d1=detector(h1,s0,t0)
 return dict(family=g['family'],core=g['core'],rep=g['rep'],height=g['height'],width=g['width'],graph_sites=len(H),has_plaquette=has,path_potential=Vp.tolist(),scattering=out,detector=d0,phase_detector=d1)
def logsinh(x):
 return float(x+np.log(-np.expm1(-2*x))-np.log(2))
def uniform(E,V,n,J0):
 k=np.arccos(1-E/(2*J0));x=1+(V-E)/(2*J0)
 if x>1:
  z=np.arccosh(x);logU=logsinh(n*z)-logsinh(z)
 elif x==1:logU=np.log(n)
 else:
  z=np.arccos(x);q=abs(np.sin(n*z)/np.sin(z));logU=np.log(max(q,1e-300))
 if V==0:return 0.
 return -float(np.logaddexp(0,2*(np.log(abs(V/(2*J0*np.sin(k))))+logU)))
def controls():
 rect=[];continuum=[]
 for v in [.06,.1,.25,.5,1.]:
  for n in [1,2,4,8,16,32,64]:
   l=uniform(.05,v,n,J);V=np.full(n,v);a,b,_,_=actions(V,.05);t=transfer(V,.05)
   assert abs(np.log(t)-l)<1e-10
   rect.append(dict(height=v,n=n,logT=l,discrete_exponent=-2*a,continuum_exponent=-2*b,logT_transfer=np.log(t)))
 for width in [1.,2.,4.,8.]:
  E=.2;V=1.;kappa=np.sqrt(V-E);lc=-np.log1p(V**2*np.sinh(kappa*width)**2/(4*E*(V-E)))
  for n in [8,16,32,64,128,256]:
   a=width/n;j=1/a**2;l=uniform(E,V,n,j);continuum.append(dict(width=width,n=n,lattice_spacing=a,logT_lattice=l,logT_continuum=float(lc),error=abs(l-lc),local_wkb_exponent=-2*kappa*width))
 return rect,continuum
def main():
 a=read(S/'FIRST_RESULTS75.json');assert sha(S/'FIRST75.npz')==a['array_sha256'];g=a['contexts'];assert len(g)==96
 with cf.ProcessPoolExecutor(max_workers=8) as pool:r=list(pool.map(run,g,chunksize=1))
 rect,cont=controls();ss=[z for q in r for z in q['scattering']]
 out=dict(scope='declared score graph Laplacian + potential, identical single-channel leads; separate closed monitored detector; phase assigned not measured',contexts=r,rectangular_controls=rect,continuum_controls=cont,source_sha=sha(S/'FIRST_RESULTS75.json'),max_flux_error=max(max(x[k]['flux_error'] for k in ('full','path','regularized')) for x in ss),max_transfer_error=max(x['transfer_error'] for x in ss),max_gauge_error=max(x['gauge_error'] for x in ss),max_reciprocity_error=max(x['phase_reciprocity_error'] for x in ss),max_regulator_sensitivity=max(x['regulator_sensitivity'] for x in ss),max_detector_unitarity=max(max(q['detector']['max_unitarity'],q['phase_detector']['max_unitarity']) for q in r))
 dump(S/'TRANSPORT75.json',out)
 for k in range(3):
  z=[q for q in r if q['family']==k];v=[x['scattering'][1] for x in z]
  print('TRANSPORT',k,'E=.15 full/path',np.mean([a['full']['T'] for a in v]),np.mean([a['path']['T'] for a in v]),'mean phase diff',np.mean([abs(a['phase_T']-a['full']['T']) for a in v]),'detector coherent/dephased',np.mean([q['detector']['coherent_hit'] for q in z]),np.mean([q['detector']['dephased_hit'] for q in z]),flush=True)
 print('AUDIT',{k:v for k,v in out.items() if k not in ('contexts','rectangular_controls','continuum_controls')},flush=True)
if __name__=='__main__':main()
