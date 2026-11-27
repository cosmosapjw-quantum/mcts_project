"""Independent matrix reference and scalar history likelihood for the registered phase instrument."""
from base61 import *
I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1.,-1.]).astype(complex);PAULI=np.array([X,Y,Z])
ARMS=['plus81','minus81','eraseplus81','eraseminus81','swap81','base81'];FAMS=['g0','g3','Rapfi']
RHO0=(I+.8*X)/2
def adj(a):return a.conj().T
def rho(v):return (I+np.einsum('i,ijk->jk',v,PAULI))/2
def bloch(r):return np.array([np.trace(r@a).real for a in PAULI])
def schedule(mode,t):
 active=32<=t<44;lam=.65 if active and mode!='base81' else 0.
 axis=2 if not active else 1 if t<36 else (2 if ((t<40)!=(mode=='swap81')) else 0)
 theta=0.;eta=1.;erase=mode.startswith('erase') and t==32
 if active and t%4==0:
  theta=(-np.pi/2 if mode in ('minus81','eraseminus81') else np.pi/2) if t==32 else .37 if t==36 else -.51
  eta=.9
 return axis,lam,theta,eta,erase
def free(r,theta,eta,erase):
 U=np.diag([np.exp(-.5j*theta),np.exp(.5j*theta)]);v=U@r@adj(U)
 if erase:return np.diag(np.diag(v))
 return (1+eta)/2*v+(1-eta)/2*Z@v@Z
def operators(m,r,axis,lam):
 P=(I+PAULI[axis])/2;Q=I-P;e0=(1-lam)*r+lam;e1=(1-lam)*r
 A=np.sqrt(m)*(np.sqrt(e0)*P+np.sqrt(e1)*Q);B=np.sqrt(m)*(np.sqrt(1-e0)*P+np.sqrt(1-e1)*Q);O=np.sqrt(1-m)*I
 return A,B,O,e0,e1
def matrix_step(mode,t,state,base,ia,ib,pick):
 ax,lam,theta,eta,erase=schedule(mode,t);pre=state.copy();v=free(state,theta,eta,erase)
 m=base[ia]+base[ib];r=base[ia]/m;A,B,O,e0,e1=operators(m,r,ax,lam)
 q=base.copy()
 if lam>0:q[ia]=np.trace(A@v@adj(A)).real;q[ib]=np.trace(B@v@adj(B)).real
 K=A if pick==ia else B if pick==ib else np.sqrt(base[pick])*I
 out=K@v@adj(K);norm=np.trace(out).real;post=out/norm
 # For no measurement, exact identity, independent of square-root roundoff.
 if lam==0 or pick not in (ia,ib):post=v.copy()
 resid=float(abs(adj(A)@A+adj(B)@B+adj(O)@O-I).max())
 return q,post,dict(pre=bloch(pre),prepared=bloch(v),post=bloch(post),pair_mass=m,population_a=r,e0=e0,e1=e1,output_a=q[ia],axis81=np.eye(3)[ax],lambda81=lam,eta81=eta,rotation=theta,erase81=erase),resid,norm
def scalar_step(mode,t,v,base,ia,ib,pick):
 # Alternative scalar formula, direct longitudinal/transverse decomposition.
 ax,lam,theta,eta,erase=schedule(mode,t);x,y,z=v
 if theta!=0 or eta!=1:
  x,y=eta*(np.cos(theta)*x-np.sin(theta)*y),eta*(np.sin(theta)*x+np.cos(theta)*y)
 if erase:x=y=0.
 v=np.array([x,y,z]);m=base[ia]+base[ib];r=base[ia]/m;z=v[ax];a=(1-lam)*r+lam;b=(1-lam)*r
 q=base.copy()
 if lam>0:
  q[ia]=m*(a+b+(a-b)*z)/2;q[ib]=m-q[ia]
  if pick in (ia,ib):
   if pick==ib:a,b=1-a,1-b
   den=(a+b+(a-b)*z)/2;v*=np.sqrt(a*b)/den;v[ax]=(a-b+(a+b)*z)/(2*den)
 return q,v
