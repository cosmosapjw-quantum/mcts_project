"""Exact integer exposed faces, KL boundary rate and authenticated replica expansion."""
from base61 import *
from scipy.special import logsumexp
from scipy.optimize import brentq
from scipy.fft import rfft2,irfft2,next_fast_len
from numpy.polynomial.hermite import hermgauss
from itertools import combinations
O=B/'mqft_restore80_20261127';P=B/'mqft_restore86_20261127'
F=['g0','g3','Rapfi'];ERR=collections.defaultdict(float)
def ck(name,a,b,tol=1e-9):
 x=float(np.max(abs(np.asarray(a)-np.asarray(b))));ERR[name]=max(ERR[name],x);assert x<tol,(name,x)
def cg(x,j):
 l=x@j;w=logsumexp(l)-np.log(len(x));q=np.exp(l-logsumexp(l));m=q@x;d=x-m;H=(d*q[:,None]).T@d
 return float(w),q,m,H
def cross(a,b,c):return int((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
def hull(y):
 p=sorted(set(map(tuple,y)));a=[]
 for t in p:
  while len(a)>=2 and cross(a[-2],a[-1],t)<=0:a.pop()
  a.append(t)
 b=[]
 for t in p[::-1]:
  while len(b)>=2 and cross(b[-2],b[-1],t)<=0:b.pop()
  b.append(t)
 h=np.array(a[:-1]+b[:-1],int);assert len(h)>=3
 # Independent certificate: each edge supports all points; all extreme vertices appear.
 for i,v in enumerate(h):
  w=h[(i+1)%len(h)]
  assert all(cross(v,w,x)>=0 for x in p)
  assert cross(h[i-1],v,w)>0
 # Compare against scipy ConvexHull on integer inputs.
 from scipy.spatial import ConvexHull
 assert set(map(tuple,h))==set(map(tuple,y[ConvexHull(y).vertices]))
 return h
def inside(y,h):
 return np.array([all(cross(v,h[(i+1)%len(h)],x)>=0 for i,v in enumerate(h)) for x in y])
def face(y,label):
 h=hull(y);rows=[];X=y/128;N=len(y)
 for i,v in enumerate(h):
  w=h[(i+1)%len(h)];u=w-v;n=np.array([u[1],-u[0]],int);support=int(v@n);gap=support-y@n;assert min(gap)==0 and np.any(gap>0)
  sel=gap==0; mass=float(sel.mean());j=u/np.linalg.norm(u);x=X[sel]@j;nf=gap[gap>0].min();out=[]
  for t in (-2.,0.,2.):
   W,q,m,H=cg(X[sel],t*j);gamma=float(t*(j@m)-W-np.log(mass));KL=float(q@np.log(q*N));ck('face_KL',gamma,KL)
   var=float(j@H@j);assert var>0
   # Independent one-dimensional inversion from exact projected scalar coordinates.
   def mom(b):
    r=np.exp(b*x-max(b*x));return float(r@x/r.sum())
   t2=brentq(lambda b:mom(b)-j@m,-64,64,xtol=1e-11);ck('face_tangent_inverse',t2,t,1e-7)
   lam=[];target=W+np.log(mass)
   for b in (0.,1.,4.,16.,64.,256.):
    l=X@(t*j)-b*gap/nf;z=logsumexp(l)-np.log(N);p=np.exp(l-logsumexp(l));normal=float(p[~sel].sum());error=float(z-target)
    ck('face_partition_identity',error,-np.log1p(-normal) if normal<1 else error,1e-9)
    assert error>=-1e-12
    lam.append(dict(normal_t=b,outside_mass=normal,partition_residual=error))
   ck('face_limit',lam[-1]['partition_residual'],0,1e-10)
   out.append(dict(tangent_t=t,mean=m.tolist(),gamma=gamma,conditional_gamma=gamma+np.log(mass),tangent_variance=var,tangent_hessian=1/var,normal_scan=lam))
  rows.append(dict(edge=[v.tolist(),w.tolist()],face_mass=mass,minimum_integer_gap=int(nf),support_count=int(sel.sum()),normal=n.tolist(),cases=out))
 vert=[dict(vertex=v.tolist(),mass=float(np.all(y==v,1).mean()),gamma=float(-np.log(np.all(y==v,1).mean()))) for v in h]
 assert all(np.isfinite(x['gamma']) for x in vert)
 # One constructed exterior point per edge, strict separating hyperplane.
 for i,v in enumerate(h):
  w=h[(i+1)%len(h)];u=w-v;n=np.array([u[1],-u[0]]);assert not inside(np.array([v+n]),h)[0]
 return dict(label=label,hull=h.tolist(),n=N,unique=len(np.unique(y,axis=0)),edges=rows,vertices=vert)
def main():
 a=read(O/'AUDIT80.json');assert sha(O/'MARKED80.npz')==a['array_sha']
 with np.load(O/'MARKED80.npz') as f:D={k:f[k] for k in f.files}
 A=read(O/'ACTION80.json');M=D['meta'];X=D['X'];Y=D['Y'];laws={};faces=[];interior=[];transfers=[]
 q=read(S/'PATH90.json')
 with np.load(S/'PATH90.npz') as z:N=z['meta'];y=z['Y'];xx=z['X']
 for g,name in enumerate(F):
  for co in (0,1):
   ix=(M[:,0]==co)&(M[:,1]==g)&(M[:,2]<32);v=Y[ix];assert len(v)==256
   laws[name,co]=v;faces.append(face(v,[name,'R80_reference' if co==0 else 'R80_matched']))
  for co in (3,4):
   z=y[(N[:,0]==co)&(N[:,1]==g)];faces.append(face(z,[name,'R84' if co==3 else 'R86']))
  h=hull(laws[name,0]);test=[('matched',laws[name,1]),('novel_R80',Y[(M[:,0]==1)&(M[:,1]==g)&(M[:,2]>=32)]),('R84',y[(N[:,0]==3)&(N[:,1]==g)]),('R86',y[(N[:,0]==4)&(N[:,1]==g)])]
  for k,z in test:transfers.append(dict(family=name,cohort=k,n=len(z),outside=int((~inside(z,h)).sum())))
  # All existing bivariate CGF/1PI cases at same source and same empirical X.
  for case in A['sources'][g]['cases']:
   j=np.array(case['J'])
   for k,sel in [('train',(M[:,0]==0)),('matched',(M[:,0]==1)&(M[:,2]<32)),('novel',(M[:,0]==1)&(M[:,2]>=32))]:
    d=X[sel&(M[:,1]==g)];W,q,m,H=cg(d,j);r=case[k]
    ck('W',W,r['W']);ck('mean',m,r['phi']);ck('H',H,r['H']);ck('inverse_H',np.linalg.inv(H),r['inverse_H'],1e-7);ck('Gamma',j@m-W,r['gamma']);ck('KL',q@np.log(q*len(q)),r['KL']);interior.append([g,k,j.tolist(),W,float(j@m-W),float(np.linalg.eigvalsh(H).min())])
 # All180 stored saddle results: independent Gaussian-Hermite, algebra, and full unaliased FFT for six M8 J0=0.
 a,w=hermgauss(5);zz=np.array([(x,y) for x in a*np.sqrt(2) for y in a*np.sqrt(2)]);wt=np.outer(w,w).ravel()/np.pi;fft_checks=0;coeff=[]
 for r in A['saddles']:
  assert r['status']=='OK';v=laws[r['family'],r['cohort']];s=np.array(r['s']);K,q,m,H=cg(v,s);M0=r['M'];k=np.array(r['k']);ck('saddle_mean',m,k/M0,1e-8);ck('saddle_H',H,r['H'],1e-7)
  eig,U=np.linalg.eigh(H);wh=(U/np.sqrt(eig))@U.T;z=(v-m)@wh;v0=z@zz.T
  c3=q@(v0**3);c4=q@(v0**4)-3*(np.sum(zz**2,axis=1))**2
  C=float(wt@(c4/24-c3*c3/72));ck('loop_coefficient',C,r['coeff'],1e-8)
  lead=1/(2*np.pi*M0*np.sqrt(np.linalg.det(H)));ck('leading',lead,r['leading']);ck('corrected',lead*(1+C/M0),r['corrected']);p0=r['wrapped_mass'];assert r['alias_bound']/p0<=1e-12
  gamma=float(s@(k/M0)-K);ck('lattice_gamma',gamma,r['gamma']);ck('negative_log_mass',gamma-np.log(p0)/M0,r['exact_action_per_replica'])
  if M0==8 and r['J0']==[0.,0.]:
   off=v.min(0);span=v.max(0)-off;shape=tuple(next_fast_len(int(M0*x+1)) for x in span);a0=np.zeros(shape);v1=v-off;np.add.at(a0,(v1[:,0],v1[:,1]),q);a0=irfft2(rfft2(a0)**M0,s=shape);kk=k-M0*off;p=float(a0[tuple(kk)])
   ck('unaliased_fft',p,p0,1e-10);fft_checks+=1
  coeff.append(dict(family=r['family'],cohort=r['cohort'],M=M0,leading_error=abs(r['leading_rel']),corrected_error=abs(r['corrected_rel']),C=C))
 assert len(faces)==12 and len(interior)==261 and len(coeff)==180 and fft_checks==6
 dump(S/'DUAL90.json',dict(status='PASS_SCOPED',laws=faces,transfer=transfers,interior_cases=len(interior),saddle_cases=len(coeff),independent_unaliased_FFT=fft_checks,coeff=coeff,errors=dict(ERR),scope='Empirical finite support boundary exact KL and normal limit, not full native support. Prior R80 FFTs reauthenticated; all coefficients independently checked, six unaliased reexecutions. No physical hbar.'))
 print('DUAL',len(faces),sum(len(r['edges']) for r in faces),sum(len(r['vertices']) for r in faces),len(interior),dict(ERR),transfers,flush=True)
if __name__=='__main__':main()
