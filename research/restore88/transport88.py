"""Rebuild transport on actual score graph, independent dense boundary and detector check."""
from base61 import *
import transport75 as T
from scipy.linalg import expm
O=B/'mqft_restore75_20261127'
def dense_model(g,phi,leads,path=False):
 v=g['path'] if path else g['component'];index={a:i for i,a in enumerate(v)}
 pot=dict(zip(g['component'],g['potential']));H=np.diag([pot[i] for i in v]).astype(complex);pl=tuple(g['plaquette'][:2]);ed=list(zip(v[:-1],v[1:])) if path else g['edges']
 for a,b in ed:
  i,j=index[a],index[b];z=np.exp(1j*phi) if not path and (a,b)==pl else np.exp(-1j*phi) if not path and (b,a)==pl else 1.
  H[i,i]+=.35;H[j,j]+=.35;H[i,j]-=.35*z;H[j,i]-=.35*np.conjugate(z)
 s,t=index[g['source']],index[g['target']]
 if leads:H[s,s]+=.35;H[t,t]+=.35
 return H,s,t
def dense_scatter(H,s,t,E):
 J=.35;k=np.arccos(1-E/(2*J));gamma=2*J*np.sin(k);sig=-J*np.exp(1j*k)
 M=(E+1e-11j)*np.eye(len(H))-H;M[s,s]-=sig;M[t,t]-=sig
 b=np.zeros(len(H),complex);b[s]=1;q=np.linalg.solve(M,b)
 return float(abs(gamma*q[t])**2),float(abs(1-1j*gamma*q[s])**2)
def dense_detector(H,s,t):
 U=expm(-1j*H);P=abs(U)**2
 a=np.eye(len(H),dtype=complex)[:,s];b=np.eye(len(H))[:,s]
 f=[];d=[]
 for k in range(64):
  a=U@a;b=P@b;f.append(float(abs(a[t])**2));d.append(float(b[t]));a[t]=0;b[t]=0
 assert abs(sum(f)+np.vdot(a,a).real-1)<1e-10 and abs(sum(d)+b.sum()-1)<1e-10
 return np.array(f),np.array(d)
def one(g):
 v=T.run(g);err=0.
 if g['core']<4:
  for i,E in enumerate(T.ENERGIES):
   for k,ph,pa in [('full',0.,False),('phase_T',np.pi/2,False),('path',0.,True)]:
    H,s,t=dense_model(g,ph,True,pa);a,b=dense_scatter(H,s,t,E);r=v['scattering'][i]
    old=r[k] if k=='phase_T' else r[k]['T'];err=max(err,abs(a-old))
    if k!='phase_T':err=max(err,abs(b-r[k]['R']))
  for k,ph in [('detector',0.),('phase_detector',np.pi/2)]:
   H,s,t=dense_model(g,ph,False);a,b=dense_detector(H,s,t)
   err=max(err,float(abs(a-v[k]['coherent_f']).max()),float(abs(b-v[k]['dephased_f']).max()))
  assert err<1e-7
 return v,err
def main():
 b=read(S/'BARRIER88.json');old=read(O/'TRANSPORT75.json');assert old['source_sha']==sha(O/'FIRST_RESULTS75.json')
 with cf.ProcessPoolExecutor(max_workers=6) as pool:out=list(pool.map(one,b['contexts'],chunksize=1))
 assert len(out)==96
 err=0.
 def compare(a,b):
  nonlocal err
  if isinstance(a,dict):
   assert a.keys()==b.keys()
   for k in a:compare(a[k],b[k])
  elif isinstance(a,list):
   assert len(a)==len(b)
   for x,y in zip(a,b):compare(x,y)
  elif isinstance(a,(float,int)) and not isinstance(a,bool):err=max(err,abs(a-b));assert abs(a-b)<1e-7
  else:assert a==b
 lookup={(x['family'],x['core']):x for x in old['contexts']}
 for x,e in out:compare(x,lookup[x['family'],x['core']])
 r,c=T.controls();compare(r,old['rectangular_controls']);compare(c,old['continuum_controls'])
 dump(S/'TRANSPORT88.json',dict(contexts=[x for x,e in out],rectangular_controls=r,continuum_controls=c,prior_error=err,independent_dense_error=max(e for x,e in out),dense_contexts=12,scope='All96 declared graph and path coherent transport references rebuilt, dense linear solve/expm for 12; same actual score landscapes, not native quantum dynamics.'))
 print('TRANSPORT',len(out),err,max(e for x,e in out),flush=True)
if __name__=='__main__':main()
