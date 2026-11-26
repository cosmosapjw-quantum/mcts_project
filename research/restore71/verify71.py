"""Independent forward tilted-matrix verification and direct holdout recomputation."""
from base61 import *
import math
A=np.load(S/'JOINT71.npz'); R=read(S/'PHYSICS_RESULTS.json'); audit=read(S/'AUDIT71.json')
assert sha(S/'JOINT71.npz')==R['array_sha256']==audit['array_sha256']
js=np.array(R['sources']); dirs=np.array(R['directions']);meta=A['meta']
maxdirect=maxforward=maxmom=maxcum=0.; checks=0
def digit(x,edges):return np.sum(np.asarray(x)[...,None]>=edges,axis=-1)
def cum(r):
 out=[]
 for z in r:
  u=z[1];v=z[2]-u*u
  out.append([u,v,z[3]-3*u*z[2]+2*u**3,z[4]-4*u*z[3]+6*u*u*z[2]-3*u**4-3*v*v])
 return np.array(out)
for g,f in enumerate(R['families']):
 ix=meta[:,0]==g;m=meta[ix];P=A['prob'][ix];Y=A['counts'][ix];Z=A['marks'][ix].sum(-1)
 X=np.concatenate([Y[:,:,:2],Z[...,None]],-1)/128
 vals=A['initial_value'][ix];V=digit(vals,np.array([-.5,0,.5])); mass=P[:,:,:2].sum(-1)
 b=digit(mass,np.array([.015,.04,.1,.25]))*4+digit(P[:,:,0]/mass,np.array([.25,.5,.75]))
 lag=np.c_[np.zeros(len(m)),Z[:,:-1]/4];st=b*4+digit(lag,np.array([-.5,0,.5]))
 train=m[:,1]%16>=4;test=~train
 assert len(np.unique(m[train,1]))==48 and len(np.unique(m[test,1]))==16
 assert set(m[train,1]).isdisjoint(set(m[test,1]))
 e=np.exp(X[test].sum(1)@js.T).mean(0); rr=X[test].sum(1)@dirs.T
 raw=np.stack([np.mean(rr**k,axis=0) for k in range(5)],axis=-1)
 maxdirect=max(maxdirect,float(np.max(abs(np.log(e)-f['W_direct']))))
 maxcum=max(maxcum,float(np.max(abs(cum(raw)-f['cumulants_direct']))))
 for model,h in enumerate(f['results']):
  n=[1,20,80][model];ss=np.zeros_like(b) if model==0 else b if model==1 else st
  # All sources, one fixed heldout core per model/family. Forward tilted row vectors, not backward recursion.
  q=h['pred'][0];core=q['core'];ids=np.flatnonzero(m[:,1]==core);i=ids[0];it=np.flatnonzero(train&(m[:,3]==m[i,3]))
  vb=V[i];init=ss[i,0];count=len(it)
  F=np.zeros((len(js),n));F[:,init]=1
  moments=np.zeros((5,len(dirs),n));moments[0,:,init]=1
  for t in range(32):
   rows=[]; root=( (V[it]==vb).astype(float)+16/count );root/=root.sum()
   for k in range(n):
    if model==0:w=root
    else:
     base_state=k if model==1 else k//4; mask=(V[it]==vb)&(b[it,t]==base_state)
     w=mask.astype(float)+8*root;w/=w.sum()
     if model==2:
      w=((V[it]==vb)&(ss[it,t]==k)).astype(float)+8*w;w/=w.sum()
    rows.append(w)
   W=np.array(rows);dest=ss[it,t+1] if t<31 else np.zeros(count,int)
   E=X[it,t];D=np.eye(n)[dest];T=np.einsum('si,ij,ik->jsk',W,np.exp(E@js.T),D,optimize=True)
   F=np.einsum('js,jsk->jk',F,T)
   z=E@dirs.T; U=np.zeros_like(moments)
   for k in range(5):
    for l in range(k+1):
     M=np.einsum('si,ij,ik->jsk',W,z**(k-l),D,optimize=True)
     U[k]+=math.comb(k,l)*np.einsum('js,jsk->jk',moments[l],M)
   moments=U
  maxforward=max(maxforward,float(np.max(abs(F.sum(1)-q['mgf']))))
  maxmom=max(maxmom,float(np.max(abs(moments.sum(-1).T-q['raw']))))
  checks+=1
assert maxdirect<1e-11 and maxcum<1e-11 and maxforward<1e-8 and maxmom<1e-10
out=dict(status='PASS_SCOPED',direct_W_max_abs=maxdirect,direct_cumulants_max_abs=maxcum,forward_matrix_MGF_max_abs=maxforward,forward_raw_moments_max_abs=maxmom,forward_cases=checks,source_cases=checks*len(js),simulation_paths=checks*8192,scope='Independent forward algorithm on one heldout root per model/family; all aggregate direct laws; not true population exactness.')
dump(S/'INDEPENDENT71.json',out);print(out,flush=True)
