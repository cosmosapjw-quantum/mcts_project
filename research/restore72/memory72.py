"""Block donor control for long-time mark memory on R71 dataset; no new holdout claim."""
from base61 import *
from scipy.special import logsumexp
from math import comb
O=B/'mqft_restore71_20261126'; R=read(O/'PHYSICS_RESULTS.json'); A={k:v for k,v in np.load(O/'JOINT71.npz').items()}
reg=read(O/'REGISTRATION.json');assert sha(O/'JOINT71.npz')==read(O/'AUDIT71.json')['array_sha256']
J=np.array(R['sources']);D=np.array(R['directions']);BLOCKS=[1,2,4,8,16,32]
def features(c):
 m=np.array(c['moves']);xy=np.column_stack([m//9,m%9]);src=np.array(c['sources']);sxy=np.column_stack([src//9,src%9])
 dist=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
 z=np.eye(len(m),dtype=bool);same=np.arange(len(m))[:,None]%2==np.arange(len(m))[None,:]%2
 return [np.mean(np.linalg.norm(xy-4,axis=1)),dist[same&~z].mean(),dist[~same].mean(),*np.min(np.linalg.norm(sxy[:,None]-xy[None,:],axis=-1),axis=1),*np.mean(np.linalg.norm(sxy[:,None]-xy[None,:],axis=-1),axis=1)]
G=np.array([features(c) for c in reg['cores']])
def run(g):
 m=A['meta'];ix=m[:,0]==g;m=m[ix];Y=A['counts'][ix];Z=A['marks'][ix].sum(-1);V=A['initial_value'][ix]
 X=np.concatenate([Y[:,:,:2],Z[...,None]],-1)/128;train=m[:,1]%16>=4;test=~train
 cores=np.unique(m[test,1]);fs=[];old=R['families'][g]
 for c in cores:
  ids=np.flatnonzero(m[:,1]==c);ii=np.flatnonzero(train&(m[:,3]==m[ids[0],3]));trcores=np.unique(m[ii,1]);assert len(trcores)==12
  # root-level descriptor normalization exclusively on train; distance uses known initial values and board
  trV=np.array([V[m[:,1]==k][0] for k in trcores]);v0=V[ids[0]]
  geo=(G[trcores]-G[c])/np.maximum(G[trcores].std(0),.05)
  vd=(trV-v0)/max(trV.std(),.1)
  dist=[np.zeros(len(trcores)),vd**2,vd**2+np.mean(geo**2,axis=1)]
  row=dict(core=int(c),true_mgf=np.exp(X[ids].sum(1)@J.T).mean(0).tolist(),true_raw=np.stack([np.mean((X[ids].sum(1)@D.T)**k,0) for k in range(5)],-1).tolist(),true_prefix=X[ids].cumsum(1).mean(0).tolist(),pred=[])
  for variant,d in enumerate(dist):
   w0=np.exp(-.5*d);w0/=w0.sum();w0=.95*w0+.05/len(w0)
   w=np.array([w0[np.searchsorted(trcores,k)] for k in m[ii,1]]);w/=w.sum()
   for H in BLOCKS:
    n=32//H;T=X[ii].reshape(len(ii),n,H,3).sum(2)
    W=np.log(np.einsum('i,inj->nj',w,np.exp(T@J.T))).sum(0)
    raw=np.zeros((len(D),5));raw[:,0]=1
    for t in range(n):
     q=T[:,t]@D.T; m0=np.stack([w@(q**k) for k in range(5)],-1)
     raw=np.stack([sum(comb(k,l)*raw[:,l]*m0[:,k-l] for l in range(k+1)) for k in range(5)],-1)
    prefix=np.einsum('i,itj->tj',w,X[ii]).cumsum(0)
    assert np.max(abs(raw[:3,1]-prefix[-1]))<1e-10
    row['pred'].append(dict(variant=variant,H=H,W=W.tolist(),raw=raw.tolist(),prefix=prefix.tolist(),donor_root_ess=float(1/(w0@w0))))
  fs.append(row)
 em=np.array([f['true_mgf'] for f in fs]);base=np.log(em.mean(0));rr=np.array([f['true_raw'] for f in fs]);pref=np.array([f['true_prefix'] for f in fs])
 def cumul(r):
  u=r[:,1];v=r[:,2]-u*u
  return np.stack([u,v,r[:,3]-3*u*r[:,2]+2*u**3,r[:,4]-4*u*r[:,3]+6*u*u*r[:,2]-3*u**4-3*v*v],-1)
 out=[]
 for k,q in enumerate(fs[0]['pred']):
  v=[f['pred'][k] for f in fs];wm=np.array([x['W'] for x in v]);W=logsumexp(wm,axis=0)-np.log(len(v));raw=np.array([x['raw'] for x in v]);pp=np.array([x['prefix'] for x in v])
  out.append(dict(variant=q['variant'],H=q['H'],W=W.tolist(),W_delta=(W-base).tolist(),W_rmse=float(np.sqrt(np.mean((W-base)**2))),mark_W_rmse=float(np.sqrt(np.mean((W-base)[np.abs(J[:,2])>0]**2))),count_W_rmse=float(np.sqrt(np.mean((W-base)[J[:,2]==0]**2))),root_prefix_rmse=float(np.sqrt(np.mean((pp-pref)**2))),cumulant_delta=(cumul(raw.mean(0))-cumul(rr.mean(0))).tolist(),mean_donor_ess=float(np.mean([x['donor_root_ess'] for x in v]))))
 return dict(family=g,label=old['label'],baseline_R71_lag_RMSE=old['results'][2]['W_rmse'],results=out,roots=fs)
if __name__=='__main__':
 with cf.ProcessPoolExecutor(max_workers=5) as pool:r=list(pool.map(run,range(5)))
 for f in r:print(f['label'],[(x['variant'],x['H'],round(x['W_rmse'],4)) for x in f['results']],flush=True)
 dump(S/'MEMORY72.json',dict(scope='R71 existing train/test split reused for structural diagnostic, not independent new validation; fixed donor models; H=32 whole-path transfer; no support smoothing',data_sha=sha(O/'JOINT71.npz'),results=r))
