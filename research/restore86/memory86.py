"""Nonstationary discrete conditional-projection memory on finite path ensembles."""
from base61 import *
from scipy.stats import t as student
from itertools import combinations
FAMS=['g0','g3','Rapfi'];ORDERS=[1,2,4,15];HORIZONS=[1,2,4,8];NT=16
def ci(a):
 a=np.asarray(a);m=a.mean(0);h=student.ppf(.975,len(a)-1)*a.std(0,ddof=1)/np.sqrt(len(a));return [m.tolist(),(m-h).tolist(),(m+h).tolist()]
def bins(d):
 c=np.cumsum(d['count'],axis=1)[:,1::2]/(8*np.arange(1,17));w=np.cumsum(d['mark'],axis=1)[:,1::2]/(8*np.arange(1,17))
 return 3*np.digitize(c,[1/16,1/4])+np.digitize(w,[-.2,.2])
def rootids(M):
 return M[:,2]+np.array([0,48,72,104,152])[M[:,0]]
def rootmean(a,c):
 return np.stack([a[c==k].mean(0) for k in np.unique(c)])
def fit(y,root):
 assert y.shape[1]==NT;w=np.array([1/(len(set(root))*sum(root==i)) for i in root]);assert abs(w.sum()-1)<1e-12
 sup=[np.unique(y[:,t]) for t in range(NT)];Z=[(y[:,i,None]==s[None,:]).astype(float) for i,s in enumerate(sup)]
 mass=[w@z for z in Z];E=[z.T*w[None,:]/p[:,None] for z,p in zip(Z,mass)];G={};K={};errs=[]
 for i in range(NT):
  for j in range(i,NT):
   G[i,j]=E[i]@Z[j];assert abs(G[i,j].sum(1)-1).max()<1e-12
   if i==j:assert abs(G[i,j]-np.eye(len(sup[i]))).max()<1e-12
 for i in range(NT-1):
  for j in range(i,NT-1):
   K[i,j]=G[i,j+1]-sum((K[i,k]@G[k+1,j+1] for k in range(i,j)),start=np.zeros_like(G[i,j+1]))
   assert abs(K[i,j].sum(1)-(1 if i==j else 0)).max()<1e-10
   # Independent direct orthogonal residual Q_j ... Q_(i+1) on future onehot.
   f=Z[j+1].copy()
   for k in range(j,i,-1):f-=Z[k]@(E[k]@f)
   e=float(abs(E[i]@f-K[i,j]).max());errs.append(e);assert e<1e-10
 pred={}
 for order in ORDERS:
  for s in range(NT):
   pred[order,s,s]=np.eye(len(sup[s]))
   for i in range(s-1,-1,-1):
    pred[order,i,s]=sum((K[i,k]@pred[order,k+1,s] for k in range(i,min(s,i+order))),start=np.zeros_like(G[i,s]))
    assert abs(pred[order,i,s].sum(1)-1).max()<1e-9
    if order==15:assert abs(pred[order,i,s]-G[i,s]).max()<1e-10
 return sup,mass,G,K,pred,max(errs)
def simplex(v):
 # Euclidean projection of each row onto probability simplex; no intermediate clipping.
 u=np.sort(v,axis=-1)[:,::-1];c=np.cumsum(u,axis=1)-1;k=np.arange(1,len(v[0])+1)
 rho=np.sum(u>c/k,axis=1);theta=c[np.arange(len(v)),rho-1]/rho
 x=np.maximum(v-theta[:,None],0);assert abs(x.sum(1)-1).max()<1e-12
 return x
def evaluate(y,root,sup,mass,G,K,pred):
 r=[]
 for h in HORIZONS:
  score={o:[] for o in ORDERS};diff=[];unsupported=[];ck=[]
  for t in range(NT-h):
   s=t+h;row=np.searchsorted(sup[t],y[:,t]);valid=np.isin(y[:,t],sup[t]);row[~valid]=0
   truth=np.eye(9)[y[:,s]];unsupported.append(~valid)
   for o in ORDERS:
    z=pred[o,t,s][row].copy();z[~valid]=mass[s];p=np.zeros((len(y),9));p[:,sup[s]]=z
    v=np.zeros_like(p);v[:,sup[s]]=simplex(z)
    # Full direct supported state, unsupported start fallback to training future marginal.
    neg=np.maximum(-p,0).sum(1);corr=.5*abs(v-p).sum(1);assert np.isfinite(v).all()
    count=v.reshape(-1,3,3).sum(2);mark=v.reshape(-1,3,3).sum(1)
    score[o].append(np.column_stack([((v-truth)**2).sum(1),((count-np.eye(3)[y[:,s]//3])**2).sum(1),((mark-np.eye(3)[y[:,s]%3])**2).sum(1),neg,corr]))
   p=G[t,s][row].copy();q=pred[1,t,s][row].copy();p[~valid]=mass[s];q[~valid]=mass[s];ck.append(.5*abs(p-q).sum(1))
  metrics={o:np.stack(v).mean(0) for o,v in score.items()};assert h!=1 or all(abs(metrics[o]-metrics[1]).max()<1e-10 for o in ORDERS)
  vals={str(o):ci(rootmean(v,root)) for o,v in metrics.items()}
  gains={str(o):ci(rootmean(metrics[1]-metrics[o],root)) for o in ORDERS[1:]}
  r.append(dict(horizon=h,fields=['joint_Brier','count_Brier','mark_Brier','negative_mass','simplex_TV'],scores=vals,gains_vs_markov=gains,unsupported_start=float(np.mean(unsupported)),CK_TV=float(np.mean(ck))))
 return r
def collisions(d):
 M=d['meta'];roots=rootids(M);C=np.cumsum(d['count'],1);W=np.cumsum(d['mark'],1);P=d['prob'];out=[]
 # Quiescent next wave at times 4,...,124, same root and evaluator, no cross-stage duplicated seeds.
 for g in range(3):
  for cohort in (0,1,2):
   ix=np.flatnonzero((M[:,1]==g)&(np.where(M[:,0]<3,0,M[:,0]-2)==cohort));rows={}
   for name in ('coarse_count_mark','exact_root_count','exact_root_NW'):
    num=bad=tree=0;maxv=0.;best=None;first_tree=None
    for c in np.unique(roots[ix]):
     z=ix[roots[ix]==c]
     for t in range(1,32):
      if name=='coarse_count_mark':
       k=(np.digitize(C[z,t-1]/(4*t),[1/16,1/4])*3+np.digitize(W[z,t-1]/(4*t),[-.2,.2]))
      else:k=d['counts_hash' if name=='exact_root_count' else 'root_nw_hash'][z,t]
      for u in np.unique(k):
       a=z[k==u]
       for i,j in combinations(a,2):
        num+=1;v=float(abs(P[i,t]-P[j,t]));td=d['prefix_multiset_hash'][i,t]!=d['prefix_multiset_hash'][j,t]
        if td:tree+=1
        if v>1e-10:bad+=1
        if v>maxv:
         maxv=v;best=dict(i=int(i),j=int(j),clock=4*t,prob=[float(P[i,t]),float(P[j,t])],count=[int(C[i,t-1]),int(C[j,t-1])],mark=[float(W[i,t-1]),float(W[j,t-1])],same_root_NW=bool(d['root_nw_hash'][i,t]==d['root_nw_hash'][j,t]),same_prefix_multiset=bool(not td))
        if td and first_tree is None:first_tree=dict(i=int(i),j=int(j),clock=4*t,prob=[float(P[i,t]),float(P[j,t])])
    if name=='exact_root_NW':assert bad==0
    rows[name]=dict(pairs=num,unequal_next_sector_probability=bad,different_prefix_multisets=tree,max_probability_TV=maxv,witness=best,first_tree_difference=first_tree)
   out.append(dict(family=FAMS[g],cohort=cohort,models=rows));print('COLLISIONS',FAMS[g],cohort,[(k,v['pairs'],v['unequal_next_sector_probability']) for k,v in rows.items()],flush=True)
 return out
def main():
 a=read(S/'AUDIT86.json');assert sha(S/'WAVES86.npz')==a['array_sha']
 with np.load(S/'WAVES86.npz') as f:d={k:f[k] for k in f.files}
 y=bins(d);M=d['meta'];r=rootids(M);out=[];models=[]
 for g in range(3):
  train=(M[:,1]==g)&(M[:,0]<3);assert train.sum()==1344 and len(set(r[train]))==104
  sup,mass,G,K,pred,e=fit(y[train],r[train]);rr=[]
  for j,group in enumerate((3,4)):
   ix=(M[:,1]==g)&(M[:,0]==group);assert len(set(r[ix]))==(48 if group==3 else 24)
   v=evaluate(y[ix],r[ix],sup,mass,G,K,pred);rr.append(dict(group=group,roots=len(set(r[ix])),paths=int(ix.sum()),horizons=v))
  mem=[]
  for lag in range(1,16):
   # weighted average L1 row norm of signed coefficient (not TV of probabilities).
   z=[float(mass[t]@abs(K[t,t+lag-1]).sum(1)) for t in range(16-lag)]
   mem.append(dict(delay=lag,mean_weighted_L1=float(np.mean(z)),max_weighted_L1=max(z)))
  out.append(dict(family=FAMS[g],projection_error=e,support_sizes=list(map(len,sup)),memory_profile=mem,external=rr))
  models.append(dict(family=FAMS[g],support=[x.tolist() for x in sup],mass=[x.tolist() for x in mass],G={f'{i},{j}':v.tolist() for (i,j),v in G.items()},K={f'{i},{j}':v.tolist() for (i,j),v in K.items()}))
  print('MEMORY',FAMS[g],e,[(x['group'],[(v['horizon'],v['gains_vs_markov']['4'][0][0],v['scores']['4'][0][4]) for v in x['horizons']]) for x in rr],flush=True)
 dump(S/'MODEL86.json',dict(status='DEVELOPMENT_ONLY',wave_array_sha=a['array_sha'],models=models,scope='Only R82 development trajectories; exact finite empirical conditional projection. Unsupported training states omitted.'))
 # Model computed from training only; no test labels used to choose order or bins.
 col=collisions(d)
 dump(S/'MEMORY86.json',dict(results=out,collisions=col,model_sha=sha(S/'MODEL86.json'),array_sha=a['array_sha'],scope='Time-dependent projection on fixed ensemble, finite conditional two-time law, not a native autonomous joint history generator. Truncated signed predictions projected only at final endpoint.'))
if __name__=='__main__':main()
