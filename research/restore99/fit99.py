from scaling99 import *
from itertools import product
MODELS=['smooth','crossover','entropy_shift','critical_width'];TARGETS=['pmax','KL_norm','concentration','Fisher'];BETAS=[0.,10/3,8.,16.]
GRID={'smooth':[()], 'crossover':list(product(range(0,17,2),[1,2,4,8])), 'entropy_shift':[(v,) for v in [.25,.5,.75,1,1.5,2,3,4]], 'critical_width':list(product(range(0,17,2),[1,2,4,8],[.25,.5,1,2]))}
RIDGE=1e-8
def design(g,kind,par):
 L=np.asarray([v['L'] for v in g],float)[:,None];b=np.asarray([v['beta'] for v in g],float)[:,None];n=np.array([np.array(v['root_values'])[:,0] for v in g])
 if kind=='smooth':x=np.broadcast_to(b/16,n.shape)
 elif kind=='entropy_shift':x=expit(par[0]*b-np.log(n-1))
 elif kind=='crossover':x=np.broadcast_to(expit((b-par[0])/par[1]),n.shape)
 else:x=np.broadcast_to(expit((b-par[0])*(L/17)**par[2]/par[1]),n.shape)
 return np.stack([np.ones_like(x),x,x*x,np.broadcast_to(17/L,x.shape)],axis=-1)
def solve(A,y):
 # Shared 4 coefficient model, non-intercept ridge. SVD augmented LS.
 R=np.diag([0,1,1,1])*np.sqrt(RIDGE)
 return np.linalg.lstsq(np.r_[A,R],np.r_[y,np.zeros(4)],rcond=None)[0]
def pred(A,c,name):
 p=A@c;return np.clip(p,0,1 if name!='Fisher' else np.inf)
def main():
 a=read(S/'SUMMARY99.json');ind=read(S/'INDEPENDENT99.json');f=a['fields'];G=a['groups'];fits=[];checks={};comparisons=[]
 for arm,h,name in product([0,1],[16,32,64],TARGETS):
  k=f.index(name);train=[g for g in G if g['cohort']==0 and g['budget']==64 and g['L']<21 and g['arm']==arm and g['h']==h and g['bi']<4];assert len(train)==16
  train.sort(key=lambda g:(g['L'],g['bi']));L=np.array([g['L'] for g in train]);y=np.array([g['ci'][0][k] for g in train]);models={}
  for kind in MODELS:
   rs=[]
   for par in GRID[kind]:
    A=design(train,kind,par).mean(1);p0=np.zeros(16)
    for ell in LS[:4]:
     ix=L!=ell;c=solve(A[ix],y[ix]);p0[~ix]=pred(A[~ix],c,name)
    rs.append(float(np.sqrt(np.mean((p0-y)**2))))
   best=int(np.argmin(rs));par=GRID[kind][best];A=design(train,kind,par).mean(1);coef=solve(A,y)
   # Verify regularized normal equation and independent QR with augmented LS.
   R=np.diag([0,1,1,1])*np.sqrt(RIDGE);AA=np.r_[A,R];yy=np.r_[y,np.zeros(4)];Q,T=np.linalg.qr(AA,mode='reduced');cc=np.linalg.solve(T,Q.T@yy)
   ck(checks,'fit_QR_prediction',A@coef,A@cc,1e-8);ck(checks,'fit_normal_equation',AA.T@(AA@coef-yy),np.zeros(4),1e-8)
   result=dict(arm=arm,h=h,observable=name,model=kind,params=list(par),coef=coef.tolist(),cv_rmse=rs[best],grid_count=len(rs),grid_rmse=rs,grid_params=[list(v) for v in GRID[kind]],training_L=LS[:4],external=[])
   for co in [1,2]:
    for scope in ['L21','all_sizes']+(['log_size'] if co==2 else []):
     test=[g for g in G if g['cohort']==co and g['budget']==64 and g['arm']==arm and g['h']==h and (g['bi']==4 if scope=='log_size' else g['bi']<4) and (g['L']==21 if scope=='L21' else True)]
     test.sort(key=lambda g:(g['L'],g['bi']));assert len(test)==(3 if co==1 else 4)*(1 if scope=='L21' else 5) if scope!='log_size' else len(test)==5
     z=np.array([np.array(g['root_values'])[:,k] for g in test]);X=design(test,kind,par);p=pred(X,coef,name);pgroup=pred(X.mean(1),coef,name);# score group predictions with the same mean-feature rule used for CV
     err=pgroup-z.mean(1);root_err=(p-z)**2
     result['external'].append(dict(cohort=co,scope=scope,L=[g['L'] for g in test],bi=[g['bi'] for g in test],beta=[g['beta'] for g in test],truth=z.mean(1).tolist(),prediction=pgroup.tolist(),root_squared_error=root_err.tolist(),RMSE=float(np.sqrt(np.mean(err**2))),RMSE_root=float(np.sqrt(root_err.mean()))))
   fits.append(result);models[kind]=result
  b=min(MODELS[:3],key=lambda t:models[t]['cv_rmse'])
  for co in [1,2]:
   get=lambda m:next(v for v in models[m]['external'] if v['cohort']==co and v['scope']=='L21')
   cc=get('critical_width');nn=get(b);er=np.array(cc['root_squared_error'])-np.array(nn['root_squared_error']);errs={m:get(m)['RMSE'] for m in MODELS}
   comparisons.append(dict(arm=arm,h=h,observable=name,cohort=co,dev_selected_noncritical=b,critical_win_all=errs['critical_width']<min(errs[m] for m in MODELS[:3]),RMSE=errs,critical_minus_dev_selected_mse_ci=ci(er.mean(0)).tolist(),critical_params=models['critical_width']['params']))
 assert len(fits)==96 and len(comparisons)==48
 # Finite effective size slopes; do not identify critical exponents.
 slopes=[]
 for co,arm,h,name in product([1,2],[0,1],[16,32,64],TARGETS):
  for bi in ([0,1,3] if co==1 else [0,1,2,3]):
   gs=sorted([g for g in G if (g['cohort'],g['arm'],g['h'],g['budget'],g['bi'])==(co,arm,h,64,bi)],key=lambda v:v['L']);assert len(gs)==5
   k=f.index(name);y=np.array([g['ci'][0][k] for g in gs]);# KL and concentration are 0 at uniform, exclude tiny positive noise
   if np.min(y)>1e-12:
    local=np.diff(np.log(y))/np.diff(np.log(LS));full=np.polyfit(np.log(LS),np.log(y),1)[0];slopes.append(dict(cohort=co,arm=arm,h=h,observable=name,bi=bi,local=local.tolist(),full=float(full)))
 # count validity of the one-draw Binder statistic, no artificial zero averaged for undefined X
 with np.load(S/'SCALE99.npz') as z:
  V=np.concatenate([z[f'V{L}'] for L in LS]);M=np.concatenate([z[f'M{L}'] for L in LS]);J=np.concatenate([z[f'J{L}'] for L in LS]);K=np.concatenate([z[f'K{L}'] for L in LS])
  assert len(V)==10400
  valid=J[:,:,5]>0;assert np.all(valid[:,[0,2,3]])
  nonzero=V[:,5]>1e-12;counts=dict(valid_Binder=valid.sum(0).tolist(),undefined_Binder=(~valid).sum(0).tolist(),tail_nonzero=int((V[:,7]>1e-9).sum()),tail_collision_nonzero=int((V[:,9]>1e-9).sum()),max_tail_KL=float(V[:,7].max()),max_spike_TV=float(V[:,10].max()),max_tail_concentration=float(V[:,9].max()),max_coarse_spike_TV=float(K[:,:,2].max()),min_tail_KL=float(V[:,7].min()),mean_tail_fraction=float(np.mean(V[nonzero,7]/V[nonzero,5])))
 # Exclude root-invalid Binder from group aggregate; record availability and use valid roots only.
 for g in G:
  m=np.array(g['moment_root']);valid=m[:,:,5]>0;g['binder_valid_roots']=valid.sum(0).tolist()
  bc=[]
  for j in range(4):
   w=m[valid[:,j],j,4]
   bc.append(ci(w).tolist() if len(w)>1 else ([float(w[0])]*3 if len(w)==1 else None))
  g['binder_ci_valid_roots']=bc
  # Overwrite the provisional Binder aggregate with valid-root summaries; undefined means null.
  for j in range(4):
   for k in range(3):g['moment_ci'][k][j][4]=bc[j][k] if bc[j] is not None else None
 a['counts']=counts;dump(S/'SUMMARY99.json',a)
 dump(S/'FITS99.json',dict(fits=fits,comparisons=comparisons,slopes=slopes,models=MODELS,targets=TARGETS,train='R76 development L13-19 fixed controls only, LOSO; R77/R87 L21 test, previously existing cohorts',scope='Predictions at actual control n, descriptive root-block comparisons, no admission of physical critical exponent'))
 ind['checks'].update(checks);ind['model_fits']=len(fits);ind['model_comparisons']=len(comparisons);ind['fit_grid_total']=sum(len(x['grid_rmse']) for x in fits);dump(S/'INDEPENDENT99.json',ind)
 print('COUNTS',counts,flush=True)
 print('FITS',len(fits),'grid',ind['fit_grid_total'],'critical_wins',sum(v['critical_win_all'] for v in comparisons),'h64',sum(v['critical_win_all'] for v in comparisons if v['h']==64),'dev_choice',dict(collections.Counter(v['dev_selected_noncritical'] for v in comparisons)),'checks',checks,flush=True)
 print('paired intervals critical worse/better',sum(v['critical_minus_dev_selected_mse_ci'][1]>0 for v in comparisons),sum(v['critical_minus_dev_selected_mse_ci'][2]<0 for v in comparisons),flush=True)
 for g in G:
  if g['cohort']==2 and g['budget']==g['h']==64 and g['arm']==1 and g['bi'] in [0,1,3] and g['L'] in [13,21]:
   z=g['ci'][0];print('FRESH DENSE',g['L'],g['bi'],{k:round(z[f.index(k)],7) for k in ['pmax','KL','binary_KL','tail_KL','concentration','spike_concentration','tail_concentration','spike_TV','Fisher']},flush=True)
if __name__=='__main__':main()
