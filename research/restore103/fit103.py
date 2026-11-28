from measure103 import *
from itertools import product
MODELS=['constant','log_budget','log_size','joint']
def design(g,kind):
 x=np.array([[1.,np.log(v['h']/16),np.log(v['L']/17)] for v in g])
 return x[:,{'constant':[0],'log_budget':[0,1],'log_size':[0,2],'joint':[0,1,2]}[kind]]
def solve(A,y):
 R=np.diag([0]+[1]*(A.shape[1]-1))*1e-4;return np.linalg.lstsq(np.r_[A,R],np.r_[y,np.zeros(len(R))],rcond=None)[0]
def main():
 m=read(S/'MEASURE103.json');G=m['groups'];err={};fits=[];conditional=[]
 def target(g,j):
  r=np.asarray(g['root_values'])[:,0,j,0];v=np.asarray(g['root_valid'])[:,0,j];return float(r[v].mean()) if v.any() else None
 for arm,bi in product(range(2),range(4)):
  train=[g for g in G if (g['arm'],g['bi'],g['cohort'],g['budget'],g['h'])==(arm,bi,0,16,16) and g['L']<21];assert len(train)==4
  risks=np.array([g['conditional_ci'][0] for g in train]).mean(0)
  choice=risks[:,:,0].argmin(1);test=[g for g in G if g['L']==21 and g['arm']==arm and g['bi']==bi and (g['cohort'] in (1,2) or (g['cohort']==0 and g['budget']==16))]
  for g in test:
   betaidx=int(np.argmin(abs(GRID-g['beta'])));assert GRID[betaidx]==g['beta']
   root=np.array(g['conditional_root']);values=np.stack([root[:,0,choice[0]],root[:,1,choice[1]],root[:,0,betaidx]],1)
   conditional.append(dict(L=21,arm=arm,bi=bi,cohort=g['cohort'],budget=g['budget'],h=g['h'],train_choice=GRID[choice].tolist(),train_risk=risks.tolist(),scores_ci=ci(values).tolist(),Q_minus_score_ci=ci(values[:,1]-values[:,0]).tolist(),known_beta=g['beta']))
 for arm,bi,j in product(range(2),range(4),range(2)):
  train=[g for g in G if (g['arm'],g['bi'],g['cohort'])==(arm,bi,0) and g['L']<21 and g['h']==g['budget'] and target(g,j) is not None]
  assert len(train)==8
  L=np.array([g['L'] for g in train]);y=np.array([target(g,j) for g in train]);test=[g for g in G if g['L']==21 and g['arm']==arm and g['bi']==bi and target(g,j) is not None]
  for kind in MODELS:
   A=design(train,kind);pred=np.empty(len(y))
   for l in LS[:-1]:
    ix=L!=l;c=solve(A[ix],y[ix]);pred[~ix]=np.clip(A[~ix]@c,-BO,BO)
   c=solve(A,y);r=np.diag([0]+[1]*(len(c)-1))*1e-8;ck(err,'ridge_normal_equation',A.T@(A@c-y)+r@c,np.zeros(len(c)),1e-8)
   p=np.clip(design(test,kind)@c,-BO,BO);z=np.array([target(g,j) for g in test])
   fits.append(dict(arm=arm,bi=bi,target=j,model=kind,coef=c.tolist(),CV_RMSE=float(np.sqrt(np.mean((pred-y)**2))),test=[dict(cohort=g['cohort'],budget=g['budget'],h=g['h'],truth=float(v),prediction=float(w),error=float(w-v),boundary=int(np.sum(np.asarray(g['statuses'])[0,j,2:])),flat=int(np.asarray(g['statuses'])[0,j,0])) for g,v,w in zip(test,z,p)]))
 assert len(fits)==64 and len(conditional)==50
 # Store valid-root conditional summary instead of zero-sentinel beta CI.
 for g in G:
  r=np.asarray(g['root_values']);valid=np.asarray(g['root_valid']);z=[]
  for a in range(3):
   t=[]
   for b in range(2):
    v=r[valid[:,a,b],a,b,0];t.append(ci(v).tolist() if len(v)>1 else ([float(v[0])]*3 if len(v)==1 else None))
   z.append(t)
  g['beta_ci_valid']=z
 # retain raw arrays; do not mutate original main output after publication. Separate group projection summaries.
 dump(S/'FIT103.json',dict(status='PASS_SCOPED',conditional=conditional,scaling=fits,models=MODELS,groups=[dict(L=g['L'],cohort=g['cohort'],arm=g['arm'],bi=g['bi'],budget=g['budget'],h=g['h'],beta_ci_valid=g['beta_ci_valid'],statuses=g['statuses']) for g in G],checks=err,scope='Grid candidates trained B16 development sizes; conditional scores on recorded histories, not policy counterfactual. Signed bounded occupancy slopes, no physical temperature scaling.'))
 print('FIT',len(conditional),len(fits),err,flush=True)
if __name__=='__main__':main()
