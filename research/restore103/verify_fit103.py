from fit103 import *
def main():
 m=read(S/'MEASURE103.json');f=read(S/'FIT103.json');q=read(S/'INDEPENDENT103.json');g=m['groups'];err={}
 # Independent reconstruct root risk and grid selections and compare all external values.
 for row in f['conditional']:
  arm,bi=row['arm'],row['bi'];train=[x for x in g if (x['arm'],x['bi'],x['cohort'],x['budget'],x['h'])==(arm,bi,0,16,16) and x['L']<21]
  r=np.mean([np.mean(x['conditional_root'],0) for x in train],0);ck(err,'grid_train_risk',r,row['train_risk'])
  ix=np.argmin(r[:,:,0],axis=1);ck(err,'grid_train_choice',GRID[ix],row['train_choice'])
  u=next(x for x in g if tuple(x[k] for k in ('L','arm','bi','cohort','budget','h'))==tuple(row[k] for k in ('L','arm','bi','cohort','budget','h')))
  v=np.asarray(u['conditional_root']);k=int(np.flatnonzero(GRID==row['known_beta'])[0]);z=np.stack([v[:,0,ix[0]],v[:,1,ix[1]],v[:,0,k]],1)
  ck(err,'heldout_grid_score',z.mean(0),row['scores_ci'][0]);ck(err,'heldout_grid_gain',(z[:,1]-z[:,0]).mean(0),row['Q_minus_score_ci'][0])
 def y0(x,j):
  a=np.asarray(x['root_values'])[:,0,j,0];b=np.asarray(x['root_valid'])[:,0,j];return a[b].mean()
 for v in f['scaling']:
  arm,bi,j=v['arm'],v['bi'],v['target'];train=[x for x in g if (x['arm'],x['bi'],x['cohort'])==(arm,bi,0) and x['L']<21 and x['h']==x['budget']]
  a=design(train,v['model']);y=np.array([y0(x,j) for x in train]);R=np.diag([0]+[1]*(a.shape[1]-1))*1e-8
  c=np.linalg.solve(a.T@a+R,a.T@y);ck(err,'scaling_independent_coef',c,v['coef'],1e-7)
  zz=[]
  for L in LS[:-1]:
   ii=np.array([x['L']!=L for x in train]);u=np.linalg.solve(a[ii].T@a[ii]+R,a[ii].T@y[ii]);zz.extend((np.clip(a[~ii]@u,-BO,BO)-y[~ii])**2)
  ck(err,'scaling_independent_CV',np.sqrt(np.mean(zz)),v['CV_RMSE'],1e-7)
  for t in v['test']:
   x=next(z for z in g if z['L']==21 and (z['arm'],z['bi'],z['cohort'],z['budget'],z['h'])==(arm,bi,t['cohort'],t['budget'],t['h']))
   pred=float(np.clip(design([x],v['model'])@c,-BO,BO)[0]);ck(err,'scaling_independent_prediction',pred,t['prediction'],1e-7);ck(err,'scaling_independent_target',y0(x,j),t['truth'])
 q['checks'].update(err);q['independence']+='; independent grid risks, test scores and normal-equation scaling predictions';dump(S/'INDEPENDENT103.json',q)
 p=read(S/'PLOT_MANIFEST103.json');p['sources']['INDEPENDENT103.json']=sha(S/'INDEPENDENT103.json');dump(S/'PLOT_MANIFEST103.json',p)
 print('FIT INDEPENDENT',err,flush=True)
if __name__=='__main__':main()
