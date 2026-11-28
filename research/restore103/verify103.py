from measure103 import *
def main():
 m=read(S/'MEASURE103.json');r=read(S/'PRIOR103.json');f=read(S/'FIT103.json');err={};src=m['source_hashes']|r['source_hashes'];raw=m['raw_receipts']|r['raw_receipts'];assert len(raw)==5280 and not set(m['raw_receipts'])&set(r['raw_receipts'])
 assert all(sha(B/k)==h for k,h in src.items()) and sha(S/'MEASURE103.npz')==m['array_sha']
 total=0;mx=0.;negative=np.zeros((3,2),int);singular=np.zeros((3,2),int)
 # independent batched bisection on all fixed features, explicit legal masks
 with np.load(S/'MEASURE103.npz') as z:
  for L in LS:
   M=z[f'M{L}'];V=z[f'V{L}'];Z=z[f'Z{L}'];F=z[f'F{L}'];C=z[f'C{L}'];mask=Z[:,0]>0;nn=mask.sum(1);assert len(M)==2080
   for a in range(3):
    s=Z[:,a+2];s=np.where(mask,s-(s*mask).sum(1)[:,None]/nn[:,None],0)
    for b in range(2):
     y=Z[:,b];target=(s*y).sum(1);vv=V[:,a,b];status=vv[:,1];lo=np.full(len(M),-BO);hi=-lo
     def p0(t):
      x=np.where(mask,t[:,None]*s,-np.inf);x-=x.max(1,keepdims=True);p=np.exp(x);return p/p.sum(1,keepdims=True)
     for _ in range(50):
      mid=(lo+hi)/2;g=(p0(mid)*s).sum(1)-target;lo=np.where(g<0,mid,lo);hi=np.where(g>=0,mid,hi)
     beta=(lo+hi)/2;beta=np.where(status==0,0,beta);beta=np.where(np.isin(status,[2,4]),-BO,beta);beta=np.where(np.isin(status,[3,5]),BO,beta)
     q=p0(beta);ref=p0(vv[:,0]);ck(err,'independent_projection_probability',q,ref,1e-8)
     kl=np.sum(xlogy(y,y)-xlogy(y,np.where(mask,q,1)),1);ck(err,'independent_projection_KL',kl,vv[:,3],1e-8)
     grad=(q*s).sum(1)-target;ck(err,'independent_gradient',grad,vv[:,5],1e-8);ck(err,'independent_TV',.5*abs(y-q).sum(1),vv[:,4],1e-8)
     mx=max(mx,float(np.max(abs(beta-vv[:,0]))));total+=len(M)
     negative[a,b]+=int(np.count_nonzero((vv[:,0]<-1e-9)&(status!=0)));singular[a,b]+=int(np.count_nonzero(status==0))
   # independent conditional-grid aggregation on all groups, root equal.
   for g in [g for g in m['groups'] if g['L']==L]:
    ii=np.all(M[:,[0,2,3,5,6]]==[g[k] for k in ['cohort','arm','bi','budget','h']],1)
    root=np.stack([C[ii&(M[:,1]==k)].mean(0) for k in range(8)])
    ck(err,'conditional_root_aggregation',root,g['conditional_root'])
 assert total==62400
 # independent exact rational 2-state example: different histories, average score zero but mixture not uniform for unequal positive gaps.
 s=np.array([[0.,1.],[0.,-2.]])
 p=np.array([prob(x,2.) for x in s]);mean=p.mean(0);q=prob(s.mean(0),2.);gap=np.mean([KL(x,mean) for x in p]);ck(err,'two_state_mixture',np.mean([KL(x,q) for x in p]),gap+KL(mean,q));assert abs(mean[0]-q[0])>.1
 fresh=[]
 for bi in [0,1,3,4]:
  a=[g for g in m['groups'] if (g['cohort'],g['arm'],g['bi'],g['budget'],g['h'])==(2,1,bi,64,64)]
  for g in a:
   z0=np.asarray(g['root_values']);valid=np.asarray(g['root_valid']);b=np.empty((3,2))
   for i in range(3):
    for j in range(2):b[i,j]=z0[valid[:,i,j],i,j,0].mean() if valid[:,i,j].any() else np.nan
   fresh.append(dict(L=g['L'],bi=bi,beta=g['beta'],beta_projection=[[None if not np.isfinite(v) else float(v) for v in row] for row in b],KL=np.asarray(g['ci'])[0,:,:,3].tolist(),gap=float(np.asarray(g['path_ci'])[0,3]),statuses=g['statuses']))
 counts=dict(raw_paths=len(raw),native_R87=m['paths'],native_R79=r['paths'],R87_draws=m['draws'],R87_identified=m['identified_draws'],R87_flat=m['flat_draws'],R79_waves=r['waves'],R79_identified=r['identified'],R79_flat=r['flat'],prefixes=m['prefixes'],projection_cases=total,prior_projection_cases=r['projections'],conditional_heldout=len(f['conditional']),scaling_models=len(f['scaling']),negative_projection=negative.tolist(),flat_projection=singular.tolist(),physical_temperature_identified=0)
 # choose by dev LOSO only and report actual test winner against others, no test selection.
 selection=[]
 for arm in range(2):
  for bi in range(4):
   for j in range(2):
    x=[g for g in f['scaling'] if (g['arm'],g['bi'],g['target'])==(arm,bi,j)];best=min(x,key=lambda v:v['CV_RMSE'])
    errors={v['model']:float(np.sqrt(np.mean([a['error']**2 for a in v['test'] if a['cohort']==2 and a['h']==64]))) for v in x};selection.append(dict(arm=arm,bi=bi,target=j,selected=best['model'],fresh64_RMSE=errors,selected_wins=errors[best['model']]<=min(errors.values())+1e-12))
 dump(S/'SUMMARY103.json',dict(status='PASS_SCOPED',counts=counts,status_counts=m['status_counts'],features=m['features'],targets=m['targets'],fresh=fresh,scaling_selection=selection,prior_heldout=[dict(family=x['family'],heldout=x['heldout']) for x in r['families']],scope='Typed counts, finite score law and occupancy projection, not physical T or universal scaling.'))
 dump(S/'INDEPENDENT103.json',dict(status='PASS_SCOPED',counts=counts,checks={**m['checks'],**{'prior_'+k:float(v) for k,v in r['checks'].items()},**f['checks'],**err},max_bisection_beta_difference=mx,raw_receipts=raw,source_hashes=src,software=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),independence='Batched independent bisection all projections, prior bisection and heldout aggregate, exact two-state mixture, all root aggregation; same agent.'))
 print('CHECKS',err,'beta diff',mx,flush=True);print('COUNTS',counts,flush=True);print('SCALING',[(x['selected'],x['selected_wins']) for x in selection],flush=True)
 for g in fresh:
  if g['L'] in (13,21):print('FRESH',g,flush=True)
 for g in f['conditional']:
  if g['cohort']==2 and g['h']==64:print('CONDITIONAL',g['arm'],g['bi'],g['train_choice'],g['scores_ci'][0],flush=True)
if __name__=='__main__':main()
