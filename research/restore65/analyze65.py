"""Independent heldout high-budget proxy utility and persistent measurement analysis."""
from audit65 import *
from scipy.optimize import least_squares
def main():
 a=read(S/'AUDIT65.json');reg=read(S/'REGISTRATION.json');assert sha(S/'PHASE65.npz')==a['array_sha'] and sha(S/'REFERENCE65.npz')==a['reference_sha']
 with np.load(S/'PHASE65.npz') as f:z={k:f[k] for k in f.files}
 with np.load(S/'REFERENCE65.npz') as f:ref={k:f[k] for k in f.files}
 m=z['meta'];arms=list(z['arms']);fields=list(z['fields']);wf=list(z['wave_fields']);cuts=z['cuts'];rng=np.random.default_rng(650201);boot=rng.integers(0,48,(10000,48))
 def stat(v):
  v=np.asarray(v,float);assert len(v)==48
  return dict(mean=v.mean(0).tolist(),ci95=np.quantile(v[boot].mean(1),[.025,.975],axis=0).tolist())
 def conditional(v,mask):
  # conditional on eligible recommendation(s), geometry bootstrap reweights sums
  v=np.asarray(v);mask=np.asarray(mask);x=(np.where(mask,v,0)).sum(1);n=mask.sum(1)
  assert n.sum()>0
  den=n[boot].sum(1);keep=den>0;u=x[boot].sum(1)[keep]/den[keep]
  return dict(mean=float(x.sum()/n.sum()),ci95=np.quantile(u,[.025,.975]).tolist(),n=int(n.sum()),roots=int((n>0).sum()),coverage=float(mask.mean()))
 assert np.array_equal(ref['meta'][:,0],np.repeat(np.arange(48),4))
 assert np.array_equal(ref['meta'][:,1],np.tile(np.arange(4),48))
 RN=ref['counts'].reshape(48,4,N);RW=ref['marks'].reshape(48,4,N);assert np.all(RN.sum(2)==1024)
 n=RN.sum(1);w=RW.sum(1);Q=np.divide(w,n,out=np.zeros_like(w),where=n>0)
 policies=n/4096
 # split consistency: independent two-seed halves
 halfN=np.stack([RN[:,:2].sum(1),RN[:,2:].sum(1)],1);halfW=np.stack([RW[:,:2].sum(1),RW[:,2:].sum(1)],1);halfQ=np.divide(halfW,halfN,out=np.zeros_like(halfW),where=halfN>0)
 top= np.argsort(-n,axis=1,kind='stable')[:,:5]
 overlap=np.array([len(set(np.argsort(-v[0],kind='stable')[:5])&set(np.argsort(-v[1],kind='stable')[:5]))/5 for v in halfN])
 halfmask=np.all(halfN>=8,axis=1);qdiff=np.sum(np.where(halfmask,abs(halfQ[:,0]-halfQ[:,1]),0),1)/np.maximum(1,halfmask.sum(1))
 refstats=dict(n_roots=48,seeds_per_root=4,backups_per_seed=1024,top1_split_match=stat((halfN[:,0].argmax(1)==halfN[:,1].argmax(1)).astype(float)),top5_split_overlap=stat(overlap),split_policy_TV=stat(abs(halfN[:,0]/2048-halfN[:,1]/2048).sum(1)/2),split_Q_MAE_on_eligible=stat(qdiff),eligible_actions={str(t):stat((n>=t).sum(1)) for t in (8,16,32)})
 obs=[];contrasts=[];utility=[];utilcontrast=[];local=[];by={}
 for g in (0,3):
  for arm in arms:
   ix=np.flatnonzero((m[:,0]==g)&(m[:,3]==arms.index(arm)));ix=ix[np.lexsort((m[ix,2],m[ix,1]))]
   assert len(ix)==384 and np.array_equal(m[ix,1],np.repeat(np.arange(48),8)) and np.array_equal(m[ix,2],np.tile(np.arange(8),48))
   A=z['rows'][ix].reshape(48,8,6,-1);W=z['waves'][ix].reshape(48,8,32,-1);N0=z['counts'][ix].reshape(48,8,N)
   ac=A[:,-1] # not used; recommendations at each cut in A
   by[g,arm]=(A,W,N0)
   for k0,k in enumerate(cuts):
    q=A[:,:,k0];res={f:stat(q[:,:,fields.index(f)].mean(1)) for f in ('Na','Nb','Npair','Hvisit','var','mean_depth','terminal_fraction')}
    res['mean_mark']=stat(-q[:,:,fields.index('E_Q')].mean(1));obs.append(dict(generation=g,arm=arm,k=int(k),metrics=res))
   if arm.startswith('persist') or arm=='reset_plus':
    d={f:stat(W[:,:,:,wf.index(f)].mean((1,2))) for f in ('abs_memory_delta','measurement_TV','probability_change_within_wave','entropy','purity','pair_mass')}
    v=W[:,:,:,wf.index('probability_change_within_wave')];d['non_iid_wave_fraction']=stat((v>1e-12).mean((1,2)))
    # Conditional density is actual state; averaged post/pre X/Y not an individual path.
    local.append(dict(generation=g,arm=arm,metrics=d,final_X=stat(W[:,:,-1,wf.index('post_X')].mean(1)),final_Y=stat(W[:,:,-1,wf.index('post_Y')].mean(1)),final_Z=stat(W[:,:,-1,wf.index('post_Z')].mean(1))))
   rec=A[:,:,-1,fields.index('recommended')].astype(int)
   assert np.array_equal(rec,N0.argmax(2))
   covered=np.take_along_axis(n,rec,axis=1);qval=np.take_along_axis(Q,rec,axis=1)
   hit=(rec[:,:,None]==top[:,None,:]).any(2)
   scores=dict(top5=stat(hit.mean(1)),top1=stat((rec==n.argmax(1)[:,None]).mean(1)),policy_TV=stat(abs(N0/128-policies[:,None,:]).sum(2).mean(1)/2),reference_policy_mass=stat(np.take_along_axis(policies,rec,axis=1).mean(1)),regret={})
   for t in (8,16,32):
    eligible=n>=t;qmax=np.max(np.where(eligible,Q,-np.inf),axis=1);assert np.isfinite(qmax).all()
    mask=covered>=t;regret=qmax[:,None]-qval
    assert np.min(regret[mask])>=-1e-13
    scores['regret'][str(t)]=conditional(regret,mask)
   utility.append(dict(generation=g,arm=arm,scores=scores))
  for left,right in [('persist_plus','persist_minus'),('persist_plus','persist_dephased'),('persist_plus','reset_plus'),('persist_plus','fresh64'),('persist_plus','baseline'),('persist_minus','baseline'),('reset_plus','fresh64')]:
   A,W,N0=by[g,left];B0,V,N1=by[g,right]
   for k0,k in enumerate(cuts):
    q=A[:,:,k0]-B0[:,:,k0];res={f:stat(q[:,:,fields.index(f)].mean(1)) for f in ('Na','Nb','Npair','Hvisit','var')};res['mean_mark']=stat(-q[:,:,fields.index('E_Q')].mean(1));contrasts.append(dict(generation=g,left=left,right=right,k=int(k),metrics=res))
   r0=N0.argmax(2);r1=N1.argmax(2);h0=(r0[:,:,None]==top[:,None,:]).any(2);h1=(r1[:,:,None]==top[:,None,:]).any(2)
   d=dict(top5_diff=stat((h0.astype(float)-h1.astype(float)).mean(1)),reference_policy_TV_diff=stat((abs(N0/128-policies[:,None,:]).sum(2)-abs(N1/128-policies[:,None,:]).sum(2)).mean(1)/2),same_recommendation=stat((r0==r1).mean(1)),regret_diff={})
   for t in (8,16,32):
    mask=(np.take_along_axis(n,r0,axis=1)>=t)&(np.take_along_axis(n,r1,axis=1)>=t)
    # same reference qmax cancels exactly; compare only paired jointly eligible recs
    diff=np.take_along_axis(Q,r1,axis=1)-np.take_along_axis(Q,r0,axis=1)
    d['regret_diff'][str(t)]=conditional(diff,mask)
   utilcontrast.append(dict(generation=g,left=left,right=right,metrics=d))
 out=dict(scope='48 new geometry blocks, eight paired seeds, six registered low-budget arms; independent g3 4x1024 reference per root, proxy not true regret; root block intervals descriptive',reference=refstats,observables=obs,contrasts=contrasts,local=local,utility=utility,utility_contrasts=utilcontrast)
 dump(S/'PHYSICS_RESULTS.json',out)
 for x in utility:print('UTILITY',x['generation'],x['arm'],x['scores']['top5']['mean'],x['scores']['regret']['16'],flush=True)
 for x in utilcontrast:
  if x['left']=='persist_plus':print('UTIL_DIFF',x['generation'],x['right'],x['metrics']['top5_diff'],x['metrics']['regret_diff']['16'],flush=True)
 for x in contrasts:
  if x['k']==128 and x['left']=='persist_plus':print('COUNT_DIFF',x['generation'],x['right'],x['metrics']['Na'],x['metrics']['mean_mark'],flush=True)
 print('REF',refstats);print('LOCAL',local)
if __name__=='__main__':main()
