from base61 import *
from audit61 import EXTRA
def main():
 a=np.load(S/'CONTROL61.npz');X=a['values'];M=a['meta'];fld=list(a['fields']);reg=read(S/'REGISTRATION.json')
 assert X.shape==(20992,27)
 ix={k:fld.index(k) for k in fld}
 rng=np.random.default_rng(610010);bs=rng.integers(0,32,(10000,32))
 def summ(v,boot=bs):
  v=np.asarray(v,float);assert len(v)==boot.shape[1]
  return dict(mean=float(v.mean()),se=float(v.std(ddof=1)/np.sqrt(len(v))),ci95=np.quantile(v[boot].mean(1),[.025,.975]).tolist())
 responses=[];coredata=[];raw={}
 obs=['C','E_Q','Hvisit','var','KL']
 for g in (0,3):
  for width in (1,4):
   for k in (4,8,16,32,64,128):
    v={}
    for h in (-35,0,35):
     mask=(M[:,1]==g)&(M[:,3]==width)&(M[:,4]==h)&(M[:,8]==0)&(M[:,9]==k)
     m=M[mask];z=X[mask];order=np.lexsort((m[:,7],m[:,2]));m=m[order];z=z[order]
     assert len(z)==256 and np.array_equal(m[:,2],np.repeat(np.arange(32),8)) and np.array_equal(m[:,7],np.tile(np.arange(8),32))
     v[h]=z.reshape(32,8,-1)
    z=v[0];S0=z[:,:,ix['score']];C=z[:,:,ix['C']];F=z[:,:,ix['fisher_pred']]
    wp=z[:,:,ix['lr_plus']];wm=z[:,:,ix['lr_minus']]
    def cov(x,y):return np.sum((x-x.mean(1,keepdims=True))*(y-y.mean(1,keepdims=True)),axis=1)/7
    def ess(w):return w.sum()**2/np.sum(w*w)
    record=dict(generation=g,width=width,k=k,roots=32,reps=8,score_mean=summ(S0.mean(1)),score2=summ((S0*S0).mean(1)),expected_fisher=summ(F.mean(1)),score2_minus_fisher=summ((S0*S0-F).mean(1)),count_variance=summ(C.var(1,ddof=1)),adaptive_drift_variance=summ(z[:,:,ix['mean_p']].var(1,ddof=1)),noise_variance=summ(S0.var(1,ddof=1)),cross_term=summ(2*cov(S0,z[:,:,ix['mean_p']])),source_occupancy=summ(C.mean(1)/k),path_action_per_backup=summ(z[:,:,ix['action']].mean(1)/k),last_count_residual=summ((z[:,:,ix['last_C']]-z[:,:,ix['last_mean']]).mean(1)),last_resid2_minus_var=summ(((z[:,:,ix['last_C']]-z[:,:,ix['last_mean']])**2-z[:,:,ix['last_var']]).mean(1)),last_wave_susceptibility=summ(z[:,:,ix['last_var']].mean(1)),weight_minus_mean=summ(wm.mean(1)),weight_plus_mean=summ(wp.mean(1)),ess_minus=float(ess(wm)),ess_plus=float(ess(wp)),n_weights=int(wp.size),observables={})
    assert abs(record['count_variance']['mean']-record['adaptive_drift_variance']['mean']-record['noise_variance']['mean']-record['cross_term']['mean'])<1e-9
    for key in obs:
     Y=z[:,:,ix[key]];minus=v[-35][:,:,ix[key]];plus=v[35][:,:,ix[key]]
     # Common random numbers; finite contrast vs derivative differ by O(h^2).
     fd=(plus-minus).mean(1)/.7;lr=cov(Y,S0)
     rw=(np.mean(Y*wp,1)-np.mean(Y*wm,1))/.7
     sn=(np.sum(Y*wp,1)/wp.sum(1)-np.sum(Y*wm,1)/wm.sum(1))/.7
     entry=dict(fd=summ(fd),score=summ(lr),reweighted_fd=summ(rw),self_normalized_fd=summ(sn),fd_minus_score=summ(fd-lr),fd_minus_reweighted=summ(fd-rw),h0=summ(Y.mean(1)))
     record['observables'][key]=entry
     coredata.append(dict(generation=g,width=width,k=k,observable=key,fd=fd.tolist(),score=lr.tolist(),rw=rw.tolist(),h0=Y.mean(1).tolist()))
    for key in FIELDS:record.setdefault('native_means',{})[key]=summ(z[:,:,ix[key]].mean(1))
    responses.append(record)
    if k==128:
     raw[(g,width)]=z
 # Contrasts between training generations paired at same independent root.
 contrasts=[]
 for width in (1,4):
  for k in (4,8,16,32,64,128):
   for key in obs:
    u=[next(x for x in coredata if x['generation']==g and x['width']==width and x['k']==k and x['observable']==key) for g in (0,3)]
    contrasts.append(dict(width=width,k=k,observable=key,fd_g3_minus_g0=summ(np.array(u[1]['fd'])-u[0]['fd']),score_g3_minus_g0=summ(np.array(u[1]['score'])-u[0]['score'])))
 # Empirical coarse count field. Normalize N/B; choose lambda on count itself.
 # Equal weight mixture over 32 roots and 8 seeds (not single-root W).
 cgfs=[]
 lambdas=np.linspace(-.8,.8,33)
 for (g,w),z in raw.items():
  c=z[:,:,ix['C']].ravel();mu=c.mean();var=c.var();f=z[:,:,ix['last_mean']].ravel()/w
  for l in lambdas:
   zz=l*c;mx=zz.max();wt=np.exp(zz-mx);P=wt/wt.sum();W=mx+np.log(wt.mean());m=P@c;v=P@((c-m)**2)
   # average conditional last-wave W; conditional Legendre evaluated pointwise per p, not a mixture Legendre.
   lastW=float(np.mean(w*np.log1p(f*np.expm1(l))))
   cgfs.append(dict(generation=g,width=w,lambda_=float(l),W=float(W),tilted_count=float(m),tilted_variance=float(v),legendre=float(l*m-W),gaussian=float(mu*l+.5*var*l*l),last_wave_W_mean=lastW))
 # D4 readout comparisons and empirical native orientation TV (2 seeds per orientation).
 perm=np.array(reg['permutations']);sym=[];symrows=[]
 for g in (0,3):
  s=read(S/f'data/infer_single{g}.json');d=read(S/f'data/infer_d4_{g}.json');assert len(s)==len(d)==256
  pp=np.array([x['p'] for x in s]).reshape(32,8,81);vv=np.array([x['v'] for x in s]).reshape(32,8)
  qq=np.array([x['p'] for x in d]).reshape(32,8,81);uu=np.array([x['v'] for x in d]).reshape(32,8)
  a0=np.stack([pp[:,o,perm[o]] for o in range(8)],1);a1=np.stack([qq[:,o,perm[o]] for o in range(8)],1)
  qref=a0.mean(1);vref=vv.mean(1);err=float(abs(a1-qref[:,None,:]).max());verr=float(abs(uu-vref[:,None]).max())
  equiv=float(abs(a1-a1[:,:1]).max());vinv=float(abs(uu-uu[:,:1]).max())
  assert err<2e-6 and verr<2e-6 and equiv<2e-6 and vinv<2e-6
  vdiff=np.sum(abs(a0-a0[:,:1]),axis=2)/2;vdiff4=np.sum(abs(a1-a1[:,:1]),axis=2)/2
  for c in range(32):symrows.append(dict(generation=g,core=c,policy_TV_single=float(vdiff[c,1:].mean()),policy_TV_d4=float(vdiff4[c,1:].mean()),value_range_single=float(np.ptp(vv[c])),value_range_d4=float(np.ptp(uu[c]))))
  sym.append(dict(generation=g,max_policy_reference_error=err,max_value_reference_error=verr,max_equivariance_error=equiv,max_value_invariance_error=vinv,mean_single_TV=float(vdiff[:,1:].mean()),mean_D4_TV=float(vdiff4[:,1:].mean()),mean_single_value_range=float(np.ptp(vv,axis=1).mean()),max_single_value_range=float(np.ptp(vv,axis=1).max()),max_D4_value_range=vinv))
  # orientation diagnostic on actual search counts, no pathwise equivariance expected.
  for d4 in (0,1):
   mask=(M[:,1]==g)&(M[:,5]==d4)&(M[:,8]==1)&(M[:,9]==64)
   m=M[mask];assert len(m)==128
   z=a['final_n'][m[:,0]]/64
   z=np.array([x[perm[o]] for x,o in zip(z,m[:,6])])
   cvals=[]
   for c in sorted(set(m[:,2])):
    u=[]
    for o in range(8):
     ix0=(m[:,2]==c)&(m[:,6]==o);assert ix0.sum()==2;u.append(z[ix0].mean(0))
    u=np.array(u);cvals.append(float(np.sum(abs(u[1:]-u[0]),axis=1).mean()/2))
   sym[-1][f'search_TV_d4_{d4}']=dict(mean=float(np.mean(cvals)),by_core=cvals,scope='2-seed empirical orientation means, sampling/order/tie effects unresolved')
 # R60 matched initial roots, deduplicate exact replicate across arena arms, all prefixes.
 p=np.load(S/'R60_PREFIX61.npz');x=p['values'];m=p['meta'];tag=p['tags'];r=read(S/'R60_REDUCTION61.json');assert sha(S/'R60_PREFIX61.npz')==r['array_sha256']
 games=r['games'];plans={}
 for t in set(tag.tolist()):
  plans[t]={j['id']:j for j in read(O/f'plans/{t}.json')}
 matched={};cohort=[]
 for idx,row in enumerate(m):
  game,tid,gen,ply,k,budget,ri=map(int,row);t=str(tag[tid])
  if t.startswith('a') and not t.startswith('arena') and ri==0:
   job=plans[t][games[game]['id']];key=(budget,gen,tuple(job['opening']),k);v=x[idx]
   if key in matched:assert np.array_equal(v,matched[key])
   else:matched[key]=v
 assert len([k for k in matched if k[3]==k[0]])==384
 for budget in (32,128):
  for gen in range(4):
   for k in (4,8,16,32,64,128):
    if k>budget:continue
    v=np.array([vv for kk,vv in matched.items() if kk[:2]==(budget,gen) and kk[3]==k]);assert len(v)==48
    cohort.append(dict(budget=budget,generation=gen,k=k,roots=48,mean={n:float(v[:,i].mean()) for i,n in enumerate(FIELDS)}))
 full=[]
 for t in tag:
  for g in range(4):
   for k in (4,8,16,32,64,128):
    ix0=(m[:,1]==list(tag).index(t))&(m[:,2]==g)&(m[:,4]==k)
    if not ix0.any():continue
    v=x[ix0];gs=m[ix0,0];uniq=np.unique(gs)
    means=np.stack([v[gs==j].mean(0) for j in uniq])
    full.append(dict(cohort=str(t),generation=g,k=k,n_games=len(uniq),n_roots=len(v),mean={n:float(means[:,i].mean()) for i,n in enumerate(FIELDS)}))
 result=dict(scope='32 new response root blocks; fixed g0/g3; exact source score identity and finite MC comparisons; D4 distribution diagnostic separate; historical matched roots only',responses=responses,contrasts=contrasts,coarse_cgf=cgfs,symmetry=sym,symmetry_core=symrows,matched_prefix=cohort,fullgame_cohorts=full)
 dump(S/'PHYSICS_RESULTS.json',result);dump(S/'CORE_EFFECTS61.json',coredata)
 print('SUMMARY',[(x['generation'],x['width'],{k:round(x['observables']['C'][k]['mean'],6) for k in ('fd','score','reweighted_fd','self_normalized_fd')},x['expected_fisher']['mean'],x['count_variance']['mean'],x['score2_minus_fisher']['ci95'],x['ess_minus'],x['ess_plus']) for x in responses if x['k']==128],flush=True)
 print('D4',sym,flush=True)
 print('CONTRASTS',[(x['width'],x['fd_g3_minus_g0'],x['score_g3_minus_g0']) for x in contrasts if x['k']==128 and x['observable']=='C'],flush=True)
if __name__=='__main__':main()
