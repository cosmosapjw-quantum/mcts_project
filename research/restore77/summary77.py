from base61 import *
C=read(S/'CLOSURE77.json');R=read(S/'RG77.json');V=read(S/'INDEPENDENT77.json');LS=[13,15,17,19,21];BT=[0.,10/3,16.]
aud=[read(S/f'AUDIT_L{L}.json') for L in LS];gen=[read(S/f'GEN_L{L}.json') for L in LS]
names=[x['model'] for x in C['strata'][0]['models']];kinds=['homogeneous','clock_conditioned'];rows=[];cols=[];rg=[];fps=[]
for b in BT:
 a=[x for x in C['strata'] if x['beta']==b];assert len(a)==10
 for name in names:
  for kind in kinds:
   m=[next(v for q in x['models'] if q['model']==name for v in q['models'] if v['kind']==kind) for x in a]
   one=np.mean([x['one_step_ci'][0] for x in m],0);W=np.mean([x['W_rmse'] for x in m]);cr=[]
   for h in (1,2,4,8):
    z=[next(q for q in x['CK'] if (q['model'],q['kind'],q['h'])==(name,kind,h)) for x in a];cr.append(dict(h=h,means=np.mean([q['ci'][0] for q in z],0).tolist()))
   rows.append(dict(beta=b,model=name,kind=kind,one=one.tolist(),mean_W_rmse=float(W),CK=cr))
   if kind=='homogeneous':
    v=[x['fixed_point'] for x in m];fps.append(dict(beta=b,model=name,support=[min(x['support'] for x in v),max(x['support'] for x in v)],max_subdominant=max(x['eigen_subdominant'] for x in v),max_dobrushin=max(x['dobrushin'] for x in v),max_stationary_residual=max(x['residual'] for x in v),max_projector_residual=max(x['projector_residual'] for x in v),max_stationary_crosscheck=max(x['stationary_crosscheck'] for x in v),max_dyadic_TV=np.max([x['dyadic_TV'] for x in v],0).tolist()))
 for name in ['exact_three_sector_count']+names[1:]:
  z=[r for q in C['collisions'] if q['beta']==b for r in q['results'] if r['cohort']==1 and r['model']==name]
  count=sum(r['pairs'] for r in z);num=sum(r['nonzero'] for r in z);cols.append(dict(beta=b,model=name,pairs=count,nonzero=num,max_TV=max(r['max_TV'] for r in z),mean_TV=sum(r['pairs']*(r['mean_TV'] or 0.) for r in z)/max(count,1)))
 a=[x for x in R['strata'] if x['beta']==b];assert len(a)==10
 for field in ('central_residual','mark'):
  for h in (1,2,4,8):
   z=[next(v for v in x[field]['blocks'] if v['h']==h) for x in a]
   rg.append(dict(beta=b,field=field,h=h,mean_W_train_error=float(np.mean([x['W_train_error'] for x in z])),mean_W_same_error=float(np.mean([x['W_same_error'] for x in z])),mean_abs_k4=float(np.mean([abs(x['cumulants'][3]) for x in z])),mean_var=float(np.mean([x['cumulants'][1] for x in z])),mean_abs_adjacent_cov=float(np.mean([abs(x['adjacent_cov']) for x in z])) if h<8 else None,mean_dependence_error=float(np.mean([x['dependence_rmse'] for x in z])) if h<8 else None,mean_marginal_asymmetry=float(np.mean([x['marginal_asymmetry_rmse'] for x in z])) if h<8 else None))
res=dict(fresh_paths=sum(a['fresh_paths'] for a in aud),fresh_backups=sum(a['fresh_backups'] for a in aud),train_paths=sum(a['train_paths'] for a in aud),train_backups=sum(a['train_backups'] for a in aud),fresh_bytes=sum(a['fresh_bytes'] for a in aud),train_bytes=sum(a['train_bytes'] for a in aud),max_errors={k:max(a['errors'].get(k,0.) for a in aud) for k in set().union(*(a['errors'] for a in aud))},generation_seconds={str(a['L']):a['seconds'] for a in gen},old_witness_counts=R['old_witnesses']['counts'],old_max_remaining_tv=R['old_witnesses']['max_remaining_tv'],heldout=rows,collisions=cols,fixed_points=fps,block_rg=rg,independent=V,scope='equal-stratum descriptive averages, 8 test geometry blocks per stratum; no aggregate population equivalence or criticality')
dump(S/'SUMMARY77.json',res)
print('ACQUISITION',{k:res[k] for k in ('fresh_paths','fresh_backups','train_paths','train_backups','fresh_bytes','train_bytes','max_errors','generation_seconds','old_witness_counts','old_max_remaining_tv')},flush=True)
for x in rows:print('MODEL',x['beta'],x['model'],x['kind'],'one',x['one'],'W',x['mean_W_rmse'],'h8',x['CK'][-1]['means'],flush=True)
for x in cols:print('COLLISION',x,flush=True)
for x in fps:print('FIXED',x,flush=True)
for x in rg:
 if x['h'] in (2,8):print('RG',x,flush=True)
