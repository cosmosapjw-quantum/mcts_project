from base61 import *
A=read(S/'AUDIT84.json');L=read(S/'LANDSCAPE84.json');F=read(S/'MODEL84.json');T=read(S/'TRANSFER84.json');V=read(S/'INDEPENDENT84.json');R=read(S/'REGISTRATION.json')
assert V['status']=='PASS_SCOPED' and A['paths']==2304 and A['prior_paths']==4032 and F['status']=='TRAIN_ONLY_FROZEN'
assert sha(S/'MODEL84.json')==T['model_sha']=='45548d7f87dcf48670a243cef6d77b37f6fed8ca3d9a6255e426820b2d170b0c'
assert len(L['rows'])==456 and len({x['orbit'] for x in L['rows']})==152
# Frozen model existed before creation of every new raw path, verified against actual metadata.
assert (S/'MODEL84.json').stat().st_mtime_ns<min((B/x['path']).stat().st_mtime_ns for x in A['raw_receipts'])
out=[]
for g,q in enumerate(T['results']):
 a=next(x for x in q['models'] if x['model']==q['selected_candidate']);b=next(x for x in q['models'] if x['model']==q['selected_baseline'])
 assert q['registered_pass'] and q['energy_gain_ci'][1]>0 and a['W_RMSE']<b['W_RMSE']
 l=[r for r in L['rows'] if r['family']==g and r['group']==3];v=[r for r in A['raw_receipts'] if r['family']==g]
 assert len(l)==48 and len(v)==768
 out.append(dict(family=q['family'],baseline=b['model'],candidate=a['model'],gain_ci=q['energy_gain_ci'],energy_baseline=b['energy_ci'][0],energy_candidate=a['energy_ci'][0],energy_reduction=1-a['energy_ci'][0]/b['energy_ci'][0],W_baseline=b['W_RMSE'],W_candidate=a['W_RMSE'],W_reduction=1-a['W_RMSE']/b['W_RMSE'],split_half_W=a['split_half_W_RMSE'],W_stress=a['W_stress_RMSE'],population_W=a['population_W_RMSE'],coverage90=a['coverage90'],min_root_ESS_regular=a['min_root_ESS_regular'],min_root_ESS_stress=a['min_root_ESS_stress'],mean_clipped_mass=a['mean_clipped_mass'],fresh_landscape_calls=sum(r['calls'] for r in l),mean_extra_child_calls=float(np.mean([r['calls']-1 for r in l])),mean_native_eval_calls=float(np.mean([r['eval_calls'] for r in v])),mean_native_elapsed_s=float(np.mean([r['elapsed_ns'] for r in v])/1e9),registered_pass=True))
q=dict(fresh_paths=2304,fresh_backups=294912,prior_paths=4032,development_roots=104,test_roots=48,raw_bytes=A['raw_bytes'],landscape_roots=152,landscape_rows=456,direct_evaluator_calls=sum(x['calls'] for x in L['rows']),direct_child_checks=V['child_checks'],timing=L['timing'],families=out,passed=3,independent=V,scope='Registered finite pre-search prediction improves relative to strongest development-only static baseline for all three evaluator families on one independent 48-root cohort. Not universal transfer, exact native fine history, or same-cost utility.')
dump(S/'SUMMARY84.json',q)
old=read(B/'mqft_restore83_20261127/CONVERGENCE83.json');prev=[]
for c in old['claims']:
 c=dict(c);c['evidence']=[('mqft_restore83_20261127/'+x if '/' not in x else x) for x in c['evidence']]
 if c['id']=='STAT-TRANSFER-04':
  c['restore84_update']='New registered bounded pre-search transfer criterion passed all three evaluators on 48 independent roots, selected combined ridge vs strongest root-only ridge. Corrected claim STAT-PRESEARCH-07 accepted. Broad arbitrary-population prediction still not admitted; absolute W error, tail/coverage and finite landscape sufficiency limits remain. Retire repeated same feature grid and move to thermal/reverse.'
  c['evidence']+=['SUMMARY84.json','TRANSFER84.json']
 prev.append(c)
new=dict(id='STAT-PRESEARCH-07',stage=84,status='ACCEPTED_CORRECTED_SCOPED',items=['L16','Q32','G07'],claim='For the registered B128 clean search law and fixed evaluator families, adding deterministic full legal-child landscape to root-only predictive features improves proper joint energy score and regular root CGF RMSE over the strongest frozen development-selected static baseline on the new independent 48-root cohort. All three family cells meet the registered gate.',evidence=['THEORY.md','LANDSCAPE84.json','MODEL84.json','TRANSFER84.json','AUDIT84.json','INDEPENDENT84.json','SUMMARY84.json'],limit='104 development geometries and 48 new geometry blocks; additional direct child evaluations outside B128, residual transport has bounded/rounded support not exact native fine-history realizability, pointwise intervals, nonzero clipping and absolute W error, coverage and tail limits; no universal transfer or matched-cost utility.')
dump(S/'CONVERGENCE84.json',dict(status='TEN_BOUNDED_CORRECTED_SUBCLAIMS_ACCEPTED_THREE_OPEN',claims=prev+[new],accepted_subclaims=10,new_accepted_subclaims=1,final_original_item_certifications=0,prior_corrected_report_artifacts=['L31'],scope='Nine accepted subclaims preserved; bounded pre-search improvement accepted. Broad STAT-TRANSFER-04 arbitrary transfer remains not admitted. No repeat feature grid absent new scientific question.'))
print('SUMMARY',q['passed'],q['direct_evaluator_calls'],[(x['family'],x['energy_reduction'],x['W_reduction'],x['mean_clipped_mass']) for x in out],flush=True)
