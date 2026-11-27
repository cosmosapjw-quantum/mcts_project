from base61 import *
R=read(S/'CRITICAL87.json');I=read(S/'INDEPENDENT87.json');E=read(S/'ERRATUM86.json');LS=[13,15,17,19,21]
assert I['status']=='PASS_SCOPED' and I['raw_paths']==400 and I['fit_models']==36 and E['status']=='CORRECTED_STARTUP_SCOPE'
aud=[read(S/f'AUDIT_L{L}.json') for L in LS];gen=[read(S/f'GEN_L{L}.json') for L in LS]
assert sum(x['fresh_paths'] for x in aud)==800 and sum(x['fresh_backups'] for x in aud)==51200 and sum(x['prior_paths'] for x in aud)==3520
g=R['gates'];cross=R['crossings']
passes=sum(x['critical_win'] for x in g);common=sum(bool(x['common_brackets']) for x in cross);critical=[x for x in R['fits'] if x['model']=='critical' and x['h']==64]
stable=len({x['params'][2] for x in critical})==1 and len({x['params'][0] for x in critical})==1 and all(0<x['params'][0]<16 for x in critical)
gate=passes==len(g) and all(x['common_brackets'] for x in cross if x['h']==64) and stable
# Summary of actual geometry-balanced descriptive averages, keep sizes/densities in source.
metrics=[]
for bi in range(5):
 for h in (16,32,64):
  z=[x for x in R['strata'] if x['cohort']==2 and x['bi']==bi and x['budget']==64 and x['h']==h]
  assert len(z)==10
  v=np.mean([x['mean'] for x in z],0);metrics.append(dict(bi=bi,h=h,trace=float(v[0]),null_trace=float(v[3]),raw_trace=float(v[6]),drift_trace=float(v[7]),centered_innovation_trace=float(v[8]),cross_trace=float(v[9]),uncentered_innovation_trace=float(v[10]),ratio=float(np.mean([x['ratio'] for x in z])),pmax=float(v[11])))
corrected=[]
for beta in (0.,10/3,16.):
 v=[x for x in E['strata'] if x['beta']==beta and x['cohort']==1]
 a=[next(q for q in x['blocks'] if q['h']==16) for x in v];corrected.append(dict(beta=beta,predictable_old=float(np.mean([x['prior_predictable_trace'] for x in a])),predictable_corrected=float(np.mean([x['predictable_trace_ci'][0] for x in a])),compensated=float(np.mean([x['martingale_trace_ci'][0] for x in a]))))
q=dict(fresh_paths=800,fresh_backups=51200,reused_paths=3520,total_paths=4320,all_B64_paths=3040,independent_B16_paths=1280,prefix_reductions=10400,cumulative_fixed_root=93336,raw_bytes=sum(x['raw_bytes'] for x in gen),generation_seconds_by_size=[x['seconds'] for x in gen],max_startup_full_action_gap=max(x['startup_max'] for x in aud),startup_changed_paths=sum(x['startup_changed'] for x in aud),raw_errors={k:max(x['errors'].get(k,0.) for x in aud) for k in set().union(*(x['errors'] for x in aud))},candidate_samples=sum(x['candidate_samples'] for x in aud),strata=len(R['strata']),registered_critical_wins=passes,registered_cells=len(g),common_crossings=common,critical_gate=bool(gate),critical_params=critical,metrics=metrics,corrected_R86=corrected,erratum=dict(variance_interval_failures=E['variance_interval_failures'],exponential_interval_failures=E['exponential_interval_failures'],max_startup_trace_overestimate=E['max_startup_trace_overestimate'],nonuniform_startup_sector_paths=E['nonuniform_startup_sector_paths']),independent=I,scope='Exact bounded innovation covariance, noninteracting entropy competition and finite-size model comparison; first-wave R86 erratum preserved. No physical criticality inferred from curve fits.')
dump(S/'SUMMARY87.json',q)
old=read(B/'mqft_restore86_20261127/CONVERGENCE86.json');claims=[]
for c in old['claims']:
 c=dict(c);c['evidence']=[('mqft_restore86_20261127/'+x if '/' not in x else x) for x in c['evidence']]
 if c['id']=='MEM-SCALE-03':
  c['restore87_correction']='R49 startup is 1+1+2, not iid width4 for adaptive beta. Original theorem scoped to actual quiescent waves; per-selection conditional covariance/exponential compensator correct all startup cases. R86 D and cross-moments unchanged, replace predictable bracket and exponential summaries by ERRATUM86.json. R86 exact 81-string enumeration evaluated averaged probability, not actual adaptive startup. Historical files immutable.'
  c['evidence']+=['ERRATUM86.json','THEORY.md']
 claims.append(c)
claims.extend([
dict(id='SCALE-BOUND-01',stage=87,status='ACCEPTED_DERIVED_AND_CHECKED',items=['L17','L26','L36','L37','L38','Q32'],claim='For fixed legal support, normalized categorical martingale innovation covariance is PSD, has constant-vector null mode, trace<=1-1/n and operator norm<=1/2, for arbitrary adapted conditional selection law and finite B. Scalar bounded mode variance <=range(f)^2/4. Actual per-selection R49 trace verified, including startup correction.',evidence=['THEORY.md','CRITICAL87.json','ERRATUM86.json','INDEPENDENT87.json','SUMMARY87.json'],limit='Normalized innovation only; no claim excluding critical behavior in drift, extensive observables, length or a singular joint limit. R49 per-selection conditional law, not all first four IID.'),
dict(id='SCALE-ENTROPY-02',stage=87,status='ACCEPTED_DERIVED_AND_CHECKED',items=['L17','L26','L36','L37','L38'],claim='A noninteracting categorical model with one score gap exhibits exact sigmoid collapse in beta*Delta-log(n-1), and a finite Gibbs concentration bound follows from score range. Actual finite-size score landscapes and declared log-size control tested separately from adaptive dynamics.',evidence=['THEORY.md','CRITICAL87.json','INDEPENDENT87.json','SUMMARY87.json'],limit='Explicit counterexample to sufficiency of collapse as evidence of spatial criticality; not proof that all actual MCTS concentration is explained by this null. No physical temperature or scaling exponent.'),
dict(id='SCALE-CRITICAL-03',stage=87,status='OPEN_CONTROLLED_CRITICAL_LIMIT',items=['L17','L26','L36','L37','L38','Q32'],claim='Native adaptive MCTS has a controlled spatial critical limit with consistent finite-size scaling and identified RG fixed point.',evidence=['CRITICAL87.json','SUMMARY87.json','THEORY.md'],limit='Registered finite-size critical admission not established unless all predeclared comparisons pass; even a pass is descriptive and requires independent asymptotic mechanism. R76/R77 negative results and nonlumpability retained. No repeating same grid without new mechanism.')
])
dump(S/'CONVERGENCE87.json',dict(status='EIGHTEEN_BOUNDED_CORRECTED_SUBCLAIMS_ACCEPTED_SIX_OPEN',claims=claims,accepted_subclaims=18,new_accepted_subclaims=2,final_original_item_certifications=0,prior_corrected_report_artifacts=['L31'],scope='16 prior accepted with explicit R86 startup correction; two new scoped analytic claims. No critical-limit admission.'))
print('SUMMARY',q['registered_critical_wins'],q['registered_cells'],q['common_crossings'],q['critical_gate'],q['corrected_R86'],flush=True)
