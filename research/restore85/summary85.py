from base61 import *
from scipy.stats import t as student
N0=read(S/'NATIVE85.json');T=read(S/'THERMAL85.json');B0=read(S/'BRANCH_ENERGY85.json');F=read(S/'STATIONARY85.json');I=read(S/'INDEPENDENT85.json')
assert I['status']=='PASS_SCOPED' and N0['paths']==2304 and len(N0['contexts'])==768 and len(B0['contexts'])==104 and len(T['cases'])==832
assert N0['auxiliary_samples']+T['auxiliary_samples']==6553600 and len(F['contexts'])==768
def ci(x):
 x=np.array(x);m=x.mean(0);h=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return [m.tolist(),(m-h).tolist(),(m+h).tolist()]
keys=['lambda_singular','exact_ift','mc_ift','mc_reverse_support','mean_work','mean_heat','mean_delta_E','mean_info','mean_sigma','jensen_lower','uncorrected_ift','mc_ess','exact_weight_ESS_fraction','max_weight','mean_final_KL']
groups=[]
for protocol in ('R73_fine_detector','R83_binary_detector'):
 for g,name in enumerate(('g0','g3','Rapfi')):
  for alpha in (.25,1.):
   for eps in (0.,.05,.2,1.):
    z=[q for q in T['cases'] if q['protocol']==protocol and q['family']==g and q['alpha']==alpha and q['eps']==eps]
    if not z:continue
    assert len(z)==(16 if protocol.startswith('R73') else 24)
    delta=[q['mc_ift']-q['exact_ift'] for q in z];c=ci(delta);groups.append(dict(protocol=protocol,family=name,alpha=alpha,eps=eps,contexts=len(z),mean={k:float(np.mean([q[k] for q in z])) for k in keys},sample_minus_exact_ci=c,pointwise_interval_contains_zero=bool(c[1]<=0<=c[2]),negative_mean_work_cases=sum(q['mean_work']<0 for q in z),min_mc_ess=min(q['mc_ess'] for q in z),max_weight=max(q['max_weight'] for q in z)))
assert len(groups)==40
flux=[]
for g,name in enumerate(('g0','g3','Rapfi')):
 a=[q for q in F['contexts'] if q['family']==g];assert len(a)==256
 x=[q['stationary_entropy_production'] for q in a];flux.append(dict(family=name,mean=float(np.mean(x)),minimum=min(x),maximum=max(x)))
q=dict(fresh_native_paths=0,reused_native_paths=2304,reused_native_backups=294912,branch_contexts=104,branch_raw_paths=2448,source_raw_total=4752,auxiliary_samples=6553600,wave_samples=N0['auxiliary_samples'],thermal_samples=T['auxiliary_samples'],thermal_cases=832,enumerated_triples=sum(q['triples'] for q in T['cases']),exact_ift_error=max(abs(q['exact_ift']+q['lambda_singular']-1) for q in T['cases']),max_LDB_error=max(q['ldb_error'] for q in T['cases']),max_energy_error=max(q['energy_error'] for q in T['cases']),max_logratio_error=max(q['logratio_error'] for q in T['cases']),negative_mean_work_cases=sum(q['mean_work']<0 for q in T['cases']),min_mc_ess=min(q['mc_ess'] for q in T['cases']),max_weight=max(q['max_weight'] for q in T['cases']),pointwise_group_ift_interval_failures=[g for g in groups if not g['pointwise_interval_contains_zero']],groups=groups,native=N0['results'],stationary=flux,independent=I,scope='Exact stopped-support path identities and declared finite thermal feedback model. No native physical heat, reservoir, clock, or unique physical reverse identified.')
dump(S/'SUMMARY85.json',q)
old=read(B/'mqft_restore84_20261127/CONVERGENCE84.json');prev=[]
for c in old['claims']:
 c=dict(c);c['evidence']=[('mqft_restore84_20261127/'+x if '/' not in x else x) for x in c['evidence']];prev.append(c)
new=[
dict(id='THERMO-LEDGER-01',stage=85,status='ACCEPTED_DERIVED_AND_CHECKED',items=['L18','Q19','Q21','Q23'],claim='A retained ordered native event ledger has an explicit inverse bookkeeping map; for a fixed quiescent native pulse wave, a normalized multinomial endpoint-deletion reverse reference induces the exact ordered categorical law. On a forward positive-hit postselection the finite RN IFT has correction 1-lambda, lambda equal reverse mass outside forward support.',evidence=['THEORY.md','NATIVE85.json','WAVE_SAMPLES85.npz','INDEPENDENT85.json','SUMMARY85.json'],limit='R74 2304 authenticated native records, 768 H32 contexts; auxiliary reverse and postselection distinct from original unconditioned native process. No reversible native engine state or physical dissipation identified.'),
dict(id='THERMO-FEEDBACK-02',stage=85,status='ACCEPTED_DERIVED_AND_CHECKED',items=['L08','L18','L32','Q19','Q20','Q21','Q23'],claim='For a declared branch-driven finite feedback cycle with explicit Gibbs energy, heat-bath LDB, detector and reverse protocol, the exact path log ratio is b W/E* + pointwise information. Generalized IFT equals one minus reverse singular mass, with nontrivial perfect-measurement correction, and first-law/Jensen identities hold. Actual R73/R83 branch scores parameterize the finite model.',evidence=['THEORY.md','BRANCH_ENERGY85.json','THERMAL85.json','THERMAL_SAMPLES85.npz','INDEPENDENT85.json','SUMMARY85.json'],limit='832 declared cases, exact finite triples and independent auxiliary sampling; no feedback device or physical native reservoir executed. Work on system sign, record preparation/reset/controller cost excluded. No general work extraction; imperfect detector rare-weight estimation failures retained.'),
dict(id='THERMO-ID-03',stage=85,status='ACCEPTED_DERIVED_AND_CHECKED',items=['L18','Q19','Q20','Q23'],claim='A Gibbs one-time marginal alone cannot identify local detailed balance or stationary entropy production: positive cycle-flux kernels and detailed-balanced iid kernel share every tested actual frozen Gibbs marginal. Energy-temperature common rescaling also leaves the native dimensionless selection law invariant.',evidence=['THEORY.md','STATIONARY85.json','INDEPENDENT85.json','SUMMARY85.json'],limit='768 actual frozen marginals, declared alternative kernels; not a claim that the alternative reproduces the authenticated native conditional transition law. Physical calibration remains absent.'),
dict(id='THERMO-PHYSICAL-04',stage=85,status='OPEN_PHYSICAL_IDENTIFICATION',items=['L08','L18','L32','Q19','Q20','Q21','Q23'],claim='Actual native MCTS admits independently calibrated physical energy, temperature, reservoir heat and microscopic time reversal.',evidence=['THEORY.md','SUMMARY85.json','ORIGINAL_SOURCE85.json'],limit='Not established; original proposals explicitly reject natural physical energy. No independent physical bath/calibration or hardware-level reverse. Retire repetitive auxiliary fluctuation checks absent a new question.')]
dump(S/'CONVERGENCE85.json',dict(status='THIRTEEN_BOUNDED_CORRECTED_SUBCLAIMS_ACCEPTED_FOUR_OPEN',claims=prev+new,accepted_subclaims=13,new_accepted_subclaims=3,final_original_item_certifications=0,prior_corrected_report_artifacts=['L31'],scope='Ten prior accepted corrected claims preserved. Three new accepted, physical identification remains open.'))
print('SUMMARY',q['enumerated_triples'],q['negative_mean_work_cases'],len(q['pointwise_group_ift_interval_failures']),q['min_mc_ess'],q['max_weight'],flush=True)
