"""Acceptance and root-level effective donor diagnostics; no refitting."""
from population82 import *
A=read(S/'AUDIT82.json');F=read(S/'MODEL82.json');T=read(S/'TRANSFER82.json');I=read(S/'INDEPENDENT82.json');C=read(S/'COMPAT82.json');U=read(S/'RNG_AUDIT82.json')
assert I['status']==C['status']=='PASS_SCOPED' and F['status']=='TRAIN_ONLY_FROZEN' and T['model_sha']==sha(S/'MODEL82.json')
d=arrays();M=d['meta'];res=[];decomp=[]
def model(z,k):return next(v for v in z['models'] if v['model']==k)
def comparison(z,a,b):return next(v for v in z['comparisons'] if v['candidate']==a and v['baseline']==b)
for z in T['results']:
 g=FAMS.index(z['family']);gr=z['group'];tr=subset(d,(M[:,0]==0)&(M[:,1]==g));te=subset(d,(M[:,0]==gr)&(M[:,1]==g));C0=setup(tr,te);cores=tr['meta'][:,2];D=np.eye(48)[cores];assert D.shape==(640,48)
 metrics={}
 for k in MODELS:
  v=model(z,k);p=weights(C0,k,F['families'][g]['bandwidth'].get(k));assert abs((p@D).sum(1)-1).max()<1e-12
  donor=tr['V']*.75 if k.startswith('prefix') or k=='root_history' else tr['X'];ex=np.exp(donor@GRID.T)
  # Root ESS after summing dependent donor histories within geometry, not number of iid observations.
  rg=p@D;ess=1/(rg*rg).sum(1);essj=[]
  for j in range(len(GRID)):
   p1=p*ex[:,j][None,:];p1/=p1.sum(1)[:,None];r=p1@D;essj.append(1/(r*r).sum(1))
  e=np.array(essj).T
  metrics[k]=dict(energy=v['energy_ci'][0],W_RMSE=v['W_RMSE'],W_stress_RMSE=v['W_stress_RMSE'],W_population_RMSE=v['W_population_RMSE'],coverage90=v['coverage90'],min_tilt_donor_ESS_regular=v['min_tilt_ESS_regular'],min_tilt_donor_ESS_stress=v['min_tilt_ESS_stress'],min_root_donor_ESS=float(ess.min()),median_root_donor_ESS=float(np.median(ess)),min_tilt_root_ESS_regular=float(e[:,:25].min()),min_tilt_root_ESS_stress=float(e[:,25:].min()))
 root=metrics['root'];ply=metrics['ply'];pool=metrics['pool'];rh=metrics['root_history'];pp=metrics['prefix_pool']
 cr=comparison(z,'root','ply')['energy_gain_ci'];crp=comparison(z,'root','pool')['energy_gain_ci'];c1=comparison(z,'root_history','prefix_pool')['energy_gain_ci'];c2=comparison(z,'root_history','root')['energy_gain_ci']
 ok_root=cr[1]>0 and root['W_RMSE']<ply['W_RMSE'];ok_hist=c1[1]>0 and c2[1]>0 and rh['W_RMSE']<pp['W_RMSE'] and rh['W_RMSE']<root['W_RMSE']
 res.append(dict(family=z['family'],group=gr,roots=z['roots'],paths=z['paths'],models=metrics,root_gain_vs_ply=cr,root_gain_vs_pool=crp,root_history_gain_vs_prefix_pool=c1,root_history_gain_vs_root=c2,root_history_gain_vs_history_only=comparison(z,'root_history','prefix_kernel')['energy_gain_ci'],root_only_registered_pass=bool(ok_root),prefix_registered_pass=bool(ok_hist),split_half_W_RMSE=z['split_half_W_RMSE'],nearest_root_distance_quantiles=z['nearest_root_distance_quantiles']))
for r in T['mixtures']:
 z=next(a for a in r['source'] if a['J']==[0,0]);c=np.array(z['cov']);b=np.array(z['between']);decomp.append(dict(family=r['family'],group=r['group'],between_trace_fraction=float(np.trace(b)/np.trace(c)),between_per_component=(b.diagonal()/c.diagonal()).tolist(),between_scaled_trace_fraction=float(np.dot(b.diagonal(),[1,.25])/np.dot(c.diagonal(),[1,.25])),max_jensen_gap=max(a['jensen_gap'] for a in r['source']),min_tilt_root_ESS=min(a['root_tilt_ESS'] for a in r['source']),max_decomposition_error=max(a['error'] for a in r['source'])))
nroot=sum(x['root_only_registered_pass'] for x in res);nhist=sum(x['prefix_registered_pass'] for x in res);assert nroot==5 and nhist==6
q=dict(paths=A['paths'],backups=A['backups'],prior_paths=A['prior_paths'],prior_backups=A['prior_backups'],raw_bytes=A['raw_bytes'],diagnostic_replays=12,diagnostic_backups=1536,roots=[48,24,32],errors=A['errors'],rng=U,independent=I,root_only_passes=nroot,prefix_passes=nhist,results=res,mixture_decomposition=decomp,scope='Frozen R79/R80 train-only donor models, two disjoint external cohorts, proper joint energy score and regular CGF errors; root-only transfer gate not met; prefix observed, not autonomous or pre-search prediction. ESS includes geometry aggregation.')
dump(S/'SUMMARY82.json',q)
old=read(B/'mqft_restore81_20261127/CONVERGENCE81.json');prev=[]
for c in old['claims']:
 c=dict(c)
 if c['stage']==81:c['evidence']=['mqft_restore81_20261127/'+x for x in c['evidence']]
 if c['id']=='STAT-TRANSFER-04':
  c['status']='OPEN_GENERALIZATION';c['restore82_update']='Root-only model meets registered criterion 5/6 family-cohort cells; Rapfi R81 lower gain CI negative, g0 fresh vs pool includes zero. H32 observed-prefix conditional forecasts improve in 6/6, distinct claim; absolute source errors remain well above split-half noise. No universal root-only transfer.'
  c['evidence']+=['SUMMARY82.json','TRANSFER82.json']
 prev.append(c)
new=[
 dict(id='STAT-MIX-05',stage=82,status='ACCEPTED_DERIVED_AND_CHECKED',items=['L16','Q32','G07'],claim='For finite root-conditioned laws the population CGF is logsumexp of conditional CGFs and its Hessian exactly decomposes into within-root covariance plus between-root tilted mean covariance. Geometry mixture and intrinsic fluctuation must be distinguished.',evidence=['THEORY.md','TRANSFER82.json','INDEPENDENT82.json','SUMMARY82.json'],limit='Exact mathematical identity and 261 authenticated empirical source cases; between-root fractions depend on coordinates, sampled population and tilt, no physical criticality or universal scale.'),
 dict(id='STAT-HISTORY-06',stage=82,status='ACCEPTED_CORRECTED_SCOPED',items=['L16','Q32','G07'],claim='For the registered finite B128 clean evaluator laws, the H32 root+history conditioned weighted suffix forecast has lower proper joint energy score than both transported unconditional suffix and root-only kernel in all six external family-cohort cells; regular root CGF errors also smaller. Exact prefix count/mark bounds and joint donor dependence retained.',evidence=['THEORY.md','MODEL82.json','TRANSFER82.json','INDEPENDENT82.json','SUMMARY82.json'],limit='Two specific external cohorts, 24/32 geometry blocks; not a universal population guarantee, not pre-search prediction or autonomous closure. Root-only gate fails 1/6; history-only can be better than root+history; limited tail ESS and high absolute W error.')]
dump(S/'CONVERGENCE82.json',dict(status='SEVEN_BOUNDED_CORRECTED_SUBCLAIMS_ACCEPTED_THREE_OPEN',claims=prev+new,accepted_subclaims=7,new_accepted_subclaims=2,final_original_item_certifications=0,prior_corrected_report_artifacts=['L31'],scope='R80/R81 accepted subclaims retained; R82 mixture identity and bounded observed-prefix prediction accepted. STAT-TRANSFER-04 remains open.'))
print('SUMMARY',nroot,nhist,'root gate, history gate',[(x['family'],x['group'],x['models']['root_history']['min_tilt_root_ESS_regular'],x['models']['root_history']['min_tilt_root_ESS_stress']) for x in res],flush=True)
print('DECOMP',[(x['family'],x['group'],x['between_per_component'],x['between_scaled_trace_fraction']) for x in decomp],flush=True)
