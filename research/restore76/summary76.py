"""Compact typed scientific summary from R76 verified aggregates."""
from base61 import *
from collections import Counter
Q=read(S/'CRITICAL76.json');V=read(S/'INDEPENDENT76.json');LS=[13,15,17,19,21];BT=[0.,10/3,8.,16.];BS=[16,64]
assert V['status']=='PASS_SCOPED' and Q['paths']==3200
A=[read(S/f'AUDIT_L{L}.json') for L in LS];gen=[read(S/f'GEN_L{L}.json') for L in LS]
groups=Q['groups'];nulls=Q['nulls'];n0=[g for g in groups if g['beta']==0];null_summary={}
for k in ['trace_B','ratio','binder']:
 obs=[];ref=[];inside=[];width=[]
 for g in n0:
  v=g['B']*g['means']['trace'] if k=='trace_B' else g[k];e=g['B']*g['means']['null_trace'] if k=='trace_B' else g['null_'+k]
  n=next(x for x in nulls if (x['L'],x['arm'],x['B'])==(g['L'],g['arm'],g['B']))[k]
  obs.append(v);ref.append(e);inside.append(n['lo']<=v<=n['hi']);width.append(n['hi']-n['lo'])
 null_summary[k]=dict(pointwise_inside=int(sum(inside)),cases=len(inside),obs_mean=float(np.mean(obs)),ref_mean=float(np.mean(ref)),rmse=float(np.sqrt(np.mean((np.array(obs)-ref)**2))),mean_null_95_width=float(np.mean(width)))
profiles=[]
for b in BS:
 for beta in BT:
  g=[x for x in groups if x['B']==b and x['beta']==beta]
  profiles.append(dict(B=b,beta=beta,**{k:float(np.mean([b*x['means'][k] for x in g])) for k in ('trace','drift','martingale','cross','bracket')},ratio_mean=float(np.mean([x['ratio'] for x in g])),binder_mean=float(np.mean([x['binder'] for x in g])),top_pair_abs=float(np.mean([x['means']['top_pair_abs'] for x in g]))))
wins=Counter();fit=[]
for ob in ('ratio','log_B_trace'):
 for L in (13,21):
  rows=[x for x in Q['heldout_fits'] if x['observable']==ob and x['holdout']==[L]]
  z={k:float(np.mean([next(f['test_rmse'] for f in x['models'] if f['model']==k) for x in rows])) for k in ('analytic','crossover','scaling')}
  win=Counter(min(x['models'],key=lambda f:f['test_rmse'])['model'] for x in rows);wins.update(win);fit.append(dict(observable=ob,holdout=L,mean_rmse=z,wins=dict(win)))
peak={k:dict(cases=len([x for x in Q['peaks'] if x['observable']==k]),interior=sum(x['interior'] for x in Q['peaks'] if x['observable']==k),maxima=dict(Counter(str(x['beta']) for x in Q['peaks'] if x['observable']==k))) for k in ('capacity','chi_top','chi_fixed')}
pulse=[dict(L=x['L'],direct=x['ci'][0][0],conditional=x['ci'][0][1],difference=[r[3] for r in x['ci']],direct_mark=x['ci'][0][4]) for x in Q['aux'] if x['kind']=='pulse']
cp=[]
for c in (.7,1.4,2.8):
 a=[x for x in Q['aux'] if x['kind']=='cpuct' and x['c']==c]
 cp.append(dict(c=c,**{k:float(np.mean([x['means'][k] for x in a])) for k in ('top_pair_abs','entropy','chi_top','capacity','top')}))
summary=dict(paths=Q['paths'],backups=Q['backups'],candidate_samples=Q['candidate_samples'],raw_bytes=Q['raw_bytes'],size_generation_seconds={str(g['L']):g['seconds'] for g in gen},max_root_score_error=max(a['max_root_score_error'] for a in A),max_root_u_error=max(a['max_root_u_error'] for a in A),max_sample_log_error=max(a['max_sample_log_error'] for a in A),max_probability_error=max(a['max_probability_error'] for a in A),max_backup_moment_error=max(a['max_backup_moment_error'] for a in A),null_summary=null_summary,profiles=profiles,crossing_pairs_with_roots=sum(bool(x['crossings']) for x in Q['crossings']),crossing_pairs=len(Q['crossings']),joint_crossings=Q['joint_crossings'],heldout_wins=dict(wins),heldout_summary=fit,peaks=peak,pulse=pulse,pulse_exact=[{k:v for k,v in r.items() if k not in ("pmf","raw_triple_sha")} for r in read(S/"PULSE_EXACT76.json")["cases"]],cpuct=cp,independent=V,figures=len(read(S/'PLOT_MANIFEST76.json')['figures']),scope='Finite native family, pointwise intervals and descriptive fits. No physical criticality or equivalence inference.')
dump(S/'SUMMARY76.json',summary)
print(json.dumps(summary,indent=2,allow_nan=False),flush=True)
