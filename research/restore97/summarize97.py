from entropy97 import *
from scipy.stats import t as student
def ci(x):
 x=np.asarray(x,float);m=x.mean(0);h=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x)) if len(x)>1 else np.zeros_like(m)
 return np.stack([m,m-h,m+h]).tolist()
def main():
 a=read(S/'ENTROPY97.json');assert sha(S/'ENTROPY97.json.gz')==a['raw_gzip_sha']
 r=json.load(gzip.open(S/'ENTROPY97.json.gz','rt'));assert len(r)==a['states']
 groups=collections.defaultdict(list)
 for x in r:groups[tuple(x[k] for k in ['family','protocol','mode','depth','xi'])].append(x)
 out=[]
 for (g,p,m,d,xi),z in sorted(groups.items()):
  assert len(z)==24 and len({x['core'] for x in z})==24
  sk=[x['mask'] for x in z[0]['subsets']];cutkeys=[x['mask'] for x in z[0]['cuts']]
  fields=['S','Hdiag','coherence','purity','success'];sf=['S','Renyi2','Hdiag'];cf0=['SA','SB','MI','conditional_A','conditional_B','N','success_weighted_N'];mf=['MI','classic_Z','chi_Z','chi_best','fixed_gap','discord_lower','discord_upper']
  assert all([s['mask'] for s in x['subsets']]==sk and [s['mask'] for s in x['cuts']]==cutkeys for x in z)
  out.append(dict(family=g,protocol=p,mode=m,depth=d,xi=xi,n=24,fields=fields,mean_ci=ci([[x[k] for k in fields] for x in z]),subset_fields=sf,subset_masks=sk,subset_ci=ci([[[v[k] for k in sf] for v in x['subsets']] for x in z]),spectrum_ci=[ci([x['subsets'][i]['spectrum'] for x in z]) for i in range(len(sk))],cut_fields=cf0,cut_masks=cutkeys,cut_ci=ci([[[v[k] for k in cf0] for v in x['cuts']] for x in z]) if cutkeys else [],measurement_fields=mf,measurement_ci=ci([[[v[k] for k in mf] for v in x['measurements']] for x in z]) if d else [],NPT_counts=[sum(x['cuts'][i]['N']>1e-10 for x in z) for i in range(len(cutkeys))],negative_cond_counts=[sum(min(x['cuts'][i]['conditional_A'],x['cuts'][i]['conditional_B'])<-1e-9 for x in z) for i in range(len(cutkeys))]))
 cuts=[v for x in r for v in x['cuts']];ms=[v for x in r for v in x['measurements']]
 counts=dict(NPT=sum(v['N']>1e-10 for v in cuts),negative_conditional=sum(min(v['conditional_A'],v['conditional_B'])<-1e-9 for v in cuts),NPT_nonnegative_conditional=sum(v['N']>1e-10 and min(v['conditional_A'],v['conditional_B'])>=-1e-9 for v in cuts),improved_Z=sum(v['chi_best']>v['chi_Z']+1e-9 for v in ms),mean_fixed_gap=float(np.mean([v['fixed_gap'] for v in ms])),mean_discord_upper=float(np.mean([v['discord_upper'] for v in ms])),min_CMI=min(v for x in r for v in x['conditional_MI']))
 a=read(S/'NATIVE_THERMAL97.json');risk=[]
 for (co,arm,ply),z in sorted(collections.defaultdict(list,{}).items()):pass
 bins=collections.defaultdict(list)
 for v in a['native']:
  if v['budget']==64:bins[v['cohort'],v['arm'],v['ply']//8*8].append(v)
 for (co,arm,ply),z in sorted(bins.items()):
  by=collections.defaultdict(list)
  for x in z:by[x['core']].append([x['H'],x['conditional_H']])
  risk.append(dict(cohort=co,arm=arm,ply=ply,searches=len(z),games=len({x['game'] for x in z}),cores=len(by),ci=ci([np.mean(v,axis=0) for v in by.values()])))
 assert sum(x['searches'] for x in risk)==853
 t=a['thermal'];theta=a['theta'];therm=dict(theta=theta,curve_ci=ci([x['curve'] for x in t]),derivative_ratio_max=max(abs(v[1]) for x in t for v in x['derivative_relative']),derivative_ratio_cases=sum(len(x['derivative_relative']) for x in t),feedback_cases=a['feedback_cases'])
 scale=read(P92/'SUMMARY92.json');ss=[dict(L=x['L'],arm=x['arm'],beta=x['beta'],h=x['h'],ci=[v[2] for v in x['metric_ci']]) for x in scale['strata'] if x['cohort']==2 and x['budget']==64 and x['h']==64]
 assert len(ss)==50
 # Prior response evidence is a separate pinned reference, not a rerun.
 p93=B/'mqft_restore93_20261128';r93=read(p93/'RUN_MANIFEST93.json');f=p93/'RESULTS_KO.md';assert sha(f)==r93['files']['RESULTS_KO.md']
 ind=read(S/'INDEPENDENT97.json');ind['source_hashes'][str(f.relative_to(B))]=sha(f)
 # Original/authority receipts and R68 old plot evidence source
 ind['source_hashes'][str((O/'ORIGINAL_SOURCE96.json').relative_to(B))]=sha(O/'ORIGINAL_SOURCE96.json')
 dump(S/'INDEPENDENT97.json',ind)
 dump(S/'SUMMARY97.json',dict(groups=out,counts=counts,risk=risk,thermal=therm,scale=ss,scope='Means and descriptive t intervals over 24 roots; historical R68 game-ply grouped by core, R92 categorical size strata separate; R91 thermal reference curves only; R93 response reference not physical FDT'))
 print('SUMMARY',len(out),counts,'risk',len(risk),'thermal',therm['derivative_ratio_cases'],flush=True)
if __name__=='__main__':main()
