from arena68 import *
from itertools import product
def main():
 a=read(S/'AUDIT68.json');reg=read(S/'REGISTRATION.json');boots=np.random.default_rng(680301).integers(0,8,(10000,8))
 def ci(v):
  v=np.array(v,float);assert len(v)==8;valid=np.isfinite(v);assert valid.any()
  den=valid[boots].sum(1);ok=den>0;xx=np.where(valid,v,0.)
  return dict(mean=float(xx.sum()/valid.sum()),ci95=np.quantile(xx[boots].sum(1)[ok]/den[ok],[.025,.975]).tolist(),blocks=int(valid.sum()))
 def blockratio(s,n):
  s=np.asarray(s,float);n=np.asarray(n,float);assert len(s)==len(n)==8;assert n.sum()>0
  dn=n[boots].sum(1);ok=dn>0;return dict(mean=float(s.sum()/n.sum()),ci95=np.quantile(s[boots].sum(1)[ok]/dn[ok],[.025,.975]).tolist(),n=int(n.sum()),blocks=int((n>0).sum()))
 def score(x):return np.nan if x['status']!='complete' else .5 if x['winner']==0 else float(x['winner']==x['job']['color'])
 matches=sorted({x['job']['match'] for x in a['rows']});by={};result=[]
 for m in matches:
  rows=[x for x in a['rows'] if x['job']['match']==m];mat=np.full((8,2),np.nan)
  for x in rows:mat[x['job']['core'],int(x['job']['color']==-1)]=score(x)
  by[m]=mat;complete=np.isfinite(mat);pairs=complete.all(1)
  pair=np.where(pairs,np.nansum(mat,axis=1)/2,np.nan)
  # missing results bounds without assumptions, all registered games
  lo=float(np.nansum(mat)/16);hi=float((np.nansum(mat)+(~complete).sum())/16)
  result.append(dict(match=m,registered=16,complete=int(complete.sum()),wins=int((mat==1).sum()),draws=int((mat==.5).sum()),losses=int((mat==0).sum()),failed=int((~complete).sum()),complete_score=float(np.nanmean(mat)),paired_opening_score=ci(pair),all_registered_score_bounds=[lo,hi],black_score=float(np.nanmean(mat[:,0])),white_score=float(np.nanmean(mat[:,1])),complete_pairs=int(pairs.sum()),pentanomial=[int(np.sum(pairs&(np.nansum(mat,axis=1)==x))) for x in (0,.5,1,1.5,2)],matrix=[[None if not np.isfinite(v) else float(v) for v in r] for r in mat]))
 contrasts=[]
 for l,r in [('ext_recovery','ext_single'),('ext_recovery','ext_clean'),('ext_single','ext_clean')]:
  x,y=by[l],by[r];ok=np.isfinite(x).all(1)&np.isfinite(y).all(1);d=np.where(ok,np.nansum(x-y,axis=1)/2,np.nan)
  v=d[ok];signs=np.array(list(product((-1,1),repeat=len(v))));obs=abs(v.sum());pv=float(np.mean(abs(signs@v)>=obs-1e-12))
  lo=float((np.nansum(x)-np.nansum(y)-np.isnan(y).sum())/16);hi=float((np.nansum(x)+np.isnan(x).sum()-np.nansum(y))/16)
  contrasts.append(dict(left=l,right=r,complete_opening_contrast=ci(d),sign_flip_p=pv,all_registered_contrast_bounds=[lo,hi]))
 # For internal matches, focal is recovery, compare with .5 under exchangeable/symmetric opening-level null.
 for m in ('internal_recovery_single','internal_recovery_clean'):
  x=by[m];d=x.mean(1)-.5;sg=np.array(list(product((-1,1),repeat=8)));pv=float(np.mean(abs(sg@d)>=abs(d.sum())-1e-12))
  contrasts.append(dict(left=m,right='neutral_.5',complete_opening_contrast=ci(d),sign_flip_p=pv))
 # Native fault rates, per-move summaries and costs by actual arm, separate external and internal.
 rows=[];temporal=[]
 for x in a['rows']:
  for r in x['native']:
   rows.append(dict(core=x['job']['core'],match=x['job']['match'],game=x['id'],complete=x['status']=='complete',focal=x['job']['color']==r['side'],winner=x['winner'],**r))
 arms=['clean','single','recovery'];summary=[];cutstats=[]
 keys=['H','Hnorm','Hprior','KL','mean_value','variance','within','between','top','neff','depth','terminal_fraction','conditional_H','surprisal']
 for cohort in ('external','internal'):
  for arm in arms:
   rr=[r for r in rows if r['arm']==arm and r['match'].startswith('ext_' if cohort=='external' else 'internal_')]
   nn=lambda f:np.array([sum(f(x) for x in rr if x['core']==i) for i in range(8)])
   n=nn(lambda x:x['fault_records']);moves=nn(lambda x:1)
   summary.append(dict(cohort=cohort,arm=arm,native_moves=len(rr),fault_records=int(n.sum()),carrier_rate=blockratio(nn(lambda x:x['carrier_flips']),3*n),sign_error_rate=blockratio(nn(lambda x:x['sign_errors']),n),syndrome_rate=blockratio(nn(lambda x:x['syndromes']),n),value_MAE=blockratio(nn(lambda x:x['value_abs_error']),n),correction_success=None if nn(lambda x:x['syndromes']).sum()==0 else blockratio(nn(lambda x:x['corrected']),nn(lambda x:x['syndromes'])),native_seconds=blockratio(nn(lambda x:x['elapsed']),moves),choice_surprisal=blockratio(nn(lambda x:x['chosen_surprisal']),moves)))
   for k0,k in enumerate((4,8,16,32,64)):
    vals={key:blockratio(nn(lambda x:x['cuts'][k0][key]),moves) for key in keys}
    vals['surprisal_residual']=blockratio(nn(lambda x:x['cuts'][k0]['surprisal']-x['cuts'][k0]['conditional_H']),moves)
    cutstats.append(dict(cohort=cohort,arm=arm,k=k,metrics=vals))
   # At-risk game-stage bins, not a balanced longitudinal panel.
   for start in (0,8,16,24,32,40,48,56):
    q=[x for x in rr if start<=x['ply']<start+8]
    if not q:continue
    den=np.array([sum(x['core']==i for x in q) for i in range(8)])
    ms={key:blockratio([sum(x['cuts'][-1][key] for x in q if x['core']==i) for i in range(8)],den) for key in ('H','KL','mean_value','variance','conditional_H')}
    temporal.append(dict(cohort=cohort,arm=arm,ply_lo=start,ply_hi=start+7,n=len(q),games=len(set(x['game'] for x in q)),metrics=ms))
 # Exact empirical mixture of observed conditional root laws over four-draw waves.
 # I(T;A)=H(mean p_t)-mean H(p_t), not unconditional path MI. Spatial connected visit field C(r).
 spaces=[];mi=[]
 y,x=np.divmod(np.arange(225),15);dist=np.abs(y[:,None]-y)+np.abs(x[:,None]-x)
 for z in a['rows']:
  with gzip.open(S/'data'/(z['id']+'.json.gz'),'rt') as f:g=json.load(f)
  for row in g['rows']:
   if row['engine']!='native':continue
   r=row['search'];b=board(r['moves']);legal=np.flatnonzero(b==0);nlegal=len(legal)
   vv=[]
   for t in range(0,64,4):
    p=np.zeros(225);law=r['events'][t]['root_selection']['law'];p[law['actions']]=law['p'];vv.append(p)
    for e in r['events'][t:t+4]:assert e['root_selection']['law']['p']==law['p']
   vv=np.array(vv);h0=entropy(vv.mean(0));h1=np.mean([entropy(q) for q in vv]);I=h0-h1;assert -1e-12<=I<=math.log(16)+1e-12
   mi.append(dict(core=z['job']['core'],match=z['job']['match'],game=z['id'],arm=row['arm'],ply=row['ply'],mi=float(I),Hmean=float(h0),Hconditional=float(h1)))
   dd=dist[np.ix_(legal,legal)]
   for c in r['cuts']:
    v=np.asarray(c['n'])[legal]/c['k'];u=v-v.mean();den=u@u/nlegal
    if den<=1e-18:continue
    mat=u[:,None]*u[None,:]/den
    corr=[float(mat[dd==d].mean()) if (dd==d).any() else None for d in range(1,13)]
    spaces.append(dict(core=z['job']['core'],match=z['job']['match'],game=z['id'],arm=row['arm'],ply=row['ply'],k=c['k'],corr=corr))
 correlations=[];inf=[]
 for cohort in ('external','internal'):
  for arm in arms:
   q=[x for x in mi if x['arm']==arm and x['match'].startswith('ext_' if cohort=='external' else 'internal_')]
   n=np.array([sum(x['core']==i for x in q) for i in range(8)])
   inf.append(dict(cohort=cohort,arm=arm,n=len(q),MI=blockratio([sum(x['mi'] for x in q if x['core']==i) for i in range(8)],n)))
   for k in (4,8,16,32,64):
    q=[x for x in spaces if x['arm']==arm and x['k']==k and x['match'].startswith('ext_' if cohort=='external' else 'internal_')]
    vals=[]
    for d in range(12):
     v=[x for x in q if x['corr'][d] is not None];n=np.array([sum(x['core']==i for x in v) for i in range(8)])
     vals.append(blockratio([sum(x['corr'][d] for x in v if x['core']==i) for i in range(8)],n))
    correlations.append(dict(cohort=cohort,arm=arm,k=k,metrics=vals))
 dump(S/'SPATIAL68.json',dict(scope='actual root visit fields centered on legal cells; Manhattan pair distance, normalized by within-root variance; no equilibrium correlation length or critical exponent',rows=spaces))
 dump(S/'INFORMATION68.json',dict(scope='empirical mixture of 16 recorded conditional wave laws, uniform wave-index T; I(T;A) not actual unconditional path or quantum MI',rows=mi))
 out=dict(scope='actual 15x15 complete-game outcomes; bootstrap blocks=8, small sample descriptive, failed games retained bounds; fixed-size actual entropy and spatial statistics',results=result,contrasts=contrasts,native=summary,cuts=cutstats,temporal=temporal,information=inf,spatial=correlations)
 dump(S/'PHYSICS_RESULTS.json',out)
 for r in result:print('MATCH',r)
 for r in contrasts:print('CONTRAST',r)
 for r in summary:print('NATIVE',r)
 print('MI',inf)
if __name__=='__main__':main()
