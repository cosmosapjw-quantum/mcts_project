from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
Q=read(S/'CRITICAL76.json');I=read(S/'INDEPENDENT76.json');LS=[13,15,17,19,21];BS=[16,64];BT=[0.,10/3,8.,16.];names=['fixed 8 stones','density near .06'];rows=Q['groups'];figs=[]
def save(fig,name,ids,scope):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=150);fig.savefig(str(p)+'.pdf');plt.close(fig)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 assert b.read_bytes()[:4]==b'%PDF'
 figs.append(dict(figure=name,ids=ids,scope=scope,png_sha=sha(a),pdf_sha=sha(b)))
def get(L,arm,B,beta):return next(x for x in rows if (x['L'],x['arm'],x['B'],x['beta'])==(L,arm,B,beta))
def panel(field,title,scale=1.):
 fig,ax=plt.subplots(2,2,figsize=(12,8))
 for arm in (0,1):
  for j,B in enumerate(BS):
   a=ax[arm,j]
   for L in LS:
    z=[get(L,arm,B,b) for b in BT];v=np.asarray([x['ci'][field] for x in z])* (B if scale=='B' else scale)
    a.errorbar(BT,v[:,0],yerr=[v[:,0]-v[:,1],v[:,2]-v[:,0]],marker='o',label=f'L={L}')
   a.set(title=f'{names[arm]} | B={B}',xlabel='selector beta',ylabel=title);a.legend(fontsize=7)
 return fig
save(panel('top_pair_abs','mean |top-prior pair occupancy contrast|'),'order_parameter',['L01','L17','Q02'],'Independent full B, same pretrained evaluator, prior-ranked pair frozen before outcomes; finite order diagnostic.')
save(panel('capacity','mean conditional score curvature'),'score_curvature',['L17','Q02','Q34'],'Conditional beta² Var(score), not physical heat capacity or transition temperature.')
save(panel('chi_top','B Var(top-pair contrast)'),'fluctuation_susceptibility',['L01','L17','Q02','Q34'],'Root-conditioned fluctuation susceptibility; not a response/FDT identification.')
save(panel('trace','B Tr(Cov occupancy)',scale='B'),'finite_budget_covariance',['L14','Q16','Q34'],'Independent B16/64; exact fixed-total covariance; not a thermodynamic limit.')
fig,ax=plt.subplots(2,2,figsize=(12,8))
for arm in (0,1):
 for j,B in enumerate(BS):
  a=ax[arm,j]
  for L in LS:
   z=[get(L,arm,B,b) for b in BT];y=[x['ratio'] for x in z];lo=[x['ratio_bootstrap']['lo'] for x in z];hi=[x['ratio_bootstrap']['hi'] for x in z]
   a.plot(BT,y,'o-',label=f'L={L}');a.fill_between(BT,lo,hi,alpha=.09,color=a.lines[-1].get_color())
   a.scatter(0,z[0]['null_ratio'],marker='x',color=a.lines[-1].get_color())
  a.set(title=f'{names[arm]} | B={B}',xlabel='selector beta',ylabel='low / (low+high) variance');a.legend(fontsize=7)
save(fig,'open_mode_ratio',['L01','L17','Q02','Q34'],'Connected DCT mode ratio; fixed total excludes ordinary zero-mode xi estimator. Paired root bootstrap, not RG-invariant theorem.')
fig,ax=plt.subplots(2,2,figsize=(12,8))
for arm in (0,1):
 for j,B in enumerate(BS):
  a=ax[arm,j]
  for L in LS:
   z=[get(L,arm,B,b) for b in BT];y=[x['binder'] for x in z];lo=[x['binder_bootstrap']['lo'] for x in z];hi=[x['binder_bootstrap']['hi'] for x in z]
   a.plot(BT,y,'o-',label=f'L={L}');a.fill_between(BT,lo,hi,alpha=.07,color=a.lines[-1].get_color());a.scatter(0,z[0]['null_binder'],marker='x',color=a.lines[-1].get_color())
  a.axhline(0,ls=':',color='k',lw=.7);a.set(title=f'{names[arm]} | B={B}',xlabel='selector beta',ylabel='Binder-type cumulant ratio');a.legend(fontsize=7)
save(fig,'binder_ratio',['L01','L17','Q02','Q34'],'Distinct-pair moment estimators, ratio of root means; broad uncertainty retained, no universal Binder crossing assumed.')
fig,ax=plt.subplots(2,3,figsize=(14,7))
for arm in (0,1):
 for j,(metric,title) in enumerate([('trace_B','B trace'),('ratio','mode ratio'),('binder','Binder-type ratio')]):
  a=ax[arm,j]
  for k,B in enumerate(BS):
   vals=[]
   for L in LS:
    g=get(L,arm,B,0.);n=next(x for x in Q['nulls'] if (x['L'],x['arm'],x['B'])==(L,arm,B))[metric]
    v=B*g['means']['trace'] if metric=='trace_B' else g[metric];ref=B*g['means']['null_trace'] if metric=='trace_B' else g['null_'+metric]
    vals.append([v,ref,n['lo'],n['hi']])
   v=np.array(vals);off=(-.12 if B==16 else .12);x=np.array(LS)+off
   a.plot(x,v[:,0],'o-',label=f'native B{B}');a.plot(x,v[:,1],':',color=a.lines[-1].get_color());a.fill_between(x,v[:,2],v[:,3],alpha=.12,color=a.lines[-1].get_color())
  a.set(title=names[arm]+' | '+title,xlabel='L',ylabel=title);a.legend(fontsize=7)
save(fig,'exact_iid_null',['L01','L17','Q02','Q34'],'Beta=0 exact uniform multinomial reference plus finite-rep synthetic null (2000 cohorts per condition).')
fig,ax=plt.subplots(2,2,figsize=(12,8))
for arm in (0,1):
 for j,B in enumerate(BS):
  a=ax[arm,j]
  for k,b in enumerate(BT):
   z=[get(L,arm,B,b) for L in LS]
   for key,ls in [('trace','-'),('martingale','--'),('drift',':'),('cross','-.')]:
    a.plot(LS,[B*g['means'][key] for g in z],ls,marker='o' if key=='trace' else None,label=f'beta={b:.3g} {key}')
  a.set(title=f'{names[arm]} | B={B}',xlabel='L',ylabel='B × covariance contribution');a.legend(fontsize=5,ncol=2)
save(fig,'covariance_decomposition',['L14','Q16','Q34'],'Exact sample drift/martingale/cross decomposition, with independent budgets.')
fig,ax=plt.subplots(2,2,figsize=(12,8))
for arm in (0,1):
 for j,B in enumerate(BS):
  a=ax[arm,j];L=21
  for b in BT:
   q=get(L,arm,B,b);cnt=np.asarray(q['radial_pair_counts']);z=np.asarray(q['radial_sum']);w=np.asarray(q['radial_null_sum']);keep=cnt>0;r=np.arange(len(cnt))[keep]
   a.plot(r/L,B*z[keep]/cnt[keep],label=f'beta={b:.3g}')
   if b==0:a.plot(r/L,B*w[keep]/cnt[keep],'k:',label='uniform null')
  a.set(title=f'L21 {names[arm]} | B={B}',xlabel='Manhattan distance / L',ylabel='B × connected pair covariance');a.set_yscale('symlog',linthresh=1e-6);a.legend(fontsize=7)
save(fig,'radial_connected_covariance',['L14','L17','Q34'],'Legal ordered pair shells; fixed-total cancellation, no exponential xi inferred.')
fig,ax=plt.subplots(2,2,figsize=(12,8))
for arm in (0,1):
 for j,B in enumerate(BS):
  a=ax[arm,j];z=[r for r in Q['crossings'] if r['arm']==arm and r['B']==B]
  for k,ob in enumerate(['ratio','binder']):
   w=[x for x in z if x['observable']==ob];xx=np.arange(4)+(k-.5)*.22
   a.bar(xx,[x['bootstrap_fraction_any'] for x in w],.22,label=ob)
  a.set_xticks(range(4),['13/15','15/17','17/19','19/21']);a.set_ylim(0,1);a.set(title=f'{names[arm]} | B={B}',xlabel='adjacent size pair',ylabel='bootstrap fraction any crossing');a.legend(fontsize=8)
save(fig,'crossing_stability',['L01','L17','Q02'],'All piecewise-linear sampled-grid crossings, not transition estimates; bootstrap prevalence and multiple crossing retained.')
fig,ax=plt.subplots(2,2,figsize=(12,7))
for a,ob,L in zip(ax.flat,['ratio','ratio','log_B_trace','log_B_trace'],[13,21,13,21]):
 z=[r for r in Q['heldout_fits'] if r['observable']==ob and r['holdout']==[L]]
 for j,k in enumerate(['analytic','crossover','scaling']):
  w=[next(x for x in r['models'] if x['model']==k)['test_rmse'] for r in z]
  a.bar(np.arange(len(z))+(j-1)*.24,w,.24,label=k)
 a.set_xticks(range(len(z)),[f"{'dense' if r['arm'] else 'fixed'} B{r['B']}" for r in z],rotation=15);a.set(title=f'{ob} | held out L{L}',ylabel='prediction RMSE');a.legend(fontsize=8)
save(fig,'heldout_model_comparison',['L28','Q18','Q34'],'Separate analytic/crossover/critical-type descriptive candidates; fit on other sizes; no exponent admission.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
for L in LS:
 z=sorted([x for x in Q['aux'] if x['kind']=='cpuct' and x['L']==L],key=lambda x:x['c'])
 for a,k in zip(ax,['top_pair_abs','entropy','chi_top']):
  v=np.array([q['ci'][k] for q in z]);a.errorbar([q['c'] for q in z],v[:,0],yerr=[v[:,0]-v[:,1],v[:,2]-v[:,0]],marker='o',label=f'L{L}')
for a,k in zip(ax,['pair contrast','occupancy entropy','fluctuation susceptibility']):a.set(xlabel='c_puct',ylabel=k,title='dense B16 beta=10/3');a.legend(fontsize=7)
save(fig,'cpuct_scan',['L01'],'Direct c_puct scan on independent runs, not a phase boundary.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
z=read(S/'PULSE_EXACT76.json')['cases']
obs=np.array([r['observed_difference_count'] for r in z]);exp=np.array([r['expected_difference_count'] for r in z]);pi=np.array([r['predictive_count_95'] for r in z])
ax[0].vlines(LS,pi[:,0],pi[:,1],color='0.65',lw=4,label='conditional 95% predictive range')
ax[0].plot(LS,obs,'o',label='observed count difference')
ax[0].plot(LS,exp,'x--',label='conditional expectation')
ax[0].set(xlabel='L',ylabel='paired last-wave count difference',title='128 paired draws per size');ax[0].legend(fontsize=8)
ax[1].plot(LS,[r['p_zero'] for r in z],'o-');ax[1].set(xlabel='L',ylabel='conditional P(zero difference)',ylim=(0,1),title='Exact conditional rare-event calibration')
save(fig,'size_source_response',['L01','L17','Q02','Q34'],'Actual paired last-wave count difference, exact conditional Poisson-binomial predictive ranges under ideal IID uniforms; not population confidence intervals or equilibrium FDT.')
fig,ax=plt.subplots(2,2,figsize=(12,7))
for arm in (0,1):
 for j,B in enumerate(BS):
  a=ax[arm,j]
  for k,ob in enumerate(['capacity','chi_top','chi_fixed']):
   z=[x for x in Q['peaks'] if (x['arm'],x['B'],x['observable'])==(arm,B,ob)];a.plot([x['L'] for x in z],[x['beta'] for x in z],'o-',label=ob)
  a.set(title=f'{names[arm]} | B={B}',xlabel='L',ylabel='sampled-grid maximum beta');a.set_yticks(BT);a.legend(fontsize=7)
save(fig,'sampled_peak_locations',['L17','Q02','Q34'],'Sampled grid maxima only, boundary peaks not resolved critical peaks.')
assert len(figs)==14 and len({f['png_sha'] for f in figs})==14
dump(S/'PLOT_MANIFEST76.json',dict(figures=figs,animations=[],sources={k:sha(S/k) for k in ['CRITICAL76.json','INDEPENDENT76.json','PULSE_EXACT76.json','plot76.py']+ [f'AUDIT_L{L}.json' for L in LS]}))
print('PLOTS',len(figs),flush=True)
