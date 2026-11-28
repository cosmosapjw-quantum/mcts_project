from measure103 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
M=read(S/'MEASURE103.json');F=read(S/'FIT103.json');Q=read(S/'SUMMARY103.json');R=read(S/'PRIOR103.json');X=read(S/'CROSS103.json')
G=M['groups'];gf={(g['L'],g['cohort'],g['arm'],g['bi'],g['budget'],g['h']):g for g in G};bf={(g['L'],g['cohort'],g['arm'],g['bi'],g['budget'],g['h']):g for g in F['groups']}
figs=[]
def save(fig,n,ids):
 fig.tight_layout();b=S/'plots'/n;fig.savefig(str(b)+'.png',dpi=145);fig.savefig(str(b)+'.pdf');plt.close(fig)
 p=Path(str(b)+'.png');a=Path(str(b)+'.pdf')
 with Image.open(p) as im:im.verify()
 subprocess.run(['pdfinfo',str(a)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=n,ids=ids,png_sha=sha(p),pdf_sha=sha(a)))
def legend(a):a.legend(fontsize=7)
def band(a,x,z,l):
 z=np.asarray(z);a.plot(x,z[:,0],'o-',ms=3,label=l);a.fill_between(x,z[:,1],z[:,2],alpha=.11)
# Exact observed score control, identified draw counts and curvature.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('L03 | recorded conditional score slope and identification, no physical temperature')
for j,betaindex in enumerate([0,1,3]):
 for L in LS:
  gs=[gf[L,2,1,betaindex,64,h] for h in HS]
  ax[0,j].plot(HS,[g['beta'] for g in gs],'o-',label=f'L{L}',ms=3)
  band(ax[1,j],HS,[np.array(g['path_ci'])[:,2] for g in gs],f'L{L}')
 ax[0,j].set(title=f'Actual beta index {betaindex}',xlabel='Backup prefix',ylabel='Recorded beta (identified draws)')
 ax[1,j].set(xlabel='Backup prefix',ylabel='Mean frozen-score Fisher');legend(ax[1,j])
save(f,'L03_identified_conditional_law_and_information',['L03','L05','L34'])
# Actual A,X and fixed features, KL residual and mixture gap.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('L25 | full legal support projections of A and X, different fixed features')
for j,bi in enumerate([0,1,3]):
 for a,l in enumerate(M['features']):
  for t,lab in [(0,'A'),(1,'X')]:
   vv=[bf[L,2,1,bi,64,64]['beta_ci_valid'][a][t] for L in LS]
   ax[0,j].plot(LS,[v[0] for v in vv],'o-' if t==0 else 'x--',ms=3,label=f'{lab}, {l}')
   v=[np.asarray(gf[L,2,1,bi,64,64]['ci'])[0,a,t,3] for L in LS]
   ax[1,j].plot(LS,v,'o-' if t==0 else 'x--',ms=3,label=f'{lab}, {l}')
 ax[0,j].axhline(gf[13,2,1,bi,64,64]['beta'],ls=':',color='black',label='Actual control');ax[0,j].set(title=f'Fresh dense beta index {bi}',xlabel='L',ylabel='Bounded projection slope');legend(ax[0,j]);ax[1,j].set(xlabel='L',ylabel='Residual KL (nat)');legend(ax[1,j])
save(f,'L25_projection_feature_and_target_comparison',['L03','L25','L34'])
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('L25 | temporal mixture gap and static feature projection residual')
for j,bi in enumerate([0,1,3]):
 for L in LS:
  g=[gf[L,2,1,bi,64,h] for h in HS]
  band(ax[0,j],HS,[np.array(v['path_ci'])[:,3] for v in g],f'L{L}')
  # average conditional KL to q_b for the A projection of mean score
  band(ax[1,j],HS,[np.array(v['ci'])[:,1,0,10] for v in g],f'L{L}')
 ax[0,j].set(title=f'Fresh dense beta index {bi}',xlabel='Prefix',ylabel='Jensen gap (nat)');ax[1,j].set(xlabel='Prefix',ylabel='Mean conditional KL to static q');legend(ax[0,j]);legend(ax[1,j])
save(f,'L25_temporal_mixture_decomposition',['L25','L05'])
# L26 budget comparison independent B16 and B64 prefixes.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('L26 | actual independent B16/B64 versus correlated prefixes')
for j,bi in enumerate([0,1,3]):
 for k,lab in [(0,'A'),(1,'X')]:
  a=ax[k,j]
  for co,bud,h,l in [(0,16,16,'Dev B16'),(0,64,16,'Dev B64 prefix16'),(0,64,64,'Dev B64 full'),(2,64,64,'Fresh B64 full')]:
   band(a,LS,[bf[L,co,1,bi,bud,h]['beta_ci_valid'][0][k] for L in LS],l)
  a.set(title=f'{lab}: beta index {bi}',xlabel='L',ylabel='Endpoint-Q slope');legend(a)
save(f,'L26_independent_budget_vs_prefix',['L26','L04'])
# L04 actual size holdout, dev selected vs model alternatives.
f,ax=plt.subplots(2,4,figsize=(16,7));f.suptitle('L04 | development-only scaling and L21 heldout signed-slope errors')
for arm in range(2):
 for bi in range(4):
  a=ax[arm,bi]
  for t,ls in [(0,'o-'),(1,'s--')]:
   s=next(v for v in Q['scaling_selection'] if (v['arm'],v['bi'],v['target'])==(arm,bi,t));a.plot(range(4),[s['fresh64_RMSE'][k] for k in F['models']],ls,label=('A' if t==0 else 'X')+' selected '+s['selected'],ms=3)
  a.set(title=f'Arm {arm}, beta index {bi}',xticks=range(4),xticklabels=['const','log B','log L','joint'],ylabel='Fresh L21 B64 absolute error');a.tick_params(axis='x',rotation=30);legend(a)
save(f,'L04_heldout_scaling_competition',['L04','L26','L34'])
# Heldout conditional NLL and KL grid, separate from occupancy.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('L03/L25 | B16-trained conditional slope candidates, L21 independent test')
for j,bi in enumerate([1,2,3]):
 for arm in range(2):
  v=[g for g in F['conditional'] if (g['cohort'],g['bi'],g['arm'])==(2,bi,arm)]
  v.sort(key=lambda g:g['h']);t=[g['h'] for g in v]
  for k,l in [(0,'score'),(1,'Q only'),(2,'known score')]:
   for ii,key in [(0,0),(1,1)]:ax[ii,j].plot(t,[g['scores_ci'][0][k][key] for g in v],'o-' if arm==0 else 's--',ms=3,label=f'{l} arm{arm}')
 for ii,l in [(0,'Actual choice NLL'),(1,'Conditional law KL')]:ax[ii,j].set(title=f'Fresh beta index {bi}',xlabel='B64 prefix',ylabel=l);legend(ax[ii,j])
save(f,'L03_L25_heldout_score_Q_likelihood',['L03','L25','L05'])
# Typed measurement quality, not summed across units.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('L05/L34 | typed measurement quality and projection status, no physical T count')
s=np.asarray(Q['status_counts']);cnt=Q['counts']
ax[0,0].bar(['Identified','Flat'],[cnt['R87_identified'],cnt['R87_flat']]);ax[0,0].set(ylabel='R87 selections')
ax[0,1].bar(['Identified','Flat'],[cnt['R79_identified'],cnt['R79_flat']]);ax[0,1].set(ylabel='R79 waves')
for t in range(2):
 a=ax[1,t];bottom=np.zeros(3)
 for k in range(6):
  v=s[:,t,k];a.bar(range(3),v,bottom=bottom,label=M['projection_status'][k]);bottom+=v
 a.set(xticks=range(3),xticklabels=['Q end','Mean score','Last score'],ylabel='Projection cases',title='A' if t==0 else 'X');legend(a)
ax[0,2].bar(['R87 prefixes','R79 endpoint cuts'],[cnt['prefixes'],cnt['prior_projection_cases']]);ax[0,2].set(ylabel='Different typed units');ax[0,2].tick_params(axis='x',rotation=15)
neg=np.asarray(cnt['negative_projection']);ax[1,2].bar(np.arange(3)-.17,neg[:,0],.34,label='A');ax[1,2].bar(np.arange(3)+.17,neg[:,1],.34,label='X');ax[1,2].set(xticks=range(3),xticklabels=['Q end','Mean score','Last score'],ylabel='Negative slope < -1e-9');legend(ax[1,2])
save(f,'L05_L34_typed_quality_and_identification',['L05','L34','L25'])
# Same / cross-history moment.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('L34 | endogenous endpoint features: same vs independent repetition')
for a,l in enumerate(M['features']):
 for j,bi in enumerate([0,1,3,4]):
  r=next(v for v in Q['cross_history']['fresh'] if v['bi']==bi);v=r['mean'][a]
  ax[0,a].bar(j-.17,v[0],.34,color='C0',label='Same' if j==0 else None);ax[0,a].bar(j+.17,v[1],.34,color='C1',label='Cross' if j==0 else None)
  gg=[x for x in X['rows'] if (x['cohort'],x['arm'],x['bi'],x['budget'],x['h'])==(2,1,bi,64,64)]
  ax[1,a].plot([g['L'] for g in gg],[g['ci'][0][a][2] for g in gg],'o-',ms=3,label=f'beta index {bi}')
 ax[0,a].set(title=l,xticks=range(4),xticklabels=['0','10/3','16','log L²'],ylabel='(X-A) dot feature');ax[0,a].axhline(0,color='gray',lw=.5);legend(ax[0,a])
 ax[1,a].set(xlabel='L',ylabel='Same minus cross');ax[1,a].axhline(0,color='gray',lw=.5);legend(ax[1,a])
save(f,'L34_endogeneity_cross_history',['L34','L25','L05'])
dump(S/'PLOT_MANIFEST103.json',dict(figures=figs,animations=[],sources={k:sha(S/k) for k in ['MEASURE103.json','MEASURE103.npz','PRIOR103.json','FIT103.json','CROSS103.json','SUMMARY103.json','INDEPENDENT103.json','ORIGINAL_SOURCE103.json']},scope='Actual score-law, full-support projections, heldout scaling and typed measurements; no physical temperature.'))
print('PLOTS',len(figs),flush=True)
