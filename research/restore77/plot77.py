from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
Q=read(S/'SUMMARY77.json');R=read(S/'RG77.json');C=read(S/'CLOSURE77.json');LS=[13,15,17,19,21];BT=[0.,10/3,16.];names=['clock','count','count_mark','count_mark_prob','count_mark_prob_lag'];short=['clock','count','+mark','+prob','+lag'];figs=[]
def save(fig,name,ids,scope):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=150);fig.savefig(str(p)+'.pdf');plt.close(fig)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 assert b.read_bytes()[:4]==b'%PDF'
 figs.append(dict(figure=name,ids=ids,scope=scope,png_sha=sha(a),pdf_sha=sha(b)))
def select(beta,model,kind):return next(q for q in Q['heldout'] if (q['beta'],q['model'],q['kind'])==(beta,model,kind))
fig,ax=plt.subplots(1,2,figsize=(11,4))
e=np.linspace(0,2,201);ax[0].plot(e,-.5+e-e*e);ax[0].axhline(0,ls=':',color='k');ax[0].set(xlabel='original epsilon',ylabel='beta_g / g at proposed point',title='Original Wilson–Fisher candidate: no root')
t=np.geomspace(.1,20,200);s=np.sqrt(1-4/t+0j)
ax[1].semilogx(t,((1+s)/2).real,label='Re s+');ax[1].semilogx(t,((1-s)/2).real,label='Re s−');ax[1].semilogx(t,-1/t**2,label='s_c');ax[1].semilogx(t,3/t,label='s_lambda at epsilon=1 (example)');ax[1].axhline(0,ls=':',color='k');ax[1].set(xlabel='frozen tau',ylabel='stability eigenvalue real part',title='Actual frozen-tau stationary candidate',ylim=(-4,12));ax[1].legend(fontsize=8)
save(fig,'original_fixed_point_audit',['L37','Q09','Q13'],'Exact original beta-function residual and conditional frozen-tau unstable point; lambda eig example epsilon=1; not native physical RG.')
fig,ax=plt.subplots(1,2,figsize=(11,4));old=R['old_witnesses'];ks=['all','mark_bin_equal','mark_prob_bin_equal','all_bin_equal'];nn=[len(old['rows'])]+[old['counts'][k] for k in ks[1:]];mx=[old['max_tv']]+[max(x['tv'] for x in old['rows'] if x[k]) for k in ks[1:]]
ax[0].bar(range(4),nn);ax[0].set(ylabel='same-state nonzero-law pairs',title='R72 exact count collision subset');ax[1].bar(range(4),mx);ax[1].set(ylabel='maximum conditional law TV',title='Retained counterexamples')
for a in ax:a.set_xticks(range(4),['count','+mark bin','+prob bin','+lag bin'],rotation=15)
save(fig,'historical_augmented_collisions',['L13','L37','L38'],'Reevaluation of 112 authenticated R72 count collisions; only nested bin comparisons, exact marks unique in this subset.')
fig,ax=plt.subplots(1,3,figsize=(13,4));ks=['exact_three_sector_count']+names[1:]
for a,b in zip(ax,BT):
 z=[next(x for x in Q['collisions'] if x['beta']==b and x['model']==k) for k in ks];a.bar(np.arange(5)-.16,[q['pairs'] for q in z],.32,label='same state');a.bar(np.arange(5)+.16,[q['nonzero'] for q in z],.32,label='nonzero TV');a.set_xticks(range(5),['exact count','count bin','+mark','+prob','+lag'],rotation=20);a.set(title=f'beta={b:.3g}',ylabel='heldout collision pairs');a.legend(fontsize=7)
save(fig,'fresh_nonlumpability',['L13','L37','L38'],'New geometry, same root/time collisions; pair counts depend on state resolution and do not estimate population prevalence.')
fig,ax=plt.subplots(2,3,figsize=(13,7))
for j,b in enumerate(BT):
 for i,k in enumerate(['homogeneous','clock_conditioned']):
  z=[select(b,m,k) for m in names];a=ax[i,j];a.plot(range(5),[q['one'][0] for q in z],'o-',label='next count-bin Brier');a.plot(range(5),[q['one'][1] for q in z],'s--',label='joint emission MSE');a.plot(range(5),[q['one'][2] for q in z],'^:',label='unseen-state fraction');a.set_xticks(range(5),short,rotation=15);a.set(title=f'{k}, beta={b:.3g}');a.legend(fontsize=7)
save(fig,'heldout_one_step',['L13','L35','L38'],'New geometry one-step common targets, separate conditional models, equal-stratum averages.')
fig,ax=plt.subplots(2,3,figsize=(13,7))
for j,b in enumerate(BT):
 for i,k in enumerate(['homogeneous','clock_conditioned']):
  a=ax[i,j]
  for m in names:
   z=select(b,m,k)['CK'];a.plot([q['h'] for q in z],[q['means'][4] for q in z],'o-',label=short[names.index(m)])
  a.set(title=f'{k}, beta={b:.3g}',xlabel='wave horizon',ylabel='count marginal CK TV');a.legend(fontsize=7)
save(fig,'chapman_kolmogorov_defect',['L29','L35','L36','L37'],'Composition of fitted one-step matrices vs direct fitted block laws; not exact lumpability theorem.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
for a,b in zip(ax,BT):
 for i,k in enumerate(['homogeneous','clock_conditioned']):
  z=[select(b,m,k) for m in names];a.bar(np.arange(5)+(i-.5)*.32,[q['mean_W_rmse'] for q in z],.32,label=k)
 a.set_xticks(range(5),short,rotation=15);a.set(title=f'beta={b:.3g}',ylabel='heldout two-vector CGF RMSE');a.legend(fontsize=7)
save(fig,'autonomous_cgf',['L29','L35','Q11','Q13'],'Forward from observed initial state only; actual heldout CGF not fed into kernels.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for b in BT:
 z=[x for x in Q['fixed_points'] if x['beta']==b]
 ax[0].plot(range(5),[q['max_subdominant'] for q in z],'o-',label=f'beta={b:.3g} |eig2|');ax[0].plot(range(5),[q['max_dobrushin'] for q in z],'s--',label=f'beta={b:.3g} delta')
 q=z[-1];ax[1].semilogy(np.arange(7),np.maximum(q['max_dyadic_TV'],1e-16),'o-',label=f'beta={b:.3g}')
ax[0].set_xticks(range(5),short,rotation=15);ax[0].set(ylabel='max contraction diagnostic',title='Supported homogeneous reference kernels');ax[0].legend(fontsize=7)
ax[1].set(xlabel='squaring step m',ylabel='max row TV to stationary projector',title='K^(2^m), full augmented state');ax[1].legend(fontsize=8)
save(fig,'reference_markov_fixed_point',['L35','L36','L37','Q11'],'Stationary projector of smoothed finite learned kernel; mathematical reference, not native RG fixed point.')
fig,ax=plt.subplots(1,2,figsize=(11,4));ref=R['reference_control'];j=np.array(R['source_J'])
for x in ref[::2]:ax[0].plot(j,x['W'],'o-',label=f"block {x['block']}")
ax[0].plot(j,ref[0]['gaussian_W'],'k:',label='Gaussian fixed');ax[0].set(xlabel='J',ylabel='log MGF',title='Independent Rademacher block RG');ax[0].legend(fontsize=8)
ax[1].loglog([x['block'] for x in ref],[-x['k4'] for x in ref],'o-',label='exact −k4');ax[1].loglog([x['block'] for x in ref],[2/x['block'] for x in ref],'k:',label='2 / block');ax[1].set(xlabel='block size',ylabel='absolute fourth cumulant',title='Irrelevant cumulant direction');ax[1].legend()
save(fig,'gaussian_probability_rg',['L37','L38','Q09','Q11'],'Exact independent-sum fixed law and finite binomial control, not physical Wilson–Fisher.')
fig,ax=plt.subplots(2,3,figsize=(13,7))
for j,b in enumerate(BT):
 for i,f in enumerate(['central_residual','mark']):
  z=[x for x in Q['block_rg'] if x['beta']==b and x['field']==f];a=ax[i,j];a.plot([x['h'] for x in z],[x['mean_W_train_error'] for x in z],'o-',label='train IID law');a.plot([x['h'] for x in z],[x['mean_W_same_error'] for x in z],'s--',label='same-test IID law')
  a.set(title=f'{f}, beta={b:.3g}',xlabel='block waves',ylabel='CGF RMSE');a.legend(fontsize=7)
save(fig,'native_block_rg_defect',['L35','L37','L38','Q11'],'Actual aligned late blocks, fixed training normalization; independent-convolution reference errors.')
fig,ax=plt.subplots(2,3,figsize=(13,7))
for j,b in enumerate(BT):
 for i,f in enumerate(['central_residual','mark']):
  z=[x for x in Q['block_rg'] if x['beta']==b and x['field']==f and x['h']<8];a=ax[i,j]
  a.plot([x['h'] for x in z],[x['mean_dependence_error'] for x in z],'o-',label='joint vs marginal product');a.plot([x['h'] for x in z],[x['mean_marginal_asymmetry'] for x in z],'s--',label='marginal asymmetry');a.plot([x['h'] for x in z],[x['mean_abs_adjacent_cov'] for x in z],'^:',label='|adjacent cov|')
  a.set(title=f'{f}, beta={b:.3g}',xlabel='half-block waves',ylabel='finite diagnostic');a.legend(fontsize=7)
save(fig,'adjacent_block_dependence',['L13','L29','L38','Q13'],'Dependence and marginal mismatch separated; mixtures can contribute, no universal population memory estimate.')
assert len(figs)==10 and len({f['png_sha'] for f in figs})==10
dump(S/'PLOT_MANIFEST77.json',dict(figures=figs,animations=[],sources={k:sha(S/k) for k in ['SUMMARY77.json','CLOSURE77.json','RG77.json','INDEPENDENT77.json','ORIGINAL_RG_AUDIT77.json','plot77.py']}))
print('PLOTS',len(figs),flush=True)
