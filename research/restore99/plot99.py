from fit99 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess,io
Q=read(S/'SUMMARY99.json');F=read(S/'FITS99.json');G=Q['groups'];fields=Q['fields'];figs=[];animations=[]
def ix(k):return fields.index(k)
def get(L,co,arm,bi,h=64,bud=64):return next(x for x in G if (x['L'],x['cohort'],x['arm'],x['bi'],x['h'],x['budget'])==(L,co,arm,bi,h,bud))
def fit(arm,h,name,kind):return next(x for x in F['fits'] if (x['arm'],x['h'],x['observable'],x['model'])==(arm,h,name,kind))
def save(fig,name,ids):
 fig.tight_layout();b=S/'plots'/name;fig.savefig(str(b)+'.png',dpi=150);fig.savefig(str(b)+'.pdf');plt.close(fig)
 a=Path(str(b)+'.png');p=Path(str(b)+'.pdf')
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(p)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,png_sha=sha(a),pdf_sha=sha(p)))
def band(a,t,z,label):
 z=np.array(z);a.plot(t,z[:,0],'o-',ms=3,label=label);a.fill_between(t,z[:,1],z[:,2],alpha=.10)
def legend(a):a.legend(fontsize=7)
# Q14 finite local size slopes, not physical exponents, and sampled finite peaks.
f,ax=plt.subplots(2,4,figsize=(16,7));f.suptitle('Q14 | finite size slopes and sampled score-control peaks (not critical exponents)')
for arm in range(2):
 for j,name in enumerate(TARGETS):
  a=ax[arm,j];g=[z for z in F['slopes'] if (z['cohort'],z['arm'],z['h'],z['observable'])==(2,arm,64,name)]
  for z in g:a.plot([14,16,18,20],z['local'],'o-',label='beta='+str(round(BETAS[z['bi']],3)))
  a.axhline(0,lw=.5,color='gray');a.set(xlabel='Adjacent-size midpoint',ylabel='log-ratio / log-size-ratio',title=f'{name} | arm{arm}');legend(a)
save(f,'Q14_finite_effective_size_slopes_partial',['Q14','Q31'])
# Q15/Q30 honest collapse diagnostics: show same fitted curve and genuinely independent L21.
f,ax=plt.subplots(2,4,figsize=(16,7));f.suptitle('Q15/Q30 | conditional collapse coordinates, trained L13-19 and tested L21')
for arm in range(2):
 for j,kind in enumerate(MODELS):
  a=ax[arm,j];name='KL_norm';m=fit(arm,64,name,kind);co=np.array(m['coef']);par=m['params']
  for L in LS:
   for cohort in ([0] if L<21 else [1,2]):
    g=sorted([v for v in G if (v['L'],v['cohort'],v['arm'],v['budget'],v['h'])==(L,cohort,arm,64,64) and v['bi']<4],key=lambda z:z['bi'])
    A=design(g,kind,par).mean(1);x=A[:,1];y=np.array([v['ci'][0][ix(name)] for v in g])-17/L*co[3]
    a.plot(x,y,'o-' if cohort==0 else 's--',ms=3,label=f'L{L}/c{cohort}')
  x=np.linspace(0,1,100);a.plot(x,co[0]+co[1]*x+co[2]*x*x,'k:',lw=1.5,label='unclipped fit')
  a.set(xlabel='Mean transformed coordinate (diagnostic)',ylabel='KL/log n minus size term',title=f'{kind}, arm {arm}');legend(a)
save(f,'Q15_Q30_heldout_collapse_diagnostic_partial',['Q15','Q30'])
# Q16 exact tail decomposition and actual categorical moment comparison.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q16 | actual finite categorical scaling functions, not equilibrium magnetization')
for arm in range(2):
 for L in LS:
  gs=[get(L,2,arm,i) for i in range(4)]
  for j,k in enumerate(['KL_norm','concentration','Fisher']):band(ax[arm,j],BETAS,[np.array(g['ci'])[:,ix(k)] for g in gs],f'L{L}')
 for j,k in enumerate(['KL/log n','Drift concentration','Frozen-score Fisher']):ax[arm,j].set(xlabel='Actual score beta',ylabel=k,title=f'fresh B64 arm {arm}');legend(ax[arm,j])
save(f,'Q16_actual_finite_scaling_functions',['Q16','Q31'])
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q16 | matched-peak diagnostic: hidden tail structure and one-draw moments')
for arm in range(2):
 for L in LS:
  gs=[get(L,2,arm,i) for i in range(4)]
  band(ax[arm,0],BETAS,[np.array(g['ci'])[:,ix('tail_KL')] for g in gs],f'L{L}')
  band(ax[arm,1],BETAS,[np.array(g['ci'])[:,ix('spike_TV')] for g in gs],f'L{L}')
  ax[arm,2].plot(BETAS,[g['binder_ci_valid_roots'][0][0] for g in gs],'o-',label=f'A L{L}',ms=3);ax[arm,2].plot(BETAS,[g['binder_ci_valid_roots'][2][0] for g in gs],'--',label=f'spike L{L}',lw=.8)
 for j,k in enumerate(['Tail KL (nat)','TV from matched peak','Centered one-draw U4']):ax[arm,j].set(xlabel='Actual score beta',ylabel=k,title=f'fresh B64 arm {arm}');legend(ax[arm,j])
save(f,'Q16_peak_tail_and_categorical_moments',['Q16','Q34'])
f,ax=plt.subplots(1,3,figsize=(13,4));f.suptitle('Q16 | analytic one-spike null: exact collapse and finite peak without spatial interactions')
for n in [16,64,169,441,1000,1000000]:
 g=[v for v in Q['controls'] if v['n']==n]
 ax[0].plot([v['beta'] for v in g],[v['M'] for v in g],label=f'n={n:g}')
 ax[1].plot([v['x'] for v in g],[v['Fisher'] for v in g],label=f'n={n:g}')
 ax[2].plot([v['x'] for v in g],[v['U4'] for v in g],label=f'n={n:g}')
for a,x,y in zip(ax,['beta (gap=1)','beta-log(n-1)','beta-log(n-1)'],['Peak mass','Fisher','Centered Bernoulli U4']):a.set(xlabel=x,ylabel=y);legend(a)
ax[2].set_yscale('symlog',linthresh=1)
save(f,'Q16_analytic_entropy_competition_null',['Q14','Q15','Q16','Q30'])
# Q31 independent B16 vs same-path prefix and independent size data.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q31 | separate budget and size comparators; no physical correlation length or spectral gap')
for arm in range(2):
 for j,k in enumerate(['X_KL','geometric_spread_A','Fisher']):
  a=ax[arm,j]
  for co,bu,h,l in [(0,16,16,'development B16'),(0,64,16,'development prefix16'),(0,64,64,'development B64'),(1,64,64,'external B64'),(2,64,64,'fresh B64')]:
   g=[get(L,co,arm,3,h,bu) for L in LS];band(a,LS,[np.array(v['ci'])[:,ix(k)] for v in g],l)
  a.set(xlabel='Board L',ylabel=k,title=f'beta16, arm {arm}');legend(a)
save(f,'Q31_independent_budget_size_partial',['Q31'])
# Q18/Q34 report all held-size-out results and selected parameters, not exponent admission.
f,ax=plt.subplots(2,4,figsize=(16,7));f.suptitle('Q18/Q34 | heldout L21 error; no selection on test data')
for arm in range(2):
 for j,name in enumerate(TARGETS):
  a=ax[arm,j];z=[g for g in F['comparisons'] if (g['arm'],g['observable'])==(arm,name)];z.sort(key=lambda g:(g['h'],g['cohort']));t=np.arange(len(z))
  for kind in MODELS:a.plot(t,[g['RMSE'][kind] for g in z],'o-',ms=3,label=kind)
  a.set(xticks=t,xticklabels=[f"H{g['h']}/c{g['cohort']}" for g in z],ylabel='Group RMSE',title=f'{name} arm{arm}');a.tick_params(axis='x',rotation=40);legend(a)
save(f,'Q18_Q34_heldout_model_errors',['Q18','Q34','Q15','Q30'])
# Q34 dyadic KL actual vs matched peak comparison, no pooling of size cohorts.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q34 | observed and peak-matched dyadic information flow, fresh B64')
for arm in range(2):
 for j,L in enumerate([13,17,21]):
  a=ax[arm,j];g=get(L,2,arm,3);v=np.array(g['scale_ci']);t=Q['scales']
  for k,l in enumerate(['KL actual','KL matched peak','TV actual vs peak']):band(a,t,v[:,:,k].T,l)
  a.set(xscale='log',xticks=t,xticklabels=t,xlabel='b (cells)',ylabel='KL nat or TV',title=f'L{L}, beta16, arm{arm}');legend(a)
save(f,'Q34_tail_structure_across_scales',['Q34','Q16'])
# Q17 finite control scan; fixed sampled beta, no interpolation to critical temperature.
def control_frame(bi):
 f,ax=plt.subplots(2,3,figsize=(12,7));f.suptitle(f'Q17 | fresh B64 finite score-control scan, beta={BETAS[bi]:.3g}')
 for arm in range(2):
  for j,k in enumerate(['KL_norm','concentration','Fisher']):
   a=ax[arm,j];g=[get(L,2,arm,bi) for L in LS];band(a,LS,[np.array(v['ci'])[:,ix(k)] for v in g],f'arm {arm}')
   a.set(xlabel='Board L',ylabel=k,ylim=(-.08,1.05 if j<2 else .55),xticks=LS);legend(a)
 return f
# Q33 actual R92 signed radial shell mass, fixed fresh beta16 dense, h prefix same trajectories.
def radial_frame(h):
 f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle(f'Q33 | actual signed covariance shell mass, fresh dense beta16 B64 prefix h={h}')
 for j,L in enumerate(LS):
  a=ax.flat[j];g=next(z for z in Q['radial_bins'] if z['L']==L and z['h']==h);v=np.array(g['bin_ci']);r=g['bin_centers']
  for k,name in [(0,'raw'),(1,'drift'),(2,'innovation'),(3,'cross'),(4,'predictable'),(5,'uniform null')]:a.plot(r,v[0,k],'o-',ms=2,label=name)
  a.axhline(0,color='gray',lw=.5);lim=max(abs(np.array(z['bin_ci'])[0]).max() for z in Q['radial_bins'] if z['L']==L);a.set(xlabel='r/L',ylabel='Signed bin mass',title=f'L{L}',ylim=(-1.08*lim,1.08*lim));legend(a)
 a=ax.flat[5];a.axis('off');a.text(.05,.8,'Same B64 histories at each prefix.\n8 root blocks, 2 repetitions/root.\nNo stationary correlation length.\nSelf shell excluded.',transform=a.transAxes,va='top')
 return f
save(control_frame(3),'Q17_finite_control_snapshot',['Q17'])
save(radial_frame(64),'Q33_actual_multisize_covariance_snapshot',['Q33','Q34'])
def gif(name,id0,frames,fn,clock):
 ims=[]
 for x in frames:
  f=fn(x);f.tight_layout();b=io.BytesIO();f.savefig(b,format='png',dpi=90);b.seek(0);ims.append(Image.open(b).convert('RGB'));plt.close(f)
 p=S/'plots'/name;ims[0].save(p,save_all=True,append_images=ims[1:],duration=850,loop=0)
 with Image.open(p) as im:
  assert im.n_frames==len(frames);hs=[]
  for i in range(im.n_frames):im.seek(i);hs.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
  assert len(set(hs))==len(frames)
 animations.append(dict(id=id0,path='plots/'+p.name,sha256=sha(p),frames=len(frames),frame_rgb_sha256=hs,clock=clock))
gif('Q17_finite_score_control.gif','Q17',list(range(4)),control_frame,BETAS)
gif('Q33_actual_spatial_prefix.gif','Q33',[16,32,64],radial_frame,[16,32,64])
# Explicit one-draw Legendre dual, no physical QFT 1PI claim.
T=read(S/'ACTION99.json')
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q16 | one-draw categorical source dual; parameterized by fixed J')
for arm in range(2):
 for j,L in enumerate([13,17,21]):
  a=ax[arm,j];g=next(v for v in T['groups'] if (v['L'],v['cohort'],v['arm'],v['bi'],v['budget'],v['h'])==(L,2,arm,3,64,64));v=np.array(g['ci'])[0]
  for k,label in enumerate(T['distributions']):a.plot(v[k,:,1],v[k,:,3],'o-',label=label)
  a.set(xlabel='Mean tilted mode (root average)',ylabel='Gamma (nat, root average)',title=f'L{L}, beta16, arm{arm}');legend(a)
save(f,'Q16_finite_categorical_legendre_dual',['Q16','Q34'])
dump(S/'PLOT_MANIFEST99.json',dict(figures=figs,animations=animations,sources={k:sha(S/k) for k in ['SUMMARY99.json','FITS99.json','SCALE99.json','SCALE99.npz','INDEPENDENT99.json','ORIGINAL_SOURCE99.json','ACTION99.json','ACTION99.npz']},scope='Actual finite scaling with analytic null, independent test, separate clocks; no thermodynamic critical point'))
print('PLOTS',len(figs),'GIFS',[(x['id'],x['frames']) for x in animations],flush=True)
