from lag102 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
M=read(S/'MARKOV102.json');L0=read(S/'LAG102.json');N=read(S/'CMI_NULL102.json');Q=read(S/'SUMMARY102.json');G=L0['groups'];figs=[]
def save(f,n,ids):
 f.tight_layout();p=S/'plots'/n;f.savefig(str(p)+'.png',dpi=145);f.savefig(str(p)+'.pdf');plt.close(f)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True);figs.append(dict(figure=n,ids=ids,png_sha=sha(a),pdf_sha=sha(b)))
def legend(a):a.legend(fontsize=7)
def get(L,co,arm,bi,h):return next(x for x in G if (x['L'],x['cohort'],x['arm'],x['bi'],x['budget'])==(L,co,arm,bi,h))
def line(a,g,k,label):
 v=np.asarray(g['lag_ci'])[:,:,k];a.plot(g['lags'],v[0],'o-',ms=3,label=label);a.fill_between(g['lags'],v[1],v[2],alpha=.12)
# L39 root/time covariance, separate empirical mixture and martingale innovation.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('L39 | actual root-clock covariances and innovation lag products, not stationary ACF')
for j,bi in enumerate([0,1,3]):
 for L in LS:
  g=get(L,2,1,bi,64);line(ax[0,j],g,0,f'L{L}');line(ax[1,j],g,4,f'L{L}')
 for i,k in [(0,'Within-root/clock covariance'),(1,'Uncentered innovation cross moment')]:
  a=ax[i,j];a.axhline(0,color='gray',lw=.5);a.set(title=f'Fresh dense beta index {bi}',xlabel='Selection lag',ylabel=k,xticks=LAGS);legend(a)
save(f,'L39_actual_lag_covariance',['L39','L48','L49'])
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('L39 | normalized finite root-clock covariance, undefined roots excluded')
for j,bi in enumerate([0,1,3]):
 for L in LS:
  g=get(L,2,1,bi,64);x=g['lags'];v=np.array([r if r is not None else [np.nan]*3 for r in g['rho_valid_ci']]);ax[0,j].plot(x,v[:,0],'o-',ms=3,label=f'L{L}');ax[0,j].fill_between(x,v[:,1],v[:,2],alpha=.1);ax[1,j].plot(x,g['rho_valid_roots'],'o-',ms=3,label=f'L{L}')
 for i,k in [(0,'Normalized covariance (valid roots)'),(1,'Defined roots / 8')]:ax[i,j].set(title=f'beta index {bi}',xlabel='Selection lag',ylabel=k,xticks=LAGS);legend(ax[i,j])
save(f,'L39_valid_normalized_lag',['L39','L42','L44'])
# L40 empirical triple diagnostics + exact bias baseline.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('L40 | empirical Markovization: CK residual and conditional information')
for j,g in enumerate(range(3)):
 for co in [-1,3,4]:
  a=next(v for v in M['groups'] if (v['family'],v['cohort'])==(g,co));z=np.array([r['metrics'] for r in a['triples']]);ax[0,j].scatter(z[:,4],z[:,2],s=14,label=f'cohort {co}')
  b=[r for r in N['rows'] if (r['family'],r['cohort'])==(g,co)];z=np.array([r['root_ci'][0] for r in b]);ax[1,j].scatter(z[:,1],z[:,0],s=14,label=f'cohort {co}')
 for i,(x,y) in enumerate([('Pinsker upper bound','Weighted CK TV'),('Exact conditional shuffle E[CMI]','Root-conditioned plug-in CMI')]):
  a=ax[i,j];lim=max(a.get_xlim()[1],a.get_ylim()[1]);a.plot([0,lim],[0,lim],'k:',lw=.7);a.set(title=FAMS[g],xlabel=x,ylabel=y);legend(a)
save(f,'L40_CMI_CK_and_finite_sample_null',['L40','L48','L49'])
# L41 analytical conditional law against actual measured counts.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('L41 | genuine width-4 wave binomial law (startup excluded)')
for j,bi in enumerate([0,1,3]):
 for L in LS:
  g=get(L,2,1,bi,64);z=np.array(g['bins_ci'])[0];ax[0,j].plot(range(5),z[0],'o-',ms=3,label=f'L{L} exact');ax[0,j].plot(range(5),z[1],'x--',ms=3,label=f'L{L} observed')
  v=np.array(g['wave_ci']);ax[1,j].errorbar(v[0,3],v[0,2],yerr=[[v[0,2]-v[1,2]],[v[2,2]-v[0,2]]],fmt='o',label=f'L{L}')
 ax[0,j].set(title=f'beta index {bi}',xlabel='Inner-sector count',ylabel='Probability mass',xticks=range(5));legend(ax[0,j])
 a=ax[1,j];lim=max(a.get_xlim()[1],a.get_ylim()[1]);a.plot([0,lim],[0,lim],'k:');a.set(xlabel='Mean conditional p²',ylabel='Observed factorial pair fraction');legend(a)
save(f,'L41_analytic_quiescent_wave_count_law',['L41','L48','L49'])
# L42 actual root-level distributions, no fabricated games or correlation times.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('L42 | observed root-level lag distribution, separate clock drift')
for j,bi in enumerate([0,1,3]):
 for i,k in [(0,0),(1,4)]:
  a=ax[i,j];v=[]
  for L in LS:
   g=get(L,2,1,bi,64);z=np.array(g['lag_root']);v.append(z[:,0,k])
  a.boxplot(v,tick_labels=LS,showfliers=True);a.axhline(0,color='gray',lw=.5);a.set(title=f'beta index {bi}',xlabel='Board L',ylabel='lag1 root-clock cov' if i==0 else 'lag1 innovation cross')
save(f,'L42_observed_root_lag_distributions',['L42','L48'])
# L43 trained kernels at fixed time and direct/composed.
f,ax=plt.subplots(3,3,figsize=(10,10));f.suptitle('L43 | fixed development kernels: direct versus composed')
for j,g in enumerate(range(3)):
 a=M['models'][g];K=np.array(a['clock']);d=np.array(a['direct']['4,8']);c=np.array(a['composed']['4,8'])
 for i,z in enumerate([K[4],d,c]):
  im=ax[i,j].imshow(z,vmin=0,vmax=1,origin='lower');ax[i,j].set(title=f'{FAMS[g]} '+['one step t4','direct t4→8','composed t4→8'][i],xlabel='Future state',ylabel='Present state',xticks=range(9),yticks=range(9));f.colorbar(im,ax=ax[i,j],shrink=.75)
save(f,'L43_clock_direct_composed_kernels',['L43','L40'])
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('L43/L48 | heldout direct vs composed multi-step forecasts')
for j,g in enumerate(range(3)):
 for co,style in [(3,'o-'),(4,'s--')]:
  v=next(x for x in M['external'] if (x['family'],x['cohort'])==(g,co));x=[r['h'] for r in v['multi']]
  for i,k in [(0,0),(1,1)]:
   a=ax[i,j];z=np.array([r['direct_gain_ci'] for r in v['multi']])[:,:,k];a.plot(x,z[:,0],style,label=f'cohort {co}');a.fill_between(x,z[:,1],z[:,2],alpha=.13);a.axhline(0,color='gray',lw=.5);a.set(title=FAMS[g],xlabel='8-backup steps',ylabel='Composed minus direct '+('log loss' if i==0 else 'Brier'),xticks=[2,4,8]);legend(a)
save(f,'L43_heldout_direct_composed_prediction',['L43','L48','L49'])
# L44 algorithmic lag; no fictitious exponential tau.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('L44 partial | clock mixture and finite lag, not environment/system time separation')
for j,bi in enumerate([0,1,3]):
 for L in LS:
  g=get(L,2,1,bi,64);line(ax[0,j],g,2,f'L{L}');line(ax[1,j],g,3,f'L{L}')
 for i,k in [(0,'Clock mixture covariance'),(1,'Pooled covariance')]:ax[i,j].set(title=f'beta index {bi}',xlabel='Selection lag',ylabel=k,xticks=LAGS);ax[i,j].axhline(0,color='gray',lw=.5);legend(ax[i,j])
save(f,'L44_finite_clock_lag_partial',['L44','L39'])
# L49 summary one-step and conditional-null subtraction.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('L49 | independent-root prediction and exact finite-sample CMI baseline')
for j,g in enumerate(range(3)):
 a=ax[0,j]
 for co,off in [(3,-.16),(4,.16)]:
  r=next(v for v in M['external'] if (v['family'],v['cohort'])==(g,co));z=np.asarray(r['one_step']['scores_ci'])[:,:,0];a.errorbar(np.arange(3)+off,z[0],yerr=[z[0]-z[1],z[2]-z[0]],fmt='o',label=f'cohort {co}')
 a.set(title=FAMS[g],xticks=range(3),xticklabels=['homog1','clock1','clock2'],ylabel='Heldout log loss');legend(a)
 a=ax[1,j]
 for co in [3,4]:
  z=[r for r in N['rows'] if (r['family'],r['cohort'])==(g,co)];a.plot(np.arange(len(z)),[r['root_ci'][0][2] for r in z],'o-',ms=2,label=f'cohort {co}')
 a.axhline(0,color='gray',lw=.5);a.set(xlabel='Registered triple index',ylabel='Empirical CMI minus shuffle expectation');legend(a)
save(f,'L49_markov_summary_and_negative_controls',['L49','L48','L40'])
dump(S/'PLOT_MANIFEST102.json',dict(figures=figs,animations=[],sources={k:sha(S/k) for k in ['MARKOV102.json','STATE102.npz','LAG102.json','SELECTION102.npz','CMI_NULL102.json','SUMMARY102.json','INDEPENDENT102.json','ORIGINAL_SOURCE102.json']},scope='Actual nonstationary finite projection, independent root predictions and quiescent wave law; no physical relaxation times.'))
print('PLOTS',len(figs),flush=True)
