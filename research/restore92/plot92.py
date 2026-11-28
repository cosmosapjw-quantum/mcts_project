from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import t as student
import subprocess
s=read(S/'SUMMARY92.json');ls=[13,15,17,19,21];betas=[0,10/3,8,16];hs=[16,32,64];f=[];n=read(S/'INDEPENDENT92.json')
def save(fig,name,ids):
 fig.tight_layout();p=S/'plots'/name
 fig.savefig(str(p)+'.png',dpi=160);fig.savefig(str(p)+'.pdf');plt.close(fig)
 png=Path(str(p)+'.png');pdf=Path(str(p)+'.pdf')
 with Image.open(png) as im:im.verify()
 subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
 f.append(dict(figure=name,ids=ids,png_sha=sha(png),pdf_sha=sha(pdf)))
def key(x):return tuple(x[k] for k in ['L','cohort','arm','bi','budget','h'])
M={key(x):x for x in s['strata']};B={key(x):x for x in s['binned']};T={(x['L'],x['cohort'],x['arm'],x['bi']):x for x in s['temporal']}
assert len(M)==len(B)==400 and len(T)==120
def m(L,arm,bi,h=64,co=2,budget=64):return np.asarray(M[L,co,arm,bi,budget,h]['metric_ci'])
def line(ax,x,y,label=None,**kw):
 y=np.asarray(y);ax.plot(x,y[:,0],'o-',label=label,ms=3,**kw);ax.fill_between(x,y[:,1],y[:,2],alpha=.10)
def metric(ax,arm,j,field='beta'):
 for L in ls:
  line(ax,betas,[m(L,arm,b)[:,j] for b in range(4)],str(L))
 ax.set(xlabel='Selector beta (dimensionless)',ylabel=s['metrics'][j]);ax.legend(title='L',fontsize=7,ncol=2)
def zero(ax):ax.axhline(0,lw=.7,color='gray')
# G06: observed concentration, finite-count corrected collision, entropy and signed modes, B and density.
fig,ax=plt.subplots(2,3,figsize=(14,8));fig.suptitle('G06 | finite-system order diagnostics (Rapfi; no spontaneous symmetry breaking)')
for i in [0,1]:
 metric(ax[i,0],i,1);ax[i,0].set_title(['Fixed 8 stones','Nested density ~.06'][i]);zero(ax[i,0])
 metric(ax[i,1],i,2);ax[i,1].set_title('Occupancy Shannon entropy / log(legal)')
for h in hs:
 line(ax[0,2],betas,[m(17,0,b,h)[:,0] for b in range(4)],f'B64 prefix {h}')
 ax[0,2].axhline(1/h,ls=':',alpha=.45)
for L in ls:
 line(ax[0,2],[0],[m(L,0,0,16,0,16)[:,0]],None,alpha=.3)
ax[0,2].set(xlabel='Selector beta',ylabel='Uncorrected concentration',title='Prefix budget and iid 1/B null; dots independent B16 at beta=0');ax[0,2].legend(fontsize=7)
for j,lab in [(4,'horizontal centered mode'),(5,'vertical centered mode'),(3,'top-two occupancy gap')]:
 line(ax[1,2],betas,[m(17,1,b)[:,j] for b in range(4)],lab)
zero(ax[1,2]);ax[1,2].set(xlabel='Selector beta',ylabel='Signed mode / gap',title='Explicit geometry; not a broken-symmetry limit');ax[1,2].legend(fontsize=7)
save(fig,'G06_order_parameter',['G06'])
# G08: signed radial shell mass and fixed total zero mode; actual covariance decomposition.
fig,ax=plt.subplots(2,3,figsize=(14,8));fig.suptitle('G08 | open-board legal-pair connected covariance (B-scaled; ordered endpoints)')
for j,bi in enumerate([0,2,3]):
 z=B[21,2,1,bi,64,64];x=z['bin_centers'];u=np.asarray(z['bin_ci'])
 for k in [0,4,5]:line(ax[0,j],x,u[:,k,:].T,s['shell_fields'][k])
 ax[0,j].set(xlabel='r/L (diagonal excluded)',ylabel='Covariance shell mass',title=f'L21 dense, beta={betas[bi]:.3g}');zero(ax[0,j]);ax[0,j].legend(fontsize=7)
z=B[21,2,1,3,64,64];u=np.asarray(z['bin_ci'])
for k in range(4):line(ax[1,0],z['bin_centers'],u[:,k,:].T,s['shell_fields'][k])
zero(ax[1,0]);ax[1,0].set(xlabel='r/L',ylabel='Signed shell mass',title='Raw = drift + innovation + cross');ax[1,0].legend(fontsize=7)
# Full shell root-level per-pair averages for representative cases, support zero pairs excluded.
from verify92 import gzread
a=gzread(S/'SCALE92.json.gz')
for bi in [0,2,3]:
 q=next(r for r in a['strata'] if key(r)==(21,2,1,bi,64,64))
 pr=np.array(q['pair_root_count']);v=np.asarray(q['pairmean']);rr=np.sqrt(np.arange(len(pr)))/21;ix=(pr>0)&(rr>0)
 ax[1,1].scatter(rr[ix],v[4,ix]*21**4,s=6,alpha=.6,label=f'beta {betas[bi]:.3g}')
ax[1,1].set(xlabel='r/L (observed legal pairs)',ylabel='L^4 × per-pair covariance',title='Predictable shell pair average');ax[1,1].legend(fontsize=7)
for bi in range(4):
 line(ax[1,2],ls,[m(L,1,bi)[:,8] for L in ls],f'beta {betas[bi]:.3g}')
ax[1,2].set(xlabel='L',ylabel='2 Var(position)/L²',title='Conditional geometric spread, not physical xi');ax[1,2].legend(fontsize=7)
save(fig,'G08_spatial_covariance',['G08'])
# G09 joint summary; empirical R87 critical gate exactly retained.
fig,ax=plt.subplots(2,3,figsize=(14,8));fig.suptitle('G09 | finite concentration, response, covariance and held-out critical tests')
for j,idx in enumerate([1,11,8]):metric(ax[0,j],1,idx)
for L in ls:
 v=np.array([m(L,1,b)[0] for b in range(4)]);ax[1,0].plot(betas,v[:,12]/(v[:,12]+v[:,13]),'o-',label=str(L))
ax[1,0].set(xlabel='Selector beta',ylabel='low/(low+high) predictable mode ratio',title='R87 ratio diagnostic (not RG invariant)');ax[1,0].legend(fontsize=7)
g=s['critical']['gates'];x=np.arange(len(g))
for i,k in enumerate(['smooth','crossover','critical']):ax[1,1].bar(x+(i-1)*.25,[q['errors'][k] for q in g],.24,label=k)
ax[1,1].set(xlabel='R87 held-out L21 test cell',ylabel='RMSE',title='Critical-type fit wins 0/8');ax[1,1].legend(fontsize=7)
c=s['critical']['crossings'];ax[1,2].bar(np.arange(len(c)),[q['bootstrap_common_fraction'] for q in c])
ax[1,2].set(xlabel='R87 density / budget / observable cell',ylabel='Root bootstrap common bracket fraction',title='Common crossing 0/12 (point estimates)')
save(fig,'G09_critical_summary',['G09'])
# Q29 spatial and temporal: no stationarity, no physical xi.
fig,ax=plt.subplots(2,3,figsize=(14,8));fig.suptitle('Q29 | spatial and temporal correlations; physical correlation length not identified')
for L in ls:
 z=B[L,2,1,3,64,64];line(ax[0,0],z['bin_centers'],np.asarray(z['bin_ci'])[:,4,:].T,str(L))
ax[0,0].set(xlabel='r/L',ylabel='Predictable shell covariance mass',title='Spatial connected covariance; beta=16');ax[0,0].legend(title='L',fontsize=7)
for L in ls:
 line(ax[0,1],betas,[m(L,1,b)[:,8] for b in range(4)],str(L))
ax[0,1].set(xlabel='Selector beta',ylabel='Geometric spread / L²',title='Finite positional spread; not correlation length');ax[0,1].legend(fontsize=7)
for bi in range(4):
 q=T[21,2,1,bi];line(ax[0,2],np.arange(17),np.array(q['lag_ci'])[:,0,:].T,f'beta {betas[bi]:.3g}')
zero(ax[0,2]);ax[0,2].set(xlabel='Selection lag (t,s≥4)',ylabel='Raw sector covariance',title='Fixed-clock covariance, lag average');ax[0,2].legend(fontsize=7)
q=T[21,2,1,3];mat=np.asarray(q['representative_matrix']);assert mat.shape==(4,64,64)
v=max(abs(mat[0]).max(),1e-12);im=ax[1,0].imshow(mat[0],origin='lower',cmap='coolwarm',vmin=-v,vmax=v,aspect='auto');fig.colorbar(im,ax=ax[1,0],shrink=.7);ax[1,0].set(xlabel='Selection clock s',ylabel='Selection clock t',title='L21 dense beta16 raw sector covariance')
for k in range(4):line(ax[1,1],np.arange(63),np.asarray(q['lag1_clock_ci'])[:,k,:].T,s['temporal_fields'][k])
ax[1,1].axvline(4,ls='--',color='gray');zero(ax[1,1]);ax[1,1].set(xlabel='t (lag=1; startup left)',ylabel='Sector covariance',title='Temporal evolution of fixed-lag correlation');ax[1,1].legend(fontsize=7)
for L in ls:
 line(ax[1,2],np.arange(17),np.array(T[L,2,1,3]['lag_ci'])[:,2,:].T,str(L))
zero(ax[1,2]);ax[1,2].set(xlabel='Lag',ylabel='Compensated innovation covariance',title='Off-lag population null; finite-sample error');ax[1,2].legend(fontsize=7)
save(fig,'Q29_spatiotemporal_correlations',['Q29'])
# Q02 partial, source asks physical T, field and Cv. Only measured finite score-law diagnostics.
fig,ax=plt.subplots(2,2,figsize=(10,8));fig.suptitle('Q02 | partial finite-selector response, NOT a thermodynamic phase diagram')
metric(ax[0,0],0,1);metric(ax[0,1],0,11);metric(ax[1,0],0,8)
for L in ls:
 x=[0,1];y=[m(L,i,3)[0,1] for i in x];ax[1,1].plot(x,y,'o-',label=str(L))
ax[1,1].set(xticks=[0,1],xticklabels=['fixed 8','nested density'],ylabel='Corrected concentration',title='Density contrast is NOT an external field');ax[1,1].legend(fontsize=7)
save(fig,'Q02_partial_phase_diagnostics',['Q02'])
# Q31 partial original dedicated gap and entanglement missing; show measured action/shannon and score gap not substitute.
fig,ax=plt.subplots(2,3,figsize=(14,8));fig.suptitle('Q31 | partial finite-size diagnostics: no quantum gap, entanglement or exponent')
for j,idx in enumerate([6,2,7,8,11,1]):
 a0=ax.ravel()[j]
 for h in hs:line(a0,ls,[m(L,1,3,h)[:,idx] for L in ls],f'B64 prefix {h}')
 a0.set(xlabel='L',ylabel=s['metrics'][idx]);a0.legend(fontsize=7)
ax[0,0].set_title('Path action / selection (not 1PI)');ax[0,1].set_title('Occupancy entropy (not entanglement)');ax[0,2].set_title('Score top-gap (not spectral gap)');ax[1,0].set_title('Geometric spread (not physical xi)');ax[1,1].set_title('Instantaneous total variance (not FDT)');ax[1,2].set_title('Collision order (not critical exponent)')
save(fig,'Q31_partial_finite_size',['Q31'])
# Additional budget discrimination, G06/G09 evidence.
fig,ax=plt.subplots(1,2,figsize=(11,4));fig.suptitle('Independent budget arm vs same-history prefix (development cohort; distinct ensembles)')
for bi in range(4):
 x=[z for z in s['budget_comparison'] if z['arm']==1 and z['bi']==bi]
 line(ax[0],ls,[np.asarray(z['difference_ci'])[:,1] for z in x],f'beta {betas[bi]:.3g}')
 line(ax[1],ls,[np.asarray(z['difference_ci'])[:,11] for z in x],f'beta {betas[bi]:.3g}')
for a0,lab in zip(ax,['Corrected concentration difference','Predictable trace difference']):
 zero(a0);a0.set(xlabel='L',ylabel=lab,title='Independent B16 minus B64 prefix16');a0.legend(fontsize=7)
save(fig,'G06_G09_budget_discrimination',['G06','G09'])
dump(S/'PLOT_MANIFEST92.json',dict(figures=f,sources={k:sha(S/k) for k in ['SUMMARY92.json','SCALE92.json.gz','INDEPENDENT92.json']},scope='95% descriptive matched-root t intervals (8 cores); no multiple-testing or physical scaling admission.'))
print('PLOTS',len(f))
