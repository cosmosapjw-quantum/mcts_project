from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
n=read(S/'NATIVE91.json');t=read(S/'THERMAL91.json');T=np.array(t['theta']);C=np.array([r['curve'] for r in t['capacity']]);rows=t['cases'];tm=t['time'];cuts=np.array([4,8,16,32,64,128]);figs=[]
def save(fig,name,ids):
 fig.tight_layout();p=S/'plots'/name
 fig.savefig(str(p)+'.png',dpi=160);fig.savefig(str(p)+'.pdf');plt.close(fig)
 png=Path(str(p)+'.png');pdf=Path(str(p)+'.pdf')
 with Image.open(png) as im:im.verify()
 subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,png_sha=sha(png),pdf_sha=sha(pdf)))
def meanband(ax,x,y,label):
 y=np.asarray(y);ax.plot(x,y.mean(0),label=label);ax.fill_between(x,*np.quantile(y,[.1,.9],axis=0),alpha=.12)
def grouped(field):
 return np.array([np.mean([r[field] for r in rows if r['eps']==eps and r['alpha']==a]) for eps in [0,.05,.2,1] for a in [.25,1]]).reshape(4,2)
# G01 actual game and fixed root, root-score energy.
f,ax=plt.subplots(1,2,figsize=(11,4));f.suptitle('G01 | score energy, not calibrated physical energy')
game=n['game_trajectories'];k=0
for g in game:
 if len(g['ply'])>=12 and k<12:
  ax[0].plot(g['ply'],g['energy'],alpha=.55,lw=1,label=f"game {g['game']}" if k<3 else None);k+=1
ax[0].set(xlabel='Game ply (each search B=32)',ylabel='-mean signed root mark',title=f'{k} actual game traces from R61');ax[0].legend(fontsize=7)
for fam,v in n['families'].items():
 e=np.array(v['energy_by_cut']);core=np.array(v['root_ids']);a=[e[core==i].mean(0) for i in sorted(set(core))];meanband(ax[1],cuts,a,fam)
ax[1].set(xscale='log',xlabel='Completed backups B',ylabel='Score energy',title='Held-out root means; 10–90% across roots');ax[1].legend();save(f,'G01_energy_evolution',['G01'])
# G02 fit conditional logit versus occupancy
f,ax=plt.subplots(1,3,figsize=(14,4));f.suptitle('G02 | three distinct temperature-related quantities')
ax[0].plot(cuts,cuts**(-.5),'o-',label=r'$B^{-1/2}$ proxy');ax[0].axhline(.3,ls='--',label='selector inverse beta (score units)');ax[0].set(xscale='log',xlabel='B',ylabel='Dimensionless proxy / score scale');ax[0].legend(fontsize=8)
for fam,v in n['families'].items():
 qs=[np.array([r['beta'] for r in v['fits'] if r['B']==b and r['status']=='interior']) for b in cuts];q=np.array([np.quantile(a,[.1,.5,.9]) for a in qs]);ax[1].plot(cuts,q[:,1],label=fam);ax[1].fill_between(cuts,q[:,0],q[:,2],alpha=.12)
 ax[2].plot(cuts,[np.mean([r['KL'] for r in v['fits'] if r['B']==b and r['KL'] is not None]) for b in cuts],label=fam)
ax[1].axhline(10/3,ls='--',color='k',label='selector beta');ax[1].set(xscale='log',xlabel='B',ylabel='Endpoint projection beta (interior only)');ax[1].legend(fontsize=8)
ax[2].set(xscale='log',xlabel='B',ylabel='Endpoint occupancy KL (incl. bounds)');ax[2].legend();save(f,'G02_temperature_evolution',['G02'])
# G04 controlled fixed energy
f,ax=plt.subplots(1,2,figsize=(10,4));f.suptitle('G04 | fixed-H declared Gibbs heat capacity')
for g,fam in enumerate(['g0','g3','Rapfi']):
 a=C[np.array([r['family']==g for r in t['capacity']])]
 if len(a):meanband(ax[0],T,a[:,:,0],fam);meanband(ax[1],T,a[:,:,1],fam)
for a,title in zip(ax,['Initial Hamiltonian','Branch-averaged fixed Hamiltonians']):
 a.set(xscale='log',xlabel=r'$k_B T/E_{unit}$ (declared)',ylabel=r'$C/k_B$',title=title);a.legend()
save(f,'G04_heat_capacity',['G04'])
# G05 full ledger
f,ax=plt.subplots(2,3,figsize=(14,7));f.suptitle('G05 | native score statistics and declared feedback thermodynamics (separate)')
for fam,v in n['families'].items():ax[0,0].plot(cuts,np.mean(v['energy_by_cut'],axis=0),label=fam)
ax[0,0].set(xscale='log',xlabel='Completed backups',ylabel='Score energy');ax[0,0].legend()
ax[0,1].plot(T,C[:,:,2].mean(0),label='Gibbs U');ax[0,1].set(xscale='log',xlabel='Declared theta',ylabel='U / score-energy unit')
ax[0,2].plot(T,C[:,:,3].mean(0));ax[0,2].set(xscale='log',xlabel='Declared theta',ylabel='Gibbs entropy / k_B')
x=np.arange(4);lab=['0','.05','.2','1']
for k in ['DE','W','Q']:ax[1,0].plot(x,grouped(k)[:,0],'o-',label=k)
ax[1,0].set(xticks=x,xticklabels=lab,xlabel='Measurement noise eps (alpha=.25)',ylabel='Energy / score unit');ax[1,0].legend()
for k in ['total','sigma_rel','D_final','I1']:ax[1,1].plot(x,grouped(k)[:,0],'o-',label=k)
ax[1,1].set(xticks=x,xticklabels=lab,xlabel='eps',ylabel='Nats');ax[1,1].legend(fontsize=8)
ax[1,2].hist([r['residual'] for r in rows],bins=30);ax[1,2].set(xlabel='Max energy/entropy identity residual',ylabel='Cases');save(f,'G05_thermodynamics_summary',['G05'])
# Q05 Q08 actual controlled evolution, signed entropy (not abs)
bvals=[1/3,1,10/3,10];S0={b:np.array([r['steps'] for r in tm if r['beta']==b]) for b in bvals}
f,ax=plt.subplots(2,2,figsize=(10,7));f.suptitle('Q05 | entropy-production consistency in declared finite relaxation')
for b,a in S0.items():
 z=a.mean(0);lab=f'beta={b:.3g}';ax[0,0].plot(np.arange(1,9),z[:,4],label=lab);ax[0,1].plot(np.arange(9),np.r_[z[0,0],z[:,1]],label=lab)
 ax[1,0].plot(np.arange(1,9),z[:,2],label=lab);ax[1,1].plot(np.arange(1,9),z[:,5],label=lab)
for a,l in zip(ax.ravel(),['Entropy production / step','Conditional entropy / k_B','Signed system entropy increment','Absolute identity residual']):
 a.set(xlabel='Declared heat-bath step',ylabel=l);a.legend(fontsize=7)
save(f,'Q05_entropy_production_validation',['Q05'])
f,ax=plt.subplots(2,2,figsize=(10,7));f.suptitle('Q08 | full signed conditional entropy balance, beta=10/3')
a=S0[10/3];z=a.mean(0);k=np.arange(1,9);v=np.array([r['steps'] for r in read(S/'INDEPENDENT91.json')['marginal_time'] if r['beta']==10/3]).mean(0)
meanband(ax[0,0],k,a[:,:,4],'sigma_rel');ax[0,0].set(ylabel='Nats / step')
ax[0,1].plot(k,v[:,0],label='dS_marginal');ax[0,1].plot(k,-v[:,1],label='-dI');ax[0,1].plot(k,-(10/3)*z[:,3],label='-beta Q');ax[0,1].plot(k,z[:,4],ls='--',label='sum = sigma');ax[0,1].set(ylabel='Signed nats / step')
ax[1,0].plot(k,a[:,:,5].max(0),label='maximum abs residual');ax[1,0].set(ylabel='Absolute error')
ax[1,1].hist(a[:,:,5].ravel(),bins=30,label='104 contexts x 8 steps');ax[1,1].set(xlabel='Absolute residual',ylabel='Cases')
for a in ax.ravel():a.legend(fontsize=7)
for a in ax.ravel()[:3]:a.set(xlabel='Declared heat-bath step')
save(f,'Q08_full_entropy_balance',['Q08'])
# Q19/Q20 partial: actual F/R auxiliary Monte Carlo, explicit information correction.
f,ax=plt.subplots(1,2,figsize=(10,4));f.suptitle('Q19 | partial: support-aware feedback IFT (declared reference)')
for eps in [0,.05,.2,1]:
 rr=[r for r in rows if r['eps']==eps];ax[0].scatter([r['exact_ift'] for r in rr],[r['ift'] for r in rr],s=8,alpha=.4,label=f'eps={eps}');ax[1].scatter([r['reverse_support'] for r in rr],[r['ess'] for r in rr],s=8,alpha=.4,label=f'eps={eps}')
ax[0].plot([0,1],[0,1],ls='--',color='k');ax[0].set(xlabel='Exact forward IFT = reverse support',ylabel='2048 forward MC estimate');ax[1].set(xlabel='MC reverse support',ylabel='Forward ESS');ax[0].legend(fontsize=8);save(f,'Q19_partial_ift',['Q19'])
f,ax=plt.subplots(1,2,figsize=(10,4));f.suptitle('Q20 | partial: work fluctuations under declared feedback')
for eps in [0,.05,.2,1]:
 rr=[r for r in rows if r['eps']==eps];ax[0].scatter([r['W'] for r in rr],[r['work_sd'] for r in rr],s=8,alpha=.5,label=f'eps={eps}');ax[1].scatter([r['W'] for r in rr],[r['work'] for r in rr],s=8,alpha=.5)
ax[0].set(xlabel='Exact mean W',ylabel='Exact SD(W)');ax[0].legend(fontsize=8);ax[1].set(xlabel='Exact mean W',ylabel='MC mean W');save(f,'Q20_partial_work',['Q20'])
dump(S/'PLOT_MANIFEST91.json',dict(figures=figs,sources={k:sha(S/k) for k in ['NATIVE91.json','THERMAL91.json','INDEPENDENT91.json']},scope='New direct item reconstructions; no native bath or physical T. Shaded quantiles descriptive, not CI.'))
print('PLOTS',len(figs),flush=True)
