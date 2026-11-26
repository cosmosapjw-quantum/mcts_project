from channel64 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
r=read(S/'PHYSICS_RESULTS.json');cal=read(S/'CALIBRATION64.json');reg=read(S/'REGISTRATION.json')
with np.load(S/'PHASE64.npz') as d:z={k:d[k] for k in d.files}
m=z['meta'];W=z['waves'];wf=list(z['wave_fields']);arms=list(z['arms']);colors=['#607A97','#C7A56D','#8A8176','#25866E','#AA5361','#7668A8']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.dpi':120})
manifest=[]
def save(fig,name,ids):
 fig.tight_layout();p=S/'plots'/name
 fig.savefig(str(p)+'.png',dpi=165);fig.savefig(str(p)+'.pdf');plt.close(fig)
 with Image.open(str(p)+'.png') as im:im.verify()
 assert Path(str(p)+'.pdf').read_bytes().startswith(b'%PDF')
 manifest.append(dict(figure=name,ids=ids,png_sha=sha(Path(str(p)+'.png')),pdf_sha=sha(Path(str(p)+'.pdf'))))
def ci(ax,xx,ys,cc,**kw):
 ys=np.asarray(ys);cc=np.asarray(cc);ax.errorbar(xx,ys,yerr=np.maximum(0,np.vstack([ys-cc[:,0],cc[:,1]-ys])),capsize=3,**kw)
# Independent calibration
with np.load(S/'CALIBRATION64.npz') as d:te={k[5:]:d[k] for k in d.files if k.startswith('test_')}
fit=cal['fitted'];p=pred(te,fit);y=2*te['y']/reg['calibration']['shots_per_quadrature']-1
fig,axs=plt.subplots(1,3,figsize=(12,4))
for i,a in enumerate(axs[:2]):
 a.scatter(p[:,i],y[:,i],s=9,alpha=.42,c=colors[3]);a.plot([-1,1],[-1,1],'k--',lw=.8);a.set(xlim=(-.85,.85),ylim=(-.85,.85),xlabel='Predicted quadrature',ylabel='Independent shot mean',title=['X readout','Y readout'][i])
a=axs[2];keys=['heldout','heldout_no_rotation','heldout_dephased'];a.bar(range(3),[cal[k]['rmse'] for k in keys],color=[colors[3],colors[4],colors[1]]);a.set(xticks=range(3),xticklabels=['Phase fit','No rotation','Dephased'],ylabel='Held-out RMSE',title='Software tomography controls');a.tick_params(axis='x',rotation=15)
fig.suptitle('Independent prepared-state calibration (software shots)',fontsize=13)
save(fig,'Q41_Q42_quadrature_calibration',['Q41','Q42'])
# Phase and dephasing clock; compare original proxy and defined coupling
fig,axs=plt.subplots(1,3,figsize=(12.5,4.3))
t=np.arange(32)/32;eta=.8*np.exp(-1.2*t);rel=np.exp(-1.2*t)
axs[0].plot(t,eta,'o-',ms=3,label='absolute visibility');axs[0].plot(t,rel,'--',label='relative decay');axs[0].set(xlabel='Completed clock τ',ylabel='Visibility',title='Independent model dephasing');axs[0].legend(fontsize=8)
axs[1].plot(t[1:],1/np.arctan2(np.sqrt(-np.expm1(-2*1.2*t[1:])),rel[1:]),'-',label='Relative-decay match')
axs[1].plot(t,1/np.arccos(eta),'--',label='Absolute-visibility match')
axs[1].axhline(1/fit[2],c=colors[3],ls=':',label='Fitted model action scale 1/κ')
axs[1].axhline(2/np.pi,c='gray',ls='-.',label='Unit-action match floor')
axs[1].set(xlabel='Completed clock τ',ylabel='Dimensionless convention',title='Distinct action-scale definitions',yscale='log');axs[1].legend(fontsize=7)
for i,g in enumerate((0,3)):
 v=W[(m[:,0]==g)&(m[:,3]==arms.index('plus'))];x=abs(v[:,:,wf.index('exposure')]);y=abs(v[:,:,wf.index('theta')]-np.pi/2)
 axs[2].scatter(x[:,::4].ravel(),y[:,::4].ravel(),s=5,alpha=.15,label=f'g{g}',c=[colors[0],colors[3]][i])
mx=axs[2].get_xlim()[1];axs[2].plot([0,mx],[0,fit[2]*mx],'k--',lw=1,label='Independent calibrated κ');axs[2].set(xlabel='Absolute integrated score exposure',ylabel='Absolute phase displacement (rad)',title='Native controller phase clock');axs[2].legend(fontsize=8)
fig.suptitle('Visibility matching is not phase calibration or physical ħ',fontsize=13)
save(fig,'Q41_operational_action_scale',['Q41','Q42'])
# Quantum register time evolution, entropy and complex quadratures on actual states
fig,axs=plt.subplots(2,3,figsize=(12.6,7.5))
for i,g in enumerate((0,3)):
 for j,arm in enumerate(arms):
  v=W[(m[:,0]==g)&(m[:,3]==j)]
  for a,key in zip(axs[i],['X','Y','entropy_vN']):
   a.plot(t,v[:,:,wf.index(key)].mean(0),label=arm,c=colors[j],ls='--' if arm=='real' else '-')
 for a,key in zip(axs[i],['X','Y','S_vN (nat)']):a.set(xlabel='Completed clock τ',ylabel=key,title=f'g{g} | {key}')
axs[0,0].legend(ncol=2,fontsize=7)
fig.suptitle('Actual conditional auxiliary density states on native search histories',fontsize=14)
save(fig,'L09_Q24_Q42_density_dynamics',['L09','Q24','Q42'])
# Full-support native action result
fig,axs=plt.subplots(2,2,figsize=(11,7.5))
for i,g in enumerate((0,3)):
 for j,arm in enumerate(arms):
  q=[x for x in r['observables'] if x['generation']==g and x['arm']==arm];xx=[x['k'] for x in q]
  for a,key in zip(axs[i],['Na','Nb']):
   a.plot(xx,[x['metrics'][key]['mean'] for x in q],'-o',ms=3,c=colors[j],ls='--' if arm=='real' else '-',label=arm)
 for a,key in zip(axs[i],['a','b']):a.set(xscale='log',xlabel='Completed backups',ylabel=f'Visit count {key}',title=f'g{g} | source {key}',xticks=[4,8,16,32,64,128],xticklabels=[4,8,16,32,64,128])
axs[0,0].legend(ncol=2,fontsize=8)
fig.suptitle('Native full-support phase intervention: same paired geometries and seeds',fontsize=13)
save(fig,'Q45_native_phase_visit_response',['Q45','Q43','L16'])
fig,axs=plt.subplots(1,2,figsize=(11,4.5))
keys=[('plus','minus'),('plus','dephased'),('minus','dephased'),('rotate0','real')]
for j,g in enumerate((0,3)):
 q=[next(x for x in r['contrasts'] if x['generation']==g and x['k']==128 and (x['left'],x['right'])==pair) for pair in keys]
 for a,key in zip(axs,['Na','mean_mark']):
  yy=[x['metrics'][key]['mean'] for x in q];cc=[x['metrics'][key]['ci95'] for x in q];ci(a,np.arange(4)+(j-.5)*.14,yy,cc,fmt=['o','s'][j],label=f'g{g}',c=[colors[0],colors[3]][j])
for a,key in zip(axs,['Source a count difference','Mean backup mark difference']):
 a.axhline(0,c='gray',ls=':');a.set(xticks=range(4),xticklabels=['+ − −','+ − deph.','− − deph.','rot0 − real'],ylabel=key,title=key+' at B=128');a.legend(fontsize=8)
fig.suptitle('Paired root-block intervals: native response and mark utility',fontsize=13)
save(fig,'Q45_phase_contrast_and_marks',['Q45','Q43'])
# local support and likelihood controls
fig,axs=plt.subplots(1,3,figsize=(12.5,4.3))
for j,g in enumerate((0,3)):
 q=[x for x in r['local'] if x['generation']==g and x['arm'] in ('plus','minus','rotate0')];xx=np.arange(3)+(j-.5)*.14
 for a,key in zip(axs[:2],['mean_abs_phase_shift','dephased_excess_nll']):
  ci(a,xx,[x[key]['mean'] for x in q],[x[key]['ci95'] for x in q],fmt=['o','s'][j],c=[colors[0],colors[3]][j],label=f'g{g}')
 a=axs[2];ci(a,xx,[1e6*x['calibrated_excess_nll']['mean'] for x in q],[[1e6*u for u in x['calibrated_excess_nll']['ci95']] for x in q],fmt=['o','s'][j],c=[colors[0],colors[3]][j],label=f'g{g}')
for a,key in zip(axs,['Mean absolute coherent shift','Dephased excess NLL / draw','Calibrated excess NLL (×10⁻⁶)']):
 a.axhline(0,c='gray',ls=':');a.set(xticks=range(3),xticklabels=['plus','minus','rotate0'],ylabel=key,title=key);a.legend(fontsize=8)
save(fig,'Q43_native_local_readout_likelihood',['Q43','Q45','L16'])
# Actual path alternatives under baseline law: selected two-path auxiliary lift
pairs=read(S/'PATH_PAIRS64.json')['rows'];p=r['path_pairs']['summary']
fig,axs=plt.subplots(1,3,figsize=(12.5,4.4))
for j,g in enumerate((0,3)):
 q=[x for x in p if x['generation']==g];xx=[x['k'] for x in q]
 for a,key in zip(axs[:2],['mean_abs_imag','mean_abs_phase_shift']):
  a.plot(xx,[x[key] for x in q],'o-',c=[colors[0],colors[3]][j],label=f'g{g}')
 a=axs[2];z0=[x for x in pairs if x['generation']==g and x['k']==128];a.scatter([abs(x['logP'][0]-x['logP'][1]) for x in z0],[abs(x['rho_imag'][0][1]) for x in z0],alpha=.45,s=9,c=[colors[0],colors[3]][j],label=f'g{g}')
for a,key in zip(axs[:2],['Mean |Im ρ_ab|','Mean absolute interference shift']):
 a.set(xscale='log',xlabel='Prefix backups',ylabel=key,xticks=[4,8,16,32,64,128],xticklabels=[4,8,16,32,64,128]);a.legend(fontsize=8)
axs[2].set(xlabel='Absolute pair log-likelihood difference',ylabel='|Im ρ_ab|',title='B=128 pair imbalance');axs[2].legend(fontsize=8)
fig.suptitle('Selected native path pairs: auxiliary phase lift, not physical path measure',fontsize=13)
save(fig,'Q43_native_path_pair_complex_lift',['Q43','L16'])
# held-out quadrature simulations on native conditional states
fig,axs=plt.subplots(1,2,figsize=(10.8,4.3))
for j,g in enumerate((0,3)):
 q=[x for x in r['readouts'] if x['generation']==g];xx=np.arange(3)
 for k,key in enumerate(['fit_mse','no_rotation_mse','dephased_mse']):
  a=axs[j];ci(a,xx+(k-1)*.18,[x[key]['mean'] for x in q],[x[key]['ci95'] for x in q],fmt=['o','s','^'][k],label=key.replace('_',' '),c=[colors[3],colors[4],colors[1]][k])
 axs[j].set(xticks=xx,xticklabels=['plus','minus','rotate0'],ylabel='Quadrature MSE',title=f'g{g} native state readouts',yscale='log');axs[j].legend(fontsize=8)
fig.suptitle('Frozen calibration predicts independent software readout shots',fontsize=13)
save(fig,'Q42_Q45_native_state_readout_validation',['Q42','Q45'])
dump(S/'PLOT_MANIFEST64.json',dict(figures=manifest,sources={p:sha(S/p) for p in ['PHYSICS_RESULTS.json','CALIBRATION64.json','CALIBRATION64.npz','PHASE64.npz','PATH_PAIRS64.json']}))
imgs=[]
for x in manifest:
 with Image.open(S/'plots'/(x['figure']+'.png')) as im:
  j=im.convert('RGB');j.thumbnail((650,480));imgs.append(j.copy())
sheet=Image.new('RGB',(1300,480*((len(imgs)+1)//2)),'white')
for j,im in enumerate(imgs):sheet.paste(im,(650*(j%2),480*(j//2)))
sheet.save(S/'logs/plot_qa.jpg',quality=88)
print('PLOT_PAIRS',len(manifest))
