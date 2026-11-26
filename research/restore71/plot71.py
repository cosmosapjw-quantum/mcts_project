from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
r=read(S/'PHYSICS_RESULTS.json'); fs=r['families']; a=np.array(r['amplitudes']); names=[f['label'] for f in fs]; labs=['clock + root','state','state + lag']
figs=[]
def save(fig,name,scope):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=160);fig.savefig(str(p)+'.pdf');plt.close(fig)
 png=Path(str(p)+'.png');pdf=Path(str(p)+'.pdf')
 with Image.open(png) as im:im.verify()
 assert pdf.read_bytes()[:4]==b'%PDF'
 figs.append(dict(figure=name,ids=['L16','Q21'],scope=scope,png_sha=sha(png),pdf_sha=sha(pdf)))
for i,f in enumerate(fs):
 fig,ax=plt.subplots(2,2,figsize=(10,7),sharex=True)
 for j,k in enumerate([0,1,2,6]):
  h=ax.flat[j]; sl=slice(k*9,(k+1)*9);h.axhline(0,color='black',lw=.7)
  for q in f['results']:
   y=np.array(q['W_delta'])[sl];lo,hi=np.array(q['W_delta_ci95'])[:,sl]
   h.plot(a,y,marker='o',ms=3,label=q['model']);h.fill_between(a,lo,hi,alpha=.1)
  h.set_title('source direction '+str(r['directions'][k]));h.set_ylabel('W model − heldout W');h.set_xlabel('source amplitude')
 ax.flat[0].legend(fontsize=8);fig.suptitle(f['label']+' | geometry-block pointwise 95% intervals')
 save(fig,'W_error_family_'+str(i),'Autonomous source residual and fixed-fit test-block uncertainty.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for j,key in enumerate(['W_rmse_moderate','W_rmse']):
 for k,lab in enumerate(labs):ax[j].bar(np.arange(5)+(k-1)*.25,[f['results'][k][key] for f in fs],width=.24,label=lab)
 ax[j].set_xticks(range(5),names,rotation=20);ax[j].set_ylabel('RMSE of W');ax[j].set_title('Moderate source' if j==0 else 'Full source grid');ax[j].legend(fontsize=8)
save(fig,'source_rmse','Direct heldout W versus autonomous fitted kernels.')
fig,ax=plt.subplots(3,5,figsize=(17,9),sharex=True)
for i,f in enumerate(fs):
 for j,label in enumerate(['count a /128','count b /128','signed mark /128']):
  h=ax[j,i];h.plot(range(1,33),np.array(f['prefix_direct'])[:,j],color='black',label='heldout')
  for q in f['results']:h.plot(range(1,33),np.array(q['prefix'])[:,j],label=q['model'])
  if j==0:h.set_title(f['label'])
  if i==0:h.set_ylabel(label)
  if j==2:h.set_xlabel('wave')
ax[0,0].legend(fontsize=7);save(fig,'prefix_means','Autonomous wave-by-wave cumulative mark/count means.')
fig,ax=plt.subplots(2,2,figsize=(11,7))
for j,key in enumerate(['joint_brier','next_state_brier','zero_joint_support','root_prefix_rmse']):
 h=ax.flat[j]
 for k,lab in enumerate(labs):
  v=[f['results'][k][key] for f in fs]; y=[x[0] if isinstance(x,list) else x for x in v]
  h.bar(np.arange(5)+(k-1)*.25,y,width=.24,label=lab)
 h.set_xticks(range(5),names,rotation=20);h.set_title(key);h.legend(fontsize=7)
save(fig,'closure_diagnostics','Teacher-forced Brier/support diagnostics separate from autonomous root-prefix RMSE.')
fig,ax=plt.subplots(2,2,figsize=(11,7))
for j in range(4):
 h=ax.flat[j]; h.axhline(0,color='black',lw=.7)
 for k,lab in enumerate(labs):
  y=[f['results'][k]['cumulant_delta'][6][j] for f in fs];lo=[f['results'][k]['cumulant_delta_ci95'][0][6][j] for f in fs];hi=[f['results'][k]['cumulant_delta_ci95'][1][6][j] for f in fs]
  h.scatter(np.arange(5)+(k-1)*.18,y,label=lab)
  h.vlines(np.arange(5)+(k-1)*.18,lo,hi)
 h.set_xticks(range(5),names,rotation=20);h.set_title('Mixed direction cumulant '+str(j+1));h.legend(fontsize=7)
save(fig,'mixed_cumulant_residuals','Pointwise block-bootstrap residuals, directional cumulants through fourth order.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for i,f in enumerate(fs):
 ess=np.array([q['ess'] for q in f['empirical']]);ax[0].plot(a,ess[:,54:63].min(0),label=f['label'])
 for k,q in enumerate(f['results']):ax[1].scatter(i+(k-1)*.15,q['mc_checks'][0]['max_abs_z'],label=labs[k] if i==0 else None)
ax[0].set_xlabel('mixed source amplitude');ax[0].set_ylabel('minimum root ESS');ax[0].legend(fontsize=8)
ax[1].set_xticks(range(5),names,rotation=20);ax[1].set_ylabel('max |MC − exact| / MC SE');ax[1].legend(fontsize=8)
save(fig,'source_ess_and_mc','Importance-weight ESS and independent simulation checks of fitted transition.')
assert len(figs)==10 and len({q['png_sha'] for q in figs})==10
dump(S/'PLOT_MANIFEST71.json',dict(figures=figs,sources={k:sha(S/k) for k in ['PHYSICS_RESULTS.json','JOINT71.npz','plot71.py']}))
print('PLOTS',len(figs),flush=True)
