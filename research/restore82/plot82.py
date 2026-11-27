from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
Q=read(S/'SUMMARY82.json');T=read(S/'TRANSFER82.json');F=Q['results'];names=['g0','g3','Rapfi'];models=['pool','ply','root','prefix_pool','prefix_kernel','root_history'];labels=['pool','ply','root','H32+pool','H32 kernel','root+H32'];figs=[]
def save(fig,name,ids,scope):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=180);fig.savefig(str(p)+'.pdf');plt.close(fig)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,scope=scope,png_sha=sha(a),pdf_sha=sha(b)))
fig,ax=plt.subplots(1,2,figsize=(11,4))
for j,k in enumerate(['source-pair count','signed mark']):
 a=ax[j]
 for g,name in enumerate(names):
  z=[x for x in Q['mixture_decomposition'] if x['family']==name]
  a.bar(np.arange(3)+(g-1)*.25,[100*x['between_per_component'][j] for x in z],.25,label=name)
 a.set_xticks(range(3),['train','R81 base','R82 fresh']);a.set(ylim=(0,105),ylabel='between-root variance (%)',title=k+' at J=0');a.legend(fontsize=8)
save(fig,'root_mixture_covariance_decomposition',['L16','Q32','G07'],'Exact empirical root-conditional covariance decomposition, per variable, at zero source; not thermodynamic susceptibility.')
fig,ax=plt.subplots(2,2,figsize=(13,8))
for gr in (1,2):
 for g,name in enumerate(names):
  z=next(x for x in F if x['family']==name and x['group']==gr)
  for a,k in [(ax[0,gr-1],'energy'),(ax[1,gr-1],'W_RMSE')]:
   a.bar(np.arange(6)+(g-1)*.25,[z['models'][m][k] for m in models],.25,label=name)
   a.set_xticks(range(6),labels,rotation=20,ha='right');a.set(title=('R81 base' if gr==1 else 'R82 fresh')+' | '+('joint energy score' if k=='energy' else 'root CGF RMSE (25 sources)'),ylabel='lower is better');a.legend(fontsize=8)
save(fig,'independent_conditional_prediction',['L16','Q32','G07'],'Frozen train-only bandwidths, two disjoint external geometry cohorts; H32 predictions condition on observed prefix, root-only generalization not globally admitted.')
fig,ax=plt.subplots(2,2,figsize=(12,8))
for i,z in enumerate(F):
 name=z['family']+' R'+str(80+z['group']);u=np.arange(6)+(i-2.5)*.12
 ax[0,0].bar(u,[z['models'][m]['min_tilt_root_ESS_regular'] for m in models],.12,label=name)
 ax[0,1].bar(u,[z['models'][m]['min_tilt_root_ESS_stress'] for m in models],.12,label=name)
 ax[1,0].scatter(i,z['models']['root_history']['W_RMSE'],label='H32 root+history' if i==0 else None);ax[1,0].scatter(i,z['split_half_W_RMSE'],label='empirical split-half' if i==0 else None)
 q=np.asarray(z['root_gain_vs_ply']);r=np.asarray(z['root_history_gain_vs_root']);ax[1,1].errorbar(i-.1,q[0],yerr=[[q[0]-q[1]],[q[2]-q[0]]],fmt='o',label='root vs ply' if i==0 else None);ax[1,1].errorbar(i+.1,r[0],yerr=[[r[0]-r[1]],[r[2]-r[0]]],fmt='s',label='root+H32 vs root' if i==0 else None)
for a,t in [(ax[0,0],'Regular source grid'),(ax[0,1],'Stress source corners')]:
 a.set_xticks(range(6),labels,rotation=20,ha='right');a.set(title=t,ylabel='minimum tilted donor-root ESS');a.legend(fontsize=7,ncol=2)
for a in ax[1]:a.set_xticks(range(6),[z['family']+'\nR'+str(80+z['group']) for z in F]);a.legend(fontsize=8)
ax[1,0].set(title='Absolute root CGF error remains',ylabel='RMSE');ax[1,1].set(title='Paired geometry-block energy gain',ylabel='positive favors candidate');ax[1,1].axhline(0,color='k',lw=.7)
save(fig,'transfer_support_and_uncertainty',['L16','Q32'],'Donor-root ESS (not independent history count), stress diagnostics and pointwise intervals; finite reference noise is not lower error bound.')
assert len(figs)==3 and len({f['png_sha'] for f in figs})==3
dump(S/'PLOT_MANIFEST82.json',dict(figures=figs,animations=[],sources={f:sha(S/f) for f in ['SUMMARY82.json','TRANSFER82.json','MODEL82.json','AUDIT82.json','INDEPENDENT82.json','CONVERGENCE82.json','plot82.py']}))
print('PLOTS',len(figs),flush=True)
