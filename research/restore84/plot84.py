from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
Q=read(S/'SUMMARY84.json');T=read(S/'TRANSFER84.json');F=Q['families'];names=[x['family'] for x in F];figs=[]
def save(fig,name,ids,scope):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=180);fig.savefig(str(p)+'.pdf');plt.close(fig)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,scope=scope,png_sha=sha(a),pdf_sha=sha(b)))
fig,ax=plt.subplots(1,3,figsize=(13,4));x=np.arange(3)
for a,k,l in [(ax[0],'energy','joint energy score'),(ax[1],'W','root CGF RMSE')]:
 a.bar(x-.18,[f[k+'_baseline'] for f in F],.36,label='root-only ridge');a.bar(x+.18,[f[k+'_candidate'] for f in F],.36,label='root+child ridge');a.set_xticks(x,names);a.set(ylabel=l,title='Fresh 48 geometries');a.legend(fontsize=8)
for i,f in enumerate(F):
 q=np.array(f['gain_ci']);ax[2].errorbar(i,q[0],yerr=[[q[0]-q[1]],[q[2]-q[0]]],fmt='o',capsize=3)
ax[2].set_xticks(x,names);ax[2].axhline(0,color='k',lw=.7);ax[2].set(title='Paired root-block gain',ylabel='baseline minus candidate ES')
save(fig,'presearch_landscape_transfer',['L16','Q32','G07'],'Frozen model, independent 48-root cohort, proper joint energy score and regular CGF RMSE; not exact fine-history law.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
for i,f in enumerate(F):
 a=next(z for z in T['results'] if z['family']==f['family']);z=next(z for z in a['models'] if z['model']==f['candidate'])
 ax[0].bar(i-.2,f['W_candidate'],.2,label='regular' if i==0 else None);ax[0].bar(i,f['W_stress'],.2,label='stress' if i==0 else None);ax[0].bar(i+.2,f['split_half_W'],.2,label='split-half regular' if i==0 else None)
 ax[1].bar(i-.18,f['coverage90'][0],.36,label='count' if i==0 else None);ax[1].bar(i+.18,f['coverage90'][1],.36,label='mark' if i==0 else None)
 ax[2].bar(i-.18,f['min_root_ESS_regular'],.36,label='regular' if i==0 else None);ax[2].bar(i+.18,f['min_root_ESS_stress'],.36,label='stress' if i==0 else None)
for a in ax:a.set_xticks(x,names);a.legend(fontsize=8)
ax[0].set(title='Absolute error retained',ylabel='W RMSE');ax[1].axhline(.9,color='k',ls=':');ax[1].set(title='Marginal interval coverage',ylim=(0,1));ax[2].set(title='Tilted donor-root concentration',ylabel='minimum root ESS')
save(fig,'presearch_calibration_and_tail',['L16','Q32'],'Absolute and stress errors, discrete marginal intervals, donor-root ESS; split-half not a formal lower bound.')
fig,ax=plt.subplots(1,2,figsize=(10,4))
ax[0].bar(x-.18,[f['mean_native_eval_calls'] for f in F],.36,label='one B128 search');ax[0].bar(x+.18,[f['mean_extra_child_calls'] for f in F],.36,label='additional child calls');ax[0].set_xticks(x,names);ax[0].set(ylabel='evaluator calls',title='Separate pre-search cost');ax[0].legend(fontsize=8)
ax[1].bar(x,[f['mean_clipped_mass'] for f in F]);ax[1].set_xticks(x,names);ax[1].set(title='Residual transport bound clipping',ylabel='mean donor probability mass')
save(fig,'presearch_cost_and_support',['L16','Q32','G07'],'Extra evaluator calls outside B128, bounded joint residual transport and clipping; no equal-time utility.')
assert len(figs)==3 and len({f['png_sha'] for f in figs})==3
dump(S/'PLOT_MANIFEST84.json',dict(figures=figs,animations=[],sources={f:sha(S/f) for f in ['SUMMARY84.json','TRANSFER84.json','MODEL84.json','LANDSCAPE84.json','AUDIT84.json','INDEPENDENT84.json','CONVERGENCE84.json','plot84.py']}))
print('PLOTS',len(figs))
