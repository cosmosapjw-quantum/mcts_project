from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
A=read(S/'ACTION80.json');Q=read(S/'SUMMARY80.json');V=read(S/'INDEPENDENT80.json')
names=['g0','g3','Rapfi'];figs=[]
def save(fig,name,ids,scope):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=180);fig.savefig(str(p)+'.pdf');plt.close(fig)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,scope=scope,png_sha=sha(a),pdf_sha=sha(b)))
fig,ax=plt.subplots(1,2,figsize=(11,4))
q=Q['saddle_convergence'];n=np.array([r['M'] for r in q]);a=ax[0]
for key,label in [('max_leading','max Gaussian'),('max_corrected','max corrected'),('mean_leading','mean Gaussian'),('mean_corrected','mean corrected')]:
 a.loglog(n,[r[key] for r in q],'o-',label=label)
a.loglog(n,.0044*256/n,'k:',label='M^-1 guide');a.loglog(n,3.1e-5*(256/n)**2,'k--',label='M^-2 guide')
a.set(xlabel='independent whole-history replicas M',ylabel='absolute relative mass error',title='Bivariate lattice expansion, 30 cases per M');a.legend(fontsize=8)
a=ax[1]
for g in names:
 z=[r for r in A['saddles'] if r['family']==g]
 a.loglog(n,[max(abs(r['corrected_rel'])*r['M']**2 for r in z if r['M']==m) for m in n],'o-',label=g)
a.set(xlabel='M',ylabel='M² × max corrected error',title='Finite tested remainder diagnostic');a.legend()
save(fig,'controlled_bivariate_saddle',['L16','Q21','Q43'],'Six finite full-lattice laws, exact tilted torus reference with analytic alias upper bound; relative approximation errors not physical quantum loops.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
for a,g in zip(ax,A['sources']):
 z=g['cases'];x=np.array([r['train']['W'] for r in z])
 for k,label in [('matched','same geometry fresh seeds'),('novel','new geometry')]:
  y=np.array([r[k]['W'] for r in z]);a.scatter(x,y,label=label,alpha=.75,s=22)
 lo=min(a.get_xlim()[0],a.get_ylim()[0]);hi=max(a.get_xlim()[1],a.get_ylim()[1]);a.plot([lo,hi],[lo,hi],'k:');a.set(xlabel='reference empirical W',ylabel='independent empirical W',title=g['family']);a.legend(fontsize=7)
save(fig,'marked_source_transfer',['L16','Q32','G07'],'All 29 source points, independent same-root vs novel-root data; empirical transfer not guaranteed.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
with np.load(S/'MARKED80.npz') as z:d={k:z[k] for k in z.files}
m=d['meta']
for i,a in enumerate(ax):
 ix=(m[:,0]==1)&(m[:,1]==i);a.scatter(d['unseen_mass'][ix],d['action'][ix]/128,s=14,alpha=.55)
 a.set(xlabel='last-wave mass of final-unvisited moves',ylabel='path action / B (nat)',title=names[i])
save(fig,'path_action_full_support',['L16'],'Observed ordered path action and unvisited positive conditional mass; endpoint visits are not path probabilities.')
assert len(figs)==3 and len({f['png_sha'] for f in figs})==3
dump(S/'PLOT_MANIFEST80.json',dict(figures=figs,animations=[],sources={f:sha(S/f) for f in ['ACTION80.json','SUMMARY80.json','AUDIT80.json','INDEPENDENT80.json','MARKED80.npz','CONVERGENCE80.json','plot80.py']}))
print('PLOTS',len(figs),flush=True)
