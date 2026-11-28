"""Direct corrected original-output plates (one unique figure for each item)."""
from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
import subprocess
G=read(S/'GAUGE88.json');T=read(S/'TOPOLOGY88.json');B0=read(S/'BARRIER88.json');Q=read(S/'TRANSPORT88.json');R=read(S/'RECOVERY88.json');L=read(S/'LANDSCAPE88.json');S0=read(S/'SUMMARY88.json')
names=['g0','g3','Rapfi'];figs=[]
def save(fig,id,name,scope):
 fig.suptitle(id+' | '+name.replace('_',' '),fontsize=12);fig.tight_layout(rect=(0,0,1,.95));p=S/'plots'/f'{id}_{name}';fig.savefig(str(p)+'.png',dpi=170);fig.savefig(str(p)+'.pdf');plt.close(fig)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],check=True,stdout=subprocess.DEVNULL)
 figs.append(dict(figure=p.name,ids=[id],scope=scope,png_sha=sha(a),pdf_sha=sha(b)))
# L22 original 3 panels plus oriented-volume discriminator.
fig,ax=plt.subplots(2,2,figsize=(11,8))
for k,name in enumerate(names):
 z=[x for x in G['contexts'] if x['family']==k];ax[0,0].hist([np.mean([a[0] for a in x['reconstructed']['SU2']['wilson']]) for x in z],bins=np.linspace(-1,1,21),alpha=.5,label=name)
 ax[0,1].scatter([max(x['reconstructed']['policy_gauge_error']) for x in z],[x['reconstructed']['policy_stability'][-1] for x in z],label=name)
 ax[1,0].scatter([x['reconstructed']['nodes'] for x in z],[x['reconstructed']['edges'] for x in z],label=name)
 ax[1,1].scatter([x['joint'][0]['triple_trace'] for x in z],[x['joint'][0]['reflected_trace'] for x in z],label=name)
ax[0,0].set(xlabel='mean normalized SU(2) Wilson trace',ylabel='contexts');ax[0,1].set(xlabel='transported gauge readout residual',ylabel='1 - TV after four updates');ax[1,0].set(xlabel='legal graph vertices',ylabel='edges');ax[1,1].plot([-1,1],[-1,1],'k:');ax[1,1].set(xlabel='triple trace',ylabel='reflected triple trace')
for a in ax.ravel():a.legend(fontsize=8)
save(fig,'L22','gauge_policy_joint_holonomy','48 actual legal graphs, declared SU2 connection and covariant effect, original 3 panel intents and triple trace test.')
# L24: discrete extrema vs dynamics, bars and structural distinction.
fig,ax=plt.subplots(2,2,figsize=(11,8))
z=[x for x in T['contexts'] if x['kind']=='R78']
for k,name in enumerate(names):
 a=[x['topology']['cubical'] for x in z if x['family']==k]
 ax[0,0].hist([len(x['critical_thresholds']) for x in a],bins=12,alpha=.5,label=name)
 ax[0,1].scatter([x['plateau_min'] for x in a],[x['plateau_max'] for x in a],label=name)
 x=[r for r in a if r['finite_mean_lifetime'] is not None]
 ax[1,0].scatter([r['final_betti'][1] for r in x],[r['finite_mean_lifetime'] for r in x],label=name)
 ax[1,1].plot(range(1,129),np.mean([np.asarray(x['prefix']['curve'])[:,5] for x in G['contexts'] if x['family']==k],0),label=name)
ax[0,0].set(xlabel='distinct persistent birth/death levels',ylabel='contexts');ax[0,1].set(xlabel='discrete minimum plateaus',ylabel='maximum plateaus (not flow stability)');ax[1,0].set(xlabel='final cubical H1',ylabel='mean positive finite lifetime (score)');ax[1,1].set(xlabel='completed backups',ylabel='mean quotient H1 (tree H1=0)')
for a in ax.ravel():a.legend(fontsize=8)
save(fig,'L24','topological_analysis_persistence','48 actual score landscapes and chronological prefix/quotient, GF2 bars and discrete extrema; no continuum Morse stability.')
# G18 explicit oblique 3D projection of actual samples, no synthetic interpolation, legal mask.
fig,ax=plt.subplots(3,2,figsize=(10,12))
for k,name in enumerate(names):
 c=next(x for x in L['contexts'] if x['family']==k and x['core']==0);v=np.array(c['vertices']);n=c['L'];xx=v%n;yy=v//n;u=np.array(c['score']);q=np.array(c['Q']);z0=u.min();zscale=(n-1)*.5/max(float(np.ptp(u)),1e-8)
 X=xx-.55*yy;Y=.34*yy;Z=Y+zscale*(u-z0);a=ax[k,0];norm=Normalize(u.min(),u.max());colors=plt.get_cmap('viridis')(norm(u))
 a.add_collection(LineCollection(np.stack([np.column_stack([X,Y]),np.column_stack([X,Z])],1),colors=colors,linewidths=.65,alpha=.7));a.scatter(X,Z,c=u,cmap='viridis',norm=norm,s=12);a.scatter(X,Y,s=3,c='gray')
 a.plot([0,n-1],[0,0],'k-');a.plot([0,-.55*(n-1)],[0,.34*(n-1)],'k-');a.plot([0,0],[0,zscale*np.ptp(u)],'k-');a.text(n-1,0,'x');a.text(-.55*(n-1),.34*(n-1),'y');a.text(0,zscale*np.ptp(u),f'U={u.max():.2f}');a.text(0,-.4,f'U={z0:.2f}');a.autoscale();a.set_aspect('equal');a.set_axis_off();a.set_title(name+' H32 score, oblique projection')
 board=np.full(n*n,np.nan);board[v]=q;im=ax[k,1].imshow(board.reshape(n,n),vmin=-1,vmax=1,cmap='coolwarm');un=v[~np.asarray(c['visited'])];ax[k,1].scatter(un%n,un//n,s=4,c='black',alpha=.6,label='unvisited Q=0');path=np.array(c['path']);ax[k,1].plot(path%n,path//n,'k-',lw=.8);ax[k,1].scatter([c['source']%n,c['target']%n],[c['source']//n,c['target']//n],marker='*',c='gold',edgecolors='k',s=55);ax[k,1].set_title(name+' signed Q; path from -U');ax[k,1].legend(fontsize=7,loc='lower right');fig.colorbar(im,ax=ax[k,1],shrink=.7)
save(fig,'G18','value_landscape_3d','Actual H32 U samples in explicit oblique projection of (x,y,U); Q with legal mask. No interpolated surface or physical energy.')
# L10 original barrier vs probability plot corrected for risk set (binary observed, binned empirical hit rates).
fig,ax=plt.subplots(1,3,figsize=(12,4))
for k,name in enumerate(names):
 z=[x for x in B0['paths'] if x['eligible'] and x['key'][0]==k];h=np.array([x['height'] for x in z]);ev=np.array([x['event'] for x in z],int);f=np.array([x['frozen_hit'] for x in z]);order=np.argsort(h);bins=np.array_split(order,8);a=ax[k]
 a.scatter(h,ev,s=4,alpha=.08);a.plot([h[i].mean() for i in bins],[ev[i].mean() for i in bins],'o-',label='observed hit / all risk');a.plot([h[i].mean() for i in bins],[f[i].mean() for i in bins],'s--',label='frozen hazard control');a.set(title=name,xlabel='minimax excess score',ylabel='hit by 96 backups',ylim=(-.05,1.05));a.legend(fontsize=7)
save(fig,'L10','tunneling_risk_corrected','All 3071 eligible records, height-sorted descriptive bins; actual target hit and frozen reference, not tunneling probability.')
# G19 native full risk survival and historical recovery.
fig,ax=plt.subplots(1,2,figsize=(11,4))
for k,name in enumerate(names):
 z=[x for x in B0['paths'] if x['eligible'] and x['key'][0]==k];roots=[]
 for core in range(32):
  a=[x for x in z if x['key'][1]==core];roots.append([np.mean([x['tau']>h for x in a]) for h in range(97)])
 ax[0].step(range(97),np.mean(roots,0),where='post',label=name)
for r in R['game']['results']:
 a=r['risk_table'];ax[1].step([0]+[x['age'] for x in a],[1.]+[x['descriptive_KM'] for x in a],where='post',label=f"drop {r['threshold']}, n={r['eligible']}, events={r['events']}")
ax[0].set(xlabel='root selections after H32',ylabel='not yet hit',ylim=(0,1.05));ax[1].set(xlabel='same-side decisions since entry',ylabel='descriptive unrecovered fraction',ylim=(0,1.05),title='Informative game-end censoring possible')
for a in ax:a.legend(fontsize=8)
save(fig,'G19','tunneling_events_risk_recovery','Actual first-hit right censor and historical same-side recovery risk sets; distinct clocks.')
# G20 barrier distributions and certificate
fig,ax=plt.subplots(1,3,figsize=(12,4))
for k,name in enumerate(names):
 z=[x for x in B0['paths'] if x['eligible'] and x['key'][0]==k]
 ax[0].hist([x['height'] for x in z],bins=24,alpha=.45,label=name);ax[1].scatter([x['graph_distance'] for x in z],[x['width'] for x in z],s=8,alpha=.12,label=name)
 a=[x for x in T['contexts'] if x['kind']=='R75' and x['family']==k];b=[x for x in B0['contexts'] if x['family']==k];ax[2].scatter([x['saddle'] for x in b],[x['merge_level'] for x in a],s=12,label=name)
ax[0].set(xlabel='excess barrier height',ylabel='records');ax[1].set(xlabel='shortest graph distance',ylabel='minimax sublevel shortest width');ax[2].plot([-1,2],[-1,2],'k:');ax[2].set(xlabel='minimax level',ylabel='first sublevel connection')
for a in ax:a.legend(fontsize=7)
save(fig,'G20','barrier_statistics','3071 actual minimax certificates and width, 96 independent sublevel connectivity checks.')
# G21 exact WKB asymptotic and continuum controls
fig,ax=plt.subplots(1,3,figsize=(13,4))
for h in sorted({x['height'] for x in Q['rectangular_controls']}):
 z=[x for x in Q['rectangular_controls'] if x['height']==h];ax[0].plot([x['n'] for x in z],[x['logT'] for x in z],'o-',label=f'V={h:g}');ax[0].plot([x['n'] for x in z],[x['discrete_exponent'] for x in z],'--',alpha=.6)
for w in sorted({x['width'] for x in Q['continuum_controls']}):
 z=[x for x in Q['continuum_controls'] if x['width']==w];ax[1].loglog([x['lattice_spacing'] for x in z],[x['error'] for x in z],'o-',label=f'width={w:g}')
for k,name in enumerate(names):
 z=[x for x in Q['contexts'] if x['family']==k];v=[r for x in z for r in x['scattering']];ax[2].scatter([-2*r['action_discrete'] for r in v],[np.log(max(r['path']['T'],1e-300)) for r in v],s=8,alpha=.5,label=name)
ax[0].set(xlabel='rectangular barrier sites',ylabel='log T, solid exact / dashed action');ax[1].set(xlabel='lattice spacing',ylabel='absolute logT continuum error');ax[2].set(xlabel='discrete local action exponent',ylabel='log exact path transmission')
for a in ax:a.legend(fontsize=7)
save(fig,'G21','wkb_analysis','Exact lead scattering and independent transfer, rectangle and continuum controls; WKB exponent not exact general T.')
# G22 summary explicitly distinguishes native and declared units.
fig,ax=plt.subplots(2,2,figsize=(10,7));x=np.arange(3);a=S0['families']
for j,(key,lbl) in enumerate([('mean_full_T_E015','full graph'),('mean_path_T_E015','minimax path')]):ax[0,1].bar(x+(j-.5)*.35,[q[key] for q in a],.35,label=lbl)
for j,(key,lbl) in enumerate([('coherent_hit_mean','coherent'),('dephased_hit_mean','dephased')]):ax[1,0].bar(x+(j-.5)*.35,[q[key] for q in a],.35,label=lbl)
ax[0,0].bar(x,[q['events']/q['eligible'] for q in a]);ax[0,0].set(title='Native first hit / all eligible',ylim=(0,1))
ax[0,1].set(title='Declared lead T at E/E*=0.15');ax[1,0].set(title='Declared detector hit by 64 steps')
ax[1,1].bar(x,[q['nonneighbor_fraction'] for q in a]);ax[1,1].set(title='Native hit preceded by non-neighbor',ylim=(0,1))
for a0 in ax.ravel():a0.set_xticks(x,names)
ax[0,1].legend(fontsize=8);ax[1,0].legend(fontsize=8)
save(fig,'G22','tunneling_summary','Actual native risk and nonlocality versus separately declared stationary and monitored coherent model.')
assert len(figs)==8 and len({r['png_sha'] for r in figs})==8
dump(S/'PLOT_MANIFEST88.json',dict(figures=figs,animations=[],sources={f:sha(S/f) for f in ['GAUGE88.json','TOPOLOGY88.json','BARRIER88.json','TRANSPORT88.json','RECOVERY88.json','LANDSCAPE88.json','SUMMARY88.json','INDEPENDENT88.json','plot88.py']}))
print('PLOTS',len(figs),flush=True)
