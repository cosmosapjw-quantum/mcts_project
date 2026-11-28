from entropy97 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess,io
Q=read(S/'SUMMARY97.json');G=Q['groups'];N=read(P94/'NATIVE94.json');figs=[];animations=[]
P=['native_repeat','pair_diagnostic']
def get(g,p,d,xi=1.,m='plus81'):
 return next(x for x in G if (x['family'],x['protocol'],x['depth'],x['xi'],x['mode'])==(g,p,d,xi,m))
def save(fig,name,ids):
 fig.tight_layout();b=S/'plots'/name
 fig.savefig(str(b)+'.png',dpi=150);fig.savefig(str(b)+'.pdf');plt.close(fig)
 a=Path(str(b)+'.png');q=Path(str(b)+'.pdf')
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(q)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,png_sha=sha(a),pdf_sha=sha(q)))
def band(a,t,x,label):
 z=np.array(x);a.plot(t,z[0],marker='o',ms=3,label=label);a.fill_between(t,z[1],z[2],alpha=.10)
def label(mask,n):return ''.join(('R'+str(i+1) if i<n-1 else 'S') for i in maskset(mask,n))
# Q35 state/purification partition entropy and original mixed cut negativity, no conflation
f,ax=plt.subplots(3,2,figsize=(13,10));f.suptitle('Q35 | declared mixed joint state: entropy and original-cut negativity')
for g in range(3):
 for j,p in enumerate(P):
  x=get(g,p,3);a=ax[g,j];z=np.array(x['cut_ci']);t=np.arange(len(x['cut_masks']))
  for k,l in [(0,'S(A) = pure A:BE cut entropy'),(1,'S(B)'),(2,'I(A:B)'),(5,'N(A:B)')]:band(a,t,z[:,:,k],l)
  a.set(xticks=t,xticklabels=[label(k,4)+'|'+label(15^k,4) for k in x['cut_masks']],ylabel='Entropy in nat; negativity dimensionless',title=FAMS[g]+' | '+p);a.tick_params(axis='x',rotation=25,labelsize=7);a.legend(fontsize=6)
save(f,'Q35_entanglement_analysis',['Q35'])
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q35 | spectra of full record-system and single system reductions')
for g in range(3):
 for p in P:
  x=get(g,p,3)
  for i,k in enumerate([15,8]):
   y=np.array(x['spectrum_ci'][x['subset_masks'].index(k)]);ax[i,g].semilogy(np.arange(1,y.shape[1]+1),np.maximum(y[0],1e-16),'o-',label=p)
 for i in range(2):ax[i,g].set(xlabel='Eigenvalue index (descending)',ylabel='Mean eigenvalue, floor 1e-16',title=FAMS[g]+(' | joint' if i==0 else ' | S'));ax[i,g].legend(fontsize=7)
save(f,'Q35_partition_spectra',['Q35'])
# Q36 finite cut-cardinality diagnostic and native categorical size; no area-law fit.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q36 | finite declared partition diagnostics; not a spatial area law')
for g in range(3):
 for p in P:
  x=get(g,p,3);z=np.array(x['subset_ci'])[0,:,0]
  xx=[k.bit_count() for k in x['subset_masks']];ax[0,g].scatter(xx,z,label=p,alpha=.65)
  for d in [1,2,3]:
   v=get(g,p,d);y=np.array(v['cut_ci'])[0,:,5];a=[min(k.bit_count(),(2**(d+1)-1^k).bit_count()) for k in v['cut_masks']]
   ax[1,g].scatter(a,y,label=p+' d='+str(d),alpha=.65)
 ax[0,g].set(xlabel='Subset cardinality (qubits)',ylabel='Mean S(A), nat',title=FAMS[g],xticks=[1,2,3,4]);ax[1,g].set(xlabel='Smaller cut cardinality',ylabel='Original mixed-cut negativity',xticks=[1,2]);ax[0,g].legend(fontsize=7);ax[1,g].legend(fontsize=5)
save(f,'Q36_finite_partition_scaling_partial',['Q36'])
f,ax=plt.subplots(1,2,figsize=(10,4));f.suptitle('Q36 | R92 independent board sizes: categorical entropy only')
for a,arm in zip(ax,[0,1]):
 for beta in sorted({x['beta'] for x in Q['scale'] if x['arm']==arm}):
  z=sorted([x for x in Q['scale'] if x['arm']==arm and x['beta']==beta],key=lambda x:x['L']);band(a,[x['L'] for x in z],np.array([x['ci'] for x in z]).T,str(round(beta,3)))
 a.set(xlabel='Board linear size L',ylabel='Normalized categorical entropy',title='R92 fresh cohort arm '+str(arm));a.legend(fontsize=7,title='score beta')
save(f,'Q36_native_size_entropy_comparator',['Q36','Q40'])
# Q37 bounds per qubit measured
f,ax=plt.subplots(3,2,figsize=(12,10));f.suptitle('Q37 | one-sided discord bounds: finite measurement search')
for g in range(3):
 for j,p in enumerate(P):
  a=ax[g,j];x=get(g,p,3);z=np.array(x['measurement_ci']);t=np.arange(4)
  for k,l in [(4,'Fixed Z gap'),(5,'Optimized discord lower'),(6,'Optimized discord upper')]:band(a,t,z[:,:,k],l)
  a.fill_between(t,z[0,:,5],z[0,:,6],alpha=.18);a.set(xticks=t,xticklabels=['R1','R2','R3','S'],ylabel='nat',title=FAMS[g]+' | '+p);a.legend(fontsize=7)
save(f,'Q37_information_flow_discord',['Q37'])
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q37 | quantum MI, actual measured classical MI, and measurement-induced gap')
for g in range(3):
 for p in P:
  for i,xi in enumerate([0.,1.]):
   x=get(g,p,3,xi);z=np.array(x['measurement_ci'])[0];t=np.arange(4)
   ax[i,g].plot(t,z[:,0],'o-',label=p+' QMI');ax[i,g].plot(t,z[:,1],'s--',label=p+' Z MI');ax[i,g].plot(t,z[:,3],'^:',label=p+' best Holevo')
 for i,xi in enumerate([0.,1.]):ax[i,g].set(xticks=range(4),xticklabels=['R1','R2','R3','S'],ylabel='nat',title=FAMS[g]+' | overlap '+str(xi));ax[i,g].legend(fontsize=5)
save(f,'Q37_measurement_comparison',['Q37'])
# Q39 independently labeled native game ply and declared record clock, no implied synchronized time.
def native_panel(a,upto=56):
 for co in ['external','internal']:
  for arm in ['clean','single','recovery']:
   z=[x for x in Q['risk'] if x['cohort']==co and x['arm']==arm and x['ply']<=upto]
   if z:a.plot([x['ply'] for x in z],[x['ci'][0][0] for x in z],'o-' if co=='external' else 's--',ms=3,label=co+' '+arm)
 a.set(xlim=(-2,58),ylim=(0,4.5),xlabel='Actual game ply bin start',ylabel='Visit Shannon entropy (nat)');a.legend(fontsize=6,ncol=2)
def risk_panel(a,upto=56):
 for co in ['external','internal']:
  for arm in ['clean','single','recovery']:
   z=[x for x in Q['risk'] if x['cohort']==co and x['arm']==arm and x['ply']<=upto]
   if z:a.plot([x['ply'] for x in z],[x['games'] for x in z],'o-' if co=='external' else 's--',ms=3,label=co+' '+arm)
 a.set(xlim=(-2,58),ylim=(0,40),xlabel='Actual game ply bin start',ylabel='Distinct contributing games in bin')
def prefix_panel(a,g=0,d=3):
 for p in P:
  z=[get(g,p,k) for k in range(d+1)];t=[32+4*k for k in range(d+1)]
  a.plot(t,[x['mean_ci'][0][0] for x in z],'o-',label=p+' joint S')
  a.plot(t,[x['subset_ci'][0][x['subset_masks'].index(1<<x['depth'])][0] for x in z],'s--',label=p+' system S')
 a.set(xlim=(31,45),ylim=(0,2.2),xlabel='Declared record clock',ylabel='von Neumann entropy (nat)',title=FAMS[g]);a.legend(fontsize=6)
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q39 | independent native and declared quantum entropy clocks')
native_panel(ax[0,0]);risk_panel(ax[0,1])
for g in range(3):
 x=np.array(N['families'][g]['features_ci']);band(ax[0,2],np.arange(32,44),x[:,0,:,8],FAMS[g])
ax[0,2].set(xlabel='Actual software-qubit selection clock',ylabel='Conditional entropy (nat)');ax[0,2].legend(fontsize=6)
for g in range(3):prefix_panel(ax[1,g],g)
save(f,'Q39_entropy_clock_snapshots',['Q39'])
def gif(name,frames,fn,clocks):
 images=[]
 for v in clocks:
  f=fn(v);f.tight_layout();b=io.BytesIO();f.savefig(b,format='png',dpi=100);b.seek(0);images.append(Image.open(b).convert('RGB'));plt.close(f)
 p=S/'plots'/name;images[0].save(p,save_all=True,append_images=images[1:],duration=700,loop=0)
 with Image.open(p) as im:
  assert im.n_frames==frames;hs=[]
  for k in range(frames):im.seek(k);hs.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
  assert len(set(hs))==frames
 animations.append(dict(id='Q39',path='plots/'+name,sha256=sha(p),frames=frames,frame_rgb_sha256=hs,clocks=clocks))
def nframe(v):
 f,ax=plt.subplots(1,2,figsize=(10,4));f.suptitle('Native R68 actual game entropy | ply bin '+str(v)+'; changing risk set');native_panel(ax[0],v);risk_panel(ax[1],v);return f
def qframe(v):
 f,ax=plt.subplots(1,3,figsize=(12,4));f.suptitle('Declared joint prefix entropy | H'+str(v)+' (not game time)')
 for g in range(3):prefix_panel(ax[g],g,(v-32)//4)
 return f
gif('Q39_native_entropy_evolution.gif',8,nframe,list(range(0,64,8)))
gif('Q39_declared_partition_evolution.gif',4,qframe,[32,36,40,44])
# Q40 comprehensive declared reference; no native temperature or physical area law.
f,ax=plt.subplots(2,2,figsize=(10,7));f.suptitle('Q40 | fixed-spectrum reference thermodynamics, not native bath')
t=Q['thermal'];v=np.array(t['curve_ci'])
for a,k,l in zip(ax.flat,range(4),['U (score-energy units)','S/kB (nat)','C/kB','F (score-energy units)']):
 band(a,t['theta'],v[:,:,k],l);a.set(xscale='log',xlabel='theta = kB T in declared energy units',ylabel=l);a.legend(fontsize=7)
save(f,'Q40_declared_thermal_reference',['Q40'])
dump(S/'PLOT_MANIFEST97.json',dict(figures=figs,animations=animations,sources={k:sha(S/k) for k in ['SUMMARY97.json','ENTROPY97.json','ENTROPY97.json.gz','NATIVE_THERMAL97.json','INDEPENDENT97.json','ORIGINAL_SOURCE97.json']},scope='Separate real visit, programmed quantum and declared reference panels; no spatial area law or native thermal calibration'))
print('PLOTS',len(figs),'GIFS',[(x['frames'],x['path']) for x in animations],flush=True)
