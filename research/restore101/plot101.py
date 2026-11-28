from field101 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io,subprocess
F=read(S/'FIELD101.json');Q=read(S/'QUANTUM101.json');A=read(S/'SUMMARY101.json')
with np.load(S/'QUANTUM101.npz') as z:M=z['meta'];V=z['values']
G=Q['groups'];rows=Q['record_rows'];fams=['g0','g3','Rapfi'];arms=['plus','minus','erase+','erase-','swap','base'];clock=np.arange(32,44);figs=[];animations=[]
def save(f,n,ids):
 f.tight_layout();p=S/'plots'/n;f.savefig(str(p)+'.png',dpi=145);f.savefig(str(p)+'.pdf');plt.close(f)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=n,ids=ids,png_sha=sha(a),pdf_sha=sha(b)))
def legend(ax):ax.legend(fontsize=7)
def group(g,a):return next(x for x in G if (x['family'],x['arm'])==(g,a))
def band(ax,y,k,label):
 y=np.asarray(y)[:,:,k];ax.plot(clock,y[0],'-o',ms=2,label=label);ax.fill_between(clock,y[1],y[2],alpha=.11)
# Q41: explicitly unit-action branch convention and channel mismatch.
f,ax=plt.subplots(1,3,figsize=(13,4));f.suptitle('Q41 | visibility convention and the inequivalence of rotation and dephasing')
for g in range(3):
 a=V[M[:,0]==g];m=a[:,:,3]>0;ax[0].scatter(a[:,:,0][m],a[:,:,2][m],s=1,alpha=.12,label=fams[g])
 ax[1].scatter(a[:,:,0].ravel(),a[:,:,6].ravel(),s=1,alpha=.1,label=fams[g])
 v=group(g,0);ax[2].plot(clock,v['finite_hbar_mean'],'o-',label=fams[g]+' plus')
ax[0].axhline(2/np.pi,color='gray',ls=':',label='principal branch floor');ax[0].set(xlabel='Local visibility gamma',ylabel='h* = 1/acos(gamma)',yscale='log')
ax[1].set(xlabel='Same X expectation',ylabel='Trace distance: dephase vs rotation')
ax[2].set(xlabel='Actual selection clock',ylabel='Finite proxy mean (unit action)')
for a in ax:legend(a)
save(f,'Q41_visibility_branch_and_state_mismatch',['Q41','Q45'])
# Q42 exact local semigroup segments and frozen sequence; no identification of global process.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q42/L20 | history-frozen unselective CPTP sequence, separate from selective trajectory')
for g in range(3):
 for a,l in [(0,'plus'),(2,'erase+'),(4,'swap'),(5,'base')]:
  v=group(g,a);band(ax[0,g],v['ci'],1,l);band(ax[1,g],v['ci'],7,l)
 ax[0,g].set(title=fams[g],xlabel='Selection clock',ylabel='Dimensionless dephase rate')
 ax[1,g].set(xlabel='Selection clock',ylabel='Order commutator Frobenius norm')
 legend(ax[0,g]);legend(ax[1,g])
save(f,'Q42_CP_segments_and_noncommutation',['Q42','L20','Q45'])
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q42/L20 | frozen sequence singular values and singular erasure')
for g in range(3):
 for a,l in [(0,'plus'),(2,'erase+'),(4,'swap'),(5,'base')]:
  v=group(g,a);band(ax[0,g],v['ci'],13,l);band(ax[1,g],v['ci'],10,l)
 ax[0,g].set(title=fams[g],xlabel='Selection clock',ylabel='Min cumulative Bloch singular value')
 ax[1,g].set(xlabel='Selection clock',ylabel='Local map determinant')
 legend(ax[0,g]);legend(ax[1,g])
save(f,'Q42_frozen_channel_spectrum',['Q42','L20'])
# L09 actual conditional selected states, contrast with erase.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('L09 | actual conditional programmed register, no global relaxation fit')
for g in range(3):
 for a,l in [(0,'plus'),(1,'minus'),(2,'erase+'),(4,'swap'),(5,'base')]:
  v=group(g,a);band(ax[0,g],v['ci'],17,l);band(ax[1,g],v['ci'],16,l)
 ax[0,g].set(title=fams[g],xlabel='Selection clock',ylabel='Conditional transverse coherence')
 ax[1,g].set(xlabel='Selection clock',ylabel='Conditional Bloch z')
 legend(ax[0,g]);legend(ax[1,g])
save(f,'L09_conditional_coherence_controls',['L09','Q42','Q45'])
# Q43: path-space W, dual, mixed response and heldout.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q43 partial | actual finite empirical path generating law and dual')
for g in range(3):
 r=[x for x in F['rows'] if x['family']==g];rs=[v for a in r for v in a['values'] if v['h']==0]
 ax[0,g].scatter([v['W'] for v in rs],[v['dual'] for v in rs],s=4,alpha=.3)
 ax[0,g].set(title=fams[g],xlabel='W(J,0)',ylabel='Gamma(mean)')
 for j,l in enumerate(['joint train','factorized train']):
  x=np.array([v['test'] for v in rs]);y=np.array([v['train' if j==0 else 'factorized_train'] for v in rs])
  ax[1,g].scatter(x,y,s=4,alpha=.25,label=l)
 xx=np.array(ax[1,g].get_xlim());ax[1,g].plot(xx,xx,'k:');ax[1,g].set(xlabel='Held-out empirical W',ylabel='Train prediction');legend(ax[1,g])
save(f,'Q43_finite_path_dual_heldout_partial',['Q43','Q45'])
f,ax=plt.subplots(1,3,figsize=(13,4));f.suptitle('Q43 partial | direct causal pulse and finite normalized path-score response')
for g in range(3):
 x=F['groups'][g];n=np.arange(4)
 for j,(k,l) in enumerate([('direct_ci','Direct finite pulse'),('reweighted_ci','Normalized reweight'),('score_ci','Score covariance')]):
  v=np.array(x[k]);ax[g].errorbar(n+.16*(j-1),v[0],yerr=np.stack([v[0]-v[1],v[2]-v[0]]),fmt='o',ms=4,label=l)
 ax[g].axhline(0,color='gray',lw=.5);ax[g].set(title=fams[g],xticks=n,xticklabels=['hit pre','mark pre','hit post','mark post'],ylabel='Mean derivative / finite difference');legend(ax[g])
save(f,'Q43_path_causal_response_partial',['Q43','Q45'])
# L11 record matrix PSD and local phase covariance; separate postselection.
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('L11/Q43 | complex record phase tag leaves diagonal and entanglement invariant')
for j,g in enumerate(range(3)):
 for p,l in [('native_repeat','Native-repeat'),('pair_diagnostic','Pair diagnostic')]:
  a=[x for x in rows if x['family']==g and x['mode']=='plus81' and x['protocol']==p and x['depth']==3]
  for i,key in [(0,'N'),(1,'phase_TV')]:
   t=[0.,.5,1.];v=[np.mean([x[key] for x in a if x['xi']==u]) for u in t];ax[i,j].plot(t,v,'o-',label=l)
 ax[0,j].set(title=fams[g],xlabel='Record overlap xi',ylabel='System:record negativity')
 ax[1,j].set(xlabel='Record overlap xi',ylabel='Fixed-H readout TV after local phase')
 legend(ax[0,j]);legend(ax[1,j])
save(f,'L11_complex_record_phase_and_entanglement',['L11','Q43','Q45'])
f,ax=plt.subplots(1,3,figsize=(13,4));f.suptitle('Q43/L11 | conditional phase readout versus success-weighted observable')
for j,g in enumerate(range(3)):
 for p,l in [('native_repeat','Native-repeat'),('pair_diagnostic','Pair diagnostic')]:
  a=[x for x in rows if x['family']==g and x['mode']=='plus81' and x['protocol']==p and x['depth']==3 and x['xi']==1]
  ax[j].scatter([x['success'] for x in a],[x['weighted_phase_TV'] for x in a],s=9,alpha=.7,label=l)
 ax[j].set(title=fams[g],xlabel='Selected record success',ylabel='Success x readout TV',xscale='log',yscale='log');legend(ax[j])
save(f,'Q43_record_operational_weight_partial',['Q43','L11'])
# Q44 actual 12 selection frames and companion plot.
def frame(k):
 f,ax=plt.subplots(1,2,figsize=(10,4));f.suptitle(f'Q44 | local visibility proxy, actual selection clock {clock[k]}')
 for g in range(3):
  v=group(g,0);ax[0].plot(clock[:k+1],v['finite_hbar_mean'][:k+1],'-o',ms=3,label=fams[g]+' plus')
  for a,l in [(0,'plus'),(5,'base')]:
   z=group(g,a);ax[1].plot(clock[:k+1],np.array(z['finite_hbar_count'][:k+1])/192,'-o',ms=2,label=fams[g]+' '+l)
 maxv=max(max(group(g,0)['finite_hbar_mean']) for g in range(3));ax[0].set(xlim=(31.8,43.2),ylim=(0,maxv*1.08),xlabel='Selection clock',ylabel='Finite dimensionless h* mean')
 ax[1].set(xlim=(31.8,43.2),ylim=(-.05,1.08),xlabel='Selection clock',ylabel='Finite proxy fraction (base undefined)')
 for a in ax:legend(a)
 return f
save(frame(11),'Q44_visibility_clock_snapshot',['Q44','Q41','Q45'])
ims=[]
for k in range(12):
 f=frame(k);f.tight_layout();b=io.BytesIO();f.savefig(b,format='png',dpi=95);b.seek(0);ims.append(Image.open(b).convert('RGB'));plt.close(f)
p=S/'plots/Q44_actual_visibility_clock.gif';ims[0].save(p,save_all=True,append_images=ims[1:],duration=550,loop=0)
with Image.open(p) as im:
 assert im.n_frames==12;hs=[]
 for i in range(12):im.seek(i);hs.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
 assert len(set(hs))==12
animations.append(dict(id='Q44',path='plots/'+p.name,sha256=sha(p),frames=12,frame_rgb_sha256=hs,clock=clock.tolist()))
dump(S/'PLOT_MANIFEST101.json',dict(figures=figs,animations=animations,sources={k:sha(S/k) for k in ['FIELD101.json','QUANTUM101.json','QUANTUM101.npz','SUMMARY101.json','INDEPENDENT101.json','ORIGINAL_SOURCE101.json']},scope='Actual finite path generating law; complex history-conditioned auxiliary instrument, not physical hbar or quantum field.'))
print('PLOTS',len(figs),'GIFS',[(a['id'],a['frames']) for a in animations],flush=True)
