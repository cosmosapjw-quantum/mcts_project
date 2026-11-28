from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
P=read(S/'PATH90.json');D=read(S/'DUAL90.json');R=read(S/'RESPONSE90.json');W=read(S/'WORK90.json');Q=read(S/'PHASE90.json');I=read(S/'INDEPENDENT90.json')
with np.load(S/'PATH90.npz') as z:A={k:z[k] for k in z.files}
p81=B/'mqft_restore81_20261127/PATHS81.npz'
with np.load(p81) as z:M81=z['meta'];ST=z['states']
F=['g0','g3','Rapfi'];C=['#2166ac','#b2182b','#25845b'];figures=[]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,'axes.grid':True,'grid.alpha':.15})
def save(fig,name,ids,scope):
 fig.tight_layout(rect=[0,0,1,.95]);b=S/'plots'/name
 fig.savefig(str(b)+'.png',dpi=180);fig.savefig(str(b)+'.pdf');plt.close(fig)
 with Image.open(str(b)+'.png') as im:im.verify()
 subprocess.run(['pdfinfo',str(b)+'.pdf'],stdout=subprocess.DEVNULL,check=True)
 figures.append(dict(figure=name,ids=ids,scope=scope,png_sha=sha(Path(str(b)+'.png')),pdf_sha=sha(Path(str(b)+'.pdf'))))
def barci(ax,x,r,label,c):
 m,l,h=r;ax.errorbar(x,m,yerr=[[m-l],[h-m]],fmt='o',color=c,label=label,capsize=3)
def pathmask(g):return (A['meta'][:,0]==4)&(A['meta'][:,1]==g)
fig,ax=plt.subplots(2,3,figsize=(13,8));fig.suptitle('L16 | Marked statistical path measure (R86 independent geometries)')
for g in range(3):
 ix=pathmask(g);ax[0,0].hist(-A['logp'][ix].sum(1),bins=24,density=True,histtype='step',color=C[g],label=F[g])
ax[0,0].set(xlabel='S = -log P(history)',ylabel='density');ax[0,0].legend()
for i,s in enumerate(P['dominant']['prefix_actions']):ax[0,1].plot(np.arange(1,129),-np.array(s),label=f'rank {i+1}')
ax[0,1].set(xlabel='completed selections',ylabel='log prefix probability',title='Five observed paths, one root');ax[0,1].legend(fontsize=7)
q=np.array(P['representative']['Q']);v=np.array(P['representative']['visited']);q[~v]=np.nan
im=ax[0,2].imshow(q.reshape(9,9),vmin=-1,vmax=1,cmap='coolwarm');fig.colorbar(im,ax=ax[0,2],shrink=.75);ax[0,2].set_title('H32 visited Q; blank = unvisited/illegal');ax[0,2].grid(False)
for g in range(3):
 z=A['overlap'][(A['overlap'][:,0]==4)&(A['overlap'][:,1]==g)];ax[1,0].scatter(z[:,3],z[:,4],s=4,alpha=.16,color=C[g],label=F[g])
ax[1,0].set(xlabel='ordered Hamming distance',ylabel='occupancy affinity',title='Classical overlap');ax[1,0].legend()
p=P['dominant']['observed_relative_weight'];ax[1,1].bar(np.arange(1,len(p)+1),p,color=C[0]);ax[1,1].set(yscale='log',xlabel='observed path rank',ylabel='relative mass within 16 paths',title='Not global mode probability')
for arm,l in [(0,'plus'),(1,'minus'),(2,'erase+')]:
 z=ST[M81[:,3]==arm];ax[1,2].plot(np.arange(1,13),np.sqrt((z[:,:,:2]**2).sum(2)).mean(0),label=l)
ax[1,2].set(xlabel='active selection',ylabel='mean transverse Bloch magnitude',title='Programmed register, post-state');ax[1,2].legend()
save(fig,'L16_path_integral',['L16'],'Actual paths; classical overlap and separately programmed register coherence.')
fig,ax=plt.subplots(2,3,figsize=(13,8));fig.suptitle('L16 | Finite-support effective action and replica expansion')
for g,f in enumerate(F):
 z=A['Y'][pathmask(g)]/128;ax[0,g].scatter(z[:,0],z[:,1],s=5,alpha=.25,label='R86')
 for lab,col in [('R80_reference','black'),('R86',C[g])]:
  h=np.array(next(x['hull'] for x in D['laws'] if x['label']==[f,lab]))/128;h=np.vstack([h,h[0]]);ax[0,g].plot(*h.T,color=col,label=lab)
 ax[0,g].set(xlabel='pair count / 128',ylabel='positive mark count / 128',title=f);ax[0,g].legend(fontsize=8)
for x in D['laws']:
 for e in x['edges']:
  s=e['cases'][1]['normal_scan'];ax[1,0].plot([t['normal_t'] for t in s],np.maximum([t['outside_mass'] for t in s],1e-120),alpha=.17,color=C[F.index(x['label'][0])])
ax[1,0].set(yscale='log',xlabel='normal tilt / minimum gap',ylabel='off-face mass',title='124 exposed edges')
ms=sorted({x['M'] for x in D['coeff']})
for key,lab in [('leading_error','leading'),('corrected_error','corrected')]:ax[1,1].loglog(ms,[max(x[key] for x in D['coeff'] if x['M']==m) for m in ms],'-o',label=lab)
ax[1,1].set(xlabel='independent history replicas M',ylabel='max relative error',title='180 stored saddle cases');ax[1,1].legend()
for g,x in enumerate(I['transfer']):
 ax[1,2].bar(g-.18,x['energy'][0],.36,color='#aaaaaa');ax[1,2].bar(g+.18,x['energy'][1],.36,color=C[g])
ax[1,2].set(xticks=range(3),xticklabels=F,ylabel='held-out energy score',title='R84 root-only vs combined')
save(fig,'L16_effective_action',['L16'],'Empirical convex support, face limit, independently checked statistical expansion and transfer.')
fig,ax=plt.subplots(1,3,figsize=(12,4));fig.suptitle('G03 | Legal-support-normalized Shannon entropies')
for g in range(3):
 z=A['profiles'][pathmask(g)];b=z[0,:,0]
 for k,l in [(1,'visit'),(2,'prior'),(3,'mean conditional')]:ax[g].plot(b,(z[:,:,k]/z[:,:,7]).mean(0),'-o',label=l)
 ax[g].set(xscale='log',xlabel='completed selections',ylabel='H / log |legal|',ylim=(0,1.05),title=F[g]);ax[g].legend(fontsize=8)
save(fig,'G03_entropy',['G03'],'R86 actual cuts, legal support, prior and conditional entropy distinguished.')
fig,ax=plt.subplots(2,3,figsize=(13,7));fig.suptitle('G07 | Matched source pulse: causal response and score identity')
for g,r in enumerate(R['results']):
 for j,title in [(0,'source fraction'),(1,'signed backup mark')]:
  a=ax[j,g]
  for k,l,c in [('direct_fd','direct finite h',C[0]),('score_derivative','score derivative',C[1]),('count_covariance','naive count cov.','#888888')]:
   v=np.array(r[k])[:,:,j];a.plot(range(32),v[0],label=l,color=c);a.fill_between(range(32),v[1],v[2],alpha=.09,color=c)
  a.axvline(8,color='black',ls=':');a.set(xlabel='4-selection wave',ylabel='response',title=F[g]+' '+title);a.legend(fontsize=7)
save(fig,'G07_response',['G07'],'R74 paired source pulse, 16 geometry blocks per family; finite h differs from derivative.')
fig,ax=plt.subplots(2,2,figsize=(11,7));fig.suptitle('Q32 | Conditional score response and corrected finite-size fluctuation')
for g,r in enumerate(R['results']):
 z=np.array(r['baseline_profiles']);ax[0,0].plot(range(32),z[0,:,7],label=F[g],color=C[g]);ax[0,1].plot(range(32),z[0,:,3],label=F[g],color=C[g])
for a,y in [(ax[0,0],'q(1-q)'),(ax[0,1],'beta^2 Var(score)')]:a.set(xlabel='wave',ylabel=y);a.legend()
for arm in [0,1]:
 z=[r for r in I['scale'] if r['arm']==arm];L=sorted({r['L'] for r in z});a=ax[1,arm]
 for beta in sorted({r['beta'] for r in z}):
  t=sorted([r for r in z if r['beta']==beta],key=lambda x:x['L']);a.plot([r['L'] for r in t],[r['trace'] for r in t],'-o',label=f'beta={beta:.3g}')
 a.set(xlabel='board size L',ylabel='per-selection innovation trace',title=f'R87 corrected arm {arm}');a.legend(fontsize=8)
save(fig,'Q32_susceptibility',['Q32'],'Dimensionless score response only; R87 corrected startup schedule; no physical temperature or critical claim.')
fig,ax=plt.subplots(2,2,figsize=(11,7));fig.suptitle('Q21 | Declared cyclic feedback work and separate phase intervention')
for prot in sorted({r['protocol'] for r in W['cases']}):
 z=[r for r in W['cases'] if r['protocol']==prot and r['alpha']==1.0 and r['eps']==.2];ax[0,0].plot(z[0]['source'],np.mean([r['logMGF'] for r in z],0),'-o',label=prot)
ax[0,0].set(xlabel='work source z',ylabel='mean log MGF',title='Declared finite heat bath');ax[0,0].legend(fontsize=8)
for eps in sorted({r['eps'] for r in W['cases']}):
 z=[r for r in W['cases'] if r['eps']==eps];ax[0,1].scatter([r['reverse_support'] for r in z],[r['ift'] for r in z],s=7,alpha=.3,label=str(eps))
ax[0,1].plot([0,1],[0,1],color='black',ls=':');ax[0,1].set(xlabel='reverse support mass',ylabel='information-corrected IFT');ax[0,1].legend(title='eps',fontsize=7)
for g,r in enumerate(Q['families']):
 barci(ax[1,0],g-.12,r['first_ci'],'observed' if g==0 else None,C[0]);barci(ax[1,0],g+.12,r['predicted_ci'],'conditional' if g==0 else None,C[1])
ax[1,0].set(xticks=range(3),xticklabels=F,ylabel='first signed contrast');ax[1,0].legend()
for k in ['native_repeat','pair_diagnostic']:
 z=[r['protocols'][k] for r in Q['records']];ax[1,1].scatter([r['conditional_tv'] for r in z],np.maximum([r['stopped_tv'] for r in z],1e-40),s=12,alpha=.6,label=k)
ax[1,1].set(yscale='log',xlabel='conditional Hadamard TV',ylabel='stopped TV',title='Declared record dilation');ax[1,1].legend(fontsize=8)
save(fig,'Q21_work_path',['Q21'],'Declared Hamiltonian/protocol plus actual programmed phase and separate complex record readout.')
fig,ax=plt.subplots(1,2,figsize=(10,4));fig.suptitle('Q43 | Partial reconstruction: physical hbar calibration missing')
for k in ['native_repeat','pair_diagnostic']:
 z=[r['protocols'][k] for r in Q['records']];ax[0].plot([0,.5,1],np.mean([r['tag_tv'] for r in z],axis=0),'-o',label=k)
ax[0].set(xlabel='declared environmental tag overlap',ylabel='conditional readout TV');ax[0].legend(fontsize=8)
for key,l in [('leading_error','leading'),('corrected_error','corrected')]:ax[1].loglog(1/np.array(ms),[max(x[key] for x in D['coeff'] if x['M']==m) for m in ms],'-o',label=l)
ax[1].set(xlabel='1/M (statistical replica scale, NOT hbar)',ylabel='max relative error');ax[1].legend()
save(fig,'Q43_partial_hbar',['Q43'],'Partial only: no physical hbar or native quantum-classical calibration.')
dump(S/'PLOT_MANIFEST90.json',dict(figures=figures,sources={k:sha(S/k) for k in ['PATH90.json','PATH90.npz','DUAL90.json','RESPONSE90.json','WORK90.json','PHASE90.json','INDEPENDENT90.json','plot90.py']},prior_sources={str(p81.relative_to(B)):sha(p81)}))
print('PLOTS',len(figures),flush=True)
