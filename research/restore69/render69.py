from records69 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.animation import FuncAnimation,PillowWriter
r=read(S/'PHYSICS_RESULTS.json');b=read(S/'BLOCK_RESULTS69.json')
groups=[x for x in r['summaries'] if x['group']=='all'];names=['Historical R65 conditional','Fresh R69 conditional','Rapfi frozen auxiliary'];col=['#527da0','#25866e','#b46b51']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.dpi':120})
man=[];an=[]
def save(fig,name,ids):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=160);fig.savefig(str(p)+'.pdf');plt.close(fig)
 with Image.open(str(p)+'.png') as im:im.verify()
 assert Path(str(p)+'.pdf').read_bytes().startswith(b'%PDF')
 man.append(dict(figure=name,ids=ids,png_sha=sha(Path(str(p)+'.png')),pdf_sha=sha(Path(str(p)+'.pdf'))))
def curve(ax,key,y):
 for i,x in enumerate(groups):ax.plot(range(1,5),x['stats'][key],'o-',ms=3,color=col[i],label=names[i])
 ax.set(xlabel='Retained draws / completed wave',ylabel=y,xticks=[1,2,3,4]);ax.legend(fontsize=7)
fig,ax=plt.subplots(1,3,figsize=(12.8,4.3))
for a,k,y in zip(ax,['coherence','Ssys','linear_entropy'],['Pointer-basis coherence','System von Neumann entropy (nat)','System linear entropy']):curve(a,k,y)
fig.suptitle('Exact nonselective instrument dynamics within actual four-draw wave')
save(fig,'G13_G14_G17_Q24_decoherence_entropy',['G13','G14','G17','Q24','Q28'])
fig,ax=plt.subplots(1,3,figsize=(12.8,4.2))
for i,x in enumerate(groups):
 for a,k,y in [(ax[0],'chi','Holevo information (nat)'),(ax[1],'chi_ratio','Information / pointer entropy')]:
  a.plot(range(1,5),x['stats'][k],'o-',color=col[i],label=names[i]+' χ')
  a.plot(range(1,5),x['stats']['classical' if k=='chi' else 'classical_ratio'],'x--',color=col[i],label=names[i]+' readout')
  a.set(xlabel='Fragment records f',ylabel=y,xticks=[1,2,3,4]);a.legend(fontsize=6)
ax[1].axhline(.9,c='gray',ls=':')
xx=np.arange(3);ax[2].bar(xx-.17,[x['stats']['chi_achieved90']/x['n'] for x in groups],.34,label='χ ≥ .9 H(Z)');ax[2].bar(xx+.17,[x['stats']['classical_achieved90']/x['n'] for x in groups],.34,label='Readout ≥ .9 H(Z)');ax[2].set(xticks=xx,xticklabels=['R65','R69','Rapfi aux'],ylabel='Fraction of wave states',title='Full wave achieves threshold');ax[2].legend(fontsize=7)
fig.suptitle('Pointer record information, not final-decision redundancy; no forced plateau')
save(fig,'L15_L30_Q25_pointer_information',['L15','L30','Q25','Q26','G16','Q28'])
fig,ax=plt.subplots(1,3,figsize=(12.9,4.2))
for a,k,y in zip(ax,['qmi','discord_Z','negativity'],['System–fragment QMI (nat)','QMI − pointer Holevo (nat)','System–fragment negativity']):curve(a,k,y)
fig.suptitle('Quantum correlation of explicit system and coherent records; conditional state')
save(fig,'L11_Q35_Q37_quantum_correlation',['L11','Q35','Q37','G27'])
fig,ax=plt.subplots(1,3,figsize=(12.8,4.2));curve(ax[0],'concurrence','System–fragment concurrence')
curve(ax[1],'CHSH','Optimal CHSH (two-qubit compression)');ax[1].axhline(2,c='gray',ls=':');xx=np.arange(3)
ax[2].bar(xx,[x['stats']['CHSH_violation_counts'][-1]/x['n'] for x in groups],color=col)
ax[2].set(xticks=xx,xticklabels=['R65','R69','Rapfi aux'],ylabel='Fraction with model CHSH > 2',title='All four coherent records')
fig.suptitle('Explicit model observables, not physical Bell experiment or branch PR proxy')
save(fig,'G25_G26_concurrence_bell',['G25','G26','G27','L11','Q35'])
with np.load(S/'RECORD_STATES69.npz') as z:
 qmi=float(z['fresh_R69_qmi'][:,0].mean());rr=float(z['fresh_R69_record_pair_MI'][:,0].mean());ne=float(z['fresh_R69_negativity'][:,0].mean())
fig,ax=plt.subplots(1,2,figsize=(10.4,4.6))
M=np.full((5,5),rr);M[0,1:]=M[1:,0]=qmi;np.fill_diagonal(M,np.nan)
im=ax[0].imshow(M,cmap='viridis',vmin=0);ax[0].set(xticks=range(5),yticks=range(5),xticklabels=['S','R1','R2','R3','R4'],yticklabels=['S','R1','R2','R3','R4'],title='Mean per-state QMI (nat)');fig.colorbar(im,ax=ax[0],fraction=.046)
for i in range(5):
 for j in range(5):
  if i!=j:ax[0].text(j,i,f'{M[i,j]:.3f}',ha='center',va='center',color='white' if M[i,j]<qmi*.6 else 'black',fontsize=8)
pos=np.r_[[[0,0]],np.stack([np.cos(np.arange(4)*np.pi/2+np.pi/4),np.sin(np.arange(4)*np.pi/2+np.pi/4)],1)]
for i in range(5):
 for j in range(i+1,5):
  ax[1].plot(pos[[i,j],0],pos[[i,j],1],'-' if i==0 else '--',c=col[1] if i==0 else '#888888',lw=1+20*M[i,j])
ax[1].scatter(pos[:,0],pos[:,1],s=650,c=[col[1]]+['#aaaaaa']*4,zorder=3)
for k,(x,y) in enumerate(pos):ax[1].text(x,y,['S','R1','R2','R3','R4'][k],ha='center',va='center',color='white',zorder=4)
ax[1].set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),aspect='equal',title=f'Record MI network\nS–R negativity={ne:.4f}; R–R exactly separable');ax[1].axis('off')
fig.suptitle('Actual R69 wave-state ensemble; mean of each defined record partition, not game-tree branch network')
save(fig,'G23_G24_record_information_network',['G23','G24','G27','Q37'])
with np.load(S/'MULTITIME69.npz') as z:
 c=z['cross'];meta=z['meta'];modes=list(z['modes'])
fig,ax=plt.subplots(2,2,figsize=(11.6,7.5))
for k,mode in enumerate(modes):
 a=c[np.flatnonzero(meta[:,0]==3),k]
 for a0,j,y in [(ax[0,0],1,'Pointer coherence'),(ax[0,1],2,'Memory entropy (nat)'),(ax[1,0],3,'Adjacent-wave record QMI (nat)'),(ax[1,1],4,'Record-basis classical MI (nat)')]:
  n=31 if j in (3,4) else 32;v=a[:,:n,j].mean(0)*(2 if j==1 else 1);a0.plot(range(n),v,label=mode);a0.set(xlabel='Frozen schedule wave',ylabel=y);a0.legend(fontsize=7)
fig.suptitle('g3 actual realized score schedules replayed as fixed external controls; record-pair negativity zero')
save(fig,'G17_multitime_noncommuting_records',['G17','Q28','Q37'])
with np.load(S/'INFLUENCE69.npz') as z:
 m=z['mass'];rr=z['r'];a=z['a'];exact=z['exact'];first=z['first'];second=z['second'];bound=z['bound']
fig,ax=plt.subplots(1,2,figsize=(10.5,4.3))
q=ax[0].scatter(m,exact,c=rr,cmap='viridis',s=5,alpha=.5);fig.colorbar(q,ax=ax[0],label='Pair population r');ax[0].set(xlabel='Actual source-pair mass m',ylabel='Exact −log overlap',title='Microscopic influence coefficient')
ix=np.argsort(a);ax[1].scatter(a[::4],(exact-first)[::4],s=5,alpha=.4,label='Linear residual');ax[1].scatter(a[::4],(exact-second)[::4],s=5,alpha=.4,label='Quadratic residual');ax[1].plot(a[ix],bound[ix],'k--',lw=1,label='Rigorous quadratic bound');ax[1].set(xlabel='a = 1−s',ylabel='Influence action residual',title='Controlled series, not continuum one-loop');ax[1].legend(fontsize=8)
save(fig,'Q24_effective_record_influence',['Q24','G13','G17'])
# Real wave-law cut-based animation; model coherent records, not physical time.
fig,ax=plt.subplots(1,2,figsize=(9.8,4.1),dpi=100);g=groups[1]['stats'];ls=[]
for a,keys,labels,y in [(ax[0],['chi','classical','qmi'],['Holevo','Record readout','QMI'],'Information (nat)'),(ax[1],['negativity','concurrence'],['Negativity','Concurrence'],'Correlation measure')]:
 for k,l in zip(keys,labels):line,=a.plot([],[],'o-',label=l);ls.append((line,k))
 a.set(xlim=(1,4),ylim=(0,max(g[k][-1] for k in keys)*1.15),xticks=range(1,5),xlabel='Fragment length',ylabel=y);a.legend(fontsize=8)
def upd(i):
 for l,k in ls:l.set_data(np.arange(1,i+2),g[k][:i+1])
 fig.suptitle(f'R69 actual conditioned instrument | fragment f={i+1}');return [l for l,k in ls]
p=S/'plots/Q27_record_fragment_evolution.gif';FuncAnimation(fig,upd,frames=4,interval=800).save(p,writer=PillowWriter(fps=2));plt.close(fig)
with Image.open(p) as im:
 assert im.n_frames==4
 for i in range(4):im.seek(i);im.load()
an.append(dict(name=p.name,ids=['Q27'],frames=4,sha256=sha(p),bytes=p.stat().st_size))
dump(S/'PLOT_MANIFEST69.json',dict(figures=man,animations=an,sources={k:sha(S/k) for k in ('PHYSICS_RESULTS.json','RECORD_STATES69.npz','MULTITIME69.npz','BLOCK_RESULTS69.json','INFLUENCE69.npz')}))
imgs=[]
for x in man:
 with Image.open(S/'plots'/(x['figure']+'.png')) as im:
  y=im.convert('RGB');y.thumbnail((650,440));imgs.append(y.copy())
sheet=Image.new('RGB',(1300,440*((len(imgs)+1)//2)),'white')
for i,im in enumerate(imgs):sheet.paste(im,(650*(i%2),440*(i//2)))
sheet.save(S/'logs/plot_qa.jpg',quality=86);print('PLOTS',len(man),'GIF',len(an),flush=True)
