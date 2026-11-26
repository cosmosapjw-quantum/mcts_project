from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
r=read(S/'PHYSICS_RESULTS.json');l=read(S/'LEGENDRE62.json');d=read(S/'SYMMETRY62.json');e=read(S/'EXACT62.json')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.dpi':120})
manifest=[]
def save(fig,name,ids):
 fig.tight_layout();p=S/'plots'/name
 fig.savefig(str(p)+'.png',dpi=160);fig.savefig(str(p)+'.pdf');plt.close(fig)
 with Image.open(str(p)+'.png') as im:im.verify()
 assert Path(str(p)+'.pdf').read_bytes().startswith(b'%PDF')
 manifest.append(dict(figure=name,ids=ids,png_sha=sha(Path(str(p)+'.png')),pdf_sha=sha(Path(str(p)+'.pdf'))))
conds=[(0,1),(0,4),(3,1),(3,4)];lab=[f'g{g}/w{w}' for g,w in conds];colors=['#4c78a8','#78a8ca','#20866d','#75b29a']
z=[next(x for x in r['responses'] if (x['generation'],x['width'],x['k'])==(g,w,128)) for g,w in conds]
def heat(ax,A,title,v):
 im=ax.imshow(A,cmap='RdBu_r',vmin=-v,vmax=v)
 ax.set(xticks=range(3),yticks=range(3),xticklabels=['a','b','rest'],yticklabels=['a','b','rest'],title=title);ax.grid(False)
 for i in range(3):
  for j in range(3):ax.text(j,i,f'{A[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if abs(A[i,j])>.62*v else 'black')
 return im
fig,axs=plt.subplots(4,3,figsize=(11,12))
for i,q in enumerate(z):
 for ax,key,n in zip(axs[i],['count_covariance','response','fisher'],['Count covariance K','Intervention R','Fisher I']):
  A=np.array(q[key]['mean']);heat(ax,A,lab[i]+' | '+n,16)
fig.suptitle('Three-group native matrices: common color scale; B=128, within geometry',fontsize=14)
save(fig,'L16_Q32_multisource_matrices',['L16','G07','Q32'])
fig,axs=plt.subplots(2,2,figsize=(10.8,7.3))
for i,q in enumerate(z):
 ax=axs.ravel()[i];A=np.array(q['response']['mean']);F=np.array(q['direct_fd']['mean']);C=np.array(q['response']['ci95']);D=np.array(q['direct_fd']['ci95'])
 xx=np.arange(4);pairs=[(0,0),(0,1),(1,0),(1,1)]
 for j,(m,ci,n,fmt) in enumerate([(A,C,'score','o'),(F,D,'direct FD','s')]):
  y=np.array([m[a,b] for a,b in pairs]);c=np.array([ci[:,a,b] for a,b in pairs]);ax.errorbar(xx+(j-.5)*.15,y,yerr=np.maximum(0,np.vstack([y-c[:,0],c[:,1]-y])),fmt=fmt,capsize=3,label=n)
 ax.set(xticks=xx,xticklabels=['Raa','Rab','Rba','Rbb'],ylabel='Response',title=lab[i]);ax.axhline(0,c='gray',ls=':')
axs[0,0].legend()
fig.suptitle('Independent source interventions ±0.2 versus path score',fontsize=13)
save(fig,'G07_Q32_direct_response_matrix',['G07','Q32','L16'])
fig,axs=plt.subplots(1,2,figsize=(11,4.6))
q=[x for x in r['field_grid'] if (x['generation'],x['width'])==(0,1)]
for key,fmt in [('fd','o-'),('score','s--'),('rw','^:'),('rw_centered','d-.')]:
 yy=np.array([x[key]['mean'][0] for x in q]);ci=np.array([np.array(x[key]['ci95'])[:,0] for x in q]);xx=[x['h'] for x in q]
 axs[0].errorbar(xx,yy,yerr=np.maximum(0,np.vstack([yy-ci[:,0],ci[:,1]-yy])),fmt=fmt,capsize=3,label=key.replace('_',' '))
axs[0].set(xlabel='Symmetric field magnitude',ylabel='Source a response',title='64 new geometries, g0/w1');axs[0].legend(fontsize=8)
q=r['legacy_replication'][-1];xx=np.arange(4)
for i,(key,oldkey) in enumerate([('fd','fd'),('score','score'),('rw','reweighted_fd'),('rw_centered',None)]):
 for j,s in enumerate([q['R61'].get(oldkey),q[key]]):
  if not s:continue
  y=s['mean'];ci=s['ci95'];axs[1].errorbar(i+(j-.5)*.18,y,yerr=np.array([[y-ci[0]],[ci[1]-y]]),fmt=['o','s'][j],capsize=3,color=['#8c8c8c','#20866d'][j],label=['R61, 8 seeds','R62, 16 fresh seeds'][j] if i==0 else None)
axs[1].set(xticks=xx,xticklabels=['FD','score','reweight','centered'],ylabel='Source count response',title='Same 32 R61 geometries; new seeds');axs[1].legend(fontsize=8)
save(fig,'G07_Q32_field_grid_reweight',['G07','Q32'])
fig,axs=plt.subplots(1,2,figsize=(10.5,4.4))
xx=np.arange(4)
for j,key in enumerate(['dispersion_raw','dispersion_center','dispersion_rb']):
 v=[x[key]['mean'][0][0] for x in z];axs[0].bar(xx+(j-1)*.25,v,width=.24,label=key.replace('dispersion_',''))
axs[0].set(xticks=xx,xticklabels=lab,yscale='log',ylabel='Mean within-core estimator variance',title='Source a diagonal score');axs[0].legend()
for j,key in enumerate(['response','crossfit_score','rao_blackwell']):
 v=[x[key]['mean'][0][0] for x in z];ci=np.array([np.array(x[key]['ci95'])[:,0,0] for x in z]);axs[1].errorbar(xx+(j-1)*.13,v,yerr=np.maximum(0,np.vstack([np.array(v)-ci[:,0],ci[:,1]-np.array(v)])),fmt=['o','s','^'][j],capsize=3,label=key)
axs[1].set(xticks=xx,xticklabels=lab,ylabel='Response Raa',title='Mean estimators (paired block intervals)');axs[1].legend(fontsize=8)
save(fig,'L41_score_variance_reduction',['L41','Q32'])
fig,axs=plt.subplots(1,3,figsize=(12,4.4))
for g,w in conds:
 q=[x for x in r['responses'] if (x['generation'],x['width'])==(g,w)];k=[x['k'] for x in q];i=conds.index((g,w))
 axs[0].plot(k,[x['K_spectrum'][1] for x in q],'-o',c=colors[i],label=lab[i]);axs[1].plot(k,[x['I_spectrum'][1] for x in q],'-o',c=colors[i]);axs[2].plot(k,[x['R_spectrum'][0] for x in q],'-o',c=colors[i])
for ax,n in zip(axs,['K largest eigenvalue','I largest eigenvalue','R largest singular value']):ax.set(xscale='log',yscale='log',xlabel='Completed backups',ylabel=n);ax.set_xticks([4,8,16,32,64,128],labels=[4,8,16,32,64,128])
axs[0].legend(fontsize=8)
save(fig,'L38_constrained_matrix_scale',['L38','L16','Q32'])
fig,axs=plt.subplots(2,2,figsize=(10.8,7.4))
for i,(g,w) in enumerate(conds):
 ax=axs.ravel()[i];q=[x for x in l['rows'] if (x['generation'],x['width'])==(g,w)]
 for x in q:
  c=x['curve'];t=[v['J'][0] for v in c];tag='mixture' if x['mixture'] else 'mean conditional'
  ax.plot(t,[v['Gamma'] for v in c],'-',label=tag);ax.plot(t,[v['quadratic'] for v in c],'--',label=tag+' quadratic')
 ax.set(xlabel='J_a along J_b=−0.75 J_a',ylabel='Empirical conjugate Γ',title=lab[i])
axs[0,0].legend(fontsize=8)
fig.suptitle('Local finite empirical effective action; mixture versus conditional geometry',fontsize=13)
save(fig,'L16_empirical_legendre_curvature',['L16'])
fig,axs=plt.subplots(2,2,figsize=(12,7.5))
for i,(g,w) in enumerate(conds):
 ax=axs.ravel()[i];q=[x for x in r['closure'] if (x['generation'],x['width'])==(g,w)]
 names=['constant','prior','occupancy','count','mark','history'];xx=np.arange(len(names))
 for j,v in enumerate(q):
  y=np.array(v['mse']['mean']);ci=np.array(v['mse']['ci95']);ax.bar(xx+(j-.5)*.34,y,width=.33,label=f"{v['k']}→{v['next_cut']}")
  ax.errorbar(xx+(j-.5)*.34,y,yerr=np.maximum(0,np.vstack([y-ci[0],ci[1]-y])),fmt='none',ecolor='#555555',capsize=2,lw=.8)
 ax.set(xticks=xx,xticklabels=names,ylabel='Held-out fraction MSE',title=lab[i]);ax.tick_params(axis='x',rotation=15);ax.legend(fontsize=8)
fig.suptitle('Conditional closure on 32 held-out geometry blocks; no future features',fontsize=13)
save(fig,'L40_L49_holdout_closure',['L40','L48','L49'])
fig,axs=plt.subplots(1,2,figsize=(10.8,4.6))
ax=axs[0];ax.bar([0,1],[d['unique_frame_comparisons'],d['stabilizer_frame_comparisons']],color=['#4c78a8','#20866d']);ax.set(xticks=[0,1],xticklabels=['Unique frame','Frame multiset'],ylabel='Exact orientation comparisons',title='Canonical path transport')
for i,n in enumerate([d['unique_frame_comparisons'],d['stabilizer_frame_comparisons']]):ax.text(i,n+4,str(n),ha='center')
ax.set_ylim(0,250)
ax=axs[1]
v=[np.array(q['asymmetry']['mean'])[0,1] for q in z];ci=np.array([np.array(q['asymmetry']['ci95'])[:,0,1] for q in z]);ax.errorbar(np.arange(4),v,yerr=np.maximum(0,np.vstack([np.array(v)-ci[:,0],ci[:,1]-np.array(v)])),fmt='o',capsize=3);ax.axhline(0,c='gray',ls=':');ax.set(xticks=range(4),xticklabels=lab,ylabel='R_ab − R_ba',title='Finite-sample response asymmetry')
save(fig,'L16_L49_symmetry_response_diagnostic',['L16','L49'])
# Separate original transition panel from direct held-out binned count transitions.
with np.load(S/'CONTROL62.npz') as a:
 M=a['meta'];C=a['C'];arm=list(a['arm_names']).index('z')
tr=read(S/'REGISTRATION.json')['cores'];test={x['id'] for x in tr if x['split']=='test'}
fig,axs=plt.subplots(2,2,figsize=(9.5,8))
trans=[]
for i,(g,w) in enumerate(conds):
 masks=[(M[:,1]==g)&(M[:,2]==w)&(M[:,5]==k)&(M[:,6]==1)&(M[:,8]==arm)&np.isin(M[:,3],list(test)) for k in (64,128)]
 ix=[np.flatnonzero(x) for x in masks];ix=[x[np.lexsort((M[x,4],M[x,3]))] for x in ix];assert np.array_equal(M[ix[0],3:5],M[ix[1],3:5])
 # fixed bins (zero; 0<r<=.02; .02<r<=.08; >.08), same units
 a,b=[np.digitize(C[x,0]/k,[1e-14,.02000000001,.08000000001]) for x,k in zip(ix,(64,128))]
 n=np.zeros((4,4),int);np.add.at(n,(a,b),1);p=np.divide(n,n.sum(1,keepdims=True),out=np.zeros_like(n,float),where=n.sum(1,keepdims=True)>0)
 ax=axs.ravel()[i];ax.imshow(p,vmin=0,vmax=1,cmap='Blues');ax.grid(False);labels=['0','≤.02','≤.08','>.08'];ax.set(xticks=range(4),yticks=range(4),xticklabels=labels,yticklabels=labels,xlabel='N_a/128 bin',ylabel='N_a/64 bin',title=lab[i])
 for aa in range(4):
  for bb in range(4):ax.text(bb,aa,f'{p[aa,bb]:.2f}\n({n[aa,bb]})',ha='center',va='center',fontsize=8,color='white' if p[aa,bb]>.65 else 'black')
 trans.append(dict(generation=g,width=w,counts=n.tolist(),transition=p.tolist(),n=384))
dump(S/'TRANSITIONS62.json',dict(scope='empirical heldout 64→128 count-fraction bin transitions; mixture over 32 geometries, not autonomous Markov transition',bins=['zero','(0,.02]','(.02,.08]','(.08,1]'],rows=trans))
fig.suptitle('Held-out count-bin transitions: measured, not assumed Markov kernel',fontsize=13)
save(fig,'L43_empirical_count_transitions',['L43','L40'])
dump(S/'PLOT_MANIFEST62.json',dict(figures=manifest,sources={p:sha(S/p) for p in ['PHYSICS_RESULTS.json','LEGENDRE62.json','SYMMETRY62.json','EXACT62.json','CONTROL62.npz','TRANSITIONS62.json']}))
imgs=[]
for x in manifest:
 with Image.open(S/'plots'/(x['figure']+'.png')) as im:
  j=im.convert('RGB');j.thumbnail((650,480));imgs.append(j.copy())
sheet=Image.new('RGB',(1300,480*((len(imgs)+1)//2)),'white')
for j,im in enumerate(imgs):sheet.paste(im,(650*(j%2),480*(j//2)))
sheet.save(S/'logs/plot_qa.jpg',quality=85)
print('PLOT_PAIRS',len(manifest))
