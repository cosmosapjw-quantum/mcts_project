from rg98 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess,io
Q=read(S/'SUMMARY98.json');G=Q['groups'];F=Q['fields'];sc=Q['scales'];figs=[];animations=[]
def ix(k):return F.index(k)
def get(L,co,arm,bi,h=64,bud=64):return next(z for z in G if (z['L'],z['cohort'],z['arm'],z['bi'],z['h'],z['budget'])==(L,co,arm,bi,h,bud))
def save(fig,name,ids):
 if not fig.get_constrained_layout():fig.tight_layout()
 b=S/'plots'/name;fig.savefig(str(b)+'.png',dpi=150);fig.savefig(str(b)+'.pdf');plt.close(fig)
 a=Path(str(b)+'.png');p=Path(str(b)+'.pdf')
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(p)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,png_sha=sha(a),pdf_sha=sha(p)))
def line(a,g,k,label,steps=False):
 n=5 if steps else 6;t=sc[:n];z=np.array(g['ci'])[:,:n,ix(k)]
 a.plot(t,z[0],'o-',label=label,ms=3);a.fill_between(t,z[1],z[2],alpha=.10);a.set(xscale='log',xticks=t,xticklabels=t,xlabel='Spatial block width b (cells)')
def legend(a):a.legend(fontsize=7)
labels=['0','10/3','8','16','log(L²)']
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q09 | actual conditional-law information and finite spatial beta field')
for j,arm in enumerate([0,1]):
 for bi in range(5):
  g=get(21,2,arm,bi)
  for i,k in enumerate(['KL_A','beta_field_rms','KL_step_loss']):line(ax[j,i],g,k,labels[bi],i>0)
 for i,k in enumerate(['KL(A_b||u_b) (nat)','Reference-weighted beta field RMS','KL lost per dyadic step (nat)']):ax[j,i].set(ylabel=k,title='L21 fresh '+('fixed stones' if arm==0 else 'dense'));legend(ax[j,i])
save(f,'Q09_exact_beta_field',['Q09'])
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q09 | frozen-history score sensitivity and coarse Fisher contraction')
for j,arm in enumerate([0,1]):
 for bi in range(5):
  g=get(21,2,arm,bi)
  for i,k in enumerate(['Fisher','Fisher_step_loss','frozen_logit_curvature_rms']):line(ax[j,i],g,k,labels[bi],i==1)
 for i,k in enumerate(['Mean conditional Fisher','Fisher lost per step','Frozen log-density curvature RMS']):ax[j,i].set(ylabel=k,title='L21 fresh '+('fixed stones' if arm==0 else 'dense'));legend(ax[j,i])
save(f,'Q09_frozen_score_fisher',['Q09'])
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q10 | finite control map, not an identified phase boundary')
for arm in range(2):
 for j,k in enumerate(['KL_A','Fisher','max_block_mass']):
  a=ax[arm,j];z=np.array([[get(L,2,arm,bi)['ci'][0][0][ix(k)] for bi in range(5)] for L in LS]);im=a.imshow(z,aspect='auto',origin='lower')
  a.set(xticks=range(5),xticklabels=labels,yticks=range(5),yticklabels=LS,xlabel='Score control beta (last size-dependent)',ylabel='L',title=k+' | '+('fixed' if arm==0 else 'dense'));f.colorbar(im,ax=a,shrink=.85)
  for y in range(5):
   for x in range(5):a.text(x,y,f'{z[y,x]:.2g}',ha='center',va='center',fontsize=6,color='white' if z[y,x]<(z.min()+z.max())/2 else 'black')
save(f,'Q10_finite_control_map_partial',['Q10','Q13'])
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q11 | projected coupling trajectory (mean per root, not autonomous)')
for arm in range(2):
 for j,L in enumerate([13,17,21]):
  a=ax[arm,j]
  for bi in range(5):
   g=get(L,2,arm,bi);z=np.array(g['ci'])[0];x=z[:,ix('theta_x')];y=z[:,ix('theta_y')];a.plot(x,y,'o-',ms=3,label=labels[bi])
   for t in range(5):
    if abs(x[t+1]-x[t])+abs(y[t+1]-y[t])>1e-10:a.annotate('',xy=(x[t+1],y[t+1]),xytext=(x[t],y[t]),arrowprops=dict(arrowstyle='->',lw=.5))
  a.set(xlabel='theta_x (cosine slope)',ylabel='theta_y (cosine slope)',title=f'L{L}, arm {arm}');legend(a)
save(f,'Q11_induced_coupling_trajectories',['Q09','Q11'])
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q11 | truncated coupling flow compared with exact pushforward')
for arm in range(2):
 for j,k in enumerate(['projection_TV','push_projection_TV','iterated_TV']):
  a=ax[arm,j]
  for bi in range(5):line(a,get(21,2,arm,bi),k,labels[bi],j==1)
  a.set(ylabel=k+' (TV distance)',title=f'L21 fresh arm {arm}');legend(a)
save(f,'Q11_projection_closure_residual',['Q11','Q13'])
sample=Q['representative'];L=sample['L'];legal=np.array(sample['legal'])
ratio=np.array(sample['q_ratio']);ph=np.array(sample['projected_ratio']);clim=max(.1,float(abs(np.log(ratio)).max()),float(abs(np.log(ph)).max()))
def board(v):
 x=np.full(L*L,np.nan);x[legal]=v;return x.reshape(L,L)
f,ax=plt.subplots(2,6,figsize=(16,6),constrained_layout=True);f.suptitle('Q12 | representative actual L21 fresh dense beta16 B64 prefix; exact vs two-slope reconstruction')
for j,b in enumerate(sc):
 for i,z in enumerate([ratio,ph]):
  a=ax[i,j];im=a.imshow(board(np.log(z[j])),vmin=-clim,vmax=clim,cmap='coolwarm');a.set(xticks=[],yticks=[],title=f'b={b} '+('exact' if i==0 else 'projected'))
f.colorbar(im,ax=ax.ravel().tolist(),shrink=.65,label='log(q_b/u_b), lifted to legal cells')
# Shared colorbar uses constrained layout.
save(f,'Q12_exact_scale_snapshots',['Q11','Q12'])
def frame(j):
 f,ax=plt.subplots(1,3,figsize=(11,4));f.suptitle(f'Actual fixed prefix; b={sc[j]} cells (spatial scale, not time)')
 for a,z,label in [(ax[0],ratio,'Exact lifted log density'),(ax[1],ph,'Two-slope reconstruction')]:
  im=a.imshow(board(np.log(z[j])),vmin=-clim,vmax=clim,cmap='coolwarm');a.set(xticks=[],yticks=[],title=label);f.colorbar(im,ax=a,shrink=.7)
 v=np.array(sample['metrics']);t=sc[:j+1]
 for k in ['KL_A','projection_TV','iterated_TV']:ax[2].plot(t,v[:j+1,ix(k)],'o-',label=k)
 ax[2].set(xscale='log',xlim=(.8,36),xticks=sc,xticklabels=sc,ylabel='nat or TV (labeled)',xlabel='Spatial b',ylim=(0,1.05*max(v[:,ix('KL_A')].max(),1)));legend(ax[2]);return f
ims=[]
for j in range(6):
 f=frame(j);f.tight_layout();b=io.BytesIO();f.savefig(b,format='png',dpi=100);b.seek(0);ims.append(Image.open(b).convert('RGB'));plt.close(f)
p=S/'plots/Q12_exact_spatial_flow.gif';ims[0].save(p,save_all=True,append_images=ims[1:],duration=700,loop=0)
with Image.open(p) as im:
 assert im.n_frames==6;hs=[]
 for j in range(6):im.seek(j);hs.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
 assert len(set(hs))==6
animations.append(dict(id='Q12',path='plots/'+p.name,sha256=sha(p),frames=6,frame_rgb_sha256=hs,scales=sc))
f,ax=plt.subplots(2,3,figsize=(13,7));f.suptitle('Q13 | independent cohort and budget comparators; no physical fixed point')
for arm in range(2):
 for j,bi in enumerate([0,1,3]):
  a=ax[arm,j]
  for co,bu,h,lab in [(0,16,16,'dev independent B16'),(0,64,16,'dev B64 prefix16'),(0,64,64,'dev B64'),(1,64,64,'external B64'),(2,64,64,'fresh B64')]:
   try:g=get(21,co,arm,bi,h,bu)
   except StopIteration:continue
   line(a,g,'iterated_TV',lab)
  a.set(ylabel='Iterated projected TV',title=f'arm {arm} beta {labels[bi]}');legend(a)
save(f,'Q13_cohort_budget_closure',['Q13'])
dump(S/'PLOT_MANIFEST98.json',dict(figures=figs,animations=animations,sources={k:sha(S/k) for k in ['SUMMARY98.json','RG98.json','RG98.npz','INDEPENDENT98.json','ORIGINAL_SOURCE98.json']},scope='Exact native categorical scale transform; two-coordinate finite projection and separate control strata; not physical critical dynamics'))
print('PLOTS',len(figs),'GIFS',[(x['frames'],x['path']) for x in animations],flush=True)
