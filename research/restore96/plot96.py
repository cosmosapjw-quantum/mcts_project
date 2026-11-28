from darwin96 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess,io
Q=read(S/'SUMMARY96.json');G=Q['groups'];N=read(R94/'NATIVE94.json');R=read(R94/'RECORD94.json');D=read(S/'DARWIN96.json');F=FAMS;P0=['native_repeat','pair_diagnostic'];figs=[]
def get(g,p,d,name,xi=1.,mode='plus81'):
 return next(x for x in G if (x['family'],x['protocol'],x['depth'],x['pointer'],x['xi'],x['mode'])==(g,p,d,name,xi,mode))
def save(fig,name,ids):
 fig.tight_layout();b=S/'plots'/name
 fig.savefig(str(b)+'.png',dpi=160);fig.savefig(str(b)+'.pdf');plt.close(fig)
 png=Path(str(b)+'.png');pdf=Path(str(b)+'.pdf')
 with Image.open(png) as im:im.verify()
 subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,png_sha=sha(png),pdf_sha=sha(pdf)))
def band(a,t,q,lab):
 x=np.array(q);a.plot(t,x[0],label=lab);a.fill_between(t,x[1],x[2],alpha=.10)
def label(x):return [''.join(x['labels'][i] for i in f) for f in x['fragments']]
# Q24 actual conditional dynamics vs defined prefix model; clock not calibrated physical time.
fig,ax=plt.subplots(3,3,figsize=(13,10));fig.suptitle('Q24 | actual software qubit and declared coherent prefix; no physical bath/time')
for j,g in enumerate(F):
 r=N['families'][j];x=np.array(r['features_ci'])
 for a in [0,1,2,4,5]:band(ax[0,j],np.arange(32,44),x[:,a,:,2],N['arms'][a])
 for k in [8,9,10]:band(ax[1,j],np.arange(32,44),x[:,0,:,k],N['fields'][k])
 for p in P0:
  z=next(v for v in R['summary'] if v['family']==g and v['protocol']==p)
  ax[2,j].plot(range(4),np.array(z['plus_ci'])[0,:,6],'o-',label=p+' record Crel')
  ax[2,j].plot(range(4),np.array(z['plus_ci'])[0,:,4],'s--',label=p+' I(S:R)')
 ax[0,j].set(title=g,ylabel='Fixed-Z coherence');ax[1,j].set(ylabel='Entropy (nat)',xlabel='Native selection clock');ax[2,j].set(xlabel='Completed waves',ylabel='Declared prefix coherence / QMI (nat)')
 for a in ax[:,j]:a.legend(fontsize=6)
save(fig,'Q24_decoherence_dynamics',['Q24'])
# Q25 all subsets, size shown without treating joint readout as disjoint observers.
fig,ax=plt.subplots(4,3,figsize=(14,12));fig.suptitle('Q25 | measured information lower bound vs Holevo/TV upper, depth3, plus81')
for j,g in enumerate(F):
 for i,(p,name) in enumerate(product(P0,['first_record','system_Z'])):
  a=ax[i,j];x=get(g,p,3,name);y=np.array(x['ratio_ci']);n=len(y);t=np.arange(n)
  for k,lab in [(0,'Z MI'),(1,'Attained'),(2,'Upper'),(4,'QMI')]:a.plot(t,y[:,0,k],'o-',label=lab,ms=3)
  a.fill_between(t,y[:,0,1],y[:,0,2],alpha=.15);a.axhline(.9,color='gray',ls='--');a.set(xticks=t,xticklabels=label(x),ylabel='Information / H(pointer)',title=f'{g} | {p} | {name}');a.tick_params(axis='x',labelsize=7);a.legend(fontsize=6,ncol=2)
save(fig,'Q25_information_proliferation',['Q25'])
# Disjoint packing 90% controls and 50/10 thresholds.
fig,ax=plt.subplots(2,3,figsize=(13,7));fig.suptitle('Q25 | exact disjoint packing: lower/upper counts of contexts with >=2 fragments')
for j,g in enumerate(F):
 for i,name in enumerate(['first_record','system_Z']):
  a=ax[i,j];xs=[get(g,p,3,name,xi,m) for p in P0 for m in ['plus81','base81'] for xi in [0,.5,1]]
  t=np.arange(len(xs));lo=[v['packing_counts'][2]['lower_ge2'] for v in xs];hi=[v['packing_counts'][2]['upper_ge2'] for v in xs]
  a.bar(t,hi,color='0.75',label='Possible (upper)');a.bar(t,lo,label='Attained (lower)')
  a.set(xticks=t,xticklabels=[('N' if x['protocol']==P0[0] else 'P')+('+' if x['mode']=='plus81' else '0')+f'/{x["xi"]:g}' for x in xs],ylim=(0,24.5),title=f'{g} | {name}',ylabel='Contexts (of 24)');a.tick_params(axis='x',labelrotation=45,labelsize=7);a.legend(fontsize=7)
save(fig,'Q25_disjoint_fragment_packing',['Q25','Q28'])
# Q26 actual decision counts, local pointer discrimination and frozen stability.
fig,ax=plt.subplots(3,3,figsize=(12,10));fig.suptitle('Q26 | local pointer stability and actual final decision comparator (not pointer energy)')
for j,g in enumerate(F):
 r=N['families'][j];s=np.array(r['frozen_stability_ci']);x=np.array(r['features_ci'])
 for k in [16,17]:band(ax[0,j],np.arange(32,44),x[:,0,:,k],N['fields'][k])
 ax[0,j].set(title=g,ylabel='Distinguishability / disturbance',xlabel='Selection clock');ax[0,j].legend(fontsize=7)
 ax[1,j].bar(range(4),s[0,0,[0,6,7,8]]);ax[1,j].set(xticks=range(4),xticklabels=['optimal','X','Y','Z'],ylim=(0,1.05),ylabel='Frozen endpoint norm')
 a=ax[2,j];r=Q['native'][j];z=np.array(r['actual_first_vs_recommendation_table']);im=a.imshow(z,cmap='viridis')
 for u,v in np.ndindex(z.shape):a.text(v,u,str(z[u,v]),ha='center',va='center',color='white' if z[u,v]<z.max()*.65 else 'black')
 a.set(xticks=range(3),yticks=range(3),xticklabels=r['labels'],yticklabels=r['labels'],xlabel='Actual final recommendation',ylabel='Actual first action',title=f"Phase reversal differs {r['recommend_disagreement']}/192")
save(fig,'Q26_pointer_states',['Q26'])
# Q28 objectivity conditions: pointer dephasing disturbance, conditional product deviation, rare event contribution
fig,ax=plt.subplots(3,3,figsize=(13,10));fig.suptitle('Q28 | objectivity diagnostics, current depth 3; conditional vs success-weighted')
for j,g in enumerate(F):
 for i,name in enumerate(['first_record','system_Z']):
  a=ax[i,j];t=np.arange(6);x=[get(g,p,3,name,xi) for p in P0 for xi in [0,.5,1]]
  a.bar(t-.17,[v['pointer_dephase_ci'][0] for v in x],.34,label='Pointer dephase TD')
  a.bar(t+.17,[v['largest_product_TD_ci'][0] for v in x],.34,label='Conditional product TD')
  a.set(xticks=t,xticklabels=[('N' if v['protocol']==P0[0] else 'P')+f'/{v["xi"]:g}' for v in x],title=f'{g} | {name}',ylabel='Trace distance');a.legend(fontsize=6)
 a=ax[2,j]
 for p in P0:
  x=[get(g,p,d,'system_Z') for d in [1,2,3]]
  a.semilogy([1,2,3],[max(v['weighted_largest_ci'][0][0],1e-40) for v in x],'o-',label=p+' lower')
  a.semilogy([1,2,3],[max(v['weighted_largest_ci'][0][1],1e-40) for v in x],'s--',label=p+' upper')
 a.set(xticks=[1,2,3],xlabel='Completed waves',ylabel='s × information (nat)',title='Success-weighted contribution');a.legend(fontsize=6)
save(fig,'Q28_objectivity_summary',['Q28'])
# Q27 actual 4 discrete frames from prefix state and actual matched state curves; no interpolation.
frames=[]
for d in range(4):
 fig,ax=plt.subplots(2,3,figsize=(12,6));fig.suptitle(f'Q27 | completed wave {d}; H{32+4*d}; actual conditional qubit and declared record prefix')
 for j,g in enumerate(F):
  x=np.array(N['families'][j]['features_ci']);t=np.arange(32,44)
  if d:
   for k in [2,8]:band(ax[0,j],t[:4*d],x[:,0,:4*d,k],N['fields'][k])
  else:
   ax[0,j].scatter([31.5],[.8],label='prepared input C_Z')
   ax[0,j].scatter([31.5],[float(ent([.9,.1]))],label='input S')
  ax[0,j].set(xlim=(31,44),ylim=(0,1.05),title=g,ylabel='Coherence / entropy',xlabel='Native selection clock');ax[0,j].legend(fontsize=7)
  for p in P0:
   z=next(v for v in R['summary'] if v['family']==g and v['protocol']==p);a=np.array(z['plus_ci'])
   ax[1,j].plot(range(d+1),a[0,:d+1,6],'o-',label=p+' Crel')
   ax[1,j].plot(range(d+1),a[0,:d+1,4],'s--',label=p+' QMI')
  ax[1,j].set(xlim=(-.1,3.1),ylim=(0,1.6),xticks=[0,1,2,3],xlabel='Completed wave',ylabel='Declared record (nat)');ax[1,j].legend(fontsize=6)
 fig.tight_layout();buf=io.BytesIO();fig.savefig(buf,format='png',dpi=100);buf.seek(0);frames.append(Image.open(buf).convert('RGB'));plt.close(fig)
gif=S/'plots'/'Q27_decoherence_evolution.gif';frames[0].save(gif,save_all=True,append_images=frames[1:],duration=900,loop=0)
with Image.open(gif) as im:
 assert im.n_frames==4
 framehash=[]
 for i in range(4):im.seek(i);framehash.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
 assert len(set(framehash))==4
# static frame at H44 for paired inventory.
fig,ax=plt.subplots(1,3,figsize=(12,3.6));fig.suptitle('Q27 | explicit prefix-clock snapshots, corresponding to four animation frames')
for j,g in enumerate(F):
 for p in P0:
  z=next(v for v in R['summary'] if v['family']==g and v['protocol']==p);a=np.array(z['plus_ci'])
  ax[j].plot([32,36,40,44],a[0,:,6],'o-',label=p+' Crel')
  ax[j].plot([32,36,40,44],a[0,:,4],'s--',label=p+' QMI')
 ax[j].set(title=g,xticks=[32,36,40,44],xlabel='Prefix clock',ylabel='Record information (nat)');ax[j].legend(fontsize=6)
save(fig,'Q27_prefix_clock_snapshots',['Q27'])
dump(S/'PLOT_MANIFEST96.json',dict(figures=figs,animations=[dict(id='Q27',path='plots/'+gif.name,sha256=sha(gif),frames=4,frame_rgb_sha256=framehash,clocks=[32,36,40,44])],sources={k:sha(S/k) for k in ['DARWIN96.json','SUMMARY96.json','INDEPENDENT96.json','ORIGINAL_SOURCE96.json']},scope='No physical time or calibrated bath; pointwise descriptive root intervals; explicit lower/upper info and disjoint packing'))
print('PLOTS',len(figs),'GIF',len(frames),flush=True)
