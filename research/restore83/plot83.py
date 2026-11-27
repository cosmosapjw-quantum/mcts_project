from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
R=read(S/'RECORD83.json');Q=read(S/'SUMMARY83.json');F=read(S/'FRAGMENTS83.json');names=['g0','g3','Rapfi'];figs=[]
def save(fig,name,ids,scope):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=180);fig.savefig(str(p)+'.pdf');plt.close(fig)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,scope=scope,png_sha=sha(a),pdf_sha=sha(b)))
fig,ax=plt.subplots(1,3,figsize=(13,4))
for g,n in enumerate(names):
 q=[x['protocols']['native_repeat'] for x in R['contexts'] if x['family']==g];s=np.array([x['success'] for x in q]);tv=np.array([x['conditional_Hadamard_TV'] for x in q]);t=np.array([x['conditional_trace_distance'] for x in q])
 ax[0].scatter(np.full(24,g)+np.linspace(-.12,.12,24),s,s=18,label=n)
 ax[1].scatter(s,tv,s=20,label=n)
 ax[2].loglog(s,s*tv,'o',ms=4,label=n)
ax[0].set(yscale='log',xticks=range(3),xticklabels=names,ylabel='conditional H32 success mass',title='All-three-wave repeat event')
ax[1].set(xscale='log',xlabel='success mass',ylabel='conditional Hadamard TV',title='Selected coherent completion');ax[1].legend()
ax[2].set(xlabel='success mass',ylabel='success-weighted Hadamard TV',title='Same abort flags included');ax[2].legend()
save(fig,'native_selective_mass_and_coherence',['L15','L16','Q24','Q43'],'Native repeat event conditional H32, exact omitted mass, conditional vs full stopped coherent readout. No physical readout performed.')
fig,ax=plt.subplots(2,2,figsize=(11,8))
for i,f in enumerate(Q['families']):
 n=f['protocols']['native_repeat'];p=f['protocols']['pair_diagnostic']
 for j,(k,a) in enumerate([('frozen_probability_TV',ax[0,0]),('conditional_Hadamard_TV',ax[0,1])]):
  a.bar(i-.18,n[k]['mean'],.36,label='native repeat' if i==0 else None);a.bar(i+.18,p[k]['mean'],.36,label='pair diagnostic' if i==0 else None)
 for j,(a,k) in enumerate([(ax[1,0],'native_repeat'),(ax[1,1],'pair_diagnostic')]):
  z=[x['protocols'][k] for x in R['contexts'] if x['family']==i]
  a.scatter(np.full(24,i)+np.linspace(-.1,.1,24),[max(v['negativity'] for v in r['splits']) for r in z],s=17)
for a in ax.ravel():a.set_xticks(range(3),names)
ax[0,0].set(title='Adaptive vs frozen branch law',ylabel='conditional probability TV');ax[0,1].set(title='Coherent vs orthogonal history tag',ylabel='Hadamard readout TV')
for a in ax[0]:a.legend(fontsize=8)
ax[1,0].set(title='Native repeat, success-conditioned',ylabel='max split negativity');ax[1,1].set(title='Distinct pair-restricted diagnostic',ylabel='max split negativity')
save(fig,'adaptive_record_vs_frozen_and_tagged',['L15','L20','Q24','Q35','Q37','Q42','Q45'],'Actual branch settings versus frozen reference; declared coherent completion vs tagged dephasing; NPT witness within model only.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for j,p in enumerate(('native_repeat','pair_diagnostic')):
 for i,n in enumerate(names):
  a=[x for x in F['contexts'] if x['family']==i and x['protocol']==p];z=np.mean([[b['holevo_fraction'] for b in x['fragments']] for x in a],0)
  ax[j].bar(np.arange(3)+(i-1)*.25,z,.25,label=n)
 ax[j].axhline(.9,color='k',ls=':',label='90% threshold');ax[j].set(xticks=range(3),xticklabels=['wave 2','wave 3','waves 2+3'],ylim=(0,1),ylabel='mean Holevo / H(first bit)',title=p.replace('_',' '));ax[j].legend(fontsize=8)
save(fig,'adaptive_record_fragment_information',['L15','Q25','Q35','Q37','Q45'],'Secondary first-bit pointer diagnostic, 72 contexts per protocol, no disjoint 90% packing; not native environment objectivity.')
assert len(figs)==3 and len({f['png_sha'] for f in figs})==3
dump(S/'PLOT_MANIFEST83.json',dict(figures=figs,animations=[],sources={f:sha(S/f) for f in ['TREE83.json','RECORD83.json','SUMMARY83.json','FRAGMENTS83.json','FRAGMENT_AUDIT83.json','AUDIT83.json','INDEPENDENT83.json','CONVERGENCE83.json','plot83.py']}))
print('PLOTS',len(figs),flush=True)
