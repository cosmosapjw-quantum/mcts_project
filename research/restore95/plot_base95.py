from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
b=read(S/'BASE_CONTROL95.json');a=read(S/'ENTANGLE95.json');q=read(S/'PLOT_MANIFEST95.json');assert len(q['figures'])==7
fig,ax=plt.subplots(2,3,figsize=(13,7));fig.suptitle('G27 | lambda=0 is not record tagging: system factorizes, record can be entangled')
for g,f in enumerate(['g0','g3','Rapfi']):
 for p in ['native_repeat','pair_diagnostic']:
  z=[x for x in b['rows'] if x['family']==g and x['protocol']==p]
  ax[0,g].plot([1,2,3],[max(x['max_RR_CHSH'] for x in z if x['depth']==d) for d in [1,2,3]],'o-',label=p)
  for mode in ['plus81','base81']:
   u=[x for x in a['rows'] if x['family']==g and x['protocol']==p and x['xi']==1 and x['mode']==mode]
   y=[]
   for d in [1,2,3]:
    x=[r for r in u if r['depth']==d];y.append(max(v[2] for r in x for pair,v in zip(r['pairs'],r['metrics']) if d in pair))
   ax[1,g].plot([1,2,3],y,'o-',label=p+' '+mode)
 ax[0,g].axhline(2,color='gray',ls='--');ax[0,g].axhline(2*np.sqrt(2),color='black',ls=':');ax[0,g].set(xticks=[1,2,3],xlabel='Prefix depth (no R-R pair at 1)',ylabel='Maximum record-record CHSH',title=f);ax[0,g].legend(fontsize=7)
 ax[1,g].set(xticks=[1,2,3],xlabel='Prefix depth',ylabel='Maximum system-pair negativity',title='Base factorization vs plus');ax[1,g].legend(fontsize=6)
fig.tight_layout();p=S/'plots'/'G27_lambda0_factorization';fig.savefig(str(p)+'.png',dpi=160);fig.savefig(str(p)+'.pdf');plt.close(fig)
png=Path(str(p)+'.png');pdf=Path(str(p)+'.pdf')
with Image.open(png) as im:im.verify()
subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
q['figures'].append(dict(figure=p.name,ids=['G27','G26'],png_sha=sha(png),pdf_sha=sha(pdf)));q['sources']['BASE_CONTROL95.json']=sha(S/'BASE_CONTROL95.json');dump(S/'PLOT_MANIFEST95.json',q);print('PLOTS',len(q['figures']))
