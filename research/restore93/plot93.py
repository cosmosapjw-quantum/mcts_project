from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
from scipy.stats import t as student
a=read(S/'RESPONSE93.json');v=read(S/'INDEPENDENT93.json');p=a['pulse'];sc=a['schedule'];w=np.arange(32);h=.35
figs=[]
def save(fig,name,ids):
 fig.tight_layout();b=S/'plots'/name
 fig.savefig(str(b)+'.png',dpi=160);fig.savefig(str(b)+'.pdf');plt.close(fig)
 png=Path(str(b)+'.png');pdf=Path(str(b)+'.pdf')
 with Image.open(png) as im:im.verify()
 subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,png_sha=sha(png),pdf_sha=sha(pdf)))
def band(ax,x,q,label):
 q=np.asarray(q);ax.plot(x,q[0],label=label,lw=1.2);ax.fill_between(x,q[1],q[2],alpha=.12)
def zero(ax):ax.axhline(0,color='gray',lw=.7)
def pulse(ax):ax.axvline(8,color='gray',ls='--',lw=.8)
O=B/'mqft_restore74_20261126'
with np.load(O/'PULSE74.npz') as z:
 m=z['meta'];f=z['features'];X=np.zeros((3,16,16,3,32,10))
 for row,x in zip(m,f):X[tuple(row)]=x
# source validation: root scatter, static and exact finite response, explicit ratios only where contact nonzero.
fig,ax=plt.subplots(2,2,figsize=(11,8));fig.suptitle('G10 | causal logit-response validation; not native equilibrium FDT')
for g,r in enumerate(p):
 q=X[g,:,:,0,8,2];qp=lambda h:q*np.exp(h)/(1+q*np.expm1(h))
 fd=((qp(.35)-qp(-.35))/.7).mean(1);d=((X[g,:,:,1,8,0]-X[g,:,:,2,8,0])/.7).mean(1)
 ax[0,0].scatter(fd,d,label=r['family'],s=20,alpha=.75);a0=np.polyfit(fd,d,1);xx=np.linspace(fd.min(),fd.max(),30);ax[0,0].plot(xx,np.polyval(a0,xx),lw=.9,label=f"{r['family']} fit slope {a0[0]:.2g}")
 xx=(q*(1-q)).mean(1);ax[0,1].scatter(xx,fd,label=r['family'],s=20)
 ax[1,0].errorbar(g,r['direct_minus_contact'][0],yerr=[[r['direct_minus_contact'][0]-r['direct_minus_contact'][1]],[r['direct_minus_contact'][2]-r['direct_minus_contact'][0]]],fmt='o')
 ax[1,1].errorbar(g,r['even_minus_contact'][0],yerr=[[r['even_minus_contact'][0]-r['even_minus_contact'][1]],[r['even_minus_contact'][2]-r['even_minus_contact'][0]]],fmt='o')
for a0 in ax[0]:
 lo=min(a0.get_xlim()[0],a0.get_ylim()[0]);hi=max(a0.get_xlim()[1],a0.get_ylim()[1]);a0.plot([lo,hi],[lo,hi],ls='--',color='gray',lw=.7);a0.legend(fontsize=7)
ax[0,0].set(xlabel='Exact conditional finite contrast',ylabel='Matched direct finite contrast',title='16 roots/family; descriptive linear fits')
ax[0,1].set(xlabel='Frozen reference variance = derivative',ylabel='Exact finite contrast',title='Finite h vs infinitesimal h')
for a0,l in zip(ax[1],['Odd finite contrast residual','Even finite contrast residual']):
 zero(a0);a0.set(xticks=range(3),xticklabels=[r['family'] for r in p],ylabel=l,title='95% pointwise root t interval')
save(fig,'G10_fdt_validation',['G10'])
# all wave causal hit kernels: raw estimate, exact contact+mediated projection, raw minus projected sampling residual.
fig,ax=plt.subplots(3,3,figsize=(12,11));fig.suptitle('G11 | all-wave causal score kernels; only wave8 directly pulsed')
for i,r in enumerate(p):
 x=np.array(r['raw_kernel'])[0,0];y=np.array(r['projected_hit_kernel'])[0];z=x-y;lim=max(abs(x).max(),abs(y).max(),abs(z).max())
 for j,(t,lab) in enumerate(zip([x,y,z],['Raw sample score covariance','Causal hit kernel estimator','Sample discrepancy'])):
  im=ax[i,j].imshow(t,origin='lower',cmap='coolwarm',vmin=-lim,vmax=lim,aspect='auto');fig.colorbar(im,ax=ax[i,j],shrink=.72)
  ax[i,j].set(xlabel='Source wave s',ylabel='Observed wave t',title=f"{r['family']} {lab}",xticks=[0,8,16,24,31],yticks=[0,8,16,24,31]);ax[i,j].axvline(8,lw=.6,color='k',ls=':')
save(fig,'G11_causal_kernel',['G11'])
# R74 first and second response, explicit drift subtraction and causal control.
fig,ax=plt.subplots(3,3,figsize=(15,10));fig.suptitle('G11 | source pulse at wave8; finite contrasts vs derivatives')
for i,r in enumerate(p):
 for j in range(2):
  a0=ax[i,j]
  for k,lab in [('direct','Direct finite'),('rw_control','Reweighted, prehistory anchor'),('score','Score derivative')]:band(a0,w,np.asarray(r[k])[:,:,j],lab)
  pulse(a0);zero(a0);a0.set(xlabel='Wave t',ylabel=['Hit fraction response','Root mark response'][j],title=r['family']);a0.legend(fontsize=7)
 a0=ax[i,2]
 for k,lab in [('even_direct','Direct even finite'),('rw_even_control','Reweighted even finite'),('score_second','Second derivative')]:band(a0,w,np.array(r[k])[:,:,0],lab)
 pulse(a0);zero(a0);a0.set(xlabel='Wave t',ylabel='Second response, hit',title='Finite curvature vs derivative');a0.legend(fontsize=7)
save(fig,'G11_time_resolved_fdt',['G11','G12'])
# Independent time-varying schedules.
fig,ax=plt.subplots(2,3,figsize=(14,7));fig.suptitle('G11 | independent R66 forward/counter source schedules; prefix path reweight')
for i,r in enumerate(sc):
 for j in range(2):
  a0=ax[i,j]
  for k,lab in [('direct','Direct F-C'),('forward_RN','F-arm RN'),('counter_RN','C-arm RN'),('balance','Equal mixture balance')]:band(a0,w,np.array(r[k])[:,:,j],lab)
  zero(a0);a0.set(xlabel='Wave t',ylabel=['Hit fraction F-C','Root mark F-C'][j],title=r['family']);a0.legend(fontsize=7)
 a0=ax[i,2]
 for k,lab in [('ess_forward','F to C ESS/n'),('ess_counter','C to F ESS/n')]:band(a0,w,r[k],lab)
 a0.set(xlabel='Wave t',ylabel='Mean per-root ESS fraction',ylim=(0,1.1),title='Independent seeds, 48 roots');a0.legend(fontsize=7)
save(fig,'G11_schedule_contrast',['G11'])
# overview correct covariance relation and comparison, no physical FDT ratio on near-zero denominators.
fig,ax=plt.subplots(2,3,figsize=(14,8));fig.suptitle('G12 | causal response summary; distinct estimands and finite-sample deviations')
for i,r in enumerate(p):
 a0=ax[0,i]
 for k,lab in [('score','Score covariance'),('naive','Count covariance'),('drift','Predictable drift correction')]:band(a0,w,np.asarray(r[k])[:,:,0],lab)
 zero(a0);pulse(a0);a0.set(xlabel='Wave t',ylabel='Response / covariance',title=r['family']);a0.legend(fontsize=7)
x=np.arange(3)
ax[1,0].bar(x-.18,[r['conditional_derivative'][0] for r in p],.36,label='Exact derivative');ax[1,0].bar(x+.18,[r['direct'][0][8][0] for r in p],.36,label='Direct finite')
ax[1,0].set(xticks=x,xticklabels=[r['family'] for r in p],ylabel='Contact response',title='Distinct infinitesimal and finite estimands');ax[1,0].legend(fontsize=7)
for r in sc:
 for k,lab in [('forward_weight_mean','F->C'),('counter_weight_mean','C->F')]:band(ax[1,1],w,r[k],r['family']+' '+lab)
ax[1,1].axhline(1,ls='--',color='gray');ax[1,1].set(xlabel='Wave t',ylabel='Mean prefix RN weight',title='Normalization in expectation, not each sample');ax[1,1].legend(fontsize=7)
for r in sc:
 band(ax[1,2],w,np.asarray(r['direct_minus_balance'])[:,:,0],r['family']+' direct − balance')
zero(ax[1,2]);ax[1,2].set(xlabel='Wave t',ylabel='Finite estimator difference',title='Balance estimate uses same data');ax[1,2].legend(fontsize=7)
save(fig,'G12_fdt_summary',['G12'])
dump(S/'PLOT_MANIFEST93.json',dict(figures=figs,sources={k:sha(S/k) for k in ['RESPONSE93.json','INDEPENDENT93.json','ORIGINAL_SOURCE93.json']},scope='Descriptive pointwise root-block intervals; exact frozen stationary reference is declared, no physical equilibrium FDT.'))
print('PLOTS',len(figs),flush=True)
