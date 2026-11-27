from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import csv,subprocess
H=read(S/'HONEST_SUMMARY79.json');C=read(S/'CALIBRATION79.json');I=read(S/'ITEM_ACCEPTANCE79.json');A=read(S/'AUDIT79.json')
assert all(sha(B/k)==v for k,v in H['inputs'].items())
F=['g0','g3','Rapfi'];cut=C['cuts'];figs=[]
def save(fig,name,ids,scope):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=180);fig.savefig(str(p)+'.pdf');plt.close(fig)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 figs.append(dict(figure=name,ids=ids,scope=scope,png_sha=sha(a),pdf_sha=sha(b)))
def data(i):return H['panels'][i-1]['data']
fig,ax=plt.subplots(3,3,figsize=(16,12))
a=ax[0,0];s=data(1);a.scatter(F,[x['scale'] for x in s],s=40,label='R78 fitted');a.axhline(10/3,color='k',ls=':',label='known selector');a.set(ylim=(3.25,3.40),ylabel='dimensionless beta',title='01 | Conditional selector scale');a.legend(fontsize=8)
a=ax[0,1]
for x in data(2):
 q=np.array([r['interior_quantiles'] for r in x['occupancy']]);a.plot(cut,q[:,1],'o-',label=x['family']);a.fill_between(cut,q[:,0],q[:,2],alpha=.10)
a.axhline(10/3,color='k',ls=':');a.set(xscale='log',xticks=cut,xlabel='completed backups',ylabel='occupancy Q fit (interior only)',title='02 | Occupancy projection, not temperature');a.legend(fontsize=8)
a=ax[0,2]
for b in [16,64]:
 q=[x for x in data(3) if x['B']==b];a.plot([x['beta'] for x in q],[x['trace'] for x in q],'o-',label=f'B{b}')
a.axhline(.9960401,color='k',ls=':',label='uniform reference');a.set(xlabel='selector beta',ylabel='B × occupancy covariance trace',title='03 | Finite crossover diagnostic');a.legend(fontsize=8)
a=ax[1,0];s=data(4);ks=['nu','eta','beta','gamma','alpha'];x=np.arange(5);a.bar(x-.18,[s['original'][k] for k in ks],.36,label='original claimed');a.bar(x+.18,[s['reference_3D_Ising'][k] for k in ks],.36,label='3D Ising reference');a.set_xticks(x,ks);a.axhline(0,color='k',lw=.6);a.set(title='04 | Ising agreement contradicted',ylabel='exponent (not new MCTS fit)');a.legend(fontsize=8)
a=ax[1,1];z=data(5)
for f,q in z['block'].items():a.plot([r['h'] for r in q],[r['mean_W_same_error'] for r in q],'o-',label=f.replace('_',' '))
a.set(xlabel='aligned block waves',ylabel='CGF RMSE vs IID sum',title='05 | Native block RG defect, beta=16');a.legend(fontsize=8)
a=ax[1,2];z=data(6);a.plot(range(1,5),z['chi_ratio'],'o-',label='Holevo / H(pointer)');a.plot(range(1,5),z['classical_ratio'],'s-',label='classical / H(pointer)');a.axhline(.9,color='k',ls=':',label='90% threshold');a.set(xlabel='records in fragment',ylim=(0,1),ylabel='mean per-wave ratio',title='06 | Model records, no redundancy plateau');a.legend(fontsize=8)
a=ax[2,0];z=data(7)['rows'];x=[r['steps'] for r in z]
for k,l in [('beta_extracted_work','feedback extracted'),('information','feedback I'),('beta_erasure_work','separate erasure cost'),('erasure_entropy_bound','erasure entropy bound')]:a.plot(x,[r[k] for r in z],'o-',label=l)
a.set(xscale='log',xlabel='declared protocol stages',ylabel='beta × work / nats',title='07 | Reference feedback and erasure');a.legend(fontsize=7)
a=ax[2,1];z=data(8);e=np.asarray(z['mean_energy']);w=np.r_[0,0,z['mean_switch_work'][0],z['mean_switch_work'][0],sum(z['mean_switch_work']),sum(z['mean_switch_work'])]
a.plot(range(6),e,'o-',label='mean energy');a.plot(range(6),e[0]+w,'x--',label='initial E + switch work');a.set(xlabel='declared switch stage',ylabel='E / E*',title='08 | Reference first-law ledger');a.legend(fontsize=8)
a=ax[2,2];a.axis('off');z=data(9);a.text(.04,.96,'09 | Typed measurement quality',weight='bold',va='top',fontsize=12,transform=a.transAxes);a.text(.04,.81,('Native: %d new searches\n%d identifiable score waves\n%d initial waves unidentifiable\n\nModel: %d record waves; %d switch contexts\n\nArchive: %d verified prior plot pairs\n94 source functions + 27 guide labels\n\nPhysical T / critical point / native phase:\nNOT IDENTIFIED or NOT ESTABLISHED' % (z['new_paths'],z['actual_conditional_slope_waves'],z['unidentifiable_initial_waves'],z['quantum_reference_record_waves'],z['reference_thermal_contexts'],z['historical_plot_pairs_verified'])),va='top',fontsize=10,transform=a.transAxes,linespacing=1.4)
fig.suptitle('MQFT | Honest physics summary: native measurements, declared references, and unresolved claims',fontsize=17,y=1.015)
save(fig,'L31_honest_physics_summary',['L31','L34','L05'],'All nine original panels reconstructed from actual authenticated numerical artifacts; separate native and declared models. Report-level acceptance only.')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for i,r in enumerate(C['results']):
 q=[x for x in r['models'] if x['model']!='known_score'];xx=np.arange(4)+(i-1)*.24
 ax[0].bar(xx,[max(x['conditional_KL'][-1][0],1e-12) for x in q],.24,label=r['family'])
 for x in q:
  if x['model']=='Q':
   d=np.array(x['excess_nll']);ax[1].plot(cut,d[:,0],'o-',label=r['family']);ax[1].fill_between(cut,d[:,1],d[:,2],alpha=.1)
ax[0].set_xticks(range(4),['PUCT score','Q only','log prior','uniform']);ax[0].set(yscale='log',ylabel='conditional KL (nat/selection)',title='Independent validation')
ax[1].set(xscale='log',xticks=cut,ylabel='Q-only excess NLL',xlabel='backup prefix',title='Actual heldout marked paths');ax[1].axhline(0,color='k',lw=.6)
for a in ax:a.legend(fontsize=8)
save(fig,'conditional_scale_calibration',['L03','L25','G02'],'Actual conditional law versus alternatives, heldout native likelihood/KL. Not physical T.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
for r in C['results']:
 o=r['occupancy'];q=np.array([x['interior_quantiles'] for x in o]);ax[0].plot(cut,q[:,1],'o-',label=r['family']);ax[0].fill_between(cut,q[:,0],q[:,2],alpha=.1)
 ax[1].plot(cut,[x['positive_interior']/x['n'] for x in o],'o-',label=r['family']);ax[2].plot(cut,[1-x['statuses'].get('interior',0)/x['n'] for x in o],'o-',label=r['family'])
ax[0].axhline(10/3,color='k',ls=':')
for a,y in zip(ax,['interior beta (5–95% range)','fraction positive interior','fraction boundary/undefined']):a.set(xscale='log',xticks=cut,xlabel='backup prefix',ylabel=y);a.legend(fontsize=7)
save(fig,'occupancy_projection_quality',['L26','L04','G02'],'Endpoint occupancy projection, finite bounds and negative/invalid fits retained; no temperature scaling law.')
fig,ax=plt.subplots(1,2,figsize=(12,5));q=I['disposition_counts'];r=I['route_counts']
for a,z,title in [(ax[0],q,'121 item dispositions (not final certification)'),(ax[1],r,'Remaining work by primary route')]:
 names=list(z);a.barh(range(len(names)),[z[k] for k in names]);a.set_yticks(range(len(names)),[k.replace('_',' ') for k in names],fontsize=8);a.invert_yaxis();a.set(xlabel='original output items',title=title)
save(fig,'original_item_quality_matrix',['L31','L34','L05'],'Authored item-specific dispositions and next work; all underlying original scientific certifications remain separate.')
assert len(figs)==4 and len({x['png_sha'] for x in figs})==4
dump(S/'PLOT_MANIFEST79.json',dict(figures=figs,animations=[],sources={k:sha(S/k) for k in ['HONEST_SUMMARY79.json','CALIBRATION79.json','ITEM_ACCEPTANCE79.json','AUDIT79.json','INDEPENDENT79.json','plot79.py']}))
dump(S/'L31_REPORT_ACCEPTANCE79.json',dict(status='PASS_CORRECTED_REPORT_ARTIFACT',original_id='L31',panels=9,plot='plots/L31_honest_physics_summary',png_sha=figs[0]['png_sha'],pdf_sha=figs[0]['pdf_sha'],inputs=H['inputs'],limitations='All nine panels present; physical observables are explicit unresolved or model quantities. Original physics final certification remains 0/121. File format/hash checked; direct visual inspection separate.'))
print('PLOTS',len(figs),flush=True)
