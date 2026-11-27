from base61 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
N0=read(S/'NATIVE78.json');G0=read(S/'GAUGE78.json');V=read(S/'INDEPENDENT78.json');A=read(S/'AUDIT78.json');O=read(S/'ORIGINAL_GAUGE_AUDIT78.json')
N=N0['results'];G=G0['contexts'];C=N0['contexts'];names=['g0','g3','Rapfi'];cuts=N0['cuts'];figs=[]
def save(fig,name,ids,scope):
 fig.tight_layout();p=S/'plots'/name;fig.savefig(str(p)+'.png',dpi=150);fig.savefig(str(p)+'.pdf');plt.close(fig)
 a=Path(str(p)+'.png');b=Path(str(p)+'.pdf')
 with Image.open(a) as im:im.verify()
 assert b.read_bytes()[:4]==b'%PDF'
 figs.append(dict(figure=name,ids=ids,scope=scope,png_sha=sha(a),pdf_sha=sha(b)))
fig,ax=plt.subplots(1,2,figsize=(10,4))
for a,key,lab in zip(ax,['prior_mean_TV','value_range'],['Mapped prior TV','Value range over 8 orientations']):
 for i,r in enumerate(N):
  q=np.array(r['ci'][key]);a.errorbar(i,q[0],yerr=[[q[0]-q[1]],[q[2]-q[0]]],fmt='o',capsize=3);z=[x[key] for x in N0['roots'] if x['family']==i];a.scatter(np.full(len(z),i)+np.linspace(-.10,.10,len(z)),z,s=12,alpha=.4)
 a.set_xticks(range(3),names);a.set(ylabel=lab,title='Native pretrained evaluator')
save(fig,'native_orientation_prior_value',['L16','L22'],'Actual initial evaluator D4 residual; native g0/g3 L9 and Rapfi L15 not pooled.')
fig,ax=plt.subplots(1,2,figsize=(10,4))
for i,r in enumerate(N):
 z=[x for x in N0['comparisons'] if x['family']==i]
 ax[0].bar(i-.18,sum(x['first_root_action_equal'] for x in z)/len(z),.36,label='first root action equal' if i==0 else None);ax[0].bar(i+.18,sum(x['first_divergence']>0 for x in z)/len(z),.36,label='first marked path equal' if i==0 else None)
 ax[1].bar(i,r['projected_readout_max_error'])
ax[0].set_xticks(range(3),names);ax[0].set(ylabel='fraction of 448 paired comparisons',title='Same seed is not a covariant coupling');ax[0].legend(fontsize=8)
ax[1].set_xticks(range(3),names);ax[1].set(ylabel='max absolute probability residual',title='Exact finite-group averaged readout')
save(fig,'seed_order_and_projection',['L16','L22'],'Uniform first selection laws; actual ordered inverse-CDF coupling differs; readout projection only.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
for a,r in zip(ax,N):
 for key,label in [('visit_mean_TV','orientation sample mean'),('paired_visit_TV','paired individual paths'),('split_noise_TV','within-orientation split')]:
  q=np.array(r['ci'][key]);a.plot(cuts,q[0],'o-',label=label);a.fill_between(cuts,q[1],q[2],alpha=.12)
 a.set(xscale='log',xticks=cuts,title=r['family'],xlabel='backup budget prefix',ylabel='TV distance');a.legend(fontsize=7)
save(fig,'orientation_occupancy_noise',['L16','L22'],'Pointwise geometry-block intervals; split noise is descriptive, not bias correction.')
fig,ax=plt.subplots(1,2,figsize=(10,4))
for a,key,lab in zip(ax,['unbiased_mean_sq_visit','unbiased_mean_sq_mark'],['Occupancy','Signed mark field']):
 for r in N:
  q=np.array(r['ci'][key]);a.plot(cuts,q[0],'o-',label=r['family']);a.fill_between(cuts,q[1],q[2],alpha=.12)
 a.axhline(0,color='k',lw=.7);a.set(xscale='log',xticks=cuts,title=lab,xlabel='backup budget prefix',ylabel='cross-seed squared mean residual');a.legend()
save(fig,'orientation_unbiased_residual',['L16','L22'],'Cross-products of independent seed-pair differences; signed finite estimates and pointwise root-block uncertainty retained.')
fig,ax=plt.subplots(1,2,figsize=(10,4))
for i,name in enumerate(names):
 z=[q for q in G if q['family']==i]
 ax[0].scatter([q['nodes'] for q in z],[q['edges'] for q in z],label=name,s=22)
 ax[1].scatter(np.arange(16)+i*.2,[q['quotient_cycles'] for q in z],label=name,s=20)
ax[0].set(xlabel='legal-grid vertices',ylabel='legal-grid links',title='Actual legal graph at H32');ax[0].legend()
ax[1].set(xlabel='registered geometry index',ylabel='cycle rank',title='Actual visited state quotient (tree rank=0)');ax[1].legend()
save(fig,'tree_quotient_legal_topology',['L22','L24'],'Actual prefix tree, transposition quotient and legal-action graph are different objects; all components retained.')
fig,ax=plt.subplots(1,2,figsize=(10,4))
w=np.array([z for q in G for z in q['U1']['wilson']]);f=np.array([z for q in G for z in q['U1_flat']['wilson']]);ax[0].scatter(f[:,0],f[:,1],s=9,alpha=.3,label='flat');ax[0].scatter(w[:,0],w[:,1],s=10,alpha=.3,label='one chord flux')
th=np.linspace(0,2*np.pi,201);ax[0].plot(np.cos(th),np.sin(th),'k:',lw=.7);ax[0].set_aspect('equal');ax[0].set(xlabel='Re W',ylabel='Im W',title='Fundamental U(1) holonomies');ax[0].legend()
for i,name in enumerate(names):
 q=np.array([x['U1_flux_TV'] for x in G if x['family']==i]);ax[1].plot(G0['times'],q.mean(0),'o-',label=name)
ax[1].set(xlabel='declared model time',ylabel='mean flux-vs-flat probability TV',title='Phase-sensitive propagation');ax[1].legend()
save(fig,'U1_cycle_flux_transport',['L22','L21'],'All fundamental cycles; assigned flux and flat connection with same actual score landscape.')
fig,ax=plt.subplots(1,2,figsize=(10,4))
for i,name in enumerate(names):
 z=[q for q in G if q['family']==i];ax[0].hist([w[0] for q in z for w in q['SU2']['wilson']],bins=np.linspace(-1,1,30),alpha=.45,density=True,label=name)
 ax[1].scatter(np.full(len(z),i)+np.linspace(-.10,.10,len(z)),[q['SU2_commutator_trace_min'] for q in z],label=name,s=18)
ax[0].set(xlabel='Tr W / 2',ylabel='density',title='SU(2) fundamental holonomy traces');ax[0].legend()
ax[1].axhline(1,color='k',ls=':');ax[1].set_xticks(range(3),names);ax[1].set(ylabel='minimum normalized commutator trace',title='Based loop noncommutativity')
save(fig,'SU2_holonomy_commutator',['L22','L21'],'Declared non-Abelian connection on actual legal graph; loop traces alone do not encode joint holonomy.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
for i,name in enumerate(names):
 z=[q for q in G if q['family']==i]
 for a,key,lab in zip(ax,['policy_stability','fixed_effect_TV','flat_link_policy_TV'],['1 − TV from initial policy','Fixed-effect frame mismatch','SU2 vs flat link policy TV']):
  q=np.array([x[key] for x in z]);a.plot(range(5),q.mean(0),'o-',label=name);a.set(xlabel='convex covariant update step',ylabel=lab)
for a in ax:a.legend(fontsize=8)
save(fig,'covariant_policy_update',['L22','L21'],'Defined density-field update and scalar readout; fixed effect is different probe, not gauge transformation.')
fig,ax=plt.subplots(1,3,figsize=(13,4))
for i,name in enumerate(names):
 z=[q for q in G if q['family']==i];ax[0].hist([np.mean([w[0] for w in q['SU2']['wilson']]) for q in z],bins=np.linspace(-1,1,25),alpha=.5,label=name)
 ax[1].scatter([max(max(q['policy_gauge_error']),1e-17) for q in z],[q['policy_stability'][-1] for q in z],label=name)
 ax[2].scatter([q['nodes'] for q in z],[q['edges'] for q in z],label=name)
ax[0].set(xlabel='context mean normalized Wilson trace',ylabel='contexts',title='Wilson loop distribution');ax[1].set(xscale='log',xlabel='fully transported gauge residual',ylabel='policy stability (model)',title='Roundoff residual, not correlation');ax[2].set(xlabel='nodes',ylabel='links',title='Network topology')
for a in ax:a.legend(fontsize=7)
save(fig,'L22_gauge_policy_analysis',['L22','L21','L24'],'Reconstructed original three-panel intent with explicit unitary connection, covariant field/effect transport and actual legal graph; no statistical correlation inferred from roundoff.')
assert len(figs)==9 and len({q['png_sha'] for q in figs})==9
dump(S/'PLOT_MANIFEST78.json',dict(figures=figs,animations=[],sources={f:sha(S/f) for f in ['NATIVE78.json','GAUGE78.json','INDEPENDENT78.json','AUDIT78.json','ORIGINAL_GAUGE_AUDIT78.json','plot78.py']}))
print('PLOTS',len(figs),flush=True)
