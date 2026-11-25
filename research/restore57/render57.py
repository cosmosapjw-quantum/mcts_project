"""Dedicated original outputs; no routine re-rendering of all grouped figures."""
from pathlib import Path
import json,statistics as st,hashlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mqft.physics57 import equilibrium_susceptibility
S=Path(__file__).resolve().parent;r=json.loads((S/'PHYSICS_RESULTS.json').read_text())
only='--g01-only' in sys.argv
out=S/'plots/original';out.mkdir(exist_ok=only);gout=S/'plots/grouped';gout.mkdir(exist_ok=only);reg={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,fig,scope,ids,folder=out):
 for ext in ('png','pdf'):fig.savefig(folder/(name+'.'+ext),dpi=150,bbox_inches='tight')
 plt.close(fig);reg[name]=dict(scope=scope,original_items=ids,files={str((folder/(name+'.'+e)).relative_to(S)):sha(folder/(name+'.'+e)) for e in ('png','pdf')})
def grid(a):a.grid(alpha=.2)
def mean(a,k):return st.mean(x[k] for x in a)
energy=r['energy'];games=r['games'];ls=(13,21);bs=(16,32,64);tau=(4.,16.,64.)
def gr(g):return sorted([x for x in energy if x['game']==g['game']],key=lambda x:x['ply'])
fig,ax=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
for L in ls:
 for B in bs:
  gg=[g for g in games if g['L']==L and g['B']==B]
  for a,k in zip(ax.ravel()[:3],('E_Q','E_PUCT','H')):
   vv=[[] for j in range(3)]
   for g in gg:
    rr=gr(g);n=len(rr)
    for j in range(3):
     z=[x['rows'][-1][k] for i,x in enumerate(rr) if min(2,3*i//n)==j]
     if z:vv[j].append(st.mean(z))
   a.plot(range(3),[st.mean(z) for z in vv],marker='o',label=f'L{L} B{B}')
  z=[x for x in energy if x['L']==L and x['B']==B]
  # Within-root differences only; one summary per complete game.
  for k0,g in enumerate(gg):
   rr=gr(g);f=st.mean(sum(v.get('field',0.) for v in x['rows']) for x in rr);d=st.mean(sum(v.get('redistribution',0.) for v in x['rows']) for x in rr)
   ax[1,1].scatter(f,d,s=20,label=f'L{L} B{B}' if k0==0 else None)
for a,y in zip(ax.ravel()[:3],('Original E_Q=-mean Q (root frame)','E_PUCT=-mean PUCT score','Occupancy entropy (nats)')):
 a.set_xticks(range(3),('early','middle','late'));a.set_ylabel(y);a.legend(fontsize=7);grid(a)
ax[1,1].set(xlabel='Mean within-root score-field component',ylabel='Mean within-root redistribution component');ax[1,1].legend(fontsize=7)
for aa,L in zip(ax[2],ls):
 for B,color in zip(bs,('C0','C1','C2')):
  gg=[g for g in games if g['L']==L and g['B']==B]
  for i,g in enumerate(gg):
   rr=gr(g)
   aa.plot([x['ply']-rr[0]['ply'] for x in rr],[x['rows'][-1]['E_Q'] for x in rr],color=color,alpha=.45,lw=.9,marker='.',ms=2,label=f'B{B}' if i==0 else None)
 aa.set(xlabel='Actual searched ply from prescribed opening',ylabel=f'L{L} E_Q per game');aa.legend(fontsize=7);grid(aa)
fig.suptitle('G01: native energy definition, 44 complete games; not physical thermal energy')
save('G01_energy_evolution',fig,'native score-energy and same-root exact balance',['G01','G03','L06','L33'])
if only:
 m=json.loads((S/'plots/manifest.json').read_text());assert m['source_sha256']==sha(S/'PHYSICS_RESULTS.json')
 m['groups'].update(reg);(S/'plots/manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 print('G01 actual ply trajectories rendered');sys.exit(0)

fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
ks=sorted({z['k'] for x in energy for z in x['rows']})
ax[0].plot(ks,[1/np.sqrt(k) for k in ks],marker='o');ax[0].set(xscale='log',yscale='log',xlabel='Completed backup k',ylabel='Original budget scale 1/sqrt(k)')
for L in ls:
 for B in bs:
  z=[x for x in r['lawfit'] if x['L']==L and x['B']==B];v=[x['beta'] for x in z if x['status']=='finite']
  ax[1].scatter([B+(-.7 if L==13 else .7)]*len(v),v,label=f'L{L} B{B}',alpha=.7)
  a=[q for x in energy if x['L']==L and x['B']==B for q in x['rows']];k0=sorted({q['k'] for q in a})
  ax[2].plot(k0,[st.mean(q['fit_status']=='finite' for q in a if q['k']==k) for k in k0],marker='.',label=f'L{L} B{B}')
ax[1].axhline(10/3,color='gray',ls=':',label='Input beta');ax[1].set(xlabel='Search budget',ylabel='Conditional likelihood beta per game');ax[1].legend(fontsize=6)
ax[2].set(xscale='log',xlabel='Completed backup k',ylabel='Finite occupancy PUCT fit fraction');ax[2].legend(fontsize=6)
fig.suptitle('G02: budget scale, conditional selector beta and occupancy-fit status are distinct')
save('G02_temperature_evolution',fig,'native original budget definition and conditional score fit',['G02','L03','L25','L45'])
t=r['thermal'];paths=np.load(S/'NONCYCLIC_PATHS.npz');ref=t[0];key=ref['key'];f=paths[key+'_F_W'][:,-1];b=paths[key+'_R_W'][:,-1];dF=ref['deltaF'];lim=max(abs(np.r_[f-dF,-b-dF]));bins=np.linspace(-lim,lim,51)
fig,ax=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
ax[0,0].hist(f,bins=35,weights=np.ones(len(f))/len(f),histtype='step');ax[0,0].axvline(dF,color='gray',ls=':');ax[0,0].set(xlabel='Forward work',ylabel='Bin probability',title=f'Specified protocol 0, tau={ref["forward"]["tau"]:g}')
ax[0,1].hist(f-dF,bins=bins,weights=np.exp(-(f-dF))/len(f),histtype='step',label='F weighted')
ax[0,1].hist(-b-dF,bins=bins,weights=np.ones(len(b))/len(b),histtype='step',ls=':',label='R reflected');ax[0,1].set(xlabel='Dissipated work (forward frame)',ylabel='Bin mass');ax[0,1].legend()
for T in tau:
 a=[x for x in t if x['forward']['tau']==T];n=np.array([16,64,256,1024,2048]);av=[];df=[]
 for x in a:
  w=paths[x['key']+'_F_W'][:,-1];e=np.exp(-(w-x['deltaF']));av.append([e[:i].mean() for i in n]);df.append([-np.log(np.exp(-w[:i]).mean())-x['deltaF'] for i in n])
 ax[0,2].plot(n,np.mean(av,0),marker='o',label=f'tau {T:g}')
 ax[1,0].plot(n,np.mean(df,0),marker='o',label=f'tau {T:g}')
ax[0,2].axhline(1,ls=':',color='gray');ax[0,2].set(xscale='log',xlabel='MC paths',ylabel='JE shifted mean');ax[0,2].legend()
ax[1,0].axhline(0,ls=':',color='gray');ax[1,0].set(xscale='log',xlabel='MC paths',ylabel='Mean free-energy estimation error');ax[1,0].legend()
for lab,sty in (('forward','-'),('reverse','--')):
 ax[1,1].plot(tau,[st.mean(x[lab]['rows'][-1]['moments'][2]-x[lab]['rows'][-1]['W']**2 for x in t if x['forward']['tau']==T) for T in tau],marker='o',ls=sty,label=lab)
 ax[1,2].plot(tau,[st.mean(x[lab]['rows'][-1]['W']-x[lab]['rows'][-1]['deltaF'] for x in t if x['forward']['tau']==T) for T in tau],marker='o',ls=sty,label=lab)
for a,y in zip(ax[1,1:],('Exact work variance','Exact mean dissipated work')):
 a.set(xscale='log',xlabel='Declared duration',ylabel=y);a.legend()
fig.suptitle('Q19: independently simulated noncyclic forward/reverse, exact Gibbs endpoints')
save('Q19_jarzynski_noncyclic',fig,'declared noncyclic Gibbs reference; not native work',['Q19'])
fig,ax=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
W=paths[key+'_F_W'];Q=paths[key+'_F_Q'];times=[z['t'] for z in ref['forward']['rows']]
for w in W[:20]:ax[0,0].plot(times,w,lw=.6,alpha=.45)
ax[0,0].plot(times,[z['W'] for z in ref['forward']['rows']],color='black',lw=2,label='Exact mean');ax[0,0].legend();ax[0,0].set(xlabel='Protocol time',ylabel='Cumulative work')
b=np.linspace(W.min(),W.max(),50)
for i in (2,4,8):ax[0,1].hist(W[:,i],bins=b,weights=np.ones(len(W))/len(W),histtype='step',label=f't={times[i]:g}')
ax[0,1].set(xlabel='Work',ylabel='Bin mass');ax[0,1].legend()
m=np.array(ref['forward']['rows'][-1]['moments']);v=m[2]-m[1]**2;c3=m[3]-3*m[1]*m[2]+2*m[1]**3;c4=m[4]-4*m[1]*m[3]+6*m[1]**2*m[2]-3*m[1]**4-3*v*v
ax[0,2].bar(range(4),[m[1],v,c3,c4]);ax[0,2].set_xticks(range(4),('mean','var','kappa3','kappa4'));ax[0,2].set_ylabel('Exact work cumulants')
for x in r['rate']:ax[1,0].plot(x['x'],x['I'],lw=1)
ax[1,0].set(xlabel='Mean of independent repeated works',ylabel='Cramer rate (tilt grid)',title='Independent repeats, not one native path')
ax[1,1].scatter(W[:,-1],Q[:,-1],s=7,alpha=.2);ax[1,1].set(xlabel='Work',ylabel='Heat into system')
# Separate legitimate equilibrium susceptibility, not Var(work)=T d<W>/dh.
fs=[equilibrium_susceptibility(z['forward']['rows'][4]['pi']) for z in t]
a=[z['variance'] for z in fs];b=[z['derivative'] for z in fs]
ax[1,2].scatter(a,b,s=18);ax[1,2].plot([min(a),max(a)],[min(a),max(a)],ls=':',color='gray');ax[1,2].set(xlabel='Equilibrium Var(f)',ylabel='Derivative of <f>',title='Conjugate static response only')
fig.suptitle('Q20: real declared work trajectories, independent rate reference, signed heat and valid susceptibility')
save('Q20_work_fluctuations',fig,'conditional work trajectories and equilibrium susceptibility; no invented exploration/quantum work split',['Q20','G10'])
fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
a=[x for y in t for tag in ('forward','reverse') for x in y[tag]['rows']]
ax[0,0].scatter([x['flux_sigma'] for x in a],[x['entropy_rate']-x['heat_rate'] for x in a],s=9,alpha=.5);mx=max(x['flux_sigma'] for x in a)
ax[0,0].plot([0,mx],[0,mx],ls=':',color='gray');ax[0,0].set(xlabel='Flux-force entropy production',ylabel='dS/dt - Qdot')
for T in tau:
 b=[x['forward']['rows'] for x in t if x['forward']['tau']==T];n=len(b[0])
 for k,sty in (('sigma','-'),('rel_entropy',':')):
  ax[0,1].plot(times,[st.mean(x[i][k] for x in b) for i in range(n)],ls=sty,label=f'{T:g} {k}')
ax[0,1].set(xlabel='Protocol time',ylabel='Entropy production / final KL');ax[0,1].legend(fontsize=7)
ax[1,0].scatter([x['entropy_rate'] for x in a],[abs(x['entropy_rate'])-x['entropy_rate'] for x in a],s=9,alpha=.4);ax[1,0].set(xlabel='Signed system entropy rate',ylabel='Error caused by absolute value')
ax[1,1].plot(times,[z['U']-ref['forward']['rows'][0]['U'] for z in ref['forward']['rows']],label='Delta U',marker='o')
ax[1,1].plot(times,[z['W']+z['Q'] for z in ref['forward']['rows']],ls=':',label='W+Q');ax[1,1].set(xlabel='Protocol time',ylabel='Energy balance');ax[1,1].legend()
fig.suptitle('Q05/Q08: signed entropy balance, positive flux production, finite-time endpoint lag')
save('Q05_Q08_entropy_balance',fig,'conditional thermodynamics, correct signed balance not original absolute-rate ansatz',['Q05','Q08'])
fig,ax=plt.subplots(2,3,figsize=(13,8),constrained_layout=True);eng=r['engines'];a=[x for x in eng if x['duration']==1.]
for x in a:
 ax[0,0].plot(x['otto']['scale'],x['otto']['force'],marker='o',alpha=.6)
 ax[0,1].plot(x['carnot']['S'],x['carnot']['T'],marker='o',alpha=.6)
ax[0,0].set(xlabel='Hamiltonian scale lambda',ylabel='Generalized force -<E_c>',title='Finite-contact Otto, not P-V')
ax[0,1].set(xlabel='Entropy',ylabel='Declared temperature / E*',title='Ideal reversible Carnot')
ax[0,2].bar([0,1],[1/3,.5]);ax[0,2].set_xticks([0,1],('Otto','Carnot'));ax[0,2].set_ylabel('Efficiency')
ds=(1.,4.,16.)
for k,label in (('Qh','Q hot'),('Qc','Q cold'),('W_in','W on system')):
 ax[1,0].plot(ds,[st.mean(x[k] for x in eng if x['duration']==d) for d in ds],marker='o',label=label)
ax[1,0].set(xscale='log',xlabel='Bath contact duration',ylabel='Energy per cycle');ax[1,0].legend()
ax[1,1].plot(ds,[st.mean(x['sigma'] for x in eng if x['duration']==d) for d in ds],marker='o');ax[1,1].set(xscale='log',xlabel='Contact duration',ylabel='Cycle entropy production')
ax[1,2].scatter([x['Qh'] for x in eng],[-x['W_in'] for x in eng],s=20);ax[1,2].set(xlabel='Heat from hot bath',ylabel='Work output')
fig.suptitle('Q03: explicit periodic finite-contact Otto and ideal Carnot references, no native bath')
save('Q03_thermodynamic_cycles',fig,'declared classical/diagonal reference; quantum efficiency/native P-V unresolved',['Q03'])
fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
p=r['mark_prediction'];ax[0].bar(range(3),[st.mean(x['mse'][i] for x in p) for i in range(3)]);ax[0].set_xticks(range(3),('zero','sector','action'));ax[0].set_ylabel('Heldout mark MSE')
for a,method in zip(ax[1:],('direct','LR')):
 z=[x for x in r['controls'] if x['method']==method]
 for law in ('iid','pl','reserve'):
  q=[x for x in z if x['law']==law];a.plot(range(5),[st.mean(x['variance'][i] for x in q) for i in range(5)],marker='o',label=law)
 a.set_xticks(range(5),z[0]['names'],rotation=45,ha='right',fontsize=7);a.set_ylabel(f'{method} future mark variance');a.legend()
fig.suptitle('Frozen R56 predictors and coefficients on new independent R57 roots')
save('X-MEMORY',fig,'native prospective fixed model, no refit',[],gout)
(S/'plots/manifest.json').write_text(json.dumps(dict(source_sha256=sha(S/'PHYSICS_RESULTS.json'),paths_sha256=sha(S/'NONCYCLIC_PATHS.npz'),groups=reg),indent=2)+'\n')
print('figures',len(reg),'dedicated',len(list(out.glob('*.png'))),'grouped',len(list(gout.glob('*.png'))))
