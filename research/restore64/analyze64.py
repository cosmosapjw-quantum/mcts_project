"""Independent root-block inference, full-support path outcomes, native path-pair lift.
No inferred physical phase/hbar: phase and dephasing are specified controller inputs."""
from channel64 import *
from scipy.special import expit
def main():
 reg=read(S/'REGISTRATION.json');a=read(S/'AUDIT64.json');fit=read(S/'CALIBRATION64.json')['fitted'];assert sha(S/'PHASE64.npz')==a['array_sha']
 with np.load(S/'PHASE64.npz') as d:z={k:d[k] for k in d.files}
 arms=list(z['arms']);fields=list(z['fields']);wf=list(z['wave_fields']);cuts=z['cuts'];m=z['meta'];A=z['rows'];W=z['waves'];H=z['hashes']
 rng=np.random.default_rng(640201);boot=rng.integers(0,64,(10000,64))
 def stat(x):
  x=np.asarray(x,float);assert len(x)==64
  return dict(mean=x.mean(0).tolist(),ci95=np.quantile(x[boot].mean(1),[.025,.975],axis=0).tolist())
 def get(g,arm):
  ix=np.flatnonzero((m[:,0]==g)&(m[:,3]==arms.index(arm)));ix=ix[np.lexsort((m[ix,2],m[ix,1]))]
  assert len(ix)==512 and np.array_equal(m[ix,1],np.repeat(np.arange(64),8)) and np.array_equal(m[ix,2],np.tile(np.arange(8),64))
  return ix
 obs=[];contrasts=[];local=[];pairs=[];density=[];phase_scale=[];control=[]
 for g in (0,3):
  by={}
  for arm in arms:
   ix=get(g,arm);x=A[ix].reshape(64,8,6,-1);v=W[ix].reshape(64,8,32,-1);by[arm]=(x,v,ix)
   f=lambda name:x[:,:,:,fields.index(name)]
   w=lambda name:v[:,:,:,wf.index(name)]
   for k,k0 in enumerate(cuts):
    vals={n:stat(f(n)[:,:,k].mean(1)) for n in ('Na','Nb','Npair','E_Q','var','Hvisit','mean_depth','terminal_fraction')}
    vals['mean_mark']=stat(-f('E_Q')[:,:,k].mean(1));vals['root_action_a_wins']=stat((f('root_best_visit_action')[:,:,k]==np.asarray([c['sources'][0] for c in reg['cores']])[:,None]).mean(1))
    obs.append(dict(generation=g,arm=arm,k=int(k0),metrics=vals))
   # Fixed-history calibration and phase contribution evaluated on realized paths.
   K=128;actual=f('path_nll')[:,:,-1]/K
   nf=f('calibrated_nll')[:,:,-1]/K;nd=f('dephased_nll')[:,:,-1]/K;nb=f('base_path_nll')[:,:,-1]/K
   local.append(dict(generation=g,arm=arm,mean_pair_mass=stat(w('mass').mean((1,2))),mean_abs_phase_shift=stat(abs(w('coherent_shift')).mean((1,2))),max_abs_phase_shift=float(abs(w('coherent_shift')).max()),mean_local_tv=stat(w('local_TV').mean((1,2))),calibrated_excess_nll=stat((nf-actual).mean(1)),dephased_excess_nll=stat((nd-actual).mean(1)),base_excess_nll=stat((nb-actual).mean(1)),native_nll=stat(actual.mean(1)),phase_sensitive_states=int(np.count_nonzero(abs(w('coherent_shift'))>1e-12)),mean_vn=stat(w('entropy_vN').mean((1,2))),mean_purity=stat(w('purity').mean((1,2)))))
   for wi in (0,7,15,31):
    d=v[:,:,wi];theta=d[:,:,wf.index('theta')];et=d[:,:,wf.index('eta')];den=d[:,:,wf.index('exposure')];r=d[:,:,wf.index('r')];Xv=d[:,:,wf.index('X')];Yv=d[:,:,wf.index('Y')]
    if arm in ('plus','minus','rotate0'):
     theta0={'plus':np.pi/2,'minus':-np.pi/2,'rotate0':0.}[arm]
     # visibility match convention vs calibrated phase-coupling scale, not physical hbar.
     proxy=1/np.arccos(et);clock=float(d[0,0,wf.index('tau')])
     phase_scale.append(dict(generation=g,arm=arm,wave=wi,tau=clock,visibility=float(et.mean()),absolute_visibility_arccos_proxy=float(proxy.mean()),relative_decay_arccos_proxy=None if clock==0 else float(1/np.arccos(np.exp(-1.2*clock))),model_action_scale=1/fit[2],mean_abs_phase_displacement=stat(abs(theta-theta0).mean(1)),mean_abs_exposure=stat(abs(den).mean(1)),mean_vn=stat(d[:,:,wf.index('entropy_vN')].mean(1))))
    density.append(dict(generation=g,arm=arm,wave=wi,mean_X=stat(Xv.mean(1)),mean_Y=stat(Yv.mean(1)),mean_entropy=stat(d[:,:,wf.index('entropy_vN')].mean(1)),mean_purity=stat(d[:,:,wf.index('purity')].mean(1))))
  for name,other in [('plus','minus'),('plus','dephased'),('minus','dephased'),('rotate0','real'),('dephased','baseline'),('real','dephased')]:
   x,v,_=by[name];y,u,_=by[other]
   for k,k0 in enumerate(cuts):
    vals={}
    for f in ('Na','Nb','Npair','E_Q','var','Hvisit','mean_depth','terminal_fraction'):
     jj=fields.index(f);diff=x[:,:,k,jj]-y[:,:,k,jj];vals[f]=stat(diff.mean(1))
    vals['mean_mark']=stat(-(x[:,:,k,fields.index('E_Q')]-y[:,:,k,fields.index('E_Q')]).mean(1))
    contrasts.append(dict(generation=g,left=name,right=other,k=int(k0),metrics=vals))
  # Paired actual path alternatives under the SAME baseline law; disjoint seed pairs,
  # conditional on the explicitly selected pair, not the complete trajectory ensemble.
  x,_,ix=by['baseline'];a0=x[:,:,:,fields.index('path_nll')];ss=x[:,:,:,fields.index('score_action')]
  hh=H[ix].reshape(64,8,6);aids=np.array([a['job']['id'] for a in a['jobs']])[ix].reshape(64,8)
  for c in range(64):
   for pair in range(4):
    i=2*pair;j=i+1
    for k,k0 in enumerate(cuts):
     if hh[c,i,k]==hh[c,j,k]:continue
     logp=np.array([-a0[c,i,k],-a0[c,j,k]])
     r=float(expit(logp[0]-logp[1]));theta=float(fit[2]*(ss[c,j,k]-ss[c,i,k]));eta=float(fit[0]*np.exp(-fit[1]*k0/128))
     rho=rho_channel(r,theta,eta);eig=np.linalg.eigvalsh(rho);assert eig.min()>-1e-14 and abs(np.trace(rho)-1)<1e-14
     out,xy=readout(rho);de=float(readout(rho_channel(r,0.,0.))[0]);inv=float(readout(rho_channel(r,theta+np.pi,eta))[0]);assert abs((out+inv)/2-de)<1e-14
     pairs.append(dict(generation=g,core=c,pair=pair,k=int(k0),ids=[aids[c,i],aids[c,j]],prefix_hashes=[hh[c,i,k],hh[c,j,k]],logP=logp.tolist(),score_integrals=[float(ss[c,i,k]),float(ss[c,j,k])],r=r,theta=theta,eta=eta,rho_real=rho.real.tolist(),rho_imag=rho.imag.tolist(),X=float(xy[0]),Y=float(xy[1]),mixed_probability=float(out),dephased_probability=de,phase_inverted_probability=inv,entropy=float(-eig[eig>0]@np.log(eig[eig>0])),purity=float(np.trace(rho@rho).real)))
 # independent heldout software quadrature shots on actual native conditional
 # states. The native fit above is already frozen, no parameters changed here.
 rrng=np.random.default_rng(640203);readouts=[]
 for g in (0,3):
  for arm in ('plus','minus','rotate0'):
   ix=get(g,arm);v=W[ix].reshape(64,8,32,-1)
   # One independently simulated pair of readouts per geometry/seed at four fixed waves.
   u=v[:,:,[0,7,15,31]];x=u[...,wf.index('X')];y=u[...,wf.index('Y')]
   shot=256;measure=2*rrng.binomial(shot,(np.stack([x,y],-1)+1)/2)/shot-1
   r=u[...,wf.index('r')];t=u[...,wf.index('tau')];A0=u[...,wf.index('exposure')];th0={'plus':np.pi/2,'minus':-np.pi/2,'rotate0':0.}[arm]
   p=pred(dict(r=r,tau=t,exposure=A0,theta0=th0),fit)
   null=pred(dict(r=r,tau=t,exposure=A0,theta0=th0),[fit[0],fit[1],0.])
   def err(p):return stat(((measure-p)**2).mean((1,2,3)))
   readouts.append(dict(generation=g,arm=arm,shots=int(measure.size*shot),fit_mse=err(p),no_rotation_mse=err(null),dephased_mse=err(np.zeros_like(p))))
   np.savez_compressed(S/f'READOUT_g{g}_{arm}.npz',observed=measure,predicted=p,truth=np.stack([x,y],-1),no_rotation=null)
 # reference identity and action support summaries
 stats=dict(scope='native results paired by root and seed; per-path likelihood diagnostics at same observed histories; independent auxiliary software calibration and readouts; path-pair lift conditional on selected native alternatives, not physical path integral',observables=obs,contrasts=contrasts,local=local,phase_scales=phase_scale,density=density,path_pairs=dict(n=len(pairs),scope='baseline paths only, distinct matched-root seed pairs, model score-integral phase; two-path normalization',summary=[dict(generation=g,k=int(k),n=len(q),mean_abs_imag=float(np.mean([abs(v['rho_imag'][0][1]) for v in q])),median_abs_imag=float(np.median([abs(v['rho_imag'][0][1]) for v in q])),mean_abs_phase_shift=float(np.mean([abs(v['mixed_probability']-v['dephased_probability']) for v in q])),mean_purity=float(np.mean([v['purity'] for v in q]))) for g in (0,3) for k in cuts if (q:=[v for v in pairs if (v['generation'],v['k'])==(g,k)])]),readouts=readouts)
 dump(S/'PATH_PAIRS64.json',dict(scope=stats['path_pairs']['scope'],rows=pairs))
 dump(S/'PHYSICS_RESULTS.json',stats)
 for x in obs:
  if x['k']==128:print('ARM',x['generation'],x['arm'],{k:round(x['metrics'][k]['mean'],6) for k in ('Na','Nb','Npair','mean_mark','var','Hvisit')})
 for x in contrasts:
  if x['k']==128 and (x['left'],x['right']) in [('plus','minus'),('plus','dephased'),('minus','dephased'),('rotate0','real')]:print('DIFF',x['generation'],x['left'],x['right'],{k:x['metrics'][k] for k in ('Na','mean_mark','Npair')})
 print('LOCAL',[(q['generation'],q['arm'],q['mean_abs_phase_shift']['mean'],q['max_abs_phase_shift'],q['calibrated_excess_nll'],q['dephased_excess_nll']) for q in local]);print('PAIRS',stats['path_pairs']);print('READOUT',readouts)
if __name__=='__main__':main()
