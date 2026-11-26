from base61 import *
from scipy.special import logsumexp
BETA=10/3
def main():
 a=read(S/'AUDIT66.json');v=read(S/'VERIFY_AUX66.json');assert sha(S/'NATIVE66.npz')==a['array_sha']
 with np.load(S/'NATIVE66.npz') as f:z={k:f[k] for k in f.files}
 m=z['meta'];A=z['rows'];W=z['waves'];fields=list(z['fields']);wf=list(z['wave_fields'])
 boot=np.random.default_rng(660401).integers(0,48,(10000,48))
 def stat(x):
  x=np.asarray(x,float);assert len(x)==48
  return dict(mean=x.mean(0).tolist(),ci95=np.quantile(x[boot].mean(1),[.025,.975],axis=0).tolist())
 def eff(x):return float(x.sum()**2/(x@x))
 native=[];curves=[]
 for g in (0,3):
  for arm in (0,1):
   ix=np.flatnonzero((m[:,0]==g)&(m[:,3]==arm));ix=ix[np.lexsort((m[ix,2],m[ix,1]))];assert len(ix)==768 and np.array_equal(m[ix,1],np.repeat(np.arange(48),16))
   aa=A[ix].reshape(48,16,6,-1);w=W[ix].reshape(48,16,32,-1)
   l=aa[:,:,-1,fields.index('L')]*(1 if arm==0 else -1);weight=np.exp(-l)
   h=w[...,wf.index('H')];sur=w[...,wf.index('surprisal')];kl=4*w[...,wf.index('KL_FC' if arm==0 else 'KL_CF')].sum(2)
   q=dict(generation=g,arm=['forward','counter'][arm],n=len(ix),log_ratio=stat(l.mean(1)),conditional_KL=stat(kl.mean(1)),martingale_residual=stat((l-kl).mean(1)),ift=stat(weight.mean(1)),ess=eff(weight.ravel()),ess_fraction=eff(weight.ravel())/len(ix),max_weight=float(weight.max()),native_entropy=stat(h.mean((1,2))),native_surprisal=stat(sur.mean((1,2))),entropy_residual=stat((sur-h).mean((1,2))),binary_record_entropy=stat(w[...,wf.index('Hindicator')].mean((1,2))),pair_record_entropy=stat(w[...,wf.index('Hpair')].mean((1,2))),js_equal_prior=stat(w[...,wf.index('JS')].mean((1,2))),visits_a=stat(aa[:,:,-1,fields.index('Na')].mean(1)))
   native.append(q)
   for k0,k in enumerate(z['cuts']):
    l=aa[:,:,k0,fields.index('L')]*(1 if arm==0 else -1)
    curves.append(dict(generation=g,arm=q['arm'],budget=int(k),log_ratio=stat(l.mean(1)),ift=stat(np.exp(-l).mean(1)),ess_fraction=eff(np.exp(-l).ravel())/len(ix)))
 tr=read(S/'THERMAL66.json');er=read(S/'ERASURE66.json');fb=read(S/'FEEDBACK66.json')
 therm=[];workhist=[];convergence=[];evo=[]
 schedules=read(S/'THERMAL_SCHEDULES.json');sm={(d['generation'],d['core']):d for d in schedules}
 bins=np.linspace(-5,8,53)
 for g in (0,3):
  for al in (.25,1.):
   rows=sorted([r for r in tr['results'] if r['generation']==g and r['alpha']==al],key=lambda r:r['core'])
   assert [r['core'] for r in rows]==list(range(48))
   def col(key):return np.array([r[key] for r in rows])
   def ex(key):return np.array([r['exact'][key] for r in rows])
   metrics={key:stat(col(key)) for key in ['meanW','meanQ','mean_sigma','mean_actual_EP']}
   metrics.update({key+'_exact':stat(ex(key)) for key in ('dF','meanW','meanQ','mean_dissipation','mean_actual_EP','final_KL','varW')})
   metrics['ift_F']=stat([r['jarzynski']['mean'] for r in rows]);metrics['ift_R']=stat([r['reverse_jarzynski']['mean'] for r in rows])
   metrics['ESS_F_fraction']=stat([r['jarzynski']['ess']/r['n'] for r in rows]);metrics['ESS_R_fraction']=stat([r['reverse_jarzynski']['ess']/r['n'] for r in rows])
   metrics['min_ESS_F_fraction']=min(r['jarzynski']['ess']/r['n'] for r in rows)
   metrics['min_ESS_R_fraction']=min(r['reverse_jarzynski']['ess']/r['n'] for r in rows)
   for key in ('df_F','df_R','df_BAR'):
    d=col(key)-ex('dF');metrics[key+'_error']=stat(d);metrics[key+'_MAE']=stat(abs(d));metrics[key+'_rmse']=float(np.sqrt(np.mean(d*d)))
   metrics['meanW_residual']=stat(col('meanW')-ex('meanW'));metrics['meanQ_residual']=stat(col('meanQ')-ex('meanQ'))
   therm.append(dict(generation=g,alpha=al,n_protocols=len(rows),n_per_direction=sum(r['n'] for r in rows),metrics=metrics))
   F=[];R=[];wf=[];wr=[];ef=[];evoW=[];evoQ=[];evoS=[];evoF=[]
   for r in rows:
    d=sm[g,r['core']];E=np.array(d['energy']);lz=logsumexp(-BETA*E,axis=1);pi=np.exp(-BETA*E-lz[:,None]);mu=pi[0].copy();w0=q0=0.;ws=[0.];qs=[0.];ss=[entropy(mu)];fs=[0.]
    for j in range(1,len(E)):
     w0+=mu@(E[j]-E[j-1]);nxt=(1-al)*mu+al*pi[j];q0+=(nxt-mu)@E[j];mu=nxt
     ws.append(float(w0));qs.append(float(q0));ss.append(entropy(mu));fs.append(float(-(lz[j]-lz[0])/BETA))
    evoW.append(ws);evoQ.append(qs);evoS.append(ss);evoF.append(fs)
    with np.load(S/'data/thermal'/r['file']) as f:
     F.append(f['sigma']);R.append(-f['sigma_reverse']);wf.append(f['W']);wr.append(-f['W_reverse']);ef.append(f['actual_EP'])
   F=np.concatenate(F);R=np.concatenate(R);N=len(F)
   hf=np.histogram(F,bins)[0];hr=np.histogram(R,bins)[0];assert hf.sum()<=N and hr.sum()<=N
   # Exact binned identity PF(bin)=E_PR[exp(sigma) 1_bin] estimated under reverse.
   weighted=np.histogram(R,bins,weights=np.exp(R))[0]
   workhist.append(dict(generation=g,alpha=al,bins=bins.tolist(),forward=hf.tolist(),reverse=hr.tolist(),reverse_weighted=weighted.tolist(),n=N,overflow_F=int(N-hf.sum()),overflow_R=int(N-hr.sum())))
   evo.append(dict(generation=g,alpha=al,W=stat(evoW),Q=stat(evoQ),S=stat(evoS),deltaF=stat(evoF)))
   # convergence of conditional normalized weights, first fixed prefix of independently sampled paths
   for num in (32,64,128,256,512,1024,2048,4096,8192):
    x=[];y=[]
    for f,r in zip(np.array(F).reshape(48,-1),np.array(R).reshape(48,-1)):
     x.append(float(np.mean(np.exp(-f[:num]))));y.append(float(np.mean(np.exp(r[:num]))))
    convergence.append(dict(generation=g,alpha=al,n=num,F=stat(x),R=stat(y)))
 erasure=[];feedback=[]
 for g in (0,3):
  for K in (1,4,16,64):
   rows=sorted([r for r in er['results'] if r['generation']==g and r['steps']==K],key=lambda r:r['core'])
   fe=sorted([r for r in fb['results'] if r['generation']==g and r['steps']==K],key=lambda r:r['core'])
   assert len(rows)==len(fe)==48
   def gather(rows,key):return stat([r[key] for r in rows])
   r0={k:gather(rows,k) for k in ('bound','mean_work_exact','mean_work_sample','mean_heat_to_system','mean_EP_exact','mean_EP_sample','final_error_rate')}
   r0['beta_work_exact']=gather([dict(r,beta_work_exact=BETA*r['mean_work_exact']) for r in rows],'beta_work_exact')
   r0['ift_F']=stat([r['jarzynski']['mean'] for r in rows]);r0['ift_R']=stat([r['reverse_jarzynski']['mean'] for r in rows])
   r0['ess_fraction']=stat([r['jarzynski']['ess']/r['n'] for r in rows])
   erasure.append(dict(generation=g,steps=K,metrics=r0))
   f0={k:gather(fe,k) for k in ('information','mean_work_exact','mean_work_sample','mean_information_sample','mean_EP_exact','efficiency_exact','efficiency_sample')}
   f0['ift_F']=stat([r['ift']['mean'] for r in fe]);f0['ift_R']=stat([r['reverse_ift']['mean'] for r in fe])
   f0['ess_fraction']=stat([r['ift']['ess']/r['n'] for r in fe]);feedback.append(dict(generation=g,steps=K,metrics=f0))
 # Thermodynamic equilibrium model heat capacity at endpoint, fixed spectrum.
 eq=[]
 for g in (0,3):
  ds=[d for d in schedules if d['generation']==g]
  for beta in (.5,1.,2.,10/3,5.,8.):
   s=[];cv=[]
   for d in ds:
    E=np.array(d['energy'])[-1];p=np.exp(-beta*E-logsumexp(-beta*E));mean=p@E
    s.append(entropy(p));cv.append(beta**2*(p@(E-mean)**2))
   eq.append(dict(generation=g,beta=beta,entropy=stat(s),heat_capacity_over_kB=stat(cv)))
 out=dict(scope='native same-order likelihood contrasted with separately defined auxiliary thermal/record models; fixed root ensemble with block intervals, not measured physical heat',native=native,native_curves=curves,thermal=therm,workhist=workhist,convergence=convergence,evolution=evo,erasure=erasure,feedback=feedback,equilibrium=eq,aux_paths=v['paths'],aux_transitions=v['transitions'])
 dump(S/'PHYSICS_RESULTS.json',out)
 for x in native:print('NATIVE',x['generation'],x['arm'],x['log_ratio'],x['ift'],x['ess_fraction'],x['native_entropy']['mean'],x['entropy_residual'],x['martingale_residual'])
 for x in therm:
  m=x['metrics'];print('THERMAL',x['generation'],x['alpha'],'IFT',m['ift_F'],m['ift_R'],'ESS',m['ESS_F_fraction']['mean'],m['ESS_R_fraction']['mean'],'DF MAE',*[m[k+'_MAE']['mean'] for k in ('df_F','df_R','df_BAR')],'EP',m['mean_dissipation_exact']['mean'],m['mean_actual_EP_exact']['mean'])
 for x in erasure:print('ERASURE',x['generation'],x['steps'],{k:round(v['mean'],6) for k,v in x['metrics'].items() if k in ('beta_work_exact','bound','mean_EP_exact','mean_work_sample','final_error_rate','ift_F')})
 for x in feedback:print('FEEDBACK',x['generation'],x['steps'],{k:round(v['mean'],6) for k,v in x['metrics'].items() if k in ('information','mean_work_exact','efficiency_exact','mean_EP_exact','ift_F')})
if __name__=='__main__':main()
