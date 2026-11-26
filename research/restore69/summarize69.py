from records69 import *
def main():
 rng=np.random.default_rng(690401);boot=rng.integers(0,16,(10000,16))
 with np.load(S/'RECORD_STATES69.npz') as z:
  pref='fresh_R69_';meta=z[pref+'meta'];out=[]
  for label,mask in [('all',np.ones(len(meta),bool)),('g0',meta[:,0]==0),('g3',meta[:,0]==3)]:
   def block(q):
    a=np.array([q[mask&(meta[:,1]==i)].mean(0) for i in range(16)]);return dict(mean=a.mean(0).tolist(),ci95=np.quantile(a[boot].mean(1),[.025,.975],axis=0).tolist())
   d={k:block(z[pref+k]) for k in ('chi','classical','qmi','discord_Z','negativity','concurrence','CHSH','chi_ratio','classical_ratio','coherence','Ssys','linear_entropy','attenuation')}
   d['count_nll_exact_minus_iid']=block(z[pref+'count_nll']-z[pref+'iid_nll'])
   d['chi90_fraction']=block(z[pref+'chi_f90']>0);d['classical90_fraction']=block(z[pref+'classical_f90']>0);d['CHSH4_gt2_fraction']=block(z[pref+'CHSH'][:,3]>2)
   out.append(dict(group=label,blocks=16,metrics=d))
  # Exact overlap as a function of pair mass and conditional pointer populations.
  with np.load(S/'NATIVE_RECORDS69.npz') as f:nat=f['native']
  m=nat[:,:,3].reshape(-1);r=nat[:,:,4].reshape(-1);p0,p1=probs(m,r)
  s=z[pref+'overlap'];c0=np.sqrt((.35*r+.65)*.35*r)+np.sqrt((.35-.35*r)*(1-.35*r));a=m*(1-c0)
  assert np.max(abs(s-(1-a)))<1e-12
  exact=-np.log(s);first=a;second=a+a*a/2
  assert np.all(exact>=second-1e-13)
  # mathematically exact nonnegative remainder bound from log power series
  bound=a**3/(3*(1-a));assert np.max(exact-second-bound)<1e-12
  info=dict(overlap_error=float(abs(s-(1-a)).max()),mean_exact=float(exact.mean()),max_exact=float(exact.max()),first_MAE=float(abs(exact-first).mean()),second_MAE=float(abs(exact-second).mean()),max_second_remainder_bound_residual=float(np.max(exact-second-bound)))
  np.savez_compressed(S/'INFLUENCE69.npz',mass=m,r=r,a=a,exact=exact,first=first,second=second,bound=bound)
  out0=dict(scope='16 geometry blocks; wave/seed dependence not treated as independent. Exact influence gamma=-log s and two power-series orders; not continuum loop correction',fresh_block_bootstrap=out,influence=info,array_sha=sha(S/'INFLUENCE69.npz'))
 dump(S/'BLOCK_RESULTS69.json',out0)
 print('INFLUENCE',info)
 for v in out:
  d=v['metrics'];print(v['group'],'NLL',d['count_nll_exact_minus_iid'],'chi90',d['chi90_fraction'],'read90',d['classical90_fraction'],'CHSH4',d['CHSH4_gt2_fraction'])
if __name__=='__main__':main()
