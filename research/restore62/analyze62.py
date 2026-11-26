"""Conditional three-group response, exact final-wave Rao-Blackwell, holdout closure."""
from base61 import *
R=B/'mqft_restore61_20261126'
def main():
 with np.load(S/'CONTROL62.npz') as z0:a={k:z0[k] for k in z0.files}
 m=a['meta'];arms=list(a['arm_names']);reg=read(S/'REGISTRATION.json');audit=read(S/'AUDIT62.json');assert audit['array_sha256']==sha(S/'CONTROL62.npz')
 keys=['C','G','F','D','preC','preG','lastP','lastF','logw','obs','prior','marks','action']
 rng=np.random.default_rng(620102);boots={n:rng.integers(0,n,(10000,n)) for n in (32,64)}
 def stat(v):
  v=np.asarray(v,float);bb=boots[len(v)];z=np.mean(v[bb],axis=1)
  return dict(mean=v.mean(0).tolist(),ci95=np.quantile(z,[.025,.975],axis=0).tolist(),se=(v.std(0,ddof=1)/np.sqrt(len(v))).tolist())
 def get(g,w,k,arm,legacy=False):
  n=32 if legacy else 64;reps=16 if legacy else 12
  mask=(m[:,1]==g)&(m[:,2]==w)&(m[:,5]==k)&(m[:,8]==arms.index(arm))&(m[:,7 if legacy else 6]==1)
  ix=np.flatnonzero(mask);ix=ix[np.lexsort((m[ix,4],m[ix,3]))]
  assert len(ix)==n*reps and np.array_equal(m[ix,3],np.repeat(np.arange(n),reps)) and np.array_equal(m[ix,4],np.tile(np.arange(reps),n))
  return {key:a[key][ix].reshape((n,reps)+a[key].shape[1:]) for key in keys}
 def outer(x,y):return x[..., :,None]*y[...,None,:]
 def cov(x,y):
  r=x.shape[1];return outer(x-x.mean(1,keepdims=True),y-y.mean(1,keepdims=True)).sum(1)/(r-1)
 def cross_baseline(c):
  n,r,d=c.shape;assert r%2==0
  base=np.empty_like(c)
  base[:,::2]=c[:,1::2].mean(1,keepdims=True);base[:,1::2]=c[:,::2].mean(1,keepdims=True)
  return base
 def ess(w):
  return float(w.sum()**2/np.sum(w*w))
 rows=[];blocks=[];curves=[];tilts=[];closure=[];legacy_rows=[]
 for g in (0,3):
  for w in (1,4):
   for k in (4,8,16,32,64,128):
    z=get(g,w,k,'z');C=z['C'];G=z['G'];F=z['F'];base=cross_baseline(C)
    K=cov(C,C);I=F.mean(1);R0=cov(C,G);GG=outer(G,G).mean(1);H=outer(G,G)-F
    # count/score tangent constraints, PSD
    for v in (K,I,GG,R0):
     assert abs(v.sum(1)).max()<1e-9 and abs(v.sum(2)).max()<1e-9
    assert min(np.linalg.eigvalsh(K).min(),np.linalg.eigvalsh(I).min())>-1e-9
    fd=[]
    for amin,aplus in [('am2','ap2'),('bm2','bp2')]:
     fd.append((get(g,w,k,aplus)['C']-get(g,w,k,amin)['C']).mean(1)/.4)
    fd=np.stack(fd,axis=-1);fd=np.concatenate([fd,-fd.sum(-1,keepdims=True)],axis=-1)
    # exact conditional last-wave RB of independently centered estimator
    M=z['preC']+w*z['lastP'];before=z['preG'];L=z['lastF']
    raw=outer(C,G);center=outer(C-base,G);rb=outer(M-base,before)+L
    hess=np.einsum('nri,nrjk->nrijk',C-base,H).mean(1)
    # paired within-root sample estimator variance (cross-fit values dependent within block).
    var_raw=np.var(raw,axis=1,ddof=1);var_center=np.var(center,axis=1,ddof=1);var_rb=np.var(rb,axis=1,ddof=1)
    desc=dict(generation=g,width=w,k=k,response=stat(R0),direct_fd=stat(fd),count_covariance=stat(K),fisher=stat(I),score_outer=stat(GG),fisher_residual=stat(GG-I),score_mean=stat(G.mean(1)),asymmetry=stat(R0-R0.transpose(0,2,1)),fd_minus_score=stat(fd-R0),crossfit_score=stat(center.mean(1)),rao_blackwell=stat(rb.mean(1)),rb_minus_center=stat(rb.mean(1)-center.mean(1)),second_response=stat(hess),dispersion_raw=stat(var_raw),dispersion_center=stat(var_center),dispersion_rb=stat(var_rb),last_wave_residual=stat((C-M).mean(1)),last_cov_residual=stat((outer(C-M,C-M)-L).mean(1)),source_count=stat(C.mean(1)),native_means=stat(z['obs'].mean(1)))
    # Constrained plane basis, 2 tangent eigenvalues.
    T=np.array([[1/np.sqrt(2),1/np.sqrt(6)],[-1/np.sqrt(2),1/np.sqrt(6)],[0.,-2/np.sqrt(6)]])
    for key,v in [('K',K),('I',I),('R',R0)]:
     A=T.T@v.mean(0)@T;desc[key+'_tangent']=A.tolist()
     desc[key+'_spectrum']=(np.linalg.eigvalsh(A) if key!='R' else np.linalg.svd(A,compute_uv=False)).tolist()
    rows.append(desc)
    if k==128:
     blocks.append(dict(generation=g,width=w,fd=fd.tolist(),score=R0.tolist(),K=K.tolist(),F=I.tolist(),rb=rb.mean(1).tolist()))
     # empirical generating function, separate quenched and mixture, gauge J3=0.
     v=C.reshape(-1,3);J=np.linspace(-.3,.3,9)
     for x in J:
      for y in J:
       j=np.array([x,y,0.]);t=C@j;mx=t.max();ww=np.exp(t-mx);p=ww.ravel()/ww.sum();mu=p@v;cv=(v-mu).T@(p[:,None]*(v-mu));W=float(mx+np.log(ww.mean()))
       mm=t.max(1);Wq=float((mm+np.log(np.exp(t-mm[:,None]).mean(1))).mean())
       assert W>=Wq-1e-12 and abs(cv.sum(0)).max()<1e-9 and abs(mu.sum()-k)<1e-9
       # third direction shifts entire W by k constant, derivatives on tangent unchanged.
       tilts.append(dict(generation=g,width=w,J=j.tolist(),W=W,W_conditional_mean=Wq,mean=mu.tolist(),covariance=cv.tolist(),legendre=float(j@mu-W),ess=ess(ww)))
     # Direct field convergence & importance replay on zero-field records
     steps=[.1,.2,.35] if (g,w)==(0,1) else [.2]
     for h in steps:
      s={.1:'1',.2:'2',.35:'35'}[h];lo=get(g,w,k,'am'+s)['C'];hi=get(g,w,k,'ap'+s)['C'];diff=(hi-lo).mean(1)/(2*h)
      i={.1:0,.2:2,.35:4}[h];wm=np.exp(z['logw'][:,:,i]);wp=np.exp(z['logw'][:,:,i+1])
      rw=(C*(wp-wm)[...,None]).mean(1)/(2*h);rc=((C-base)*(wp-wm)[...,None]).mean(1)/(2*h)
      curves.append(dict(generation=g,width=w,h=h,fd=stat(diff),score=stat(R0[:,:,0]),rw=stat(rw),rw_centered=stat(rc),fd_minus_rw=stat(diff-rw),fd_minus_centered=stat(diff-rc),fd_minus_score=stat(diff-R0[:,:,0]),weight_mean=[float(wm.mean()),float(wp.mean())],ess=[ess(wm),ess(wp)]))
   # heldout closure on 32 geometry blocks (fixed split).
   tr=np.array([c['split']=='train' for c in reg['cores']]);assert tr.sum()==32
   ply=np.array([len(c['moves']) for c in reg['cores']])/81
   for k in (32,64):
    z=get(g,w,k,'z');nxt=get(g,w,2*k,'z');y=(nxt['C']-z['C'])/k;assert np.min(y)>=0 and np.max(abs(y.sum(-1)-1))<1e-12
    p=z['prior'];count=z['C']/k;dens=np.broadcast_to(ply[:,None,None],(64,12,1))
    # three nested feature families. Current mark moment and mean depth only;
    # third adds integrated probabilities as explicit history summary.
    x0=np.concatenate([p[:,:,:2],count[:,:,:2],dens],-1)
    x1=np.concatenate([x0,z['marks'],z['obs'][:,:,[3,4,5,6,9,10,11]]],-1)
    x2=np.concatenate([x1,z['D'][:,:,:2]/k,z['G'][:,:,:2]/k,z['action'][:,:,None]/k],-1)
    def fit(x):
     xx=x[tr].reshape(-1,x.shape[-1]);yy=y[tr].reshape(-1,3);mu=xx.mean(0);sd=xx.std(0);sd[sd<1e-12]=1;z0=(xx-mu)/sd;A=np.column_stack([np.ones(len(z0)),z0])
     pen=np.diag([0.]+[.02]*x.shape[-1]);coef=np.linalg.solve(A.T@A/len(A)+pen,A.T@yy/len(A))
     # independent augmented least squares audit
     L=np.diag(np.sqrt(np.diag(pen)));coef2=np.linalg.lstsq(np.vstack([A/np.sqrt(len(A)),L]),np.vstack([yy/np.sqrt(len(A)),np.zeros((len(L),3))]),rcond=None)[0];assert abs(coef-coef2).max()<1e-10
     f=np.column_stack([np.ones((~tr).sum()*12),(x[~tr].reshape(-1,x.shape[-1])-mu)/sd])@coef
     f=np.maximum(f,0);f/=f.sum(1,keepdims=True);return f.reshape(32,12,3)
    const=np.broadcast_to(y[tr].mean((0,1)),(32,12,3));pred=[const,p[~tr]/p[~tr].sum(-1,keepdims=True),count[~tr],fit(x0),fit(x1),fit(x2)]
    err=np.stack([np.mean((v-y[~tr])**2,axis=(1,2)) for v in pred],1)
    ce=np.stack([np.mean(-np.sum(y[~tr]*np.log(np.clip(v,1e-12,1)),axis=-1),1) for v in pred],1)
    closure.append(dict(generation=g,width=w,k=k,next_cut=2*k,train_roots=32,test_roots=32,train_samples=384,test_samples=384,names=['constant','prior','occupancy','count_ridge','mark_ridge','history_ridge'],mse=stat(err),cross_entropy=stat(ce),mark_minus_count=stat(err[:,4]-err[:,3]),history_minus_mark=stat(err[:,5]-err[:,4]),scope='prediction of future empirical group count fraction, not Markov sufficiency or heldout physical exponent'))
 # legacy exact root population replication, fresh seeds; compare R61 h=.35.
 for k in (32,64,128):
  z=get(0,1,k,'z',True);C=z['C'][:,:,0];G=z['G'][:,:,0];base=cross_baseline(z['C'])[:,:,0]
  lo=get(0,1,k,'am35',True)['C'][:,:,0];hi=get(0,1,k,'ap35',True)['C'][:,:,0]
  fd=(hi-lo).mean(1)/.7;score=cov(C[...,None],G[...,None])[:,0,0]
  wm=np.exp(z['logw'][:,:,4]);wp=np.exp(z['logw'][:,:,5]);rw=(C*(wp-wm)).mean(1)/.7;rc=((C-base)*(wp-wm)).mean(1)/.7
  old=next(x for x in read(R/'PHYSICS_RESULTS.json')['responses'] if (x['generation'],x['width'],x['k'])==(0,1,k))
  legacy_rows.append(dict(k=k,fd=stat(fd),score=stat(score),rw=stat(rw),rw_centered=stat(rc),fd_minus_rw=stat(fd-rw),fd_minus_centered=stat(fd-rc),fd_minus_score=stat(fd-score),weight_mean=[float(wm.mean()),float(wp.mean())],ess=[ess(wm),ess(wp)],R61=old['observables']['C'],scope='same raw R61 g0/w1 geometries, fresh 16 seeds/geometry; scalar field and orientation unchanged'))
 dump(S/'PHYSICS_RESULTS.json',dict(responses=rows,field_grid=curves,legacy_replication=legacy_rows,closure=closure,tilts=tilts,scope='within-geometry covariance; 64 new geometry blocks, 12 seeds; descriptive paired block bootstrap; empirical tilt J vs path intervention h'))
 dump(S/'BLOCK_EFFECTS62.json',blocks)
 print('FINAL_MATRIX',[(x['generation'],x['width'],x['response']['mean'],x['direct_fd']['mean'],x['count_covariance']['mean'],x['fisher']['mean']) for x in rows if x['k']==128],flush=True)
 print('GRID',[(x['generation'],x['width'],x['h'],x['fd']['mean'][0],x['score']['mean'][0],x['rw']['mean'][0],x['rw_centered']['mean'][0],x['ess']) for x in curves],flush=True)
 print('LEGACY',legacy_rows[-1],flush=True)
 print('CLOSURE',[(x['generation'],x['width'],x['k'],x['mse']['mean'],x['mark_minus_count']['ci95'],x['history_minus_mark']['ci95']) for x in closure],flush=True)
if __name__=='__main__':main()
