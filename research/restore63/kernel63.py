"""Time-inhomogeneous Markov-additive count generator, trained on prior R62 only.
One-step exact law and reduced bath normalization; no teacher forcing in propagation.
"""
from base61 import *
from itertools import product
from scipy.special import gammaln,logsumexp
from scipy.stats import binom
import time
W=4
TH=np.array([.01,.03,.10])
Y=np.array([(a,b,W-a-b) for a in range(W+1) for b in range(W-a+1)],int)
def states(v):
 return np.digitize(v[...,0],TH)*4+np.digitize(v[...,1],TH)
def logmass(p,y):
 y=np.asarray(y);n=y.sum(-1)
 return gammaln(n+1)-gammaln(y+1).sum(-1)+np.sum(y*np.log(np.maximum(p,1e-300)),axis=-1)
def masses(p,y=Y):
 return np.exp(logmass(np.asarray(p)[...,None,:],y))
def counts_grid(k):
 return np.array([(a,b,k-a-b) for a in range(k+1) for b in range(k-a+1)],int)
def fullfit(P,C,Z):
 # P: (n,33,3); C cumulative counts; Z: discrete state.
 EM=np.zeros((32,16,15));K=np.zeros((32,15,16,16));ns=np.zeros((32,16),int)
 # actual count increment index
 y=np.diff(C,axis=1).astype(int);lookup={(a,b):i for i,(a,b,c) in enumerate(Y)}
 ids=np.array([[lookup[tuple(v[:2])] for v in row] for row in y])
 for t in range(32):
  p=P[:,t];mult=masses(p);assert abs(mult.sum(-1)-1).max()<1e-12
  for s in range(16):
   ix=np.flatnonzero(Z[:,t]==s);ns[t,s]=len(ix)
   if not len(ix):ix=np.arange(len(P))
   # smooth empirical conditional mixture of frozen multinomials
   em=(mult[ix].sum(0)+.5*masses(p[ix].mean(0)))/(len(ix)+.5)
   EM[t,s]=em
   dest=Z[ix,t+1];freq=np.bincount(dest,minlength=16)/len(ix)
   yy=ids[ix,t];joint=np.zeros((15,16));np.add.at(joint,(yy,dest),1)
   T=(joint+.5*freq)/(joint.sum(1,keepdims=True)+.5)
   K[t,:,s,:]=em[:,None]*T
  assert abs(K[t].sum((0,2))-1).max()<1e-12
  assert abs(K[t].sum(2).T-EM[t]).max()<1e-12
 return EM,K,ns
def propagate_p(K,t,h):
 D=np.eye(16).reshape(1,1,16,16)
 for step in range(h):
  d=D.shape[0];E=np.zeros((d+W,d+W,16,16))
  for j,(a,b,c) in enumerate(Y):E[a:a+d,b:b+d]+=D@K[t+step,j]
  D=E
 M=D.sum(-1).transpose(2,0,1)
 assert abs(M.sum((1,2))-1).max()<1e-12
 return M
def propagate_count(EM,t,h,starts):
 # count-consistent propagation: update exact cumulative count then rebin.
 # starts unique actual (a,b); batch of count states
 D=np.ones((len(starts),1,1))
 for s in range(h):
  d=D.shape[1];a,b=np.indices((d,d))
  C=np.stack([starts[:,None,None,0]+a,starts[:,None,None,1]+b],-1)/(W*(t+s))
  st=states(C);E=np.zeros((len(starts),d+W,d+W))
  for j,(aa,bb,cc) in enumerate(Y):E[:,aa:aa+d,bb:bb+d]+=D*EM[t+s,st,j]
  D=E
 assert abs(D.sum((1,2))-1).max()<1e-12
 return D
def bath_inputs(C,Q,prior,legal):
 # EXACT relative source logits, approximate unknown rest log-normalizer.
 n=C[:,:,:2];t=np.arange(C.shape[1],dtype=float)*W
 pa=prior[:,None,:2]
 U=(Q+1.4*pa*np.sqrt(t[None,:,None])/(1+n))*(10/3)
 nt=np.maximum(t,1)
 a=np.broadcast_to(prior[:,None,:2],n.shape)
 l=np.broadcast_to(legal[:,None,None],(len(C),C.shape[1],1))
 tt=np.broadcast_to((t/128)[None,:,None],l.shape)
 x=np.concatenate([tt,l/81,np.sqrt(tt),np.log(np.maximum(a,1e-12)),a,n/nt[None,:,None],Q,(n>0).astype(float),U],axis=-1)
 return x,U
def bath_features(P,C,Q,prior,legal):
 x,U=bath_inputs(C,Q,prior,legal)
 # bath target reconstructed from exact conditional law; not used in holdout predictor.
 target=U[:,:,0]+np.log(P[:,:,2]/P[:,:,0])-np.log(legal[:,None]-2)
 err=abs((np.log(P[:,:,0]/P[:,:,1])-(U[:,:,0]-U[:,:,1]))).max()
 assert err<2e-6,err
 return x,U,target,float(err)
def bath_fit(P,C,Q,prior,legal):
 x,U,y,err=bath_features(P,C,Q,prior,legal);X=x[:,:32].reshape(-1,x.shape[-1]);z=y[:,:32].ravel()
 mu=X.mean(0);sd=X.std(0);sd[sd<1e-12]=1.;A=np.column_stack([np.ones(len(X)),(X-mu)/sd])
 pen=np.diag([0.]+[.02]*X.shape[-1]);coef=np.linalg.solve(A.T@A/len(A)+pen,A.T@z/len(A))
 L=np.diag(np.sqrt(np.diag(pen)));r=np.linalg.lstsq(np.vstack([A/np.sqrt(len(A)),L]),np.r_[z/np.sqrt(len(A)),np.zeros(len(L))],rcond=None)[0]
 assert abs(r-coef).max()<1e-10
 return dict(mu=mu.tolist(),sd=sd.tolist(),coef=coef.tolist(),relative_logit_error=err)
def bath_predict(model,C,Q,prior,legal):
 x,U=bath_inputs(C,Q,prior,legal)
 z=(x-np.array(model['mu']))/np.array(model['sd']);coef=np.array(model['coef'])
 bath=np.log(legal[:,None]-2)+coef[0]+z@coef[1:]
 logits=np.concatenate([U,bath[:,:,None]],axis=-1)
 return np.exp(logits-logsumexp(logits,axis=-1,keepdims=True))
def summary(x):
 # x=64 geometry blocks, each already seed averaged
 x=np.asarray(x);rng=np.random.default_rng(630301);ix=rng.integers(0,len(x),(10000,len(x)))
 return dict(mean=x.mean(0).tolist(),ci95=np.quantile(x[ix].mean(1),[.025,.975],axis=0).tolist())
def score_distribution(prob,yidx,Ygrid):
 assert abs(prob.sum(1)-1).max()<1e-10
 p=prob[np.arange(len(prob)),yidx]
 assert np.all(p>0)
 nll=-np.log(p);brier=np.sum(prob*prob,1)-2*p+1
 mu=prob@Ygrid
 y=Ygrid[yidx];err=y-mu
 return np.column_stack([nll,brier,np.mean(err*err,1),err])
def eval_one(P,C,Q,prior,legal,model):
 pred=bath_predict(model,C,Q,prior,legal)
 _,_,_,err=bath_features(P,C,Q,prior,legal)
 return pred,err
def main():
 a=read(S/'AUDIT63.json');assert sha(S/'WAVES63.npz')==a['wave_sha'] and sha(S/'FINAL63.npz')==a['final_sha']
 with np.load(S/'WAVES63.npz') as d:w={k:d[k] for k in d.files}
 with np.load(S/'FINAL63.npz') as d:f={k:d[k] for k in ('prior','meta')}
 tags=list(w['tags']);m=w['meta'];reg=read(S/'REGISTRATION.json');old=read(B/'mqft_restore62_20261126/REGISTRATION.json')
 def load(tag):
  ix=np.flatnonzero(m[:,1]==tags.index(tag));ix=ix[np.lexsort((m[ix,4],m[ix,3]))]
  tagname=str(tag);n=12 if tagname.startswith('response') else 16
  assert len(ix)==64*n and np.array_equal(m[ix,3],np.repeat(np.arange(64),n)) and np.array_equal(m[ix,4],np.tile(np.arange(n),64))
  p,c,q=[w[k][ix] for k in ('wave_p','wave_c','wave_q')]
  legal=np.repeat([81-len(x['moves']) for x in (old['cores'] if n==12 else reg['holdout_cores'])],n)
  prior=f['prior'][m[ix,0]];assert p.shape==(64*n,33,3) and abs(p.sum(-1)-1).max()<1e-10 and np.array_equal(np.diff(c,axis=1),w['wave_y'][ix])
  return p,c,q,prior,legal
 models={}
 # freeze model using previous R62 data only
 for g in (0,3):
  P,C,Q,prior,legal=load(f'response{g}w4')
  zP=states(P);zC=states(C/np.maximum(1,4*np.arange(33))[None,:,None])
  emP,KP,nP=fullfit(P,C,zP);emC,KC,nC=fullfit(P,C,zC)
  bath=bath_fit(P,C,Q,prior,legal)
  np.savez_compressed(S/f'KERNEL_MODEL_g{g}.npz',emP=emP,KP=KP,nP=nP,emC=emC,KC=KC,nC=nC)
  models[str(g)]=dict(bath=bath,kernel_sha=sha(S/f'KERNEL_MODEL_g{g}.npz'))
 dump(S/'MODEL_LOCK63.json',dict(source='only R62 zero field w4 64*12 per generation',old_audit_sha=reg['old_audit_sha'],models=models,code_sha=sha(Path(__file__)),registration_sha=sha(S/'REGISTRATION.json'),observed_new_holdout_results_before_fit=False))
 result=[];cal=[];comp=[]
 for g in (0,3):
  P,C,Q,prior,legal=load(f'hold{g}');zP=states(P);zC=states(C/np.maximum(1,4*np.arange(33))[None,:,None])
  with np.load(S/f'KERNEL_MODEL_g{g}.npz') as d:K={k:d[k] for k in d.files}
  BATH,err=eval_one(P,C,Q,prior,legal,models[str(g)]['bath'])
  n=len(P);assert n==1024
  # oracle exact local wave calibration: compare moment residual and randomized PIT marginal.
  YY=np.diff(C,axis=1).astype(int);p=P[:,:32];mean=W*p;v=W*(np.eye(3)[None,None,:,:]*p[:,:,:,None]-p[:,:,:,None]*p[:,:,None,:])
  delta=YY-mean
  residual=(delta[:,:,:,None]*delta[:,:,None,:]-v).mean(1).reshape(64,16,3,3).mean(1)
  rr=(delta.mean(1)).reshape(64,16,3).mean(1)
  rng=np.random.default_rng(630350+g);pit=binom.cdf(YY[:,:,0]-1,W,p[:,:,0])+rng.random((n,32))*binom.pmf(YY[:,:,0],W,p[:,:,0])
  hist=np.stack([np.histogram(pit[i],bins=np.linspace(0,1,11))[0] for i in range(n)]).reshape(64,16,10).sum(1)/512
  # exact moment finite enumeration on every observed conditional p
  M=masses(p)
  assert abs(M.sum(-1)-1).max()<1e-12 and abs(M@Y-mean).max()<1e-11
  dy=Y[None,None,:,:]-mean[:,:,None,:];cov=np.einsum('nty,ntyi,ntyj->ntij',M,dy,dy);errcov=float(abs(cov-v).max());assert errcov<1e-11
  cal.append(dict(generation=g,n_waves=n*32,mean_residual=summary(rr),covariance_residual=summary(residual),pit_histogram=summary(hist),max_enumerated_cov_error=errcov,relative_logit_error=err))
  for t in (4,8,16,24):
   for h in (1,2,4,8):
    N=W*h;grid=counts_grid(N);lookup={(a,b):i for i,(a,b,c) in enumerate(grid)}
    y=(C[:,t+h]-C[:,t]).astype(int);iy=np.array([lookup[tuple(v[:2])] for v in y])
    # Baselines only use initial t, not future states.
    occ=(C[:,t]+1)/(W*t+3);frozen=P[:,t]
    # Exact multinomial over N assuming frozen p for h>1 (not oracle for h>1).
    p0=masses(occ,grid);p1=masses(frozen,grid)
    dist=propagate_p(K['KP'],t,h);p2=dist[zP[:,t]][:,grid[:,0],grid[:,1]]
    st,inv=np.unique(C[:,t,:2].astype(int),axis=0,return_inverse=True)
    distC=propagate_count(K['emC'],t,h,st);p3=distC[inv][:,grid[:,0],grid[:,1]]
    names=['occupancy_frozen','full_p_frozen','p_state_composed','count_consistent_composed'];probs=[p0,p1,p2,p3]
    if h==1:names+=['source_Q_bath_ridge'];probs+=[masses(BATH[:,t],grid)]
    vals=np.stack([score_distribution(q,iy,grid) for q in probs],1)
    block=vals.reshape(64,16,len(names),6).mean(1)
    diffs={name:summary(block[:,j,0]-block[:,0,0]) for j,name in enumerate(names)}
    result.append(dict(generation=g,start_wave=t,horizon=h,n_test=n,names=names,metrics=['nll','brier','mean_count_mse','residual_a','residual_b','residual_rest'],scores=summary(block),nll_minus_occupancy=diffs,p_minus_full_p=summary(block[:,2,0]-block[:,1,0]),count_minus_full_p=summary(block[:,3,0]-block[:,1,0]),p_unseen=float(np.mean(K['nP'][t,zP[:,t]]==0)),count_unseen=float(np.mean(K['nC'][t,zC[:,t]]==0))))
    # generating function from fitted kernel versus heldout observed increments;
    # report local J, conditional distribution prediction vs mixture across test starts.
    for J in (np.array([-.1,.05,0.]),np.zeros(3),np.array([.1,-.05,0.])):
     phi=np.exp(grid@J);emp=np.exp(y@J);pred=[q@phi for q in probs]
     row=dict(generation=g,start_wave=t,horizon=h,J=J.tolist(),names=names,empirical_W=float(np.log(emp.mean())),pred_W=[float(np.log(v.mean())) for v in pred],mgf_residual=summary(np.column_stack([v-emp for v in pred]).reshape(64,16,len(names)).mean(1)))
     comp.append(row)
  print('KERNEL',g,[(x['start_wave'],x['horizon'],np.round(np.array(x['scores']['mean'])[:,0],5).tolist()) for x in result if x['generation']==g],flush=True)
 dump(S/'KERNEL_RESULTS63.json',dict(scope='time-dependent Markov-additive kernel, training R62 only, test 64 new roots; no future p supplied in composition; fixed-source counts; exact oracle only one wave',results=result,oracle_calibration=cal,tilt_composition=comp,models=models))
 print('KERNEL COMPLETE',len(result),len(comp),flush=True)
if __name__=='__main__':main()
