from base61 import *
import gzip,time
from scipy.special import expit
from scipy.stats import t as student
from itertools import product,combinations
O74=B/'mqft_restore74_20261126';O66=B/'mqft_restore66_20261126';H=.35;BETA=10/3
FAMILIES=['g0','g3','Rapfi']
def ci(x):
 x=np.asarray(x);m=x.mean(0);e=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return np.stack([m,m-e,m+e]).tolist()
def cov(x,y):
 x=np.asarray(x);y=np.asarray(y);return (x-x.mean(0)).T@(y-y.mean(0))/(len(x)-1)
def soft(s):
 z=BETA*(s-s.max());v=np.exp(z);return v/v.sum()
def one(arg):
 kind,r=arg
 if kind==74:
  f=B/r['path']
 else:f=O66/'data'/r['tag']/(r['job']['id']+'.json.gz')
 assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256']
 with gzip.open(f,'rt') as z:d=json.load(z)
 j=d['input'];es=d['trace']['events'];assert len(es)==128 and j['seed']==(r['seed'] if kind==74 else r['job']['seed'])
 if kind==74:
  g=2 if '/rapfi/' in r['path'] else int(j['generation']==3);arm=['zero74','plus74','minus74'].index(j['arm'])
 else:
  assert j==r['job'];g=int(j['generation']==3);arm=int(j['arm']=='counter')
 src=j['sources'][0];rows=[];prefix=[];lp=0.;cum=0.;err=0.;el=0.
 for t in range(32):
  ee=es[4*t:4*t+4];law=ee[0]['root_selection']['law'];a=np.array(law['actions']);u=np.array(law['scores']);p=soft(u);idx=int(np.flatnonzero(a==src)[0]);q=float(p[idx])
  h=( [0,H,-H][arm] if t==8 else 0.) if kind==74 else (-.45+.9*t/31)*(1 if arm==0 else -1)
  p1=p.copy();p1[idx]*=math.exp(h);p1/=p1.sum();hit=0;marks=[]
  for k,e in enumerate(ee):
   l=e['root_selection']['law'];assert l['t']==4*t+k and l['actions']==law['actions'] and l['scores']==law['scores'] and l['beta_p']==BETA and abs(l['logit_h']-h)<1e-14 and e['complete_seq']==4*t+k
   assert np.all(p1>0) and abs(p1.sum()-1)<1e-12 and all(l['active']);err=max(err,float(abs(np.array(l['p'])-p1).max()))
   pick=l['picked'];assert a[pick]==e['actions'][0] and int(np.searchsorted(np.cumsum(l['p']),l['u'],side='right'))==pick
   ll=math.log(l['p'][pick]);lp+=ll;el=max(el,abs(ll-l['log_p']),abs(lp-l['log_p_path']))
   hit+=int(a[pick]==src);mark=e['root_value'];assert mark==(-1)**len(e['actions'])*e['leaf_value'];marks.append(mark)
   if t<8:prefix.append((e['actions'],mark,l['u'],e['leaf_hash']))
  if kind==66:
   hf=-.45+.9*t/31
   # same-history log PF/PC; independent direct log softmax ratio per selected action
   lf=2*hf*hit-4*(np.log1p(q*np.expm1(hf))-np.log1p(q*np.expm1(-hf)))
   ff=p.copy();cc=p.copy();ff[idx]*=np.exp(hf);cc[idx]*=np.exp(-hf);ff/=ff.sum();cc/=cc.sum()
   ll=sum(np.log(ff[e['root_selection']['law']['picked']]/cc[e['root_selection']['law']['picked']]) for e in ee)
   assert abs(ll-lf)<1e-10;cum+=lf
  rows.append([hit/4,float(np.mean(marks)),q,float(p1[idx]),q*(1-q),float(cum)])
 assert err<2e-12 and el<1e-9
 return (g,j['core'],j['rep'],arm),np.array(rows),hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest(),str(f.relative_to(B)),r['sha256'],err,el
def readraw():
 begin=time.monotonic();a=read(O74/'AUDIT74.json');b=read(O66/'AUDIT66.json');assert sha(O74/'PULSE74.npz')==a['array_sha256'] and sha(O66/'NATIVE66.npz')==b['array_sha']
 res={};rec={};maxerr=0.;maxlog=0.;ph={}
 for kind,rr in [(74,a['raw_receipts']),(66,b['jobs'])]:
  data={}
  with cf.ProcessPoolExecutor(max_workers=10) as pool:
   for i,(k,v,h,p,sh,e,l) in enumerate(pool.map(one,[(kind,r) for r in rr],chunksize=8)):
    assert k not in data and p not in rec;data[k]=v;rec[p]=sh;maxerr=max(maxerr,e);maxlog=max(maxlog,l);ph[k]=h
    if (i+1)%768==0:print('RAW',kind,i+1,flush=True)
  res[kind]=data
  if kind==74:
   for g,c,r in product(range(3),range(16),range(16)):assert len({ph[g,c,r,a] for a in range(3)})==1
   with np.load(O74/'PULSE74.npz') as z:
    xx=np.stack([data[tuple(m)] for m in z['meta']]);old=z['features'];assert abs(xx[:,:,:3]-old[:,:,:3]).max()<1e-10
    assert abs(xx[:,:,4]-old[:,:,7]).max()<1e-10
  else:
   with np.load(O66/'NATIVE66.npz') as z:
    xx=np.stack([data[int(m[0]==3),m[1],m[2],m[3]] for m in z['meta']]);old=z['waves'];assert abs(xx[:,:,3]-old[:,:,5]).max()<1e-10 and abs(xx[:,:,5]-old[:,:,9]).max()<1e-9
 assert len(rec)==5376
 return res,rec,maxerr,maxlog,time.monotonic()-begin
def pulse(data):
 results=[];errors=collections.defaultdict(float)
 def err(k,x,y):errors[k]=max(errors[k],float(abs(np.asarray(x)-np.asarray(y)).max()))
 for g in range(3):
  X=np.array([[[data[g,c,r,a] for a in range(3)] for r in range(16)] for c in range(16)]);assert X.shape==(16,16,3,32,6)
  Y=X[:,:,0,:,:2];q=X[:,:,0,:,2];k=4*Y[:,:,:,0];S0=k-4*q;V=4*q*(1-q);Z=S0*S0-V
  assert np.array_equal(X[:,:,0,:8,:],X[:,:,1,:8,:]) and np.array_equal(X[:,:,0,:8,:],X[:,:,2,:8,:])
  assert np.array_equal(X[:,:,0,8,2],X[:,:,1,8,2]) and np.array_equal(X[:,:,0,8,2],X[:,:,2,8,2])
  raw=np.array([[cov(Y[c,:,:,i],S0[c]) for i in range(2)] for c in range(16)]) # c,i,t,s
  naive=np.array([[cov(Y[c,:,:,i],k[c]) for i in range(2)] for c in range(16)])
  drift=np.array([[cov(Y[c,:,:,i],4*q[c]) for i in range(2)] for c in range(16)])
  err('raw_naive_drift',raw,naive-drift)
  med=np.array([cov(q[c],S0[c]) for c in range(16)]);rr=np.tril(med,-1);ii=np.arange(32);rr[:,ii,ii]=(q*(1-q)).mean(1)
  # identity at population level; sample difference is exactly centered innovation covariance.
  scov=np.array([cov(S0[c],S0[c])/4 for c in range(16)])
  err('hit_kernel_sample_decomposition',raw[:,0],med+scov)
  err('triangular_decomposition',raw[:,0]-rr,med-np.tril(med,-1)+scov-np.eye(32)[None,:,:]*(V.mean(1)[:,None,:]/4))
  # independent sample pairwise covariance subset: all 16 roots, 2 observables and all waves.
  for c in range(16):
   pd=np.mean([(Y[c,i,:,0]-Y[c,j,:,0])[:,None]*(S0[c,i]-S0[c,j])[None,:]/2 for i,j in combinations(range(16),2)],axis=0)
   err('pair_covariance',pd,raw[c,0])
  # full score covariance and predictable information: off-diag zero in population, not imposed on sample.
  q8=q[:,:,8];k8=k[:,:,8];ss=S0[:,:,8];vv=V[:,:,8];zz=Z[:,:,8]
  Lp=np.exp(H*k8-4*np.log1p(q8*np.expm1(H)));Lm=np.exp(-H*k8-4*np.log1p(q8*np.expm1(-H)))
  w=(Lp-Lm)/(2*H);we=(Lp+Lm-2)/(H*H)
  # History-measurable anchor: pre-wave observables themselves before pulse, preceding mark and q8 afterward.
  anchor=Y.copy();anchor[:,:,8:,0]=q8[:,:,None];anchor[:,:,8:,1]=Y[:,:,7:8,1]
  rw=np.mean(Y*w[:,:,None,None],1);rwc=np.mean((Y-anchor)*w[:,:,None,None],1)
  rwe=np.mean(Y*we[:,:,None,None],1);rwec=np.mean((Y-anchor)*we[:,:,None,None],1)
  direct=((X[:,:,1,:,:2]-X[:,:,2,:,:2])/(2*H)).mean(1)
  even=((X[:,:,1,:,:2]+X[:,:,2,:,:2]-2*Y)/(H*H)).mean(1)
  sec=np.array([[cov(Y[c,:,:,i],zz[c,:,None])[:,0] for i in range(2)] for c in range(16)]).transpose(0,2,1)
  qt=lambda h:q8*np.exp(h)/(1+q8*np.expm1(h))
  cond=(qt(H)-qt(-H))/(2*H);conde=(qt(H)+qt(-H)-2*q8)/(H*H);der=q8*(1-q8);second=der*(1-2*q8)
  assert abs(direct[:,:8]).max()==0 and abs(even[:,:8]).max()==0 and abs(rwc[:,:8]).max()==0 and abs(rwec[:,:8]).max()==0
  # Enumerate five binomial outcomes independently for each pre-pulse q. Check exact LR normalization, response and curvature.
  n=np.arange(5);prob=np.stack([math.comb(4,int(i))*q8**i*(1-q8)**(4-i) for i in n],-1)
  sr=n-4*q8[...,None];br=sr**2-4*q8[...,None]*(1-q8[...,None])
  wp=np.exp(H*n-4*np.log1p(q8[...,None]*np.expm1(H)));wm=np.exp(-H*n-4*np.log1p(q8[...,None]*np.expm1(-H)))
  err('binomial_normalization',prob.sum(-1),1);err('binomial_Lplus',(prob*wp).sum(-1),1);err('binomial_Lminus',(prob*wm).sum(-1),1)
  err('binomial_score_zero',(prob*sr).sum(-1),0);err('binomial_curvature_zero',(prob*br).sum(-1),0)
  err('binomial_contact',(prob*sr*n/4).sum(-1),der);err('binomial_second',(prob*br*n/4).sum(-1),second)
  err('binomial_finite_contact',(prob*(wp-wm)*n/4/(2*H)).sum(-1),cond);err('binomial_even_contact',(prob*(wp+wm-2)*n/4/H**2).sum(-1),conde)
  # Compare old R74 sample estimates, not independent test data.
  old=read(O74/'RESPONSE74.json')['results'][g];err('R74_direct',ci(direct),old['direct_fd']);err('R74_raw_score',ci(raw[:,:,:,8].transpose(0,2,1)),old['score_derivative']);err('R74_reweighted',ci(rw),old['reweighted_fd'])
  ess=lambda x:(x.sum(1)**2/(x*x).sum(1))/x.shape[1]
  results.append(dict(family=FAMILIES[g],roots=16,reps=16,raw_kernel=ci(raw),naive_kernel=ci(naive),drift_kernel=ci(drift),projected_hit_kernel=ci(rr),predictable_drift_hit_kernel=ci(med),sample_innovation_kernel=ci(scov),bracket=ci(V.mean(1)/4),direct=ci(direct),even_direct=ci(even),rw=ci(rw),rw_control=ci(rwc),rw_even=ci(rwe),rw_even_control=ci(rwec),score=ci(raw[:,:,:,8].transpose(0,2,1)),score_second=ci(sec),naive=ci(naive[:,:,:,8].transpose(0,2,1)),drift=ci(drift[:,:,:,8].transpose(0,2,1)),conditional_fd=ci(cond.mean(1)),conditional_derivative=ci(der.mean(1)),conditional_even=ci(conde.mean(1)),conditional_second=ci(second.mean(1)),direct_minus_rw=ci(direct-rw),direct_minus_control=ci(direct-rwc),direct_minus_contact=ci(direct[:,8,0]-cond.mean(1)),even_minus_contact=ci(even[:,8,0]-conde.mean(1)),Lplus=ci(Lp.mean(1)),Lminus=ci(Lm.mean(1)),ess_plus=ci(ess(Lp)),ess_minus=ci(ess(Lm)),prepulse_max_direct=float(abs(direct[:,:8]).max()),prepulse_max_control=float(abs(rwc[:,:8]).max())))
 assert max(errors.values())<1e-9,errors
 return results,dict(errors)
def schedule(data):
 out=[];err=0.
 for g in range(2):
  X=np.array([[[data[g,c,r,a] for a in range(2)] for r in range(16)] for c in range(48)]) # c,r,arm,t,f
  assert X.shape==(48,16,2,32,6)
  y=X[:,:,:,:,:2];L=X[:,:,:,:,5];f=y[:,:,0];c=y[:,:,1];lf=L[:,:,0];lc=L[:,:,1]
  direct=(f-c).mean(1);f_to_c=np.mean(f*np.exp(-lf)[...,None],1);c_to_f=np.mean(c*np.exp(lc)[...,None],1)
  # F-C computed in either source arm with raw normalized-in-expectation RN. Mixture balance weights bounded [0,2].
  fd_f=f.mean(1)-f_to_c;fd_c=c_to_f-c.mean(1)
  wf=2*expit(L);wc=2*expit(-L);err=max(err,float(abs(wf+wc-2).max()))
  assert wf.min()>=0 and wc.min()>=0 and wf.max()<=2 and wc.max()<=2
  bal=np.mean(y*(wf-wc)[...,None],axis=(1,2));bal_f=np.mean(y*wf[...,None],axis=(1,2));bal_c=np.mean(y*wc[...,None],axis=(1,2))
  err=max(err,float(abs(bal-bal_f+bal_c).max()))
  # paired root averages only, independent seeds between arms
  ess=lambda w:w.sum(1)**2/(w*w).sum(1)/16
  out.append(dict(family=FAMILIES[g],roots=48,independent_reps_per_arm=16,direct=ci(direct),forward_RN=ci(fd_f),counter_RN=ci(fd_c),balance=ci(bal),direct_minus_forward=ci(direct-fd_f),direct_minus_counter=ci(direct-fd_c),direct_minus_balance=ci(direct-bal),forward_weight_mean=ci(np.exp(-lf).mean(1)),counter_weight_mean=ci(np.exp(lc).mean(1)),ess_forward=ci(ess(np.exp(-lf))),ess_counter=ci(ess(np.exp(lc))),balance_norm_f=ci(wf.mean((1,2))),balance_norm_c=ci(wc.mean((1,2))),h_forward=(-.45+.9*np.arange(32)/31).tolist(),h_counter=(.45-.9*np.arange(32)/31).tolist()))
 assert err<1e-12
 return out,err
def tiny():
 # two adaptive Bernoulli waves: q0=.3, q1=.2 if first 0, .8 if first 1; Y=second bit. w=1 each.
 q0=.3;q1=[.2,.8]
 def expected(h):
  z=0.
  for a,b in product(range(2),repeat=2):
   p0=expit(math.log(q0/(1-q0))+h[0]);p1=expit(math.log(q1[a]/(1-q1[a]))+h[1])
   z+=(p0 if a else 1-p0)*(p1 if b else 1-p1)*b
  return z
 rr=np.zeros(2);HH=np.zeros((2,2));naive=0.;mu=0.;ex=0.
 for a,b in product(range(2),repeat=2):
  p=(q0 if a else 1-q0)*(q1[a] if b else 1-q1[a]);s=np.array([a-q0,b-q1[a]]);v=np.array([q0*(1-q0),q1[a]*(1-q1[a])])
  rr+=p*b*s;HH+=p*b*(np.outer(s,s)-np.diag(v));mu+=p*b;ex+=p*b*b
 h=1e-4;E=np.eye(2)*h;fd=np.array([(expected(e)-expected(-e))/(2*h) for e in E]);hess=np.array([[(expected(a+b)-expected(a-b)-expected(-a+b)+expected(-a-b))/(4*h*h) for b in E] for a in E])
 assert abs(rr-fd).max()<2e-6 and abs(HH-hess).max()<2e-6
 naive=ex-mu*mu;assert abs(naive-rr[1])>1e-2
 return dict(response=rr.tolist(),hessian=HH.tolist(),fd_error=float(abs(rr-fd).max()),hessian_fd_error=float(abs(HH-hess).max()),naive_second_bit_variance=naive,contact_response=float(rr[1]))
def main():
 data,rec,e,l,sec=readraw();p,errs=pulse(data[74]);s,se=schedule(data[66]);a=tiny()
 dump(S/'RESPONSE93.json',dict(status='PASS_SCOPED',pulse=p,schedule=s,raw_paths=len(rec),pulse_paths=2304,schedule_paths=3072,matched_pulse_triples=768,raw_receipts=rec,max_probability_error=e,max_log_error=l,elapsed_raw_s=sec,checks=errs,mixture_error=se,tiny=a,scope='Finite native logit response. Exact score/Hessian and controlled estimates; not equilibrium FDT. Root-block pointwise descriptive intervals, no simultaneous significance.'))
 print('COMPLETE',len(rec),e,l,errs,'mixture',se,'tiny',a,flush=True)
if __name__=='__main__':main()
