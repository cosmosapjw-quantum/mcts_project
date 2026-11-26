"""Actual pulse response with paired direct native runs and exact path-score identity."""
from base61 import *
import gzip
from scipy.stats import t as student
from audit_native74 import one as audit9
from audit_rapfi74 import audit_search
H=.35;beta=10/3;ARMS=['zero74','plus74','minus74'];FEATURES=['source_hit','mean_root_mark','p0_source','conditional_capacity','conditional_entropy','participation','mean_score_energy','conditional_response','score_quench','score_redistribution']
def one(arg):
 tag,j=arg;p=S/'data'/tag/(j['id']+'.json.gz')
 with gzip.open(p,'rt') as f:d=json.load(f)
 assert d['input']==j;tr=d['trace'];src=j['sources'][0];h=[0.,H,-H][ARMS.index(j['arm'])]
 if tag=='rapfi':
  assert tr['source74']==src and tr['pulse74']==h
  a=audit_search(tr,j['moves'],'clean',j['seed'],128,src,h);receipt={k:a[k] for k in ('max_probability','max_score','max_log','evals')};g=2
 else:
  a=audit9((tag,j));receipt={k:a[k] for k in ('max_probability','max_score','max_log','evaluations')};g=int(j['generation']==3)
 rows=[];pulse=[];lastE=lastP=None;maxcv=maxbal=0.;ids=[]
 for t in range(32):
  ev=tr['events'][4*t:4*t+4];law=ev[0]['root_selection']['law'];ix=np.argsort(law['actions']);legal=np.array(law['actions'])[ix];u=np.array(law['scores'])[ix];z=beta*u;z-=z.max();p0=np.exp(z);p0/=p0.sum();i=int(np.searchsorted(legal,src));assert legal[i]==src
  prob=p0[i];p1=p0.copy();p1[i]*=np.exp(h if t==8 else 0.);p1/=p1.sum()
  for e in ev:
   l=e['root_selection']['law'];assert np.array_equal(np.array(l['actions'])[ix],legal) and np.max(abs(np.asarray(l['p'])[ix]-p1))<2e-11 and np.array_equal(np.asarray(l['scores'])[ix],u)
  hit=np.mean([e['actions'][0]==src for e in ev]);mark=np.mean([e['root_value'] for e in ev])
  cv=beta**2*(p0@(u-p0@u)**2)
  def E(b):
   x=np.exp(b*(u-u.max()));return -float(x@u/x.sum())
  dt=1e-4;fd=-beta**2*(E(beta+dt)-E(beta-dt))/(2*dt);err=abs(fd-cv);maxcv=max(maxcv,err);assert err<2e-7
  energy=-u;mean=float(p0@energy)
  if lastE is None:w=q=0.
  else:
   w=float(lastP@(energy-lastE));q=float((p0-lastP)@energy);err=abs(mean-lastP@lastE-w-q);maxbal=max(maxbal,err);assert err<1e-12
  rows.append([hit,mark,prob,cv,entropy(p0),1/(p0@p0),mean,prob*(1-prob),w,q])
  if t==8:
   n=4*hit;lp=np.array([x*n-4*np.log1p(prob*np.expm1(x)) for x in (H,-H)])
   pulse=[n,4*prob,n-4*prob,*lp]
  lastE=energy;lastP=p0
  ids.extend([(e['actions'],e['root_value'],e['root_selection']['law']['u']) for e in ev])
 receipt.update(path=str(p.relative_to(B)),sha256=sha(p),bytes=p.stat().st_size,seed=j['seed'],max_capacity_fd_error=maxcv,max_score_balance_error=maxbal)
 return [g,j['core'],j['rep'],ARMS.index(j['arm'])],rows,pulse,hashlib.sha256(json.dumps(ids[:32],separators=(',',':')).encode()).hexdigest(),receipt
def ci(v):
 x=np.asarray(v);m=x.mean(0);d=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return [m.tolist(),(m-d).tolist(),(m+d).tolist()]
def main():
 R=read(S/'REGISTRATION.json');jobs=[]
 for tag in ('schedule0','schedule3','rapfi'):
  p=S/'plans'/f'{tag}.json';assert sha(p)==R['plans'][tag]['sha256'];q=read(p)
  assert len(q)==768 and {x.name for x in (S/'data'/tag).glob('*.gz')}=={j['id']+'.json.gz' for j in q}
  jobs.extend((tag,j) for j in q)
 M=[];F=[];P=[];A=[];hashes={}
 with cf.ProcessPoolExecutor(max_workers=12) as pool:
  for i,(m,f,p,h,a) in enumerate(pool.map(one,jobs,chunksize=8)):
   M.append(m);F.append(f);P.append(p);A.append(a);hashes[tuple(m)]=h
   if (i+1)%768==0:print('AUDIT',i+1,flush=True)
 M=np.array(M);F=np.array(F);P=np.array(P);assert len(M)==2304
 assert len(hashes)==2304
 for g in range(3):
  for c in range(16):
   for r in range(16):assert len({hashes[g,c,r,a] for a in range(3)})==1
 np.savez_compressed(S/'PULSE74.npz',meta=M,features=F,pulse=P)
 results=[]
 for g in range(3):
  shape=(16,16,3,32,len(FEATURES));x=np.empty(shape);p=np.empty((16,16,3,5))
  for m,f,z in zip(M,F,P):
   gg,c,r,a=m
   if gg==g:x[c,r,a]=f;p[c,r,a]=z
  assert np.array_equal(x[:,:,0,:8],x[:,:,1,:8]) and np.array_equal(x[:,:,0,:8],x[:,:,2,:8])
  assert np.array_equal(p[:,:,0,1],p[:,:,1,1]) and np.array_equal(p[:,:,0,1],p[:,:,2,1])
  y=x[:,:,0,:,:2];score=p[:,:,0,2];hit=p[:,:,0,0];drift=p[:,:,0,1]
  def cov(s):
   return np.einsum('crtk,cr->ctk',y-y.mean(1,keepdims=True),s-s.mean(1,keepdims=True))/15
  s=cov(score);c=cov(hit);d=cov(drift);assert np.max(abs(s-c+d))<1e-12
  direct=((x[:,:,1,:,:2]-x[:,:,2,:,:2])/(2*H)).mean(1)
  w=np.exp(p[:,:,0,3:]);rw=np.einsum('crtk,cr->ctk',y,(w[:,:,0]-w[:,:,1])/(2*H))/16
  # Exact conditional same-wave finite difference and derivative integrated over pre-pulse histories.
  pr=p[:,:,0,1]/4
  pp=lambda h:pr*np.exp(h)/(1+pr*np.expm1(h))
  cond=(pp(H)-pp(-H))/(2*H);deriv=pr*(1-pr)
  # Same pulse history law predicts exact finite-h reweighting. Test direct against reweighted finite contrast.
  revp=np.exp(-p[:,:,1,3]);revm=np.exp(-p[:,:,2,4])
  ep_p=p[:,:,1,3];ep_m=p[:,:,2,4]
  result=dict(family=['g0','g3','Rapfi'][g],roots=16,replicates=16,direct_fd=ci(direct),score_derivative=ci(s),count_covariance=ci(c),predictable_drift_covariance=ci(d),reweighted_fd=ci(rw),direct_minus_reweighted=ci(direct-rw),direct_minus_score=ci(direct-s),same_wave_exact_conditional_fd=ci(cond.mean(1)),same_wave_exact_derivative=ci(deriv.mean(1)),same_wave_direct_minus_conditional=ci(direct[:,8,0]-cond.mean(1)),pre_pulse_max_direct=float(abs(direct[:,:8]).max()),mean_score=ci(score.mean(1)),baseline_Lplus=ci(w[:,:,0].mean(1)),baseline_Lminus=ci(w[:,:,1].mean(1)),plus_inverse_L=ci(revp.mean(1)),minus_inverse_L=ci(revm.mean(1)),plus_mean_logLR=ci(ep_p.mean(1)),minus_mean_logLR=ci(ep_m.mean(1)),baseline_profiles=ci(x[:,:,0].mean(1)),mean_effective_sample_size=[float((w[:,:,k].sum()**2/(w[:,:,k]**2).sum())) for k in range(2)])
  results.append(result)
  print('RESPONSE',result['family'],'same wave direct',result['direct_fd'][0][8][0],'score',result['score_derivative'][0][8][0],'cond',result['same_wave_exact_conditional_fd'],'last direct',result['direct_fd'][0][-1],'last score',result['score_derivative'][0][-1],'last drift',result['predictable_drift_covariance'][0][-1],flush=True)
 audit=dict(paths=2304,backups=2304*128,raw_bytes=sum(x['bytes'] for x in A),max_probability=max(x['max_probability'] for x in A),max_score=max(x['max_score'] for x in A),max_log=max(x['max_log'] for x in A),max_capacity_fd_error=max(x['max_capacity_fd_error'] for x in A),max_score_balance_error=max(x['max_score_balance_error'] for x in A),matched_prepulse_triples=768,raw_receipts=A,array_sha256=sha(S/'PULSE74.npz'))
 dump(S/'AUDIT74.json',audit);dump(S/'RESPONSE74.json',dict(scope='native matched source pulse, conditional finite score Gibbs capacity not physical heat capacity, score response vs naive covariance, block pointwise CI',pulse_h=H,wave=8,features=FEATURES,results=results,array_sha256=audit['array_sha256']))
 print('AUDIT',{k:v for k,v in audit.items() if k!='raw_receipts'},flush=True)
if __name__=='__main__':main()
