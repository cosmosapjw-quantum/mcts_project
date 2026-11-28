"""Independently re-reduce R74 actual paired source pulse and response."""
from base61 import *
import gzip
O=B/'mqft_restore74_20261126';H=.35;BETA=10/3
def one(r):
 f=B/r['path'];assert sha(f)==r['sha256'] and f.stat().st_size==r['bytes']
 with gzip.open(f,'rt') as z:d=json.load(z)
 j=d['input'];es=d['trace']['events'];src=j['sources'][0];arm=['zero74','plus74','minus74'].index(j['arm']);h=[0,H,-H][arm];g=2 if '/rapfi/' in r['path'] else int(j['generation']==3)
 rows=[];pulse=None;hash0=[];pmax=0.
 for t in range(32):
  e=es[4*t:4*t+4];l=e[0]['root_selection']['law'];a=np.array(l['actions']);u=np.array(l['scores']);v=np.exp(BETA*(u-u.max()));p=v/v.sum();i=np.flatnonzero(a==src)[0];q=p[i];actual=p.copy();actual[i]*=np.exp(h if t==8 else 0);actual/=actual.sum()
  for w in e:
   m=w['root_selection']['law'];assert m['actions']==l['actions'] and m['scores']==l['scores'];pmax=max(pmax,float(abs(actual-m['p']).max()));assert pmax<2e-12
   if t<8:hash0.append([w['actions'],w['root_value'],m['u']])
  hit=sum(w['actions'][0]==src for w in e)/4;mark=np.mean([w['root_value'] for w in e])
  rows.append([hit,mark,q,BETA*BETA*(p@(u-p@u)**2),-p@np.log(p),1/(p@p),-p@u,q*(1-q)])
  if t==8:pulse=[4*hit,4*q,4*(hit-q),*[4*hit*x-4*np.log1p(q*np.expm1(x)) for x in (H,-H)]]
 return [g,j['core'],j['rep'],arm],rows,pulse,hashlib.sha256(json.dumps(hash0,separators=(',',':')).encode()).hexdigest(),pmax
def main():
 A=read(O/'AUDIT74.json');R=read(O/'RESPONSE74.json');assert sha(O/'PULSE74.npz')==A['array_sha256']==R['array_sha256']
 with np.load(O/'PULSE74.npz') as z:m=z['meta'];f=z['features'];p=z['pulse']
 ix={tuple(x):i for i,x in enumerate(m)};F=np.empty(f[:,:,:8].shape);P=np.empty(p.shape);prefix={};e=0.
 with cf.ProcessPoolExecutor(max_workers=6) as pool:
  for k,v,u,h,er in pool.map(one,A['raw_receipts'],chunksize=8):
   i=ix[tuple(k)];F[i]=v;P[i]=u;prefix[tuple(k)]=h;e=max(e,er)
 assert len(prefix)==2304 and abs(F-f[:,:,:8]).max()<1e-10 and abs(P-p).max()<1e-10
 rows=[]
 for g,r in enumerate(R['results']):
  X=np.zeros((16,16,3,32,8));S0=np.zeros((16,16,3,5))
  for k,i in ix.items():
   if k[0]==g:X[k[1:]]=F[i];S0[k[1:]]=P[i]
  for k in range(16):
   for j in range(16):assert len({prefix[g,k,j,a] for a in range(3)})==1
  D=(X[:,:,1,:,:2]-X[:,:,2,:,:2])/(2*H);Y=X[:,:,0,:,:2];score=S0[:,:,0,2];cnt=S0[:,:,0,0];drift=S0[:,:,0,1]
  # Independent covariance via centered product of all pair differences.
  def cov(z):return np.sum((Y-Y.mean(1,keepdims=True))*(z-z.mean(1,keepdims=True))[:,:,None,None],axis=1)/15
  direct=D.mean(1);der=cov(score);naive=cov(cnt);cor=cov(drift);assert abs(der-naive+cor).max()<1e-12
  weights=np.exp(S0[:,:,0,3:]);rw=(Y*(weights[:,:,0]-weights[:,:,1])[:,:,None,None]/(2*H)).mean(1)
  from scipy.stats import t as st
  def ci(z):
   m=z.mean(0);v=st.ppf(.975,len(z)-1)*z.std(0,ddof=1)/np.sqrt(len(z));return np.array([m,m-v,m+v])
  out=dict(family=r['family'],direct_fd=ci(direct).tolist(),score_derivative=ci(der).tolist(),count_covariance=ci(naive).tolist(),predictable_drift_covariance=ci(cor).tolist(),reweighted_fd=ci(rw).tolist(),same_wave_exact_derivative=r['same_wave_exact_derivative'],same_wave_exact_conditional_fd=r['same_wave_exact_conditional_fd'],baseline_profiles=ci(X[:,:,0].mean(1)).tolist())
  for k in ('direct_fd','score_derivative','count_covariance','predictable_drift_covariance','reweighted_fd'):assert abs(np.array(out[k])-r[k]).max()<1e-10
  rows.append(out)
 dump(S/'RESPONSE90.json',dict(status='PASS_SCOPED',paths=2304,matched_prefix_triples=768,max_probability_error=e,max_feature_error=float(abs(F-f[:,:,:8]).max()),max_pulse_error=float(abs(P-p).max()),results=rows,raw_receipts=A['raw_receipts'],scope='Actual H32 source pulse +/-0.35, paired root blocks, exact likelihood score and finite contrast; no equilibrium FDT or physical temperature. Prior full path audit inherited.'))
 print('RESPONSE COMPLETE',e,float(abs(F-f[:,:,:8]).max()),[(x['family'],x['direct_fd'][0][8],x['score_derivative'][0][8]) for x in rows],flush=True)
if __name__=='__main__':main()
