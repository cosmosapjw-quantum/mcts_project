from instrument81 import *
from itertools import combinations,product
import time,gzip
from scipy.special import xlogy
P=B/'mqft_restore95_20261128';A=B/'mqft_restore81_20261127';R94=B/'mqft_restore94_20261128';R83=B/'mqft_restore83_20261127';R73=B/'mqft_restore73_20261126'
ERR=collections.defaultdict(float);DELTA=[.1,.5,.9]
def norm(a):return float(np.max(abs(np.asarray(a))))
def check(k,a,b,tol=1e-9):
 e=norm(np.asarray(a)-np.asarray(b));ERR[k]=max(ERR[k],e);assert e<tol,(k,e)
def ent(a,axis=-1):
 a=np.maximum(np.asarray(a).real,0);return -np.sum(xlogy(a,a),axis=axis)
def vn(a):
 v=np.linalg.eigvalsh(a);assert v.min()>-1e-10;return float(ent(v))
def td(a,b):return float(.5*np.sum(abs(np.linalg.eigvalsh(a-b))))
def red(a,keep,n):
 rest=[i for i in range(n) if i not in keep];p=list(keep)+rest;d=2**len(keep);e=2**len(rest)
 return np.trace(a.reshape([2]*(2*n)).transpose(p+[i+n for i in p]).reshape(d,e,d,e),axis1=1,axis2=3)
def MI(j):
 j=np.asarray(j).real;assert j.min()>-1e-10
 return ent(j.sum(-1))+ent(j.sum(-2))-ent(j.reshape(j.shape[:-2]+(-1,)))
# deterministic Fibonacci directions plus Cartesian axes (no random adaptive fit)
t=(np.arange(256)+.5)/256;az=np.arange(256)*np.pi*(3-np.sqrt(5));z=1-2*t
DIR=np.vstack([np.c_[np.sqrt(1-z*z)*np.cos(az),np.sqrt(1-z*z)*np.sin(az),z],np.eye(3)])
def measurement(p,r):
 # actual joint outcome probabilities of each candidate POVM/projective measurement
 d=r.shape[-1];w=p[:,None,None]*r
 diag=np.diagonal(w,axis1=1,axis2=2).real
 cz=float(MI(diag));best=cz;kind='Z';axis=None
 for label,Q in [('prior_helstrom',w[0]-w[1]),('equal_helstrom',r[0]-r[1])]:
  e,v=np.linalg.eigh(Q)
  probs=np.einsum('ij,bik,kj->bj',v.conj(),w,v).real
  # eigenbasis (rank-one projective); grouped positive/negative eigenspace
  for tag,j in [(label+'_basis',probs),(label+'_binary',np.stack([probs[:,e>=0].sum(1),probs[:,e<0].sum(1)],axis=1))]:
   val=float(MI(j))
   if val>best:best=val;kind=tag
 if d==2:
  v=np.array([[np.trace(x@a).real for a in PAULI] for x in r])
  D=v[0]-v[1];D2=p[0]*v[0]-p[1]*v[1]
  dirs=np.vstack([DIR,*[x[None,:]/np.linalg.norm(x) for x in (D,D2) if np.linalg.norm(x)>1e-14]])
  prob=(1+np.einsum('ki,bi->kb',dirs,v))/2;prob=np.clip(prob,0,1)
  j=p[None,:,None]*np.stack([prob,1-prob],-1)
  vals=MI(j);k=int(vals.argmax())
  if vals[k]>best:best=float(vals[k]);kind='sphere_projective';axis=dirs[k].tolist()
 return cz,best,kind,axis
def packing(masks,ok):
 # exact enumeration of every disjoint subcollection, max cardinality
 best=0
 for k in range(1,len(masks)+1):
  for ix in combinations(range(len(masks)),k):
   if all(ok[i] for i in ix) and sum(masks[i].bit_count() for i in ix)==(sum(1<<q for q in set(j for i in ix for j in range(3) if masks[i]&(1<<j)))).bit_count():best=max(best,k)
 return best
def ensemble(J,ptr,frag,n,s):
 a=red(J,[ptr]+frag,n);d=2**len(frag);u=a.reshape(2,d,2,d);r=np.stack([u[i,:,i,:] for i in range(2)])
 p=np.trace(r,axis1=1,axis2=2).real;check('pointer_norm',p.sum(),1);assert p.min()>-1e-10
 p=np.maximum(p,0);p/=p.sum();H=float(ent(p));avg=r.sum(0)
 if p.min()<1e-15:
  # absent pointer outcome has no ensemble information, ratio not defined
  return dict(fragment=frag,H=H,QMI=vn(red(a,[0],len(frag)+1))+vn(avg)-vn(a),chi=0.,classical=0.,lower=0.,upper=0.,trace_distance=None,kind='degenerate',axis=None,weighted_lower=0.,weighted_upper=0.,conditional_product_TD=None)
 r=np.array([x/w for x,w in zip(r,p)]);assert min(np.linalg.eigvalsh(r).ravel())>-1e-10
 check('conditional_trace',np.trace(r,axis1=1,axis2=2),1)
 chi=vn(avg)-sum(p[i]*vn(r[i]) for i in range(2))
 qmi=vn(red(a,[0],len(frag)+1))+vn(avg)-vn(a)
 D=td(r[0],r[1]);cz,lo,kind,axis=measurement(p,r);hi=min(chi,H*D)
 assert -1e-9<=cz<=lo+1e-9 and lo<=hi+1e-9 and chi<=H+1e-9 and chi<=qmi+1e-9 and D<=1+1e-9
 check('ensemble_avg',avg,np.einsum('b,bij->ij',p,r))
 prodtd=None
 if len(frag)>1:
  a0=[]
  for v in r:
   pp=np.ones((1,1),complex)
   for i in range(len(frag)):pp=np.kron(pp,red(v,[i],len(frag)))
   a0.append(td(v,pp))
  prodtd=float(p@a0)
 return dict(fragment=frag,H=H,QMI=qmi,chi=chi,classical=cz,lower=lo,upper=hi,trace_distance=D,kind=kind,axis=axis,weighted_lower=s*lo,weighted_upper=s*hi,conditional_product_TD=prodtd)
def one(arg):
 g,c,protocol,mode,d,xi,s,J=arg;n=d+1;bits=np.array(list(product(range(2),repeat=d)));dist=np.sum(bits[:,None,:]!=bits[None,:,:],-1);Jx=J*np.kron(xi**dist,np.ones((2,2)));check('tag_diag',Jx.diagonal(),J.diagonal());assert np.linalg.eigvalsh(Jx).min()>-1e-10
 out=[]
 for name,ptr,fs in [('first_record',0,list(range(1,d))),('system_Z',d,list(range(d)))]:
  if not fs:continue
  z=red(Jx,[ptr],n);pp=z.diagonal().real;H=float(ent(pp));j=Jx.reshape([2]*(2*n));bits0=np.arange(2**n);mask=((bits0[:,None]^bits0[None,:])&(1<<(n-1-ptr)))!=0
  deph=Jx.copy();deph[mask]=0
  fragments=[ensemble(Jx,ptr,list(f),n,s) for k in range(1,len(fs)+1) for f in combinations(fs,k)]
  check('pointer_H',np.array([v['H'] for v in fragments]),H)
  # local mask within fragment universe, exact partition packing by recursive search
  masks=[sum(1<<fs.index(i) for i in v['fragment']) for v in fragments]
  def pack(ok):
   def f(i,used):
    if i==len(masks):return 0
    a=f(i+1,used)
    if ok[i] and not used&masks[i]:a=max(a,1+f(i+1,used|masks[i]))
    return a
   return f(0,0)
  packings=[]
  if H>1e-6:
   for frac in DELTA:
    lower=pack([v['lower']>=frac*H+1e-9 for v in fragments]);upper=pack([v['upper']>=frac*H-1e-9 for v in fragments]);assert lower<=upper
    packings.append([lower,upper])
  out.append(dict(pointer=name,H=H,pointer_dephase_TD=td(Jx,deph),fragments=fragments,packing=packings,labels=['R'+str(i+1) for i in range(d)]+['S']))
 return dict(family=g,core=c,protocol=protocol,mode=mode,depth=d,xi=xi,success=s,pointers=out),dict(ERR)
def rehash(x):
 path,h=x;assert sha(B/path)==h;return path,h
def controls():
 # classical binary channels including biased priors; exact overlap bound
 ans=[]
 for p0 in [.1,.5,.9]:
  p=np.array([p0,1-p0]);H=float(ent(p))
  for q in [0,.25,.5,1.]:
   r=np.array([np.diag([1.,0]),np.diag([1-q,q])]);chi=vn(np.einsum('b,bij->ij',p,r))-sum(p[i]*vn(r[i]) for i in range(2));cz,lo,_,_=measurement(p,r);check('commuting',cz,chi);check('commuting_lower',lo,chi);assert chi<=H*td(*r)+1e-9
   ans.append([p0,q,chi,H*td(*r)])
 # equal-prior pure overlap c; analytic accessible I for Helstrom projective, upper bound distinct
 for c in [0.,.2,.6,.9,1.]:
  v=np.array([c,np.sqrt(1-c*c)]);r=np.array([np.diag([1.,0]),np.outer(v,v)]);p=np.array([.5,.5]);D=np.sqrt(1-c*c);chi=ent([(1+c)/2,(1-c)/2]);expected=np.log(2)-ent([(1+D)/2,(1-D)/2]);cz,lo,_,_=measurement(p,r);check('pure_overlap_lower',lo,expected);check('pure_overlap_entropy',vn((r[0]+r[1])/2),chi);assert expected<=min(chi,np.log(2)*D)+1e-9;ans.append([.5,c,lo,chi])
 # ideal 3-copy classical broadcast, GHZ coherent counterpart: perfect fragment info, pointer dephase disturbance different
 v=np.zeros(8);v[[0,-1]]=1/np.sqrt(2);ghz=np.outer(v,v);broadcast=np.diag(v*v)
 for a in [ghz,broadcast]:
  for f in [[1],[2],[1,2]]:
   e=ensemble(a,0,f,3,1);check('broadcast_information',e['lower'],np.log(2));check('broadcast_product',e['conditional_product_TD'] or 0,0)
 return ans
def main():
 start=time.monotonic();a=read(P/'ENTANGLE95.json');r=read(R94/'RECORD94.json');assert sha(P/'JOINT95.npz')==a['joint_sha']
 with cf.ProcessPoolExecutor(max_workers=8) as pool:receipts=dict(pool.map(rehash,a['raw_receipts'].items(),chunksize=8))
 assert len(receipts)==720
 with np.load(P/'JOINT95.npz') as z:D={k:z[k] for k in z.files}
 index={(int(g),int(c),p,m):i for i,(g,c,p,m) in enumerate(D['meta'])};args=[]
 for row in a['rows']:
  g,c,p,m,d,xi,s=[row[k] for k in ['family','core','protocol','mode','depth','xi','success']];J=D['J'+str(d)][index[g,c,p,m]]
  args.append((g,c,p,m,d,xi,s,J))
 assert len(args)==5184;rows=[]
 with cf.ProcessPoolExecutor(max_workers=10) as pool:
  for i,(v,e) in enumerate(pool.map(one,args,chunksize=12)):
   rows.append(v)
   for k,x in e.items():ERR[k]=max(ERR[k],x)
   if (i+1)%1728==0:print('STATE',i+1,flush=True)
 assert len(rows)==5184;ct=controls()
 # compare earlier R83 first bit Holevo in same final conditional ensemble
 f=read(R83/'FRAGMENTS83.json');lookup={(x['family'],x['core'],x['protocol']):x for x in f['contexts']}
 for row in rows:
  if row['depth']==3 and row['mode']=='plus81' and row['xi']==1.:
   old=lookup[row['family'],row['core'],row['protocol']];first=row['pointers'][0];assert first['pointer']=='first_record';check('R83_H',first['H'],old['pointer_entropy'])
   for x,y in zip(first['fragments'],old['fragments']):check('R83_chi',x['chi'],y['holevo'])
 dump(S/'DARWIN96.json',dict(status='PASS_SCOPED',raw_receipts=receipts,contexts=72,controlled_states=len(rows),ensembles=sum(len(x['pointers']) for x in rows),fragment_evaluations=sum(len(y['fragments']) for x in rows for y in x['pointers']),thresholds=DELTA,rows=rows,errors=dict(ERR),controls=ct,elapsed_s=time.monotonic()-start,scope='Conditional model, attainable projective lower and certified Holevo/TV upper; not optimized general POVM or physical objectivity'))
 print('DONE',len(rows),dict(ERR),flush=True)
if __name__=='__main__':main()
