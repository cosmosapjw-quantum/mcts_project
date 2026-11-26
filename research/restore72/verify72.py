"""Independent raw-witness, matrix pushforward, and block-MGF verification."""
from base61 import *
import gzip,itertools
from scipy.special import logsumexp
Llist=[13,15,17,19,21]
R=read(S/'SCALE_RESULTS72.json');mem=read(S/'MEMORY72.json');old=B/'mqft_restore71_20261126';assert sha(old/'JOINT71.npz')==mem['data_sha']
def mat(L,m):
 C=np.zeros((m*m+1,L*L+1));h=L//2;k=m//2
 for a in range(L*L):
  y,x=divmod(a,L);u,v=y-h,x-h
  C[(u+k)*m+v+k if max(abs(u),abs(v))<=k else m*m,a]=1
 C[-1,-1]=1;assert np.all(C.sum(0)==1)
 return C
ncomp=0;err=0.;errc=0.;werr=0.;quiescent=0;new=[];cross=[];group_dist={}
for L in Llist:
 A={k:v for k,v in np.load(S/f'SCALE_L{L}.npz').items()};meta=A['meta'];act=A['actions'];P=A['p']
 # source-independent matrix composition.
 for k in [x for x in [3,7,13,15,17,19,21] if x<=L]:
  for m in [x for x in [3,7,13,15,17,19,21] if x<=k]:
   assert np.array_equal(mat(k,m)@mat(L,k),mat(L,m));ncomp+=1
 C=mat(L,13);p=np.pad(P,((0,0),(0,0),(0,1)))@C.T
 for c in range(8):
  for arm in (0,1):
   ii=np.flatnonzero((meta[:,0]==c)&(meta[:,1]==arm));ii=ii[np.argsort(meta[ii,2])]
   for t in (0,7,15,31):
    a=p[ii,t];m=a.mean(0);noise=.5*np.abs(a[:2].mean(0)-a[2:].mean(0)).sum()
    group_dist[L,c,arm,t]=(m,float(noise))
 for q in [x for x in R['witnesses'] if x['L']==L]:
  rr=[];prefix=[];cl=[]
  reg=read(S/f'REGISTRATION_L{L}.json');plan=read(S/'plans'/f'L{L}.json')
  for rep in (q['rep1'],q['rep2']):
   j=next(j for j in plan['jobs'] if j['position_id']==f'c{q["core"]:02d}_'+('dense' if q['arm'] else 'fixed') and j['replicate']==rep)
   f=S/'data'/f'L{L}'/(j['id']+'.jsonl.gz')
   with gzip.open(f,'rt') as z:hd,row,end=map(json.loads,z)
   es=sorted(row['trace']['replicas'][0]['events'],key=lambda e:e['root_selection']['clock']);t=q['t'];law=es[t]['root_selection']['law']
   def rg(a):
    y,x=divmod(a,L);r=max(abs(y-L//2),abs(x-L//2));return 0 if r<=1 else 1 if r<=3 else 2 if r<=6 else 3
   n=np.bincount([rg(e['actions'][0]) for e in es[:t]],minlength=4)
   scores=np.asarray(law['scores']);pr=np.exp(law['beta_p']*(scores-scores.max()));pr/=pr.sum();p0=np.bincount([rg(a) for a in law['actions']],weights=pr,minlength=4)
   assert np.array_equal(n,q['counts']);rr.append(p0);prefix.append(tuple(e['actions'][0] for e in es[:t]));cl.append(all(e['root_backup_clock']<es[t]['root_selection']['clock'] for e in es[:t]))
  tv=.5*np.abs(rr[0]-rr[1]).sum();werr=max(werr,float(abs(tv-q['tv'])));assert tv>1e-9
  assert prefix[0]!=prefix[1]
  quiescent+=int(all(cl));new.append(dict(L=L,core=q['core'],arm=q['arm'],t=q['t'],tv=float(tv),quiescent=bool(all(cl))))
for L,M in zip(Llist[:-1],Llist[1:]):
 for c in range(8):
  for arm in (0,1):
   for t in (0,7,15,31):
    p,n=group_dist[L,c,arm,t];q,m=group_dist[M,c,arm,t]
    cross.append(dict(L=L,Lnext=M,core=c,arm=arm,t=t,conditional_mean_TV=float(.5*abs(p-q).sum()),half_split_noise_L=n,half_split_noise_next=m,outside_delta=float(q[-1]-p[-1])))
A={k:v for k,v in np.load(old/'JOINT71.npz').items()};J=np.array(read(old/'PHYSICS_RESULTS.json')['sources'])
for f in mem['results']:
 g=f['family'];ix=A['meta'][:,0]==g;meta=A['meta'][ix];X=np.concatenate([A['counts'][ix,:,:2],A['marks'][ix].sum(-1)[...,None]],-1)/128
 for root in f['roots']:
  ids=(meta[:,1]==root['core']);direct=np.exp(X[ids].sum(1)@J.T).mean(0);err=max(err,float(np.max(abs(direct-root['true_mgf']))))
  # For uniform root donor model, whole-path and block path products verified by independent explicit sums.
  train=(meta[:,1]%16>=4)&(meta[:,3]==meta[ids,3][0]);x=X[train]
  for q in root['pred']:
   if q['variant']!=0:continue
   H=q['H'];factors=np.stack([np.exp(x[:,t:t+H].sum(1)@J.T).mean(0) for t in range(0,32,H)])
   w=np.log(factors.prod(0));errc=max(errc,float(np.max(abs(w-q['W']))))
assert ncomp==R['composition_checks'] and len(new)==len(R['witnesses']) and max(werr,err,errc)<1e-10
out=dict(status='PASS_SCOPED',matrix_composition_checks=ncomp,raw_witness_checks=len(new),quiescent_witnesses=quiescent,max_witness_TV_error=werr,direct_MGF_max_error=err,uniform_block_MGF_max_error=errc,witness_replay=new,cross_conditional=cross,scope='matrix, raw witness, and explicit block checks; same-agent independent implementation; cross-size conditional TV diagnostic with split noise, not causal proof')
dump(S/'INDEPENDENT72.json',out)
print({k:v for k,v in out.items() if k not in ('witness_replay','cross_conditional')},flush=True)
