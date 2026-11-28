"""Actual clean ordered-history reductions and full legal Shannon observables."""
from base61 import *
import gzip
from itertools import combinations
O=B/'mqft_restore86_20261127';O80=B/'mqft_restore80_20261127';O82=B/'mqft_restore82_20261127';O84=B/'mqft_restore84_20261127'
CUTS=[4,8,16,32,64,128]
def ent(p):
 p=np.asarray(p);p=p[p>0];return float(-p@np.log(p))
def one(r):
 f=B/r['path'];assert sha(f)==r['sha256'] and f.stat().st_size==r['bytes']
 with gzip.open(f,'rt') as z:d=json.load(z)
 j=d['input'];tr=d['trace'];es=tr['events'];cs={x['k']:x for x in tr['cuts']};L=15 if r['family']==2 else 9;N=L*L
 assert j['seed']==r['seed'] and j['core']==r['core'] and j['rep']==r['rep'] and len(es)==128 and list(sorted(cs))==CUTS
 a=np.array([e['actions'][0] for e in es]);marks=np.array([e['root_value'] for e in es]);h=np.isin(a,j['sources']);n=np.zeros(N,int);w=np.zeros(N);s2=np.zeros(N);lp=0.;logs=[];H=[];C=[];P=[];UN=[];ng=0;scoreerr=0.;perr=0.;last=None;pr=np.array(cs[4]['prior']);legal=np.flatnonzero(pr>0)
 assert set(legal)==set(range(N))-set(j['moves'])
 for k,e in enumerate(es):
  if k%4==0:
   q=np.divide(w.astype(np.float32),n.astype(np.float32),out=np.zeros(N,np.float32),where=n>0)
   sc=q+np.float32(1.4)*pr.astype(np.float32)*np.sqrt(np.float32(k))/(np.float32(1)+n.astype(np.float32))
  l=e['root_selection']['law'];acts=np.array(l['actions']);s=np.array(l['scores']);p=np.array(l['p']);ix=np.argsort(acts)
  assert np.array_equal(acts[ix],legal) and all(l['active']) and l['beta_p']==10/3 and l['logit_h']==0 and l['t']==k and e['complete_seq']==k
  scoreerr=max(scoreerr,float(abs(s-sc[acts]).max()));v=np.exp((s-s.max())*10/3);v/=v.sum();perr=max(perr,float(abs(v-p).max()));assert min(p)>0 and abs(p.sum()-1)<1e-12
  if k%4==0:last=p[ix];P.append(float(p[np.isin(acts,j['sources'])].sum()))
  else:assert abs(last-p[ix]).max()<2e-12
  pick=l['picked'];assert acts[pick]==a[k] and np.searchsorted(np.cumsum(p),l['u'],side='right')==pick
  ell=math.log(p[pick]);lp+=ell;logs.append(ell);assert abs(lp-l['log_p_path'])<1e-9 and abs(ell-l['log_p'])<1e-12
  H.append(ent(p));ng+=int(s[pick]<s.max()-1e-8)
  assert marks[k]==(-1)**len(e['actions'])*e['leaf_value']
  if k%4==3:
   for i in range(k-3,k+1):n[a[i]]+=1;w[a[i]]+=marks[i];s2[a[i]]+=marks[i]**2
   if k+1 in cs:
    c=cs[k+1];assert np.array_equal(n,c['n']) and abs(w-c['w']).max()<1e-9 and abs(s2-c['s2']).max()<1e-9 and np.array_equal(pr,c['prior'])
    C.append([k+1,ent(n/(k+1)),ent(pr),np.mean(H),-lp,float(n[j['sources']].sum()/(k+1)),float(w.sum()/(k+1)),float(np.log(len(legal)))])
 assert scoreerr<2e-6 and perr<2e-12
 unseen=legal[n[legal]==0];mass=float(last[np.isin(legal,unseen)].sum())
 return dict(group=r['group'],family=r['family'],core=r['core'],stage=r['stage'],rep=r['rep'],path=r['path'],sha256=r['sha256'],bytes=r['bytes'],score_error=scoreerr,prob_error=perr),a,marks,logs,C,[h.sum()/128,marks.mean()],[int(h.sum()),int((marks>0).sum())],[len(unseen),mass,ng],np.array(P),np.array([h.reshape(32,4).sum(1),marks.reshape(32,4).sum(1)])
def main():
 r=read(O/'AUDIT86.json');rec=r['raw_receipts'];assert len(rec)==7488 and len(set(x['path'] for x in rec))==7488
 out=[]
 with cf.ProcessPoolExecutor(max_workers=8) as pool:
  for i,x in enumerate(pool.map(one,rec,chunksize=8)):
   out.append(x)
   if (i+1)%936==0:print('PATH',i+1,flush=True)
 rows=[x[0] for x in out];M=np.array([[x[k] for k in ['group','family','core','stage','rep']] for x in rows]);A,Z,LP,C,X,Y,U,P,W=map(np.stack,zip(*(x[1:] for x in out)))
 assert len(set(map(tuple,M)))==7488
 with np.load(O/'WAVES86.npz') as d:
  assert sha(O/'WAVES86.npz')==r['array_sha'] and np.array_equal(M,d['meta'])
  assert np.array_equal(W[:,0],d['count']) and abs(W[:,1]-d['mark']).max()<1e-14 and abs(P-d['prob']).max()<1e-14
 idx={tuple(m):i for i,m in enumerate(M)}
 for o,name in [(O82,'POPULATION82.npz'),(O84,'FRESH84.npz')]:
  with np.load(o/name) as z:
   ii=[idx[tuple(m)] for m in z['meta']];assert abs(X[ii]-z['X']).max()<1e-14
 a80=read(O80/'AUDIT80.json');lookup={r['path']:i for i,r in enumerate(rows)};ii=[lookup[x['path']] for x in a80['raw_receipts']]
 with np.load(O80/'MARKED80.npz') as z:
  assert sha(O80/'MARKED80.npz')==a80['array_sha'] and np.array_equal(A[ii],z['root_actions']) and np.array_equal(Z[ii],z['root_marks']) and abs(LP[ii]-z['root_logp']).max()<1e-12 and abs(X[ii]-z['X']).max()<1e-14 and np.array_equal(Y[ii],z['Y']) and abs(U[ii,1]-z['unseen_mass']).max()<1e-14
 # Classical overlaps only: Hamming distance on action histories vs occupancy affinity. Rootwise pair enumeration.
 pairs=[];summary=[]
 for g in range(3):
  for co in range(5):
   ix=np.flatnonzero((M[:,1]==g)&(M[:,0]==co));roots=np.unique(M[ix,2]);N=225 if g==2 else 81
   for root in roots:
    z=ix[M[ix,2]==root];cnt=np.array([np.bincount(A[i],minlength=N)/128 for i in z]);aff=np.sqrt(cnt)@np.sqrt(cnt).T
    for j,k in combinations(range(len(z)),2):
     same=A[z[j]]==A[z[k]];first=np.flatnonzero(~same);prefix=(int(first[0]) if len(first) else 128)/128
     pairs.append([co,g,int(root),float(1-same.mean()),float(aff[j,k]),prefix])
   summary.append(dict(group=co,family=g,paths=len(ix),roots=len(roots),mean_action=float(-LP[ix].sum(1).mean()),action_range=[float(-LP[ix].sum(1).max()),float(-LP[ix].sum(1).min())],mean_unseen_mass=float(U[ix,1].mean()),mean_nongreedy=float(U[ix,2].mean()),mean_profile=C[ix].mean(0).tolist()))
 # Deterministic representative root at fresh R86 g0/core0, exact score landscape H32.
 i=next(i for i,m in enumerate(M) if tuple(m)==(4,0,0,86,0));f=B/rows[i]['path']
 with gzip.open(f,'rt') as z:d=json.load(z)
 cut=next(c for c in d['trace']['cuts'] if c['k']==32);n=np.array(cut['n']);w=np.array(cut['w']);q=np.divide(w,n,out=np.zeros(81),where=n>0)
 rep=dict(path=rows[i]['path'],moves=d['input']['moves'],Q=q.tolist(),visited=(n>0).tolist(),legal=np.flatnonzero(np.array(cut['prior'])>0).tolist())
 z=np.flatnonzero((M[:,0]==4)&(M[:,1]==0)&(M[:,2]==0));la=LP[z].sum(1);j=np.argsort(-la)[:10];p=np.exp(la-la.max());p/=p.sum()
 dominant=dict(group=4,family=0,core=0,observed_paths=len(z),rank_action=(-la[j]).tolist(),observed_relative_weight=p[j].tolist(),distinct_ordered_histories=len({tuple(A[i]) for i in z}),prefix_actions=(-np.cumsum(LP[z[j[:5]]],1)).tolist())
 np.savez_compressed(S/'PATH90.npz',meta=M,actions=A,marks=Z,logp=LP,profiles=C,X=X,Y=Y,unseen=U,overlap=np.array(pairs))
 dump(S/'PATH90.json',dict(status='PASS_SCOPED',paths=len(rows),backups=len(rows)*128,raw_receipts=rows,array_sha=sha(S/'PATH90.npz'),summary=summary,overlap_pairs=len(pairs),representative=rep,dominant=dominant,max_score=max(x['score_error'] for x in rows),max_probability=max(x['prob_error'] for x in rows),scope='All clean actual histories; ideal categorical action vs empirical endpoint law; classical occupancy overlap not quantum. Root pairs reused, not iid. All historical RNG/downstream audits inherited.'))
 print('PATH COMPLETE',len(rows),len(pairs),flush=True)
if __name__=='__main__':main()
