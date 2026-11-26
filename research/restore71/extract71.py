"""Read every registered raw search, verify grouped law, count and signed mark.
Full independent native replay on one seed per root/arm/family."""
from base61 import *
import gzip,sys
from scipy.special import gammaln
from audit65 import audit_one
from audit68 import audit_search
G=np.array([(i,j,4-i-j) for i in range(5) for j in range(5-i)],int)
LCO=gammaln(5)-gammaln(G+1).sum(1)
def mass(p):return np.exp(np.log(p)@G.T+LCO)
def one(arg):
 tag,j=arg;f=S/'data'/tag/(j['id']+'.json.gz')
 with gzip.open(f,'rt') as ff:d=json.load(ff)
 assert d['input']==j;tr=d['trace'];es=tr['events'];assert len(es)==128
 a,b=j['sources'];nboard=225 if tag=='rapfi' else 81
 Y=[];P=[];Z=[];F=[];alln=np.zeros(nboard,int);allw=np.zeros(nboard);alls2=np.zeros(nboard);maxerr=0.
 for t in range(32):
  ev=es[4*t:4*t+4];l=ev[0]['root_selection']['law'];acts=l['actions'];p=np.asarray(l['p']);ia,ib=acts.index(a),acts.index(b);p3=np.array([p[ia],p[ib],1-p[ia]-p[ib]])
  assert np.all(p3>0)
  if tag!='rapfi':
   h=l['phase65'];v=h['prepared'];m=h['pair_mass'];u=(1+v[1]*np.sin(.7)+v[2]*np.cos(.7))/2
   p0=np.array([m*h['e0'],m*(1-h['e0']),1-m]);p1=np.array([m*h['e1'],m*(1-h['e1']),1-m])
   err=np.max(abs(u*p0+(1-u)*p1-p3));assert err<2e-10;maxerr=max(maxerr,float(err))
   q=u*mass(p0)+(1-u)*mass(p1)
  else:q=mass(p3)
  assert abs(q.sum()-1)<1e-10 and np.all(q>0)
  c=np.zeros(3,int);z=np.zeros(3)
  for k,e in enumerate(ev):
   law=e['root_selection']['law'];ps=np.asarray(law['p']);act=e['actions'][0];val=e['root_value'];u=law['u'];picked=int(law['picked'])
   assert e['complete_seq']==4*t+k and e['phase']==t+1
   assert abs(ps.sum()-1)<1e-11 and np.all(ps>0) and int(np.searchsorted(np.cumsum(ps),u,side='right'))==picked and law['actions'][picked]==act
   assert abs(np.log(ps[picked])-law['log_p'])<1e-10
   assert abs(val)<=1+1e-7 and abs(val-(-1)**len(e['actions'])*e['leaf_value'])<1e-7
   i=0 if act==a else 1 if act==b else 2;c[i]+=1;z[i]+=val
   alln[act]+=1;allw[act]+=val;alls2[act]+=val*val
  Y.append(c);P.append(p3);Z.append(z);F.append(q)
 cut=tr['cuts'][-1];assert cut['k']==128
 assert np.array_equal(alln,cut['n']) and np.max(abs(allw-cut['w']))<1e-8 and np.max(abs(alls2-cut['s2']))<1e-8
 fam=4 if tag=='rapfi' else 2*(j['generation']==3)+int(j['arm']=='persist_minus')
 full=None
 if j['rep']==0:
  if tag=='rapfi':
   x=audit_search(tr,j['moves'],'clean',j['seed'],128);full={k:x[k] for k in ('max_probability','max_score','max_log')}
  else:
   x=audit_one((tag,j));full={k:x[k] for k in ('max_probability','max_score','max_log','max_state')}
 return [fam,j['core'],j['rep'],len(j['moves'])],Y,P,Z,F,tr['evaluations'][0]['delivered' if tag=='rapfi' else 'value'],dict(path=str(f.relative_to(B)),sha256=sha(f),bytes=f.stat().st_size,seed=j['seed'],id=j['id'],tag=tag,max_group_law_error=maxerr,full_replay=full)
def main():
 r=read(S/'REGISTRATION.json');jobs=[]
 for tag in ('phase0','phase3','rapfi'):
  p=S/'plans'/f'{tag}.json';q=read(p);assert sha(p)==r['plans'][tag]['sha256'] and len(q)==r['plans'][tag]['n']
  if tag!='rapfi':assert {x.name for x in (S/'data'/tag).glob('*.gz')}=={j['id']+'.json.gz' for j in q}
  jobs.extend((tag,j) for j in q)
 A=[[] for _ in range(7)]
 import time
 with cf.ProcessPoolExecutor(max_workers=10) as pool:
  for i,x in enumerate(pool.map(one,jobs[:8192],chunksize=8)):
   for k,y in enumerate(x):A[k].append(y)
   if (i+1)%1024==0:print('EXTRACT',i+1,flush=True)
  deadline=time.monotonic()+180
  while time.monotonic()<deadline:
   f=S/'RAPFI_RECEIPTS71.json'
   if f.exists() and len(read(f))==1024:break
   time.sleep(2)
  assert len(read(S/'RAPFI_RECEIPTS71.json'))==1024
  for i,x in enumerate(pool.map(one,jobs[8192:],chunksize=8)):
   for k,y in enumerate(x):A[k].append(y)
   if (i+1)%256==0:print('EXTRACT RAPFI',i+1,flush=True)
 meta,Y,P,Z,F,V=map(np.asarray,A[:6]);rec=A[6]
 assert len(meta)==9216 and np.all(Y.sum(-1)==4) and np.all(abs(Z)<=Y+1e-7)
 assert len({(x[0],x[1],x[2]) for x in meta})==9216
 for g in range(5):
  for c in range(64):
   mask=(meta[:,0]==g)&(meta[:,1]==c);assert mask.sum()==(16 if g==4 else 32) and np.ptp(V[mask])<1e-7
 np.savez_compressed(S/'JOINT71.npz',meta=meta,counts=Y,prob=P,marks=Z,conditional_pmf=F,initial_value=V,grid=G)
 full=[x['full_replay'] for x in rec if x['full_replay'] is not None]
 result=dict(status='verified_raw_all_and_full_replay_subset',paths=len(rec),backups=len(rec)*128,waves=len(rec)*32,raw_bytes=sum(x['bytes'] for x in rec),full_replay_n=len(full),max_group_law_error=max(x['max_group_law_error'] for x in rec),max_full_probability_error=max(x['max_probability'] for x in full),max_full_state_error=max(x.get('max_state',0) for x in full),array_sha256=sha(S/'JOINT71.npz'),raw_receipts=rec)
 dump(S/'AUDIT71.json',result);print('AUDIT',{k:v for k,v in result.items() if k!='raw_receipts'},flush=True)
if __name__=='__main__':main()
