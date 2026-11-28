"""Reauthenticate and independently decode real R67/R68 sign records; actual outcomes."""
from base61 import *
import gzip
from itertools import product
O67=B/'mqft_restore67_20261126';O68=B/'mqft_restore68_20261126'
MASK=2**64-1;MS=list(product((0,1),repeat=3))
def rand64(seed,i,k):
 a=((seed^0x67f00dca12345678)+0x9e3779b97f4a7c15*(5*i+k+1))%2**64
 for s,b in [(30,0xbf58476d1ce4e5b9),(27,0x94d049bb133111eb)]:a=((a^(a>>s))*b)%2**64
 return (a^(a>>31))//2048/2**53
def faults(rec,ev,arm,seed,correction_field):
 f=np.zeros(8);by=np.zeros((8,4));clean=arm in ('clean','reference');recover=arm.startswith('rep') or arm=='recovery'
 for i,(r,e) in enumerate(zip(rec,ev)):
  assert r['id']==e['id']==i
  v=r['clean'];m=r['mask'];mode=(arm=='rep_common' or arm.endswith('_mix') and rand64(seed,i,0)<.5)
  want=[int(i>0 and not clean and rand64(seed,i,1 if mode else k+1)<.12) for k in range(3)];assert m==want
  b=int(math.copysign(1.,v)<0);rxb=[b^k for k in m];s=(rxb[0]^rxb[1],rxb[1]^rxb[2])
  # explicit syndrome table calculated from all single-bit patterns, not stored corrections.
  c=next(j for j,z in enumerate([(0,0),(1,0),(1,1),(0,1)]) if z==s) if recover else 0
  if c:rxb[c-1]^=1
  residual=rxb[0]^b;deliv=math.copysign(abs(v),-1 if rxb[0] else 1)
  assert list(s)==r['syndrome'] and c==r[correction_field] and deliv==r['delivered']
  if 'residual' in r:assert residual==r['residual']
  if 'value' in e:assert e['value']==deliv
  if recover:assert residual==int(sum(m)>=2)
  if i==0:assert not any(m) and c==0;continue
  err=abs(v-deliv);f+=np.array([1,sum(m),m[0],int(any(s)),residual,err,int(any(s) and not residual),int(sum(m)==3)])
  by[MS.index(tuple(m))]+=np.array([1,residual,err,abs(v)])
 return f,by
def cuts(tr,n,budget,rec):
 es=tr['events'];assert len(es)==budget
 N=np.zeros(n,int);W=np.zeros(n);S2=np.zeros(n);ids=[];err=0.
 by={c['k']:c for c in tr['cuts']}
 for k,e in enumerate(es,1):
  assert e['complete_seq']==k-1
  v=e['root_value'];lv=e['leaf_value'];assert v==(-1)**len(e['actions'])*lv
  if e['terminal']:assert e['eval_id'] is None
  else:
   i=e['eval_id'];assert i>0 and rec[i]['delivered']==lv;ids.append(i)
  a=e['actions'][0];N[a]+=1;W[a]+=v;S2[a]+=v*v
  if k in by:
   c=by[k];assert np.array_equal(N,c['n']);err=max(err,float(abs(W-c['w']).max()),float(abs(S2-c['s2']).max()))
 assert sorted(ids)==list(range(1,len(rec))) and err<1e-9
 return N,W,err
def one67(x):
 j=x['job'];p=O67/'data'/x['tag']/(j['id']+'.json.gz');assert sha(p)==x['sha256'] and p.stat().st_size==x['bytes']
 with gzip.open(p,'rt') as h:d=json.load(h)
 assert d['input']==j;tr=d['trace'];r=tr['fault67'];e=tr['evaluations'];assert len(r)==len(e)==x['evaluations']
 f,m=faults(r,e,j['arm'],j['seed'],'correction');assert abs(f[1:]-x['fault_counts']).max()<1e-9
 N,W,err=cuts(tr,81,j['budget'],r)
 return dict(generation=j['generation'],core=j['core'],rep=j['rep'],arm=j['arm'],budget=j['budget'],records=len(r),fault=f.tolist(),mask=m.tolist(),recommendation=int(N.argmax()),receipt=dict(path=str(p.relative_to(B)),sha256=x['sha256'],bytes=x['bytes']),cut_error=err),N,W
def won(m,L):
 if not m:return False
 a=m[-1];color=(len(m)-1)%2;v=set(m[color::2]);y,x=divmod(a,L)
 for dy,dx in ((1,0),(0,1),(1,1),(1,-1)):
  n=1
  for z in (-1,1):
   j=1
   while 0<=y+z*j*dy<L and 0<=x+z*j*dx<L and (y+z*j*dy)*L+x+z*j*dx in v:n+=1;j+=1
  if n>=5:return True
 return False
def one68(x):
 j=x['job'];p=O68/'data'/(j['id']+'.json.gz');assert sha(p)==x['sha256'] and p.stat().st_size==x['bytes']
 with gzip.open(p,'rt') as h:d=json.load(h)
 assert d['job']==j and d['status']==x['status'] and d['winner']==x['winner']
 m=list(j['opening']);assert len(m)==len(set(m)) and all(not won(m[:i],15) for i in range(1,len(m)+1))
 results=[];term=False
 for row in d['rows']:
  assert not term and row['ply']==len(m) and row['side']==(1 if len(m)%2==0 else -1) and row['action'] not in m and 0<=row['action']<225
  if row['engine']=='native':
   tr=row['search'];arm=row['arm'];assert tr['arm']==arm and tr['moves']==m and tr['action']==row['action']
   sd=int.from_bytes(hashlib.sha256(json.dumps([j['seed'],len(m)],separators=(',',':')).encode()).digest()[:8],'little');assert sd==tr['seed']
   e=tr['evaluations'];f,by=faults(e,e,arm,sd,'corrected');N,W,err=cuts(tr,225,j['budget'],e)
   assert N.argmax()==row['action']
   results.append(dict(arm=arm,ply=len(m),records=len(e),fault=f.tolist(),mask=by.tolist(),cut_error=err,seconds=tr['elapsed_ns']/1e9))
  m.append(row['action']);term=won(m,15) or len(m)==225
 assert m==d['moves'] and len(m)==x['plies'] and len(results)==x['native']
 assert term==(x['status']=='complete')
 if term:assert x['winner']==(1 if len(m)%2==1 else -1) if won(m,15) else x['winner']==0
 else:assert x['winner'] is None
 return dict(id=j['id'],core=j['core'],color=j['color'],match=j['match'],status=x['status'],winner=x['winner'],records=results,receipt=dict(path=str(p.relative_to(B)),sha256=x['sha256'],bytes=x['bytes']))
def main():
 a=read(O67/'AUDIT67.json');reg=read(O67/'REGISTRATION.json')
 assert sha(O67/'NATIVE67.npz')==a['array_sha'] and sha(O67/'REFERENCE67.npz')==a['reference_sha']
 for k,v in reg['plans'].items():assert sha(O67/'plans'/f'{k}.json')==v['sha256']
 with np.load(O67/'NATIVE67.npz') as z:A={k:z[k] for k in z.files}
 with np.load(O67/'REFERENCE67.npz') as z:R={k:z[k] for k in z.files}
 lo={tuple(x):i for i,x in enumerate(A['meta'])};rf={tuple(x):i for i,x in enumerate(R['meta'])};rows=[]
 with cf.ProcessPoolExecutor(max_workers=6) as pool:
  for i,(x,N,W) in enumerate(pool.map(one67,a['jobs'],chunksize=8)):
   g,k,r,arm=x['generation'],x['core'],x['rep'],x['arm'];d=R if arm=='reference' else A;ix=rf[k,r] if arm=='reference' else lo[g,k,r,list(A['arms']).index(arm)]
   assert np.array_equal(N,d['counts'][ix]) and abs(W-d['marks'][ix]).max()<1e-9
   if arm!='reference':assert abs(np.asarray(x['fault'])-A['faults'][ix]).max()<1e-9
   rows.append(x)
   if (i+1)%800==0:print('R67',i+1,flush=True)
 assert len(rows)==3200 and sum(x['records'] for x in rows)==525140 and sum(x['budget'] for x in rows)==524288
 c=read(O68/'CAMPAIGN68.json');a68=read(O68/'AUDIT68.json');old={x['id']:x for x in a68['rows']};games=[]
 with cf.ProcessPoolExecutor(max_workers=6) as pool:
  for x in pool.map(one68,c,chunksize=1):
   prior=old[x['id']]['native'];assert len(prior)==len(x['records'])
   for r,u in zip(x['records'],prior):
    assert r['arm']==u['arm'] and r['ply']==u['ply'] and r['records']==u['evals']
    f=r['fault'];assert np.allclose([f[0],f[1],f[3],f[4],f[5],f[6]],[u[k] for k in ('fault_records','carrier_flips','syndromes','sign_errors','value_abs_error','corrected')],rtol=0,atol=1e-9)
   games.append(x)
 assert len(games)==80 and sum(x['status']=='complete' for x in games)==78 and sum(len(x['records']) for x in games)==853 and sum(r['records'] for x in games for r in x['records'])==53886
 dump(S/'NATIVE89.json',dict(status='PASS_SCOPED',root_records=rows,game_records=games,raw_receipts=[x['receipt'] for x in rows+games],max_cut_error=max([x['cut_error'] for x in rows]+[r['cut_error'] for x in games for r in x['records']]),scope='All R67/R68 actual raw decoded codewords, deterministic noise and marked backup cut consistency, previous array aggregates and actual game legality/winner; prior full root law/evaluator audit inherited.'))
 print('NATIVE',len(rows),len(games),sum(x['records'] for x in rows),sum(r['records'] for x in games for r in x['records']),flush=True)
if __name__=='__main__':main()
