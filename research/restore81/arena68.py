"""Bounded 15x15 complete-game arena, using pinned pretrained MixNet and separate Rapfi opponent."""
from base61 import *
import sys,os,subprocess,gzip,time,selectors
P=B/'mqft_restore39_20261125/repo';sys.path.insert(0,str(P/'python'))
from mqft.rapfi import Rapfi,verify,parse,protocol
L=15;N=225
def board(m):
 b=np.zeros(N,np.int8)
 for i,a in enumerate(m):
  assert 0<=a<N and b[a]==0;b[a]=i%2+1
 return b
def won(b,a):
 y,x=divmod(a,L);v=b[a]
 for dy,dx in ((1,0),(0,1),(1,1),(1,-1)):
  k=1
  for sign in (-1,1):
   yy=y+sign*dy;xx=x+sign*dx
   while 0<=yy<L and 0<=xx<L and b[yy*L+xx]==v:k+=1;yy+=sign*dy;xx+=sign*dx
  if k>=5:return True
 return False
def seed(*x):return int.from_bytes(hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).digest()[:8],'little')
def orbit15(b):
 a=b.reshape(15,15);return min(x.tobytes() for i in range(4) for x in (np.rot90(a,i),np.fliplr(np.rot90(a,i)))).hex()
def plan():
 r=read(B/'mqft_restore67_20261126/REGISTRATION.json');rng=np.random.default_rng(680001);cores=[];seen=set()
 pool=[y*15+x for y in range(4,11) for x in range(4,11)]
 for i in range(8):
  while True:
   m=list(map(int,rng.choice(pool,4+2*(i%3),replace=False)));b=board(m)
   if any(won(board(m[:j+1]),a) for j,a in enumerate(m)):continue
   key=orbit15(b)
   if key in seen:continue
   seen.add(key);break
  cores.append(dict(id=i,moves=m,orbit=key,seed=seed(680001,i)))
 jobs=[]
 for c in cores:
  for match,left,right in [('ext_clean','clean','rapfi'),('ext_single','single','rapfi'),('ext_recovery','recovery','rapfi'),('internal_recovery_single','recovery','single'),('internal_recovery_clean','recovery','clean')]:
   for color in (1,-1):
    jobs.append(dict(id=f'{match}_c{c["id"]:02d}_s{color+1}',core=c['id'],opening=c['moves'],seed=c['seed'],match=match,focal=left,opponent=right,color=color,budget=64,external_ms=50))
 reg=dict(parent='f52b593f6a3ed5682b3b583b927ac865ea594d0e',binary_sha=sha(S/'mqft_train'),prior_registration_sha=sha(B/'mqft_restore67_20261126/REGISTRATION.json'),external=verify(P/'.external/rapfi'),prior_9x9_weights=r['weights'],cores=cores,jobs=jobs,workers=8,budget=64,external_ms=50,game_wall_cap_seconds=240,rule='freestyle15',root_law='iid wave4 beta=10/3',evaluator='pinned direct MixNet policy and WDL (not lookahead v1)',seed_contract='SHA256(680001,core), then SHA256(game seed,absolute ply); same ordinal across conditions',scope='80 complete-game attempts, 8 opening blocks, two colors; independent external search but shared pretrained NNUE; no equal-time claim')
 dump(S/'REGISTRATION.json',reg);print('REGISTER',len(jobs),sha(S/'REGISTRATION.json'),flush=True)
class Native:
 def __init__(self,log):
  env=os.environ.copy();env.update(MQFT_RAPFI_DIR=str(P/'.external/rapfi'),MQFT_RAPFI_WORKERS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
  self.err=open(log,'wb');self.p=subprocess.Popen([str(S/'mqft_phase81'),'arena68'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,env=env,bufsize=0);self.buf=b''
 def query(self,m,sd,arm,B,phase81=None,pair81=None):
  req=dict(moves=m,seed=sd,arm=arm,budget=B);req.update(phase81=phase81,pair81=pair81);self.p.stdin.write((json.dumps(req)+'\n').encode());self.p.stdin.flush()
  deadline=time.monotonic()+30
  with selectors.DefaultSelector() as sel:
   sel.register(self.p.stdout,selectors.EVENT_READ)
   while b'\n' not in self.buf:
    left=deadline-time.monotonic()
    if left<=0 or not sel.select(left):raise TimeoutError('native response timeout')
    q=os.read(self.p.stdout.fileno(),65536)
    if not q:raise RuntimeError('native process exited '+str(self.p.poll()))
    self.buf+=q
    if len(self.buf)>8<<20:raise RuntimeError('native response limit')
  line,self.buf=self.buf.split(b'\n',1);q=json.loads(line);assert q['ok'],q
  r=q['result'];assert all(r[k]==v for k,v in req.items());return r
 def close(self):
  if self.p.poll() is None:
   self.p.stdin.close()
   try:self.p.wait(timeout=5)
   except subprocess.TimeoutExpired:self.p.kill();self.p.wait()
  self.err.close()
  if self.p.stdout:self.p.stdout.close()
def one(j):
 p=S/'data'/(j['id']+'.json.gz');assert not p.exists()
 t=time.monotonic();m=j['opening'].copy();rows=[];status='failed';winner=None;error=None
 n=Native(S/'logs'/(j['id']+'.stderr.log'));ext=Rapfi(P/'.external/rapfi',timeout_ms=j['external_ms'],threads=1,wall_timeout=15) if j['opponent']=='rapfi' else None
 try:
  b=board(m)
  for _ in range(len(m),225):
   if time.monotonic()-t>240:raise TimeoutError('game wall cap')
   side=1 if len(m)%2==0 else -1;arm=j['focal'] if side==j['color'] else j['opponent'];sd=seed(j['seed'],len(m))
   if arm=='rapfi':
    st=time.monotonic();r=ext.best(m);a=r['action'];row=dict(engine='rapfi',ply=len(m),side=side,arm=arm,action=a,external=r,elapsed=time.monotonic()-st)
   else:
    r=n.query(m,sd,arm,j['budget']);a=r['action'];row=dict(engine='native',ply=len(m),side=side,arm=arm,action=a,search=r)
   assert b[a]==0;rows.append(row);m.append(a);b[a]=1 if side==1 else 2
   if won(b,a):winner=side;status='complete';break
   if len(m)==225:winner=0;status='complete';break
 except Exception as e:error=repr(e)
 finally:n.close()
 raw=dict(schema='mqft.arena68.game.v1',job=j,status=status,error=error,winner=winner,moves=m,rows=rows,elapsed=time.monotonic()-t)
 tmp=p.with_suffix('.part')
 with gzip.open(tmp,'wt',compresslevel=1) as f:json.dump(raw,f,separators=(',',':'),allow_nan=False)
 tmp.replace(p)
 return dict(id=j['id'],job=j,status=status,error=error,winner=winner,plies=len(m),native=sum(x['engine']=='native' for x in rows),external=sum(x['engine']=='rapfi' for x in rows),elapsed=raw['elapsed'],sha256=sha(p),bytes=p.stat().st_size)
def main():
 if '--plan' in sys.argv:plan();return
 if '--smoke' in sys.argv:
  n=Native(S/'logs/smoke.stderr.log')
  try:
   t=time.monotonic();r=n.query([112,113,97,98],123,'recovery',64);print('SMOKE',r['action'],len(r['events']),len(r['evaluations']),time.monotonic()-t,flush=True);dump(S/'SMOKE68.json',r)
  finally:n.close()
  return
 reg=read(S/'REGISTRATION.json');assert sha(S/'mqft_train')==reg['binary_sha'] and verify(P/'.external/rapfi')==reg['external']
 out=[]
 with cf.ProcessPoolExecutor(max_workers=reg['workers']) as pool:
  fs={pool.submit(one,j):j for j in reg['jobs']}
  for f in cf.as_completed(fs):
   x=f.result();out.append(x);print('GAME',len(out),x['id'],x['status'],x['winner'],x['plies'],round(x['elapsed'],2),x['error'],flush=True)
   dump(S/'CAMPAIGN68_partial.json',sorted(out,key=lambda x:x['id']))
 assert len(out)==len(reg['jobs'])
 dump(S/'CAMPAIGN68.json',sorted(out,key=lambda x:x['id']))
 print('CAMPAIGN COMPLETE',len(out),sum(x['status']=='complete' for x in out),flush=True)
if __name__=='__main__':main()
