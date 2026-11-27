"""Reconstruct R60 actual prefix observables from immutable raw games."""
import json,hashlib,math,collections,concurrent.futures as cf
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;B=S.parent;O=B/'mqft_restore60_20261126'
N=81;L=9;G=np.arange(N).reshape(L,L)
# same transform convention as Rust, old-index -> new-index
PERM=np.array([np.argsort(np.rot90(g,k=-k).ravel()) for g in (G,np.fliplr(G)) for k in range(4)])
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(4<<20),b''):h.update(c)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,d):
 p=Path(p);q=p.with_suffix(p.suffix+'.part');q.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n');q.replace(p)
def entropy(p):p=p[p>0];return float(-p@np.log(p))
def board(moves):
 b=np.zeros(N,np.int8)
 for t,a in enumerate(moves):assert 0<=a<N and b[a]==0;b[a]=1+t%2
 return b
def win(b,a):
 y,x=divmod(a,L);p=b[a]
 for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
  n=1
  for s in (-1,1):
   yy=y+s*dy;xx=x+s*dx
   while 0<=yy<L and 0<=xx<L and b[yy*L+xx]==p:n+=1;yy+=s*dy;xx+=s*dx
  if n>=5:return True
 return False
def orbit(b):return min(b[np.argsort(p)].tobytes() for p in PERM)
FIELDS=['Hvisit','Hprior','KL','E_Q','var','within','between','top','neff','mean_depth','visited_prefix_edges','root_value_initial','terminal_fraction']
def observ(n,w,s2,p,depth,k,visited,initial,terminal):
 n=np.asarray(n);w=np.asarray(w);s2=np.asarray(s2);p=np.asarray(p);pi=n/k;u=n>0
 q=np.divide(w,n,out=np.zeros(N),where=u);m=w.sum()/k
 v=s2.sum()/k-m*m;wi=(s2[u]-w[u]*q[u]).sum()/k;be=(pi*(q-m)**2).sum()
 assert abs(v-wi-be)<1e-9 and min(v,wi,be)>-1e-9 and abs(pi.sum()-1)<1e-12 and np.all(p[u]>0)
 return [entropy(pi),entropy(p),float(pi[u]@np.log(pi[u]/p[u])),-m,max(0.,v),max(0.,wi),max(0.,be),pi.max(),1/(pi@pi),depth/k,visited,initial,terminal/k]
def reduce_one(arg):
 tag,pair,game=arg;p=O/'data'/tag/(game['id']+'.json');assert p.stat().st_size==game['bytes'] and sha(p)==game['sha256']
 d=read(p);assert len(d['rows'])==game['searched'];mvs=d['moves'];r0=len(d['job']['opening']);assert d['terminal'] and len(mvs)==game['plies']
 meta=[];vals=[];orbits=set();roots=[];nb=0
 for i,r in enumerate(d['rows']):
  t=r0+i;b=board(mvs[:t]);assert r['board']==b.tolist() and r['ply']==t and r['side']==(1 if t%2==0 else -1)
  orbits.add(orbit(b).hex());roots.append(r['state_hash']);g=pair[r['model']];p0=np.asarray(r['prior'],float)
  assert abs(p0.sum()-1)<2e-6
  events=r['events'];BUD=len(events);nb+=BUD;assert BUD==d['job']['budget']
  n=np.zeros(N,int);w=np.zeros(N);s2=np.zeros(N);depth=0;visited=set();term=0
  for k,e in enumerate(events,1):
   a=e['actions'];z=e['root_value'];assert a and z==(-1)**len(a)*e['leaf_value']
   n[a[0]]+=1;w[a[0]]+=z;s2[a[0]]+=z*z;depth+=len(a);term+=int(e['terminal']);visited.update(tuple(a[:j]) for j in range(1,len(a)+1))
   if k in (4,8,16,32,64,128):
    vals.append(observ(n,w,s2,p0,depth,k,len(visited),r['evaluations'][0]['value'],term));meta.append((g,t,k,BUD,i))
  assert np.array_equal(n,r['counts']) and np.max(abs(w-np.array(r['w'])))<1e-9 and np.max(abs(s2-np.array(r['s2'])))<1e-9
 assert nb==game['backups'] and roots==game['roots']
 return meta,vals,sorted(orbits),dict(tag=tag,id=game['id'],sha256=game['sha256'],searched=game['searched'],backups=nb)
def reduce():
 a=read(O/'AUDIT60.json');scale=read(O/'SCALE_REGISTRATION.json')
 models={f'self{i}':(i-1,i-1) for i in (1,2,3)}|{f'arena{i}':(i,i-1) for i in (1,2)}|{x['tag']:(x['new'],x['old']) for x in scale['arms']}
 args=[]
 for tag,pair in sorted(models.items()):
  v=read(O/f'{tag.upper()}_VERIFIED.json');assert v['n']==len(v['games'])
  args.extend((tag,pair,q) for q in v['games'])
 X=[];M=[];keys=set();rec=[];tags=sorted(models)
 with cf.ProcessPoolExecutor(max_workers=12) as pool:
  for idx,(m,x,o,r) in enumerate(pool.map(reduce_one,args,chunksize=4)):
   X.extend(x);M.extend([[idx,tags.index(r['tag']),*row] for row in m]);keys.update(o);rec.append(r)
   if (idx+1)%256==0:print('REDUCE',idx+1,flush=True)
 assert len(rec)==1592 and sum(r['searched'] for r in rec)==11956 and sum(r['backups'] for r in rec)==1132064
 X=np.asarray(X,float);M=np.asarray(M,int);assert np.isfinite(X).all()
 np.savez_compressed(S/'R60_PREFIX61.npz',values=X,meta=M,fields=np.array(FIELDS),meta_fields=np.array(['game','tag','generation','ply','cut','budget','row']),tags=np.array(tags))
 dump(S/'R60_REDUCTION61.json',dict(games=rec,fields=FIELDS,meta_fields=['game','tag','generation','ply','cut','budget','row'],tags=tags,n_rows=len(X),n_games=len(rec),searched=11956,backups=1132064,source_audit_sha256=sha(O/'AUDIT60.json'),array_sha256=sha(S/'R60_PREFIX61.npz'),orbit_count=len(keys)))
 (S/'R60_ORBITS.txt').write_text('\n'.join(sorted(keys))+'\n')
 print('REDUCTION_COMPLETE',len(X),len(keys),flush=True)
if __name__=='__main__':reduce()
