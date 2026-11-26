"""Actual full-support native first-passage from frozen quiescent H32 geometry."""
from base61 import *
import gzip,heapq
from collections import deque
from scipy.stats import t as student, spearmanr
from audit_native74 import one as audit9
from audit_rapfi74 import audit_search
T=96
def neighbors(a,L,legal):
 y,x=divmod(int(a),L)
 return sorted(i for i in [a-L if y>0 else -1,a+L if y<L-1 else -1,a-1 if x>0 else -1,a+1 if x<L-1 else -1] if i in legal)
def search(s,adj,V,cap=math.inf):
 if V[s]>cap:return {},{}
 dist={s:0};prev={};q=deque([s])
 while q:
  a=q.popleft()
  for b in adj[a]:
   if b not in dist and V[b]<=cap:dist[b]=dist[a]+1;prev[b]=a;q.append(b)
 return dist,prev
def landscape(legal,score,L):
 legal=list(map(int,legal));V={a:-float(v) for a,v in zip(legal,score)};ls=set(legal);adj={a:neighbors(a,L,ls) for a in legal}
 s=min(legal,key=lambda a:(V[a],a));dist,_=search(s,adj,V)
 cand=[a for a,d in dist.items() if d>=4 and all(V[a]<=V[b] for b in adj[a]) and any(V[a]<V[b] for b in adj[a])]
 fallback=not cand
 if fallback:cand=[a for a,d in dist.items() if d>=4]
 if not cand:return None
 costs={s:V[s]};queue=[(V[s],s)]
 while queue:
  c,a=heapq.heappop(queue)
  if c!=costs[a]:continue
  for b in adj[a]:
   cc=max(c,V[b])
   if cc<costs.get(b,math.inf):costs[b]=cc;heapq.heappush(queue,(cc,b))
 def height(a):return max(0.,costs[a]-max(V[s],V[a]))
 a=min(cand,key=lambda a:(-height(a),V[a],a))
 cap=costs[a];dd,prev=search(s,adj,V,cap);assert a in dd and a not in search(s,adj,V,np.nextafter(cap,-np.inf))[0]
 path=[a]
 while path[-1]!=s:path.append(prev[path[-1]])
 path=path[::-1];assert len(path)-1==dd[a] and max(V[b] for b in path)==cap
 comp=sorted(dist);edges=[[b,c] for b in comp for c in adj[b] if b<c];shift=max(V[s],V[a])
 plaq=[]
 for b in comp:
  y,x=divmod(b,L)
  if y<L-1 and x<L-1 and all(c in dist for c in [b,b+1,b+L,b+L+1]):plaq=[b,b+1,b+L+1,b+L];break
 return dict(L=L,source=s,target=a,height=height(a),saddle=cap,endpoint_gap=V[a]-V[s],width=len(path)-1,graph_distance=dist[a],fallback=fallback,path=path,component=comp,edges=edges,potential=[V[b]-shift for b in comp],score=[-V[b] for b in comp],plaquette=plaq)
def one(arg):
 tag,j=arg;f=S/'data'/tag/(j['id']+'.json.gz')
 with gzip.open(f,'rt') as fp:d=json.load(fp)
 assert d['input']==j;tr=d['trace']
 if tag=='rapfi':
  a=audit_search(tr,j['moves'],'clean',j['seed'],128,j['sources'][0],0.);g=2;L=15
 else:a=audit9((tag,j));g=int(j['generation']==3);L=9
 es=tr['events'];l=es[32]['root_selection']['law'];geom=landscape(l['actions'],l['scores'],L)
 receipt=dict(path=str(f.relative_to(B)),sha256=sha(f),bytes=f.stat().st_size,seed=j['seed'],family=g,core=j['core'],rep=j['rep'],max_probability=a['max_probability'],max_score=a['max_score'],max_log=a['max_log'])
 if geom is None:return [g,j['core'],j['rep']],None,None,receipt
 target=geom['target'];probs=[];act=[]
 for e in es:
  q=e['root_selection']['law'];ii=q['actions'].index(target);probs.append(q['p'][ii]);act.append(e['actions'][0])
 p=np.array(probs[32:]);hit=np.asarray(act[32:])==target;ix=np.flatnonzero(hit)
 tt=int(ix[0]+1) if len(ix) else T+1
 at=np.arange(1,T+1);risk=at<=tt;ev=at==tt;comp=risk*p;br=risk*p*(1-p)
 frozen=1-(1-p[0])**at
 jump=None if tt>T else (abs(act[31+tt]//L-act[30+tt]//L)+abs(act[31+tt]%L-act[30+tt]%L)!=1)
 geom.update(family=g,core=j['core'],rep=j['rep'],event=tt<=T,tau=tt,first_hit_nonneighbor=jump,p0=float(p[0]),compensator=float(comp.sum()),bracket=float(br.sum()),frozen_hit=float(frozen[-1]),frozen_RMST=float(np.sum((1-p[0])**np.arange(T))),rmst=min(tt,T),source_path=receipt['path'],source_sha=receipt['sha256'],target_previsits=sum(a==target for a in act[:32]))
 rows=np.stack([ev.astype(float),risk.astype(float),comp,br,frozen,p],axis=-1)
 return [g,j['core'],j['rep']],geom,rows,receipt
def ci(x):
 x=np.asarray(x);m=x.mean(0);s=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return [m.tolist(),(m-s).tolist(),(m+s).tolist()]
def summarize(meta,rows,G):
 out=[]
 for g in range(3):
  select=meta[:,0]==g;ix=np.flatnonzero(select);f=[G[i] for i in ix];m=meta[ix]
  assert len(set(m[:,1]))==32
  d=rows[ix];hits=d[:,:,0].cumsum(1);c=d[:,:,2].cumsum(1);br=d[:,:,3].sum(1);frozen=d[:,:,4];Y=d[:,:,1]
  cs=[];S0=[];B0=[];diff=[];RM=[];H=[];nativejump=[]
  for core in range(32):
   z=m[:,1]==core;cs.append((hits[z]-c[z]).mean(0));S0.append(hits[z].mean(0));B0.append(frozen[z].mean(0));diff.append((hits[z]-frozen[z]).mean(0));RM.append([Y[z].sum(1).mean(),np.mean([v['frozen_RMST'] for v,keep in zip(f,z) if keep])]);H.append(float(br[z].mean()))
  a=np.array([v['height'] for v in f]);b=np.array([v['tau'] for v in f]);hit=np.array([v['event'] for v in f]);j=[v['first_hit_nonneighbor'] for v in f if v['event']]
  strat=[]
  for name,z in [('positive_barrier',a>1e-10),('zero_barrier',a<=1e-10)]:
   if z.any():strat.append(dict(name=name,paths=int(z.sum()),hit_fraction=float(hit[z].mean()),mean_frozen=float(frozen[z,-1].mean()),mean_RMST=float(Y[z].sum(1).mean()),mean_height=float(a[z].mean()),mean_width=float(np.mean([v['width'] for v,k in zip(f,z) if k]))))
  # Root-level mean association; no iid trajectory inference for Spearman.
  rbar=np.array([[a[m[:,1]==k].mean(),hit[m[:,1]==k].mean()] for k in range(32)])
  cor=spearmanr(rbar[:,0],rbar[:,1]).statistic
  out.append(dict(family=['g0','g3','Rapfi'][g],paths=len(ix),roots=32,positive_barrier=int((a>1e-10).sum()),fallback=sum(v['fallback'] for v in f),events=int(hit.sum()),censored=int((~hit).sum()),hit_rate=ci(np.array(S0)[:,-1]),frozen_hit_rate=ci(np.array(B0)[:,-1]),hit_minus_frozen=ci(np.array(diff)[:,-1]),RMST=ci(RM),cum_hit=ci(S0),cum_compensator=ci(np.array(S0)-np.array(cs)),compensator_residual=ci(cs),bracket_mean=float(np.mean(H)),nonneighbor_hits=sum(j),nonneighbor_fraction=float(np.mean(j)) if j else None,mean_height=float(a.mean()),height_max=float(a.max()),mean_width=float(np.mean([v['width'] for v in f])),root_mean_height_hit_spearman=float(cor) if np.isfinite(cor) else None,strata=strat))
 return out
def main():
 R=read(S/'REGISTRATION.json');jobs=[]
 for tag in ('schedule0','schedule3','rapfi'):
  p=S/'plans'/f'{tag}.json';assert sha(p)==R['plans'][tag]['sha256'];q=read(p)
  assert len(q)==1024 and {x.name for x in (S/'data'/tag).glob('*.gz')}=={j['id']+'.json.gz' for j in q}
  jobs.extend((tag,j) for j in q)
 M=[];G=[];F=[];aud=[];ineligible=[]
 with cf.ProcessPoolExecutor(max_workers=12) as pool:
  for i,(m,g,f,a) in enumerate(pool.map(one,jobs,chunksize=8)):
   aud.append(a)
   if g is None:ineligible.append(m)
   else:M.append(m);G.append(g);F.append(f)
   if (i+1)%512==0:print('AUDIT',i+1,flush=True)
 M=np.array(M);F=np.array(F);assert len(aud)==3072 and len(M)+len(ineligible)==3072
 np.savez_compressed(S/'FIRST75.npz',meta=M,features=F)
 contexts=[g for g in G if g['rep']==0]
 result=dict(scope='first subsequent selection after H32; geometrical minimax barrier not native transition constraint; eligible all paths summarized; no causal quantum claim',results=summarize(M,F,G),eligible=len(G),ineligible=ineligible,contexts=contexts,paths=[{k:v for k,v in g.items() if k not in ['component','edges','potential','score','path','plaquette']} for g in G],array_sha256=sha(S/'FIRST75.npz'))
 dump(S/'FIRST_RESULTS75.json',result)
 audit=dict(paths=len(aud),backups=128*len(aud),raw_bytes=sum(x['bytes'] for x in aud),max_probability=max(x['max_probability'] for x in aud),max_score=max(x['max_score'] for x in aud),max_log=max(x['max_log'] for x in aud),eligible=len(G),contexts=len(contexts),minimax_certificate_count=len(G),raw_receipts=aud,array_sha256=result['array_sha256'])
 dump(S/'AUDIT75.json',audit)
 for r in result['results']:print(r['family'],'event',r['events'],'censored',r['censored'],'barrier',r['positive_barrier'],'jump',r['nonneighbor_fraction'],'hit',r['hit_rate'],'frozen',r['frozen_hit_rate'],'residual',np.asarray(r['compensator_residual'])[:,-1],flush=True)
 print('AUDIT',{k:v for k,v in audit.items() if k!='raw_receipts'},flush=True)
if __name__=='__main__':main()
