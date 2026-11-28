"""Independent R75 risk/geometry reduction by Kruskal minimax instead of Dijkstra."""
from base61 import *
import gzip
from collections import deque
O=B/'mqft_restore75_20261127'
def adj_grid(v,L):
 V=set(v);a={i:[] for i in v}
 for i in v:
  for j in (i+1 if i%L<L-1 else -1,i+L if i//L<L-1 else -1):
   if j in V:a[i].append(j);a[j].append(i)
 return {i:sorted(a[i]) for i in v}
def bfs(s,a,V=None,level=math.inf):
 if V is not None and V[s]>level:return {},{}
 d={s:0};p={};q=deque([s])
 while q:
  i=q.popleft()
  for j in a[i]:
   if j not in d and (V is None or V[j]<=level):d[j]=d[i]+1;p[j]=i;q.append(j)
 return d,p
def merge_levels(a,V,s):
 # Kruskal minimum spanning forest. Tree path bottleneck equals minimum of graph path bottlenecks.
 p={i:i for i in a};tree={i:[] for i in a}
 def find(i):
  while p[i]!=i:p[i]=p[p[i]];i=p[i]
  return i
 for h,i,j in sorted((max(V[i],V[j]),i,j) for i in a for j in a[i] if i<j):
  u,v=find(i),find(j)
  if u!=v:p[v]=u;tree[i].append((j,h));tree[j].append((i,h))
 c={s:V[s]};q=[s]
 for i in q:
  for j,h in tree[i]:
   if j not in c:c[j]=max(c[i],h);q.append(j)
 return c
def one(arg):
 r,expected=arg;f=B/r['path'];assert sha(f)==r['sha256']
 with gzip.open(f,'rt') as z:raw=json.load(z)
 j=raw['input'];tr=raw['trace'];e=tr['events'];g=r['family'];L=15 if g==2 else 9
 assert len(e)==128 and j['core']==r['core'] and j['rep']==r['rep'] and j['seed']==r['seed']
 l=e[32]['root_selection']['law'];V={a:-s for a,s in zip(l['actions'],l['scores'])};legal=sorted(V)
 assert set(legal)==set(range(L*L))-set(j['moves'])
 a=adj_grid(legal,L);s=min(legal,key=lambda i:(V[i],i));dist,_=bfs(s,a);c=merge_levels(a,V,s);assert set(c)==set(dist)
 ca=[i for i in dist if dist[i]>=4 and all(V[i]<=V[k] for k in a[i]) and any(V[i]<V[k] for k in a[i])];fallback=not ca
 if not ca:ca=[i for i in dist if dist[i]>=4]
 key=(g,r['core'],r['rep'])
 if not ca:assert expected is None;return dict(key=key,eligible=False,receipt=r),None,None
 assert expected is not None
 t=min(ca,key=lambda i:(-max(0,c[i]-max(V[i],V[s])),V[i],i));h=c[t];height=max(0,h-max(V[s],V[t]));dd,prev=bfs(s,a,V,h)
 assert t in dd and t not in bfs(s,a,V,np.nextafter(h,-np.inf))[0]
 path=[t]
 while path[-1]!=s:path.append(prev[path[-1]])
 path=path[::-1];assert max(V[i] for i in path)==h
 acts=[x['actions'][0] for x in e];p=np.array([x['root_selection']['law']['p'][x['root_selection']['law']['actions'].index(t)] for x in e[32:]])
 where=np.flatnonzero(np.array(acts[32:])==t);tau=int(where[0]+1) if len(where) else 97
 risk=np.arange(1,97)<=tau;event=np.arange(1,97)==tau;hit=tau<=96;jump=None if not hit else abs(acts[31+tau]//L-acts[30+tau]//L)+abs(acts[31+tau]%L-acts[30+tau]%L)!=1
 F=np.stack([event,risk,risk*p,risk*p*(1-p),1-(1-p[0])**np.arange(1,97),p],-1)
 z=dict(key=key,eligible=True,L=L,source=s,target=t,height=height,saddle=h,width=len(path)-1,graph_distance=dist[t],fallback=fallback,endpoint_gap=V[t]-V[s],event=hit,tau=tau,rmst=min(tau,96),first_hit_nonneighbor=jump,p0=float(p[0]),compensator=float(F[:,2].sum()),bracket=float(F[:,3].sum()),frozen_hit=float(F[-1,4]),frozen_RMST=float(np.sum((1-p[0])**np.arange(96))),target_previsits=acts[:32].count(t))
 err=0.
 for k,v in z.items():
  if k in ('key','eligible'):continue
  w=expected[k]
  if isinstance(v,(float,np.float64)):err=max(err,abs(v-w));assert abs(v-w)<1e-10,(key,k,v,w)
  else:assert v==w,(key,k,v,w)
 z['prior_scalar_error']=err
 g0=None
 if r['rep']==0:
  comp=sorted(dist);shift=max(V[s],V[t]);edges=[[i,k] for i in comp for k in a[i] if i<k];plaq=[]
  for i in comp:
   if i%L<L-1 and i//L<L-1 and {i,i+1,i+L,i+L+1}<=set(comp):plaq=[i,i+1,i+L+1,i+L];break
  g0=dict(z,family=g,core=r['core'],rep=0,component=comp,edges=edges,path=path,potential=[V[i]-shift for i in comp],score=[-V[i] for i in comp],plaquette=plaq,source_path=r['path'],source_sha=r['sha256'],moves=j['moves'])
 return z,F,g0
def main():
 r=read(O/'AUDIT75.json');a=read(O/'FIRST_RESULTS75.json');assert sha(O/'FIRST75.npz')==a['array_sha256']==r['array_sha256']
 with np.load(O/'FIRST75.npz') as z:meta=z['meta'];features=z['features']
 expected={(x['family'],x['core'],x['rep']):x for x in a['paths']};lookup={tuple(m):i for i,m in enumerate(meta)};assert len(expected)==len(lookup)==3071
 receipts=r['raw_receipts'];assert len(receipts)==3072 and len({x['path'] for x in receipts})==3072
 args=[(x,expected.get((x['family'],x['core'],x['rep']))) for x in receipts];rows=[];ctx=[];err=0.
 with cf.ProcessPoolExecutor(max_workers=6) as pool:
  for i,(z,F,g) in enumerate(pool.map(one,args,chunksize=8)):
   rows.append(z)
   if F is not None:err=max(err,float(abs(F-features[lookup[tuple(z['key'])]]).max()))
   if g:ctx.append(g)
   if (i+1)%512==0:print('RISK',i+1,flush=True)
 assert err<1e-10 and len(ctx)==96 and sum(x['eligible'] for x in rows)==3071
 oldctx={(x['family'],x['core']):x for x in a['contexts']}
 for c in ctx:
  old=oldctx[c['family'],c['core']]
  for k in ['component','edges','path','potential','score','plaquette']:assert c[k]==old[k],(c['key'],k)
 dump(S/'BARRIER88.json',dict(status='PASS_SCOPED',paths=rows,contexts=ctx,raw_receipts=receipts,feature_error=err,prior_scalar_error=max(x['prior_scalar_error'] for x in rows if x['eligible']),scope='R75 full risk set and minimax recomputed via spanning-forest bottleneck, all prior scalar and feature checks. One nonentry retained. No new native searches.'))
 print('BARRIER',len(rows),len(ctx),err,flush=True)
if __name__=='__main__':main()
