"""Exact GF(2) persistence, independent threshold Betti checks and merge certificates."""
from base61 import *
from collections import deque
from barrier88 import adj_grid,bfs
def persistence(v,edges,V,L,faces=True):
 es=sorted({tuple(sorted(x)) for x in edges});vs=set(v);eset=set(es)
 fs=[(i,i+1,i+L,i+L+1) for i in sorted(v) if i%L<L-1 and i//L<L-1 and {i,i+1,i+L,i+L+1}<=vs] if faces else []
 # cells sorted by (filtration, dimension, vertices), ensuring all boundaries precede cell.
 cells=sorted([(V[i],0,(i,)) for i in v]+[(max(V[i] for i in e),1,e) for e in es]+[(max(V[i] for i in f),2,f) for f in fs])
 idx={(d,x):i for i,(_,d,x) in enumerate(cells)};red={};piv={};zero=[];pairs=[];boundary={}
 for j,(b,d,x) in enumerate(cells):
  if d==0:bd=[]
  elif d==1:bd=[(0,(i,)) for i in x]
  else:
   a,c,f,g=x;bd=[(1,e) for e in [(a,c),(a,f),(c,g),(f,g)]]
  bits=sum(1<<idx[t] for t in bd);assert not bits or bits.bit_length()-1<j
  boundary[j]=bits
  if d==2:
   h=0
   for k in bd:h^=boundary[idx[k]]
   assert h==0
  while bits and (i:=bits.bit_length()-1) in piv:bits^=red[piv[i]]
  if bits:
   i=bits.bit_length()-1;assert i in zero and i not in piv;red[j]=bits;piv[i]=j
   pairs.append([cells[i][1],cells[i][0],b])
  else:zero.append(j)
 for i in zero:
  if i not in piv:pairs.append([cells[i][1],cells[i][0],None])
 # Independent component enumeration, Euler relation, planar H2=0, all distinct thresholds.
 # A finite collection of unit squares in the plane has no nonzero 2-cycle (outermost face boundary).
 curve=[]
 for b in sorted(set(V.values())):
  active={i for i in v if V[i]<=b};ee=[e for e in es if set(e)<=active];ff=[f for f in fs if set(f)<=active];a={i:[] for i in active}
  for i,j in ee:a[i].append(j);a[j].append(i)
  left=set(active);c=0
  while left:
   k=min(left);seen=set(bfs(k,a)[0]);left-=seen;c+=1
  rank=len(ee)-len(active)+c-len(ff);assert rank>=0
  pred=[sum(d==k and a0<=b and (b0 is None or b<b0) for d,a0,b0 in pairs) for k in range(3)]
  assert pred==[c,rank,0],(b,pred,c,rank)
  curve.append([b,c,rank])
 # Plateau extrema: connected equal-valued sets; isolated/flat component not counted.
 a=adj_grid(v,L);left=set(v);strict_min=sum(all(V[i]<V[j] for j in a[i]) and bool(a[i]) for i in v);strict_max=sum(all(V[i]>V[j] for j in a[i]) and bool(a[i]) for i in v)
 extrema=[0,0,0]
 while left:
  i=min(left);p={i};queue=[i]
  for j in queue:
   for k in a[j]:
    if k not in p and V[k]==V[i]:p.add(k);queue.append(k)
  left-=p;outside={k for j in p for k in a[j] if k not in p}
  if not outside:extrema[2]+=1
  elif all(V[k]>V[i] for k in outside):extrema[0]+=1
  elif all(V[k]<V[i] for k in outside):extrema[1]+=1
 pos=[x for x in pairs if x[2] is not None and x[2]>x[1]];crit=sorted({a for d,a,b in pairs if b is None or b>a}|{b for d,a,b in pos})
 return dict(vertices=len(v),edges=len(es),faces=len(fs),final_betti=curve[-1][1:],bars=pairs,positive_bars=len(pos),zero_bars=sum(b is not None and b==a for d,a,b in pairs),finite_mean_lifetime=float(np.mean([b-a for d,a,b in pos])) if pos else None,finite_H0=sum(d==0 for d,a,b in pos),finite_H1=sum(d==1 for d,a,b in pos),critical_thresholds=crit,strict_min=strict_min,strict_max=strict_max,plateau_min=extrema[0],plateau_max=extrema[1],flat_components=extrema[2],curve=curve)
def prefix_graph(events,moves):
 tree={()};edges=set();root=(tuple(sorted(moves[::2])),tuple(sorted(moves[1::2])));q={root};qe=set();births=[];rows=[]
 def state(p):
  a=moves+list(p);return tuple(sorted(a[::2])),tuple(sorted(a[1::2]))
 for t,e in enumerate(events,1):
  seq=tuple(e['actions']);assert len(set(seq))==len(seq) and not set(seq)&set(moves)
  for k in range(1,len(seq)+1):
   u,v=seq[:k-1],seq[:k]
   if (u,v) not in edges:
    assert u in tree and v not in tree;edges.add((u,v));tree.add(v)
   i,j=state(u),state(v);assert i in q and sum(map(len,j))==sum(map(len,i))+1
   if (i,j) not in qe:
    if j in q:births.append(t)
    qe.add((i,j));q.add(j)
  assert len(edges)==len(tree)-1 and len(qe)-len(q)+1==len(births)
  rows.append([t,len(tree),len(edges),len(q),len(qe),len(births)])
 return dict(curve=rows,quotient_cycle_births=births,scope='Observed ordered prefixes and colored-board quotient, undirected H1 only, chronological graph inclusion, all H1 births essential; directed graph acyclic.')
def graph_pair(v,edges,score,L):
 V=dict(zip(v,[-x for x in score]));g=persistence(v,edges,V,L,False);c=persistence(v,edges,V,L,True)
 assert g['final_betti'][0]==c['final_betti'][0] and g['final_betti'][1]-c['final_betti'][1]==c['faces']
 assert [x[:2] for x in g['curve']]==[x[:2] for x in c['curve']]
 return dict(graph=g,cubical=c)
def main():
 g=read(S/'GAUGE88.json');a=read(S/'BARRIER88.json');rows=[]
 for c in g['contexts']:
  p=graph_pair(c['vertices'],c['edges'],c['score'],c['L'])
  rows.append(dict(kind='R78',family=c['family'],core=c['core'],topology=p,prefix=c['prefix']))
 for c in a['contexts']:
  p=graph_pair(c['component'],c['edges'],c['score'],c['L']);V=dict(zip(c['component'],[-s for s in c['score']]));a0=adj_grid(c['component'],c['L'])
  # First connecting sublevel, independent from Kruskal and prior Dijkstra.
  levels=sorted(set(V.values()));lo=0;hi=len(levels)-1
  while lo<hi:
   mid=(lo+hi)//2
   if c['target'] in bfs(c['source'],a0,V,levels[mid])[0]:hi=mid
   else:lo=mid+1
  level=levels[lo];assert level==c['saddle'] and max(0,level-max(V[c['source']],V[c['target']]))==c['height']
  rows.append(dict(kind='R75',family=c['family'],core=c['core'],topology=p,merge_level=level,height=c['height']))
 assert len(rows)==144
 dump(S/'TOPOLOGY88.json',dict(contexts=rows,threshold_checks=sum(len(x['topology'][k]['curve']) for x in rows for k in ('graph','cubical')),scope='GF2 graph and unit-square lower-star filtrations, exact threshold BFS and Euler checks; chronological observed quotient cycles; no continuum Morse or physical topology.'))
 print('TOPOLOGY',len(rows),sum(x['topology']['graph']['final_betti'][1] for x in rows),sum(x['topology']['cubical']['final_betti'][1] for x in rows),flush=True)
if __name__=='__main__':main()
