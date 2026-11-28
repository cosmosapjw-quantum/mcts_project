"""Independent inclusion-rank verification, stronger than single-threshold Betti agreement."""
from base61 import *
G=read(S/'GAUGE88.json');B0=read(S/'BARRIER88.json');T=read(S/'TOPOLOGY88.json')
src={('R78',c['family'],c['core']):(c['vertices'],c['edges'],c['score'],c['L']) for c in G['contexts']}
src.update({('R75',c['family'],c['core']):(c['component'],c['edges'],c['score'],c['L']) for c in B0['contexts']})
def rank(cols):
 p={}
 for v in cols:
  while v:
   k=v.bit_length()-1
   if k not in p:p[k]=v;break
   v^=p[k]
 return len(p)
def comp(v,e):
 p={i:i for i in v}
 def root(i):
  while p[i]!=i:i=p[i]
  return i
 for i,j in e:
  a,b=root(i),root(j);p[a]=b
 return {i:root(i) for i in v}
checks=0
for r in T['contexts']:
 v,e,score,L=src[r['kind'],r['family'],r['core']];V=dict(zip(v,[-x for x in score]));e=[tuple(sorted(x)) for x in e];edge={q:i for i,q in enumerate(e)}
 faces=[(i,i+1,i+L,i+L+1) for i in v if i%L<L-1 and i//L<L-1 and {i,i+1,i+L,i+L+1}<=set(v)]
 def bits(f):
  a,b,c,d=f;return sum(1<<edge[q] for q in [(a,b),(a,c),(b,d),(c,d)])
 levels=sorted(set(V.values()));times=[levels[i] for i in sorted(set(np.linspace(0,len(levels)-1,6,dtype=int)))]
 for i,s in enumerate(times):
  v0={i for i in v if V[i]<=s};e0=[q for q in e if set(q)<=v0];c0=len(set(comp(v0,e0).values()));z1=len(e0)-len(v0)+c0
  mask=sum(1<<edge[q] for q in e0)
  for t in times[i:]:
   vt={i for i in v if V[i]<=t};et=[q for q in e if set(q)<=vt];ct=comp(vt,et)
   ft=[f for f in faces if set(f)<=vt];h0=len({ct[i] for i in v0})
   # B1(K_t) basis: planar square boundaries independent. Image in coordinates absent in K_s.
   rr=rank([bits(f)&~mask for f in ft]);h1=z1-len(ft)+rr
   for kind,expected in [('graph',[h0,z1]),('cubical',[h0,h1])]:
    bar=r['topology'][kind]['bars'];observed=[sum(k==d and b<=s and (a is None or t<a) for k,b,a in bar) for d in (0,1)]
    assert observed==expected,(r['kind'],r['core'],s,t,kind,observed,expected);checks+=1
dump(S/'PERSISTENCE_RANK88.json',dict(status='PASS_SCOPED',contexts=144,rank_checks=checks,scope='Independent H0 inclusion image by components, H1 image by restricted target face-boundary rank on edges absent in source. Up to six exact filtration thresholds/context, all ordered pairs, graph and cubical. Complements threshold Betti and GF2 column reduction; not full pairwise rank grid.'))
print('PERSISTENCE INCLUSION RANK',checks)
