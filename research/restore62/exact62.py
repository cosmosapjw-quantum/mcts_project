"""Exact finite adaptive three-category counterexample and gauge audit."""
from fractions import Fraction as Q
from base61 import *
p=[Q(1,3)]*3
q=[[Q(1,2),Q(1,3),Q(1,6)],[Q(1,6),Q(1,2),Q(1,3)],[Q(1,3),Q(1,6),Q(1,2)]]
rows=[]
for a in range(3):
 for b in range(3):
  C=[int(a==i)+int(b==i) for i in range(3)]
  G=[Q(C[i])-p[i]-q[a][i] for i in range(3)]
  F=[[int(i==j)*(p[i]+q[a][i])-p[i]*p[j]-q[a][i]*q[a][j] for j in range(3)] for i in range(3)]
  rows.append((p[a]*q[a][b],C,G,F))
E=lambda f:sum(w*f(c,g,h) for w,c,g,h in rows)
mean=[E(lambda c,g,h:c[i]) for i in range(3)]
R=[[E(lambda c,g,h:c[i]*g[j]) for j in range(3)] for i in range(3)]
K=[[E(lambda c,g,h:(c[i]-mean[i])*(c[j]-mean[j])) for j in range(3)] for i in range(3)]
I=[[E(lambda c,g,h:h[i][j]) for j in range(3)] for i in range(3)]
assert all(E(lambda c,g,h:g[i])==0 for i in range(3))
assert I==[[E(lambda c,g,h:g[i]*g[j]) for j in range(3)] for i in range(3)]
assert all(sum(x)==0 for a in (R,K,I) for x in a) and all(sum(x)==0 for a in (R,K,I) for x in zip(*a))
assert R[0][1]!=R[1][0] and R!=K and R!=I
# Exact Hessian identity (normalization)
assert all(E(lambda c,g,h:g[i]*g[j]-h[i][j])==0 for i in range(3) for j in range(3))
out=dict(scope='two adaptive choices, explicit three-category cyclic conditional probability, common source field shifts logits; exact Fraction evaluation',p=[str(x) for x in p],q=[[str(x) for x in r] for r in q],R=[[str(x) for x in r] for r in R],K=[[str(x) for x in r] for r in K],I=[[str(x) for x in r] for r in I],antisymmetric01=str(R[0][1]-R[1][0]),gauge=True)
dump(S/'EXACT62.json',out);print(out)
