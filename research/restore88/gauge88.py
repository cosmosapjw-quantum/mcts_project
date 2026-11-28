"""Reconstruct actual native graph and complete joint SU(2) holonomy invariants."""
from base61 import *
import gzip
import gauge78 as G
from topology88 import prefix_graph
O=B/'mqft_restore78_20261127';I=G.I;P=np.array([G.X,G.Y,G.Z])
def quats(W):
 a=np.trace(W,axis1=1,axis2=2)/2;b=np.array([np.trace(W@p,axis1=1,axis2=2)/(2j) for p in P]).T
 err=max(float(abs(a.imag).max()),float(abs(b.imag).max()));assert err<1e-10
 a=a.real;b=b.real;assert abs(a*a+(b*b).sum(1)-1).max()<1e-10
 return a,b
def joint(W,Wg):
 a,b=quats(W);c,d=quats(Wg);gram=b@b.T
 err=max(float(abs(a-c).max()),float(abs(gram-d@d.T).max()))
 assert err<1e-10
 # independent rank-aware proper rotation matching via orthogonal Procrustes
 U,s,Vh=np.linalg.svd(b.T@d);T=U@np.diag([1.,1.,np.linalg.det(U@Vh)])@Vh
 e=float(abs(b@T-d).max());assert e<1e-9 and abs(np.linalg.det(T)-1)<1e-10
 rank=int(np.linalg.matrix_rank(b,tol=1e-10))
 i=int(np.argmax(np.linalg.norm(b,axis=1)));j=int(np.argmax(np.linalg.norm(np.cross(b[i],b),axis=1)));k=int(np.argmax(abs(b@np.cross(b[i],b[j]))))
 vol=float(np.linalg.det(b[[i,j,k]]));vg=float(np.linalg.det(d[[i,j,k]]))
 assert abs(vol-vg)<1e-10
 tr=float((np.trace(W[i]@W[j]@W[k])/2).real)
 ref=a[i]*a[j]*a[k]-a[i]*gram[j,k]-a[j]*gram[i,k]-a[k]*gram[i,j]+vol
 assert abs(tr-ref)<1e-10
 # Reflect vector z. Same scalar, Gram and pair commutator traces; opposite oriented triple.
 br=b*np.array([1,1,-1]);Wr=a[:,None,None]*I+1j*np.einsum('ni,ijk->njk',br,P)
 ar,bb=quats(Wr);assert abs(ar-a).max()<1e-10 and abs(bb@bb.T-gram).max()<1e-10
 rr=float((np.trace(Wr[i]@Wr[j]@Wr[k])/2).real)
 assert abs(tr-rr-2*vol)<1e-10
 return dict(rank=rank,loops=len(W),scalar_gram_error=err,proper_rotation_error=e,rotation_det=float(np.linalg.det(T)),triple=[i,j,k],oriented_volume=vol,volume_gauge_error=abs(vol-vg),triple_trace=tr,reflected_trace=rr,reflected_trace_gap=abs(tr-rr),scalar=a.tolist(),vectors=b.tolist(),transformed_vectors=d.tolist())
def one(c):
 f=B/c['source_path'];assert sha(f)==c['source_sha']
 with gzip.open(f,'rt') as h:r=json.load(h)
 e=r['trace']['events'];j=r['input'];l=e[32]['root_selection']['law'];assert j['moves']==c['moves'] and l['actions']==c['vertices'] and l['scores']==c['score'] and l['p']==c['p']
 p=prefix_graph(e,j['moves']);q=p['curve'][-1];assert q[1]==c['tree_vertices'] and q[2]==c['tree_edges'] and q[3]==c['quotient_vertices'] and q[4]==c['quotient_edges'] and q[5]==c['quotient_cycles']
 # independently regenerate legal adjacency in board coordinates.
 v=c['vertices'];vs=set(v);L=c['L'];edges=sorted([i,k] for i in vs for k in (i+1 if i%L<L-1 else -1,i+L if i//L<L-1 else -1) if k in vs)
 assert edges==sorted(c['edges'])
 # R78 declared link law, all fundamental loops and transformed field effects.
 a=G.context(c);idx={k:i for i,k in enumerate(v)};ee=sorted(tuple(sorted((idx[i],idx[j]))) for i,j in edges);_,roots,tree,chords,paths=G.forest(len(v),ee)
 s=np.array(c['score']);u={(i,j):G.su(G.X if abs(v[i]-v[j])==1 else G.Z,.35+.2*np.tanh(s[i]+s[j])) for i,j in ee}
 g=np.array([G.su(G.Y,.8*np.sin(.41*i))@G.su(G.Z,.7*np.cos(.37*i)) for i in range(len(v))]);up=G.transform(u,g)
 joints=[]
 for root in roots:
  ch=[(i,j) for i,j in chords if paths[i][0]==root]
  if not ch:continue
  W=[];Wg=[]
  for i,j in ch:
   path=paths[i]+[j]+paths[j][-2::-1];W.append(G.loop(u,path));Wg.append(G.loop(up,path))
  joints.append(dict(root=root,**joint(np.array(W),np.array(Wg))))
 return dict(c,prefix=p,reconstructed=a,joint=joints)
def main():
 r=read(O/'NATIVE78.json');a=read(O/'GAUGE78.json');assert a['source_sha']==sha(O/'NATIVE78.json')
 # R78 context graph was built from the full event list, not H32; compare final counts.
 with cf.ProcessPoolExecutor(max_workers=4) as pool:out=list(pool.map(one,r['contexts'],chunksize=1))
 assert len(out)==48
 old={(q['family'],q['core']):q for q in a['contexts']};err=0.
 for c in out:
  q=old[c['family'],c['core']];v=c['reconstructed']
  for k in ('policy_stability','policy_gauge_error','fixed_effect_TV','flat_link_policy_TV','policy_initial','policy_final'):
   err=max(err,float(abs(np.array(v[k])-q[k]).max()))
  for k in ('U1','SU2'):
   err=max(err,float(abs(np.array(v[k]['wilson'])-q[k]['wilson']).max()))
 assert err<1e-9
 # Exact three-loop counterexample.
 W=1j*P;Wr=W.copy();Wr[2]*=-1
 qa,b=quats(W);qb,d=quats(Wr);assert np.array_equal(qa,qb) and np.array_equal(b@b.T,d@d.T)
 assert np.trace(W[0]@W[1]@W[2])/2==1 and np.trace(Wr[0]@Wr[1]@Wr[2])/2==-1
 for i in range(3):
  for j in range(i):
   f=lambda A:np.trace(A[i]@A[j]@G.adj(A[i])@G.adj(A[j]))/2
   assert f(W)==f(Wr)==-1
 raw=[dict(path=c['source_path'],sha256=c['source_sha'],role='R78_identity_rep0') for c in out]
 dump(S/'GAUGE88.json',dict(contexts=out,raw_receipts=raw,prior_error=err,toy=dict(triple_trace=[1,-1],pair_commutator_trace=-1),scope='Declared SU2 connection on actual legal graph, joint holonomy classification by scalars/Gram/oriented volume. Native operational gauge not identified.'))
 print('GAUGE',len(out),err,sum(x['rank']==3 for c in out for x in c['joint']),flush=True)
if __name__=='__main__':main()
