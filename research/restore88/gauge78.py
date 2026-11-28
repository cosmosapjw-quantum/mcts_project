"""Actual legal graph, genuine U(1)/SU(2) connections, gauge-covariant policy field update."""
from base61 import *
from collections import deque
I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1.,-1.]).astype(complex)
J=.35;PHI=.65;ALPHA=.25;TIMES=[.5,2.,8.]
def adj(a):return a.conj().swapaxes(-1,-2)
def su(axis,a):return np.cos(a)*I+1j*np.sin(a)*axis
def tv(a,b):return float(.5*np.sum(abs(a-b)))
def forest(n,edges):
 a=[[] for _ in range(n)]
 for i,j in edges:a[i].append(j);a[j].append(i)
 parent={};paths={};tree=[];roots=[]
 for i in range(n):
  if i in parent:continue
  parent[i]=None;paths[i]=[i];roots.append(i);q=deque([i])
  while q:
   u=q.popleft()
   for v in sorted(a[u]):
    if v in parent:continue
    parent[v]=u;paths[v]=paths[u]+[v];tree.append((u,v));q.append(v)
 ts={tuple(sorted(x)) for x in tree};chords=[(i,j) for i,j in edges if (i,j) not in ts]
 assert len(tree)==n-len(roots) and len(chords)==len(edges)-n+len(roots)
 return a,roots,tree,chords,paths
def link(u,i,j):return u[i,j] if i<j else adj(u[j,i])
def transform(u,G):return {(i,j):G[i]@v@adj(G[j]) for (i,j),v in u.items()}
def loop(u,path):
 q=np.eye(next(iter(u.values())).shape[-1],dtype=complex)
 for i,j in zip(path,path[1:]):q=q@link(u,i,j)
 return q
def audit_links(u,n,roots,tree,chords,paths,G):
 dim=next(iter(u.values())).shape[-1];I0=np.eye(dim);err=max(float(abs(adj(v)@v-I0).max()) for v in u.values())
 up=transform(u,G);fix=np.tile(I0.astype(complex),(n,1,1))
 for i,j in tree:fix[j]=fix[i]@link(u,i,j)
 uf=transform(u,fix);te=max((float(abs(link(uf,i,j)-I0).max()) for i,j in tree),default=0.)
 W=[];mat=[];e=0.;ef=0.
 for i,j in chords:
  path=paths[i]+[j]+paths[j][-2::-1];root=path[0];assert path[-1]==root
  w=loop(u,path);w1=loop(up,path);w2=loop(uf,path);mat.append(w);W.append(np.trace(w)/dim)
  e=max(e,float(abs(w1-G[root]@w@adj(G[root])).max()));ef=max(ef,float(abs(w2-w).max()),float(abs(link(uf,i,j)-w).max()))
 assert max(err,te,e,ef)<1e-11
 return dict(unitarity=err,tree_fix=te,loop_covariance=e,fixed_loop_error=ef,wilson=[[float(z.real),float(z.imag)] for z in W]),mat
def hamiltonian(n,edges,u,score):
 H=np.diag(-score).astype(complex)
 for i,j in edges:
  H[i,i]+=J;H[j,j]+=J;H[i,j]-=J*u[i,j][0,0];H[j,i]-=J*u[i,j][0,0].conj()
 assert abs(H-adj(H)).max()<1e-12;return H
def evolve(H,s):
 val,V=np.linalg.eigh(H);a=V[s].conj()
 return np.stack([abs(V@(a*np.exp(-1j*t*val)))**2 for t in TIMES])
def covariant_update(rho,u,a):
 v=rho.copy()
 for i,js in enumerate(a):
  if js:v[i]=(1-ALPHA)*rho[i]+ALPHA*sum(link(u,i,j)@rho[j]@adj(link(u,i,j)) for j in js)/len(js)
 return v
def readout(r,E,p):
 v=np.einsum('nij,nji->n',E,r).real*p;assert np.all(v>0)
 return v/v.sum()
def context(g):
 v=g['vertices'];idx={a:i for i,a in enumerate(v)};n=len(v);edges=sorted(tuple(sorted((idx[i],idx[j]))) for i,j in g['edges']);a,roots,tree,chords,paths=forest(n,edges)
 score=np.array(g['score']);p=np.array(g['p']);p/=p.sum();s=idx[g['source']];assert n>1 and len(edges)>0
 # exact score-offset gauge on categorical law, not a local U(1) link phase.
 soft=lambda x:np.exp(x-x.max())/np.exp(x-x.max()).sum()
 scalarerr=max(float(abs(soft((10/3)*(score+k))-p).max()) for k in [-2.,0.,3.])
 # U(1) flat gradient and single fundamental chord flux.
 theta=.37*np.sin(np.arange(n)*.73);G1=np.exp(1j*.83*np.cos(np.arange(n)*.61))[:,None,None]
 flat={(i,j):np.array([[np.exp(1j*(theta[i]-theta[j]))]]) for i,j in edges};u={k:x.copy() for k,x in flat.items()}
 if chords:u[chords[0]]*=np.exp(1j*PHI)
 aa,wa=audit_links(u,n,roots,tree,chords,paths,G1);af,wf=audit_links(flat,n,roots,tree,chords,paths,G1)
 assert all(abs(w[0,0]-1)<1e-10 for w in wf)
 if chords:assert abs(wa[0][0,0]-np.exp(1j*PHI))<1e-10 and all(abs(w[0,0]-1)<1e-10 for w in wa[1:])
 H=hamiltonian(n,edges,u,score);H0=hamiltonian(n,edges,flat,score);Hg=hamiltonian(n,edges,transform(u,G1),score)
 hcov=float(abs(Hg-G1[:,0,0,None]*H*G1[:,0,0].conj()[None,:]).max());d=evolve(H,s);d0=evolve(H0,s);dg=evolve(Hg,s)
 de=float(abs(d-dg).max());assert max(de,hcov,float(abs(d.sum(1)-1).max()),scalarerr)<1e-10
 se=float(abs(np.linalg.eigvalsh(H)-np.linalg.eigvalsh(Hg)).max());assert se<1e-10
 # SU(2) link angles derived from actual frozen score, but a declared model choice.
 u2={(i,j):su(X if abs(v[i]-v[j])==1 else Z,.35+.2*np.tanh(score[i]+score[j])) for i,j in edges}
 G2=np.array([su(Y,.8*np.sin(.41*i))@su(Z,.7*np.cos(.37*i)) for i in range(n)])
 ab,wb=audit_links(u2,n,roots,tree,chords,paths,G2)
 det=max(float(abs(np.linalg.det(w)-1)) for w in u2.values());assert det<1e-11
 rho=(I+.4*np.tanh(np.log(p*n))[:,None,None]*X+.4*np.tanh(score)[:,None,None]*Z)/2
 E=np.tile((I+.7*X)/2,(n,1,1));r0=readout(rho,E,p);rr=rho.copy();rt=G2@rho@adj(G2);Et=G2@E@adj(G2);ut=transform(u2,G2);rf=rho.copy();Id={k:I for k in u2}
 errors=[];st=[];fixed=[];changes=[];ener=[]
 def cost(r,u):return float(sum(np.linalg.norm(r[i]-link(u,i,j)@r[j]@adj(link(u,i,j)))**2 for i,j in edges)/2)
 for step in range(5):
  read=readout(rr,E,p);rd=readout(rt,Et,p);rfix=readout(rt,E,p);flatpol=readout(rf,E,p)
  err=max(float(abs(rt-G2@rr@adj(G2)).max()),float(abs(read-rd).max()),abs(cost(rr,u2)-cost(rt,ut)))
  errors.append(err);st.append(1-tv(read,r0));fixed.append(tv(read,rfix));changes.append(tv(read,flatpol));ener.append(cost(rr,u2))
  assert np.min(np.linalg.eigvalsh(rr))>=-1e-12 and abs(np.trace(rr,axis1=1,axis2=2)-1).max()<1e-12 and err<1e-10
  if step<4:rr=covariant_update(rr,u2,a);rt=covariant_update(rt,ut,a);rf=covariant_update(rf,Id,a)
 comm=[]
 # Same base component only, two first noncommuting cycles.
 for i in range(min(len(chords),16)):
  for j in range(i+1,min(len(chords),16)):
   if paths[chords[i][0]][0]==paths[chords[j][0]][0]:
    w=wb[i]@wb[j]@adj(wb[i])@adj(wb[j]);comm.append(float((np.trace(w)/2).real))
 L=g['L'];V=set(v);squares=sum(i%L<L-1 and i//L<L-1 and {i,i+1,i+L,i+L+1}<=V for i in v)
 return dict(family=g['family'],core=g['core'],nodes=n,edges=len(edges),components=len(roots),cycles=len(chords),plaquettes=squares,tree_vertices=g['tree_vertices'],tree_edges=g['tree_edges'],tree_cycles=g['tree_cycles'],quotient_vertices=g['quotient_vertices'],quotient_edges=g['quotient_edges'],quotient_cycles=g['quotient_cycles'],scalar_shift_error=scalarerr,U1=aa,U1_flat=af,U1_H_covariance=hcov,U1_spectrum_error=se,U1_evolution_error=de,U1_flux_TV=[tv(x,y) for x,y in zip(d,d0)],SU2=ab,SU2_det_error=det,SU2_commutator_trace_min=min(comm,default=None),policy_gauge_error=errors,policy_stability=st,fixed_effect_TV=fixed,flat_link_policy_TV=changes,link_cost=ener,policy_initial=r0.tolist(),policy_final=readout(rr,E,p).tolist(),forest_roots=roots,tree_edges_index=tree,chords_index=chords)
def toy():
 # Explicit original failure witnesses, without claiming original code executed.
 A=I+.01*Z;non=float(np.linalg.norm(adj(A)@A-I))
 G0=su(Z,.3);G1=su(Z,.9);G2=su(Z,-.2)
 # triangle 0->1->2->0 but missing 1->2 identity not transformed.
 w=G0@adj(G1)@I@G2@adj(G0);missing=float(np.trace(w).real/2)
 w_good=G0@adj(G1)@G1@adj(G2)@G2@adj(G0)
 assert non>0 and abs(missing-1)>1e-3 and abs(w_good-I).max()<1e-14
 # Hermitian part of commutator of Hermitian matrices vanishes.
 c=X@Z-Z@X;assert abs((c+adj(c))/2).max()==0 and abs(c).max()>0
 # Open link commutator with local field not a tensor under endpoint gauges.
 v=su(X,.4);r=Z;old=v@r-r@v;vp=G0@v@adj(G1);rp=G0@r@adj(G0);new=vp@rp-rp@vp;cov=float(abs(new-G0@old@adj(G0)).max())
 assert cov>1e-3
 return dict(nonunitary_defect=non,missing_link_fake_wilson=missing,valid_link_wilson=float(np.trace(w_good).real/2),open_commutator_covariance_defect=cov,hermitian_commutator_projection=0.,U1_complex_W=[float(np.cos(.65)),float(np.sin(.65))])
def main():
 r=read(S/'NATIVE78.json');assert len(r['contexts'])==48
 with cf.ProcessPoolExecutor(max_workers=8) as pool:out=list(pool.map(context,r['contexts'],chunksize=1))
 dump(S/'GAUGE78.json',dict(scope='Native legal graph topology, declared U1/SU2 unitary connection, model policy-field update and transported readout; native phase not measured',contexts=out,source_sha=sha(S/'NATIVE78.json'),toy=toy(),J=J,phi=PHI,alpha=ALPHA,times=TIMES))
 print('GAUGE',len(out),max(max(r['policy_gauge_error']) for r in out),max(r['U1_evolution_error'] for r in out),flush=True)
if __name__=='__main__':main()
