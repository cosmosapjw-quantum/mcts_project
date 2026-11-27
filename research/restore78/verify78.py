"""Separate coordinate, graph topology, explicit exponential and Bloch rotation verification."""
from base61 import *
import gzip
import networkx as nx
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply
C=read(S/'NATIVE78.json');G=read(S/'GAUGE78.json');A=read(S/'AUDIT78.json');R=read(S/'REGISTRATION.json')
assert G['source_sha']==sha(S/'NATIVE78.json')
def coordinates(L,o):
 out=[]
 for i in range(L*L):
  y,x=divmod(i,L)
  # PERM maps original to image; reflection then clockwise index-grid rotation convention.
  if o>=4:x=L-1-x
  for _ in range(o%4):y,x=x,L-1-y
  out.append(y*L+x)
 return np.array(out)
# For R60 PERM, image grid rot90(k=-k) followed by argsort: forward physical clockwise rotation.
def raw(r):
 p=B/r['path'];assert sha(p)==r['sha256']
 with gzip.open(p,'rt') as f:d=json.load(f)
 tr=d['trace'];g=r['family'];o=r['orientation'];L=15 if g==2 else 9;perm=coordinates(L,o);j=d['input'];assert j['seed']==r['seed'] and j['core']==r['core'] and j['rep']==r['rep']
 m=R['cores'][j['core']]['moves'];ps=coordinates(9,o)[m]
 if L==15:ps=(ps//9+3)*15+(ps%9+3)
 assert j['moves']==ps.tolist()
 cuts=sorted(tr['cuts'],key=lambda v:v['k']);assert [v['k'] for v in cuts]==[4,8,16,32,64,128]
 prior=np.array(cuts[0]['prior'],float)[perm];prior/=prior.sum()
 value=tr['evaluations'][0]['clean' if g==2 else 'value'];n=np.array([np.array(v['n'])[perm]/v['k'] for v in cuts]);w=np.array([np.array(v['w'])[perm]/v['k'] for v in cuts])
 return (g,j['core'],o,j['rep']),prior,value,n,w
def gcheck(pair):
 g,r=pair;assert g['family']==r['family'] and g['core']==r['core']
 p=B/g['source_path'];assert sha(p)==g['source_sha']
 with gzip.open(p,'rt') as f:d=json.load(f)
 tr=d['trace'];es=tr['events'];L=g['L'];m=g['moves'];law=es[32]['root_selection']['law'];assert g['score']==law['scores'] and g['p']==law['p'] and g['vertices']==law['actions'] and d['input']['moves']==m
 # actual prefix tree and quotient via color bit sets, independent edge representation.
 T=nx.Graph();Q=nx.Graph()
 def state(x):
  a=m+list(x);return (sum(1<<i for i in a[::2]),sum(1<<i for i in a[1::2]))
 for e in es:
  x=tuple(e['actions'])
  for k in range(len(x)):
   u,v=x[:k],x[:k+1];T.add_edge(u,v);Q.add_edge(state(u),state(v))
 assert nx.is_tree(T) and nx.is_connected(Q)
 assert (T.number_of_nodes(),T.number_of_edges(),Q.number_of_nodes(),Q.number_of_edges())==(r['tree_vertices'],r['tree_edges'],r['quotient_vertices'],r['quotient_edges'])
 assert len(nx.cycle_basis(Q))==r['quotient_cycles']
 v=g['vertices'];n=len(v);xy=np.array([[i//L,i%L] for i in v]);adj=abs(xy[:,None]-xy[None,:,:]).sum(-1)==1
 graph=nx.from_numpy_array(adj);assert graph.number_of_nodes()==r['nodes'] and graph.number_of_edges()==r['edges'] and nx.number_connected_components(graph)==r['components']
 assert len(nx.cycle_basis(graph))==r['cycles']
 edges=sorted(graph.edges());tree=nx.Graph();tree.add_nodes_from(range(n));tree.add_edges_from(r['tree_edges_index']);assert nx.is_forest(tree) and nx.number_connected_components(tree)==r['components']
 chords=[e for e in edges if not tree.has_edge(*e)];assert [list(x) for x in chords]==r['chords_index']
 p=np.array(g['p']);p/=p.sum();sc=np.array(g['score']);s=v.index(g['source'])
 # Independent U1 magnetic incidence construction and propagation with expm_multiply.
 theta=.37*np.sin(np.arange(n)*.73);bb=np.zeros((n,len(edges)),complex);bf=bb.copy();flux=.65
 for k,(i,j) in enumerate(edges):
  a=theta[i]-theta[j];z=np.exp(1j*(a+(flux if chords and (i,j)==chords[0] else 0.)))
  bb[i,k]=1;bb[j,k]=-z.conj();bf[i,k]=1;bf[j,k]=-np.exp(-1j*a)
 H=.35*bb@bb.conj().T-np.diag(sc);Hf=.35*bf@bf.conj().T-np.diag(sc);e=np.eye(n,dtype=complex)[:,s]
 tvs=[]
 for t in (.5,2.,8.):
  a=abs(expm_multiply(csc_matrix(-1j*t*H),e))**2;b=abs(expm_multiply(csc_matrix(-1j*t*Hf),e))**2;tvs.append(.5*np.sum(abs(a-b)))
 # Independently compute SU2 products and adjoint action on Bloch vectors.
 X=np.array([[0,1],[1,0]],complex);Z=np.diag([1.,-1.]).astype(complex);I=np.eye(2,dtype=complex);U={}
 for i,j in edges:
  ang=.35+.2*np.tanh(sc[i]+sc[j]);axis=X if abs(v[i]-v[j])==1 else Z
  U[i,j]=expm(1j*ang*axis);U[j,i]=U[i,j].conj().T
 w=[]
 for i,j in chords:
  path=nx.shortest_path(tree,j,i);path=[i]+path;q=I.copy()
  for a,b in zip(path,path[1:]):q=q@U[a,b]
  w.append([float(np.trace(q).real/2),float(np.trace(q).imag/2)])
 # In SU2 conjugation by exp(i angle sigma_axis) is rotation by -2 angle.
 z=np.column_stack([.4*np.tanh(np.log(p*n)),np.zeros(n),.4*np.tanh(sc)])
 def prob(z):
  a=p*(1+.7*z[:,0]);return a/a.sum()
 p0=prob(z);st=[1.]
 for _ in range(4):
  zz=z.copy()
  for i in range(n):
   js=list(graph.neighbors(i))
   if not js:continue
   a0=[]
   for j in js:
    a=np.array([1.,0.,0.]) if abs(v[i]-v[j])==1 else np.array([0.,0.,1.])
    q=(-2 if i<j else 2)*(.35+.2*np.tanh(sc[i]+sc[j]));b=z[j]
    a0.append(b*np.cos(q)+np.cross(a,b)*np.sin(q)+a*np.dot(a,b)*(1-np.cos(q)))
   zz[i]=.75*z[i]+.25*np.mean(a0,0)
  z=zz;st.append(1-.5*np.sum(abs(prob(z)-p0)))
 return [float(abs(np.array(tvs)-r['U1_flux_TV']).max()),float(abs(np.array(w)-r['SU2']['wilson']).max()) if w else 0.,float(abs(np.array(st)-r['policy_stability']).max()),float(abs(prob(z)-r['policy_final']).max())]
def main():
 with cf.ProcessPoolExecutor(max_workers=12) as pool:raws=dict((k,(p,v,n,w)) for k,p,v,n,w in pool.map(raw,A['raw_receipts'],chunksize=8))
 assert len(raws)==1536
 e=collections.defaultdict(float)
 def chk(key,a,b):e[key]=max(e[key],float(abs(np.asarray(a)-np.asarray(b)).max()))
 for g in range(3):
  for c in range(16):
   q=next(x for x in C['roots'] if x['family']==g and x['core']==c)
   pri=np.array([raws[g,c,o,0][0] for o in range(8)]);v=np.array([raws[g,c,o,0][1] for o in range(8)]);n=np.array([[raws[g,c,o,r][2] for r in range(4)] for o in range(8)]);w=np.array([[raws[g,c,o,r][3] for r in range(4)] for o in range(8)])
   chk('prior_projected',pri.mean(0),q['prior_projected']);chk('prior_mean_TV',.5*abs(pri[1:]-pri[0]).sum(-1).mean(),q['prior_mean_TV']);chk('value_range',v.max()-v.min(),q['value_range'])
   for field,a in [('visit',n),('mark',w)]:
    d=a[1:]-a[0];D=d.sum(1);u=((D*D).sum(-1)-(d*d).sum((1,3)))/12
    chk('unbiased_'+field,u.mean(0),q['unbiased_mean_sq_'+field])
   chk('visit_TV',.5*abs(n[1:].mean(1)-n[0].mean(0)).sum(-1).mean(0),q['visit_mean_TV'])
 pairs=list(zip(C['contexts'],G['contexts']))
 with cf.ProcessPoolExecutor(max_workers=8) as pool:out=np.array(list(pool.map(gcheck,pairs,chunksize=1)))
 for k,x in zip(['U1_propagator','SU2_loop_trace','Bloch_stability','Bloch_readout'],out.max(0)):e[k]=float(x)
 assert max(e.values())<1e-9,dict(e)
 dump(S/'INDEPENDENT78.json',dict(status='PASS_SCOPED',raw_paths=len(raws),graph_contexts=len(out),errors=dict(e),scope='Separate coordinate permutation and all raw cut reductions; networkx tree/quotient/cycle checks, magnetic incidence expm_multiply, SU2 expm and independent Bloch updates. Same-agent separate implementation.'))
 print('VERIFY',dict(e),flush=True)
if __name__=='__main__':main()
