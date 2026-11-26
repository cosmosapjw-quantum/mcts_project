"""Independent raw threshold-connectivity and explicit lead-matching audit."""
from base61 import *
import gzip
from scipy.linalg import solve,expm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
J=.35
def load(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def matrix(g,path=False,phi=0.,lead=False):
 v=g['path'] if path else g['component'];idx={a:i for i,a in enumerate(v)}
 V=dict(zip(g['component'],g['potential']));edges=list(zip(v[:-1],v[1:])) if path else g['edges']
 A=np.zeros((len(v),len(v)),complex)
 for x,y in edges:
  a,b=idx[x],idx[y];z=np.exp(1j*phi) if not path and g['plaquette'] and [x,y]==g['plaquette'][:2] else 1.
  A[a,b]=z;A[b,a]=z.conj() if hasattr(z,'conj') else z
 H=np.diag([V[x] for x in v])+J*(np.diag(np.count_nonzero(A,axis=1))-A)
 s,t=idx[g['source']],idx[g['target']]
 if lead:H[s,s]+=J;H[t,t]+=J
 return H,s,t
def explicit(H,s,t,E):
 n=len(H);k=np.arccos(1-E/(2*J));A=np.zeros((n+2,n+2),complex);b=np.zeros(n+2,complex)
 A[:n,:n]=(E+1e-11j)*np.eye(n)-H
 A[s,n]=J;A[t,n+1]=J;b[s]=-J
 A[n,s]=1;A[n,n]=-np.exp(-1j*k);b[n]=np.exp(1j*k)
 A[n+1,t]=1;A[n+1,n+1]=-np.exp(-1j*k)
 q=solve(A,b,check_finite=False);assert np.linalg.norm(A@q-b)<1e-6
 return float(abs(q[n+1])**2),float(abs(q[n])**2)
def audit(pair):
 g,r=pair;assert (g['family'],g['core'],g['rep'])==(r['family'],r['core'],r['rep'])
 p=B/g['source_path'];assert sha(p)==g['source_sha'];raw=load(p);es=raw['trace']['events'];law=es[32]['root_selection']['law'];L=g['L'];legal=np.array(law['actions']);V=-np.array(law['scores']);ids=np.argsort(legal);legal=legal[ids];V=V[ids];n=len(V)
 xy=np.column_stack([legal//L,legal%L]);adj=(abs(xy[:,None]-xy[None,:]).sum(-1)==1);dist=shortest_path(csr_matrix(adj),directed=False,unweighted=True,indices=int(np.argmin(V)))
 s=int(np.argmin(V));comp=np.flatnonzero(np.isfinite(dist));assert np.array_equal(legal[comp],g['component'])
 # Minimax connectivity: compute minimum threshold to connect each candidate by component unions.
 parent=np.arange(n);active=np.zeros(n,bool);cost=np.full(n,np.inf)
 def find(i):
  while parent[i]!=i:i=parent[i]
  return i
 for h in np.unique(V):
  for a in np.flatnonzero(V==h):
   active[a]=True
   for b in np.flatnonzero(adj[a]&active):
    x,y=find(a),find(b);parent[x]=y
  if active[s]:
   root=find(s)
   for a in np.flatnonzero(active):
    if find(a)==root and not np.isfinite(cost[a]):cost[a]=h
 cand=[i for i in comp if dist[i]>=4 and np.all(V[i]<=V[adj[i]]) and np.any(V[i]<V[adj[i]])]
 if not cand:cand=[i for i in comp if dist[i]>=4]
 t=min(cand,key=lambda i:(-max(0,cost[i]-max(V[s],V[i])),V[i],legal[i]))
 assert legal[s]==g['source'] and legal[t]==g['target'] and cost[t]==g['saddle']
 assert abs(max(0,cost[t]-max(V[s],V[t]))-g['height'])<1e-13
 sub=(V<=cost[t]);a=adj&sub[:,None]&sub[None,:];dd=shortest_path(csr_matrix(a),directed=False,unweighted=True,indices=s)
 assert dd[t]==g['width']
 pr=[];hit=[]
 for e in es[32:]:
  l=e['root_selection']['law'];pr.append(dict(zip(l['actions'],l['p']))[int(legal[t])]);hit.append(e['actions'][0]==legal[t])
 tt=next((i+1 for i,z in enumerate(hit) if z),97);assert tt==g['tau']
 comp_err=abs(sum(pr[:min(tt,96)])-g['compensator']);assert comp_err<1e-10
 err=0.;phaseerr=0.
 for path in (False,True):
  H,i,j=matrix(g,path=path,lead=True)
  for a in r['scattering']:
   T,R=explicit(H,i,j,a['E']);q=a['path' if path else 'full'];err=max(err,abs(T-q['T']),abs(R-q['R']))
 H,i,j=matrix(g,phi=np.pi/2,lead=True)
 for a in r['scattering']:
  T,_=explicit(H,i,j,a['E']);phaseerr=max(phaseerr,abs(T-a['phase_T']))
 H,i,j=matrix(g);U=expm(-1j*H);K=abs(U)**2
 psi=np.zeros(len(H),complex);psi[i]=1
 p0=np.zeros(len(H));p0[i]=1;f=[];z=[]
 for k in range(64):
  psi=U@psi;p0=K@p0;f.append(abs(psi[j])**2);z.append(p0[j]);psi[j]=0;p0[j]=0
 de=max(np.max(abs(np.array(f)-r['detector']['coherent_f'])),np.max(abs(np.array(z)-r['detector']['dephased_f'])))
 assert max(err,phaseerr)<1e-7 and de<1e-10
 return [comp_err,err,phaseerr,float(de)]
def main():
 R=read(S/'FIRST_RESULTS75.json');T=read(S/'TRANSPORT75.json');assert sha(S/'FIRST75.npz')==R['array_sha256'] and sha(S/'FIRST_RESULTS75.json')==T['source_sha']
 with cf.ProcessPoolExecutor(max_workers=8) as pool:errs=np.array(list(pool.map(audit,zip(R['contexts'],T['contexts']),chunksize=1)))
 with np.load(S/'FIRST75.npz') as a:m=a['meta'];f=a['features']
 mx=0.
 for k,g in enumerate(R['results']):
  for name,arr in [('cum_hit',f[:,:,0].cumsum(1)),('cum_compensator',f[:,:,2].cumsum(1)),('compensator_residual',(f[:,:,0]-f[:,:,2]).cumsum(1))]:
   y=np.mean([arr[(m[:,0]==k)&(m[:,1]==c)].mean(0) for c in range(32)],0);mx=max(mx,float(abs(y-np.array(g[name][0])).max()))
 assert mx<1e-12
 G=read(S/'GAMES75.json')
 for q in G['results']:
  for g in q['rows']:
   y=np.array(g['values'])[:,1];k=np.flatnonzero(np.maximum.accumulate(y)-y>=q['threshold']);assert bool(len(k))==g['eligible']
   if len(k):
    i=k[0];peak=y[:i+1].max();assert g['entry_index']==i and peak==g['peak']
    ix=np.flatnonzero(y[i+1:]>=peak);assert g['event']==bool(len(ix))
    assert g['followup']==(ix[0]+1 if len(ix) else len(y)-1-i)
 # Independent continuum analytical formula from complex matching for rectangular slab; no native calibration.
 ce=0.
 for q in T['continuum_controls']:
  E=.2;V=1.;k=np.sqrt(E);kap=np.sqrt(V-E);w=q['width'];M=np.array([[np.cosh(kap*w),np.sinh(kap*w)/kap],[kap*np.sinh(kap*w),np.cosh(kap*w)]])
  C=np.array([[1,1],[1j*k,-1j*k]]);D=np.linalg.solve(C,M@C);l=-2*np.log(abs(D[1,1]));ce=max(ce,abs(l-q['logT_continuum']))
 assert ce<1e-10
 result=dict(status='PASS_SCOPED',raw_contexts=len(errs),full_and_path_scattering_cases=len(errs)*6,phase_scattering_cases=len(errs)*3,max_compensator_error=float(errs[:,0].max()),max_lead_matching_error=float(errs[:,1].max()),max_phase_matching_error=float(errs[:,2].max()),max_detector_error=float(errs[:,3].max()),max_aggregate_error=mx,continuum_reference_error=ce,scope='independent union-find threshold certificate and explicit lead boundary matching, expm detector, raw context subset, full root aggregate, historical episode and continuum matching; same-agent independent implementation')
 dump(S/'INDEPENDENT75.json',result);print(result,flush=True)
if __name__=='__main__':main()
