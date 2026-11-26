"""Separate projector-basis density construction, raw conditional and heldout checks."""
from base61 import *
import gzip
from scipy.special import logsumexp
I=np.eye(2,dtype=complex);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1.,-1.]);X=np.array([[0,1],[1,0]])
P=(I+np.sin(.7)*Y+np.cos(.7)*Z)/2;Q=I-P
eta=np.exp(-1.2/32)
def load(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def adj(x):return x.conj().swapaxes(-1,-2)
def K(l,legal,src):
 z=np.array(l['scores']);z=np.exp(l['beta_p']*(z-z.max()));z/=z.sum()
 v=dict(zip(l['actions'],z));a,b=src;m=v[a]+v[b];r=v[a]/m
 x=.35*r+.65;y=.35*r
 return np.array([np.sqrt(m)*(np.sqrt(x)*P+np.sqrt(y)*Q) if i==a else np.sqrt(m)*(np.sqrt(1-x)*P+np.sqrt(1-y)*Q) if i==b else np.sqrt(v[i])*I for i in legal])
def rho(v):return (I+v[0]*X+v[1]*Y+v[2]*Z)/2
def law(e):return e['root_selection']['law']
def F(phi):
 U=np.diag(np.exp(-.5j*phi*np.array([1,-1])))
 return U
def direct(k,l,phi,r,group,pairs):
 n=len(k);i,j=pairs.T;out=np.zeros((9,9,2,2),complex)
 for a,b in pairs:
  h=k[a]@r@adj(k[b]);v=F(phi[a])@h@adj(F(phi[b]));v=(1+eta)/2*v+(1-eta)/2*Z@v@Z
  d=l[a,i]@v@adj(l[b,j])
  np.add.at(out,(group[a]*3+group[i],group[b]*3+group[j]),d)
 return out.transpose(0,2,1,3).reshape(18,18)
def run(z):
 c,row=z;src=row['source_pair'];legal=np.array(c['legal']);g=row['generation'];n=len(legal);a,b=np.searchsorted(legal,src)
 p0=B/c['source_path'];assert sha(p0)==c['source_sha256'];tr=load(p0)['trace'];l0=law(tr['events'][15]);k=K(l0,legal,src);r=rho(l0['phase65']['prepared'])
 q=[];ph=[];pr=[]
 for x in row['raw_receipts']:
  p=B/x['path'];assert sha(p)==x['sha256'];d=load(p);u=law(d['trace']['events'][16]);q.append(K(u,legal,src));ph.append(u['phase65']['rotation']);pr.append([dict(zip(u['actions'],u['p']))[a] for a in legal])
 q=np.array(q);ph=np.array(ph);pr=np.array(pr)
 group=np.full(n,2,int);group[a]=0;group[b]=1
 pairs=np.array([(i,i) for i in range(n)]+[(a,b),(b,a)])
 out=direct(k,q,ph,r,group,pairs)
 stored=np.array(row['adaptive']['system_record_density_real'])+1j*np.array(row['adaptive']['system_record_density_imag'])
 err=float(np.max(abs(out-stored)));assert err<1e-11
 rec=np.einsum('asbs->ab',out.reshape(9,2,9,2));ev=np.linalg.eigvalsh(rec);assert ev.min()>-1e-11
 pt=np.empty((9,9),complex)
 for i in range(9):
  for j in range(9):
   a0,b0=divmod(i,3);c0,d0=divmod(j,3);pt[i,j]=rec[a0*3+d0,c0*3+b0]
 neg=float(np.maximum(-np.linalg.eigvalsh(pt),0).sum());errneg=abs(neg-row['adaptive']['record_neg']);assert errneg<1e-11
 ix=int(np.searchsorted(legal,c['observed']))
 frozen=direct(k,np.broadcast_to(q[ix],q.shape),np.broadcast_to(ph[ix],ph.shape),r,group,pairs)
 f=np.array(row['frozen']['system_record_density_real'])+1j*np.array(row['frozen']['system_record_density_imag']);errf=float(np.max(abs(f-frozen)));assert errf<1e-11
 # Independent native diagonal by scalar weights, without density amplitudes.
 pp=np.array([dict(zip(l0['actions'],l0['p']))[a] for a in legal]);j3=np.zeros((3,3))
 np.add.at(j3,(group[:,None],group[None,:]),pp[:,None]*pr)
 errd=float(np.max(abs(j3-np.diag(rec).real.reshape(3,3))));assert errd<1e-11
 return [err,errf,errneg,errd,abs(np.trace(out)-1).real]
def main():
 a=read(S/'INSTRUMENT73.json');reg=read(S/'BRANCH_REGISTRATION73.json');assert len(a['contexts'])==64
 with cf.ProcessPoolExecutor(max_workers=8) as pool:
  q=np.array(list(pool.map(run,zip(reg['contexts'],a['contexts']),chunksize=1)))
 assert q.max()<1e-10
 d=read(S/'DECISION_RESULTS73.json');raw=np.load(S/'DECISION73.npz');m=raw['meta'];assert len(m)==10112
 e=0.
 for r in d['results']:
  fresh=r['dataset']=='fresh_R73';ii=(m[:,0]==fresh)&(m[:,1]==r['generation'])&(m[:,2]==r['arm'])
  if not fresh:ii&=m[:,3]%16<4
  assert ii.sum()==r['paths'] and np.array_equal(np.bincount(m[ii,5],minlength=3),r['decision_counts'])
  ll=np.array(r['log_loss'])[:,0];gg=np.array(r['gain'])[:,0];e=max(e,float(abs(ll[0]-ll-gg).max()))
  assert abs(ll[0]-ll-gg).max()<1e-12
 out=dict(status='PASS_SCOPED',contexts=64,branch_member_hashes=4544,max_adaptive_matrix_error=float(q[:,0].max()),max_frozen_matrix_error=float(q[:,1].max()),max_negativity_error=float(q[:,2].max()),max_diagonal_error=float(q[:,3].max()),max_trace_error=float(q[:,4].max()),decision_groups=len(d['results']),max_decision_gain_identity_error=e,scope='independent projector-based direct cross-block calculation, raw branch hashes, scalar probabilities and partial transpose; decision group/gain identity only, not independent predictor fit; same-agent alternate implementation')
 dump(S/'INDEPENDENT73.json',out);print(out,flush=True)
if __name__=='__main__':main()
