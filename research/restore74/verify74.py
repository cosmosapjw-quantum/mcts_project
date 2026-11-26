"""Independent matrix-function and Bayesian TPM audit; native raw pulse audit subset."""
from base61 import *
import gzip
O=B/'mqft_restore73_20261126';I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1.,-1.]).astype(complex)
U=np.cos(.35)*I-1j*np.sin(.35)*X;HP=(np.sin(.7)*Y+np.cos(.7)*Z)/2;HZ=Z/2;eta=np.exp(-1.2/32)
def adj(x):return x.conj().swapaxes(-1,-2)
def tr(x):return np.trace(x,axis1=-2,axis2=-1).real
def law(e):return e['root_selection']['law']
def load(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def K(l,legal,src):
 a,b=src;h=l['phase65'];m=h['pair_mass'];e0,e1=h['e0'],h['e1'];p=dict(zip(l['actions'],l['p']))
 A=np.sqrt(m)*adj(U)@np.diag(np.sqrt([e0,e1]))@U
 C=np.sqrt(m)*adj(U)@np.diag(np.sqrt([1-e0,1-e1]))@U
 return np.array([A if i==a else C if i==b else np.sqrt(p[i])*I for i in legal])
def eigvals(A):
 c=tr(A)/2;d=np.sqrt(np.maximum(0,((A[...,0,0]-A[...,1,1]).real/2)**2+abs(A[...,0,1])**2));return c,d
def fun(A,mode):
 c,d=eigvals(A);p=c+d;m=c-d;assert np.all(m>0)
 if mode=='log':
  a=(np.log(p)+np.log(m))/2;b=np.divide(np.log(p)-np.log(m),2*d,out=np.array(1/c),where=d>1e-14*c)
 else:
  a=(1/np.sqrt(p)+1/np.sqrt(m))/2;b=np.divide(1/np.sqrt(p)-1/np.sqrt(m),2*d,out=np.array(-.5*c**(-1.5)),where=d>1e-14*c)
 return a[...,None,None]*I+b[...,None,None]*(A-c[...,None,None]*I)
def relative(a,b):
 return float(np.sum(tr(a@(fun(a,'log')-fun(b,'log')))))
def one(args):
 c,q,stored=args;legal=np.array(c['legal']);n=len(legal);src=c['job']['sources'];r0=load(B/c['source_path'])['trace'];l0=law(r0['events'][15]);v=l0['phase65']['prepared'];r=(I+v[0]*X+v[1]*Y+v[2]*Z)/2;k=K(l0,legal,src)
 l=[];f=[]
 for a in q['raw_receipts']:
  u=law(load(B/a['path'])['trace']['events'][16]);l.append(K(u,legal,src));V=np.diag(np.exp(-.5j*u['phase65']['rotation']*np.array([1,-1])));f.append([np.sqrt((1+eta)/2)*V,np.sqrt((1-eta)/2)*Z@V])
 l=np.array(l);f=np.array(f);M=(l[:,:,None]@f[:,None]@k[:,None,None]).reshape(n*n,2,2,2)
 def out(a):return np.sum(M@a@adj(M),axis=1)
 O0=out(r);errors=[]
 for s in stored['reference']:
  th=s['theta'];p=np.exp(-th*np.array([1.,-1.])/2);p/=p.sum();g=np.diag(p).astype(complex);G=out(g)
  loss=relative(r,g)-relative(O0,G);inv=fun(G,'isqrt');R0=inv@O0@inv
  gs=np.diag(np.sqrt(p));rec=gs@np.sum(adj(M)@R0[:,None]@M,axis=(0,1))@gs;td=float(.5*np.sum(abs(np.linalg.eigvalsh(rec-r))))
  error=[abs(loss-s['quantum_rel_loss']),abs(td-s['recovery_TD'])]
  if th>0:
   # Classical transition matrix in reference eigenbasis, reverse by normalized Bayes rule.
   e,V=np.linalg.eigh(G);T=np.sum(abs(adj(V)[:,None]@M)**2,axis=1);rp=np.diag(r).real
   joint=T*rp;fin=joint.sum(-1);back=T*p/e[:,:,None];assert abs(back.sum(-1)-1).max()<1e-10
   reverse=back*fin[:,:,None];assert abs(joint.sum()-1)<1e-11 and abs(reverse.sum()-1)<1e-11
   q0=joint>0;ep=float(np.sum(joint[q0]*np.log(joint[q0]/reverse[q0])))
   error.extend([abs(ep-s['TPM_mean_EP']),abs(np.sum(reverse)-s['TPM_IFT'])])
  else:error.extend([0.,0.])
  errors.append(error)
 assert max(np.max(errors),abs(tr(O0).sum()-1))<1e-8
 return errors
def main():
 R=read(O/'BRANCH_REGISTRATION73.json');A=read(O/'INSTRUMENT73.json');P=read(S/'PETZ74.json')
 with cf.ProcessPoolExecutor(max_workers=8) as pool:err=np.array(list(pool.map(one,zip(R['contexts'],A['contexts'],P['contexts']),chunksize=1)))
 # Independent root-block direct response and source probability inversion on 144 raw paths.
 d=read(S/'AUDIT74.json');assert sha(S/'PULSE74.npz')==d['array_sha256']
 with np.load(S/'PULSE74.npz') as z:m=z['meta'];v=z['features'];p=z['pulse']
 rr=read(S/'RESPONSE74.json');R0=read(S/'REGISTRATION.json');pathh={x['path']:x['sha256'] for x in d['raw_receipts']}
 er=0.;nraw=0
 for g,tag in enumerate(('schedule0','schedule3','rapfi')):
  ix=np.flatnonzero(m[:,0]==g);mu={}
  for a in range(3):
   q=m[ix][m[ix,3]==a];vv=v[ix][m[ix,3]==a]
   mu[a]=np.mean(vv[:,:,:2],0)
  direct=(mu[1]-mu[2])/.7;er=max(er,float(abs(direct-np.array(rr['results'][g]['direct_fd'][0])).max()))
  # independent empirical covariance using numpy cov; no same code reused.
  z=[]
  for c in range(16):
   ix=np.flatnonzero((m[:,0]==g)&(m[:,1]==c)&(m[:,3]==0));a=np.column_stack([v[ix,:,:2].reshape(len(ix),64),p[ix,0]-p[ix,1]]);z.append(np.cov(a,rowvar=False)[-1,:-1].reshape(32,2))
  er=max(er,float(abs(np.mean(z,0)-np.array(rr['results'][g]['score_derivative'][0])).max()))
  for j in read(S/'plans'/f'{tag}.json'):
   if j['rep']!=0:continue
   f=S/'data'/tag/(j['id']+'.json.gz');assert sha(f)==pathh[str(f.relative_to(B))];raw=load(f)['trace'];a=['zero74','plus74','minus74'].index(j['arm']);i=np.flatnonzero(np.all(m==[g,j['core'],j['rep'],a],axis=1))[0]
   l=law(raw['events'][32]);src=j['sources'][0];p1=dict(zip(l['actions'],l['p']))[src];h=[0.,.35,-.35][a];p0=p1/(p1+(1-p1)*np.exp(h));n=sum(e['actions'][0]==src for e in raw['events'][32:36])
   lp=[x*n-4*np.log1p(p0*np.expm1(x)) for x in (.35,-.35)];er=max(er,float(abs(p[i]-[n,4*p0,n-4*p0,*lp]).max()));nraw+=1
 assert er<1e-10 and nraw==144
 out=dict(status='PASS_SCOPED',contexts=64,reference_cases=192,TPM_Bayes_cases=128,theta_zero_TPM_scope='reference eigenbasis degeneracy, original exact direct Kraus check only; independent loss/recovery all theta',max_relative_loss_error=float(err[...,0].max()),max_recovery_TD_error=float(err[...,1].max()),max_TPM_EP_error=float(err[...,2].max()),max_TPM_IFT_error=float(err[...,3].max()),max_native_response_error=er,independent_raw_pulse_paths=nraw,scope='separate rotated-basis Kraus, analytic 2x2 matrix functions and Bayesian reverse, all 64 contexts; independent raw subset and aggregate response; not independent human/subagent')
 dump(S/'INDEPENDENT74.json',out);print(out,flush=True)
if __name__=='__main__':main()
