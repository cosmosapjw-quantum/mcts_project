"""Petz reverse on the measured fine R73 two-record channel, not a reverse MCTS tree."""
from base61 import *
import gzip
from scipy.special import xlogy
O=B/'mqft_restore73_20261126'
I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1.,-1.]).astype(complex)
P=(I+np.sin(.7)*Y+np.cos(.7)*Z)/2;Q=I-P
HP=P-I/2;HZ=Z/2;ETA=np.exp(-1.2/32);THETAS=[0.,1.,3.]
def dag(a):return a.swapaxes(-1,-2).conj()
def rho(v):return (I+v[0]*X+v[1]*Y+v[2]*Z)/2
def load(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def law(e):return e['root_selection']['law']
def tr(a):return np.trace(a,axis1=-2,axis2=-1).real
def spectral(a,power=None,log=False):
 e,U=np.linalg.eigh(a);assert np.min(e)>0
 y=np.log(e) if log else e**power
 return (U*y[...,None,:])@dag(U),e,U
def vn(a):
 e=np.linalg.eigvalsh(a);assert e.min()>-1e-12
 return -np.sum(xlogy(np.maximum(e,0),np.maximum(e,0)),axis=-1)
def rel(a,b):
 la,_,_=spectral(a,log=True);lb,_,_=spectral(b,log=True)
 return float(np.sum(tr(a@(la-lb))))
def K(l,legal,src):
 z=np.array(l['scores'])*l['beta_p'];z=np.exp(z-z.max());z/=z.sum();v=dict(zip(l['actions'],z));a,b=src;m=v[a]+v[b];r=v[a]/m;e0=.35*r+.65;e1=.35*r
 out=np.array([np.sqrt(m)*(np.sqrt(e0)*P+np.sqrt(e1)*Q) if i==a else np.sqrt(m)*(np.sqrt(1-e0)*P+np.sqrt(1-e1)*Q) if i==b else np.sqrt(v[i])*I for i in legal])
 assert abs(np.sum(dag(out)@out,axis=0)-I).max()<1e-12
 return out
def free(phi):
 u=np.diag(np.exp(-.5j*phi*np.array([1,-1])))
 return np.array([np.sqrt((1+ETA)/2)*u,np.sqrt((1-ETA)/2)*Z@u])
def channel(M,r):return np.sum(M@r@dag(M),axis=-3)
def weighted_entropy(block):
 p=tr(block);return float(np.sum(p*vn(block/p[...,None,None])))
def work(k,f,l,r):
 a=k@r@dag(k);p=tr(a);post=a/p[:,None,None];m=np.sum(f@post[:,None]@dag(f),axis=1)
 b=l@m[:,None]@dag(l);q=tr(b);assert abs(q.sum(1)-1).max()<1e-12
 E0=float(tr(r@HP));E1=float(p@tr(post@HP));w1=float(p@tr(post@(HZ-HP)))
 E2=float(p@tr(m@HZ));heat_free=E2-(E1+w1);w2=float(p@tr(m@(HP-HZ)))
 E3=float(np.sum(p[:,None]*tr(b@HP)));heat1=E1-E0;heat2=E3-(E2+w2)
 s0=float(vn(r));s1=float(p@vn(post));s2=float(p@vn(m));s3=float(np.sum(p[:,None]*q*vn(b/q[...,None,None])))
 assert max(abs(heat1),abs(heat2),abs(heat_free),abs(E3-E0-w1-w2))<2e-11
 return dict(energy=[E0,E1,E1+w1,E2,E2+w2,E3],switch_work=[w1,w2],free_energy_exchange=heat_free,measurement_mean_energy_exchange=[heat1,heat2],first_law_error=abs(E3-E0-w1-w2),conditional_entropy=[s0,s1,s2,s3],first_holevo=float(vn(a.sum(0))-s1),second_holevo=float(p@vn(b.sum(1))-s3),free_entropy_gain=s2-s1,reference_scope='H_P and H_Z switch; E* unit; no bath heat inferred')
def evaluate(M,r,theta):
 eig=np.exp(-theta*np.array([1.,-1.])/2);eig/=eig.sum();g=np.diag(eig).astype(complex);gs=np.diag(np.sqrt(eig))
 Gamma=channel(M,g);Omega=channel(M,r)
 assert abs(tr(Gamma).sum()-1)<1e-12 and abs(tr(Omega).sum()-1)<1e-12
 gi,e,U=spectral(Gamma,power=-.5)
 R=gs@dag(M)@gi[:,None]
 complete=np.sum(dag(R)@R,axis=1);cerr=float(abs(complete-I).max());assert cerr<2e-10
 recovery=np.sum(R@Gamma[:,None]@dag(R),axis=(0,1));rerr=float(abs(recovery-g).max());assert rerr<2e-10
 recovered=np.sum(R@Omega[:,None]@dag(R),axis=(0,1));assert abs(tr(recovered)-1)<2e-10
 recover_td=float(.5*np.sum(abs(np.linalg.eigvalsh(recovered-r))))
 d0=rel(r,g);d1=rel(Omega,Gamma);loss=d0-d1;assert loss>=-2e-10
 # TPM instrument: input diagonal in reference basis, output measured in reference block eigenbases.
 rp=np.diag(r).real;rd=np.diag(rp).astype(complex);omegad=channel(M,rd)
 fin=np.diagonal(dag(U)@omegad@U,axis1=-2,axis2=-1).real
 assert np.all(rp>0) and np.all(fin>0) and abs(fin.sum()-1)<1e-12
 a=abs(dag(U)[:,None]@M)**2 # block, bath, output m, input n
 b=abs(R@U[:,None])**2
 b=b.swapaxes(-1,-2)
 f=a*rp[None,None,None,:];rev=b*fin[:,None,:,None]
 fref=a*eig[None,None,None,:];rref=b*e[:,None,:,None]
 db=float(abs(fref-rref).max());assert db<1e-11
 ratio=(rp/eig)[None,None,None,:]/(fin/e)[:,None,:,None]
 # Eq ratio without dividing tiny/zero transition probabilities.
 terr=float(abs(f-rev*ratio).max());assert terr<1e-11
 assert abs(f.sum()-1)<2e-11 and abs(rev.sum()-1)<2e-11
 sig=np.log(ratio);ift=float(np.sum(f*np.exp(-sig)));ep=float(np.sum(f*sig))
 ref_ep=float(np.sum(rp*np.log(rp/eig))-np.sum(fin*np.log(fin/e)))
 assert abs(ift-1)<2e-10 and abs(ep-ref_ep)<2e-10 and ep>=-1e-10
 # Here output reference eigenbasis is fixed by reference; changing theta changes this TPM.
 return dict(theta=theta,quantum_input_rel=d0,quantum_output_rel=d1,quantum_rel_loss=loss,recovery_TD=recover_td,TPM_mean_EP=ep,TPM_IFT=ift,TPM_variance=float(np.sum(f*(sig-ep)**2)),TPM_reference_EP=ref_ep,TPM_detailed_balance_max_error=db,TPM_ratio_max_error=terr,reverse_completeness_max_error=cerr,reference_recovery_error=rerr,min_reference_eig=float(e.min()),model_heat_capacity=float(theta**2/(2+2*np.cosh(theta))))
def one(arg):
 c,q=arg;legal=np.array(c['legal']);n=len(legal);src=c['job']['sources'];root=B/c['source_path'];assert sha(root)==c['source_sha256']
 l0=law(load(root)['trace']['events'][15]);k=K(l0,legal,src);r=rho(l0['phase65']['prepared']);F=[];L=[];pp=[]
 for x in q['raw_receipts']:
  path=B/x['path'];assert sha(path)==x['sha256'];b=law(load(path)['trace']['events'][16]);F.append(free(b['phase65']['rotation']));L.append(K(b,legal,src));pp.append([dict(zip(b['actions'],b['p']))[a] for a in legal])
 F=np.array(F);L=np.array(L);M=(L[:,:,None]@F[:,None]@k[:,None,None]).reshape(n*n,2,2,2)
 out=channel(M,r);p=tr(k@r@dag(k));j=p[:,None]*np.array(pp);diagerr=float(abs(tr(out).reshape(n,n)-j).max());assert diagerr<1e-11
 # Fine action record exchange diagnostic: not a physical backward trajectory.
 ep_order=float(np.sum(j*np.log(j/j.T)));ift_order=float(np.sum(j*(j.T/j)));assert ep_order>=-1e-12 and abs(ift_order-1)<1e-12
 w=work(k,F,L,r);w0=work(k,np.broadcast_to(np.array([I,np.zeros_like(I)]),F.shape),L,r)
 return dict(generation=c['job']['generation'],arm=c['job']['arm'],core=c['job']['core'],fine_blocks=n*n,diag_error=diagerr,reference=[evaluate(M,r,t) for t in THETAS],switch_ledger=w,identity_free_control=w0,record_exchange_KL=ep_order,record_exchange_IFT=ift_order,source=c['source_path'])
def main():
 reg=read(O/'BRANCH_REGISTRATION73.json');a=read(O/'INSTRUMENT73.json')
 with cf.ProcessPoolExecutor(max_workers=8) as pool:rows=list(pool.map(one,zip(reg['contexts'],a['contexts']),chunksize=1))
 assert len(rows)==64
 res=dict(scope='measured fine two-record channel N; explicit reference-dependent reverse (Petz), TPM dephased input, no native inverse tree; H switch conditional model; record swap KL only',contexts=rows,reference_theta=THETAS,branch_hashes_checked=4544,blocks=sum(q['fine_blocks'] for q in rows),input_hashes={str(p.relative_to(B)):sha(p) for p in [O/'BRANCH_REGISTRATION73.json',O/'INSTRUMENT73.json',O/'INDEPENDENT73.json']})
 dump(S/'PETZ74.json',res)
 for t in THETAS:
  a=[r for q in rows for r in q['reference'] if r['theta']==t]
  print('THETA',t,'loss',np.mean([x['quantum_rel_loss'] for x in a]),'TPM EP',np.mean([x['TPM_mean_EP'] for x in a]),'recovery TD',np.mean([x['recovery_TD'] for x in a]),'IFT error',max(abs(x['TPM_IFT']-1) for x in a),'complete',max(x['reverse_completeness_max_error'] for x in a),flush=True)
 print('LEDGER',max(q['switch_ledger']['first_law_error'] for q in rows),'W mean',np.mean([sum(q['switch_ledger']['switch_work']) for q in rows]),'free entropy mean',np.mean([q['switch_ledger']['free_entropy_gain'] for q in rows]),'swap KL mean',np.mean([q['record_exchange_KL'] for q in rows]),'blocks',res['blocks'],flush=True)
if __name__=='__main__':main()
