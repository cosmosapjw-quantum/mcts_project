"""Exact Stinespring record complement of R65's actual commuting selective wave.
No new arbitrary phase. All logs and entropies are nats. Frozen Rapfi schedule
is a separate counterfactual auxiliary channel, not an actual native phase intervention."""
from base61 import *
from scipy.special import gammaln,xlogy
from itertools import product
I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1.,-1.]).astype(complex)
U=np.cos(.35)*I-1j*np.sin(.35)*X
eta=np.exp(-1.2/32)
def H(p):
 p=np.clip(p,0,1);return -np.sum(xlogy(p,p),axis=-1)
def h(p):return -xlogy(p,p)-xlogy(1-p,1-p)
def S2(d,c):
 r=np.sqrt(np.clip((2*d-1)**2+4*abs(c)**2,0,1));return h((1+r)/2)
def mat(v):
 x,y,z=v;return (I+x*X+y*Y+z*Z)/2
def pointer(v):
 v=np.asarray(v);d=(1+np.sin(.7)*v[...,1]+np.cos(.7)*v[...,2])/2
 c=(v[...,0]-1j*(np.cos(.7)*v[...,1]-np.sin(.7)*v[...,2]))/2
 assert np.min(d)>-1e-12 and np.max(d)<1+1e-12 and np.max(abs(c)**2-d*(1-d))<1e-12
 return np.clip(d,0,1),c
def probs(m,r):
 e0=.35*r+.65;e1=.35*r
 return np.stack([m*e0,m*(1-e0),1-m],-1),np.stack([m*e1,m*(1-e1),1-m],-1)
def compact(d,c,s,f,w):
 a=s**f;b=np.sqrt(np.maximum(0,1-a*a));u=np.zeros(d.shape+(4,),complex);v=u.copy()
 u[...,0]=1;v[...,2]=a;v[...,3]=b
 outer=lambda a,b:a[..., :,None]*b[...,None,:].conj()
 cf=c*s**(w-f)
 return d[...,None,None]*outer(u,u)+(1-d)[...,None,None]*outer(v,v)+cf[...,None,None]*outer(u,v)+cf.conj()[...,None,None]*outer(v,u)
def eigen_entropy(r):
 e=np.linalg.eigvalsh(r);assert e.min()>-2e-11 and abs(e.sum(-1)-1).max()<2e-11
 return H(e)
def neg(r,da,db):
 shape=r.shape[:-2];q=r.reshape(shape+(da,db,da,db)).swapaxes(-3,-1).reshape(shape+(da*db,da*db))
 return np.maximum(0,-np.linalg.eigvalsh(q)).sum(-1)
PAULI=np.array([X,Y,Z]);OPS=np.array([[np.kron(a,b) for b in PAULI] for a in PAULI])
def chsh(r):
 T=np.einsum('...ab,ijba->...ij',r,OPS,optimize=True).real
 v=np.linalg.eigvalsh(T.swapaxes(-1,-2)@T);return 2*np.sqrt(np.maximum(0,v[...,-1]+v[...,-2]))
def countgrid(f):return np.array([(i,j,f-i-j) for i in range(f+1) for j in range(f+1-i)],int)
def pmf(p,f):
 g=countgrid(f);return np.exp(np.log(p)@g.T+gammaln(f+1)-gammaln(g+1).sum(1))
def classical_mi(d,p0,p1,f):
 M0=pmf(p0,f);M1=pmf(p1,f);M=d[:,None]*M0+(1-d[:,None])*M1
 assert abs(M.sum(1)-1).max()<1e-11
 val=H(M)-d*H(M0)-(1-d)*H(M1);return np.maximum(0,val),M
def calc(v,p0,p1,y=None):
 d,c=pointer(v);s=np.sqrt(p0*p1).sum(1);assert np.all((s>0)&(s<1));hd=h(d);S0=S2(d,c)
 out={k:[] for k in ('Ssys','linear_entropy','coherence','chi','classical','qmi','discord_Z','negativity','concurrence','CHSH','record_pair_MI','record_pair_negativity','chi_ratio','classical_ratio','dephased_qmi','dephased_negativity','dephased_CHSH')}
 w=4
 for f in range(1,w+1):
  rho=compact(d,c,s,f,w);cf=c*s**(w-f);sf=S2(d,np.sqrt(d*(1-d))*s**f)
  ss=S2(d,c*s**w);ssF=S2(d,cf);mi=ss+sf-ssF;di=ss-ssF
  cl,M=classical_mi(d,p0,p1,f)
  assert np.min(di)>-1e-12 and np.max(cl-sf)<1e-10 and np.max(sf-hd)<1e-10
  de=compact(d,np.zeros_like(c),s,f,w)
  # exact pair record state is separable, with conditional product form
  recmi=2*S2(d,np.sqrt(d*(1-d))*s)-S2(d,np.sqrt(d*(1-d))*s**2)
  vals=dict(Ssys=S2(d,c*s**f),linear_entropy=2*d*(1-d)-2*abs(c*s**f)**2,coherence=2*abs(c)*s**f,chi=sf,classical=cl,qmi=mi,discord_Z=np.maximum(0,di),negativity=neg(rho,2,2),concurrence=2*abs(cf)*np.sqrt(1-s**(2*f)),CHSH=chsh(rho),record_pair_MI=recmi,record_pair_negativity=np.zeros_like(d),chi_ratio=np.divide(sf,hd,out=np.zeros_like(d),where=hd>1e-10),classical_ratio=np.divide(cl,hd,out=np.zeros_like(d),where=hd>1e-10),dephased_qmi=sf,dephased_negativity=neg(de,2,2),dephased_CHSH=chsh(de))
  for k,x in vals.items():out[k].append(x)
 assert np.max(out['dephased_negativity'])<1e-12 and np.max(out['dephased_CHSH'])<=2+1e-12
 out={k:np.stack(x,axis=1) for k,x in out.items()}
 out.update(d=d,c=c,overlap=s,H_pointer=hd,initial_S=S0,initial_coherence=2*abs(c),attenuation=-np.log(s))
 # f90 for w=4, actual disjoint record packing, do not fill missing target
 valid=hd>1e-10
 for k in ('chi','classical'):
  passed=out[k+'_ratio']>=.9;anypass=passed.any(1)&valid
  f=np.where(anypass,np.argmax(passed,1)+1,0);out[k+'_f90']=f
  out[k+'_packing']=np.divide(w,f,out=np.zeros_like(d),where=f>0).astype(int)
 if y is not None:
  g=countgrid(4);ix=np.argmax(np.all(y[:,None,:]==g,axis=-1),axis=1);assert np.array_equal(g[ix],y)
  M0=pmf(p0,4);M1=pmf(p1,4);p=d[:,None]*M0+(1-d[:,None])*M1
  pbar=d[:,None]*p0+(1-d[:,None])*p1;iid=pmf(pbar,4)
  out['count_nll']=-np.log(p[np.arange(len(d)),ix]);out['iid_nll']=-np.log(iid[np.arange(len(d)),ix])
 return out
def direct_check(arg):
 v,p0,p1=arg;d,c=pointer(v);s=float(np.sqrt(p0*p1).sum());d=float(d);c=complex(c)
 rr=U@mat(v)@U.conj().T;assert abs(rr-np.array([[d,c],[c.conjugate(),1-d]])).max()<1e-12
 K=[np.diag(np.sqrt([a,b])) for a,b in zip(p0,p1)]
 V=np.stack(K,axis=1).reshape(6,2);assert abs(V.conj().T@V-I).max()<1e-12
 # independently construct every ordered record history from Kraus products
 A=[]
 for seq in product(range(3),repeat=4):
  a=I.copy()
  for i in seq:a=K[i]@a
  A.append(a)
 Vw=np.stack(A,axis=1).reshape(162,2)
 q=Vw@rr@Vw.conj().T;assert abs(np.trace(q)-1)<1e-12
 P=np.diag(q).real.reshape(2,81).sum(0)
 pred=np.array([d*np.prod(p0[list(x)])+(1-d)*np.prod(p1[list(x)]) for x in product(range(3),repeat=4)])
 eprob=float(abs(P-pred).max())
 r0=calc(np.asarray(v)[None,:],np.asarray(p0)[None,:],np.asarray(p1)[None,:])
 err=[];Y2=np.kron(Y,Y)
 for f in range(1,5):
  df=3**f;de=3**(4-f)
  sf=np.einsum('aibcjd->',np.zeros((1,1,1,1,1,1))) if False else np.einsum('aibcjb->aicj',q.reshape(2,df,de,2,df,de)).reshape(2*df,2*df)
  RF=np.einsum('aiaj->ij',sf.reshape(2,df,2,df));RS=np.einsum('aibi->ab',sf.reshape(2,df,2,df))
  nn=neg(sf,2,df);mi=eigen_entropy(RF)+eigen_entropy(RS)-eigen_entropy(sf)
  rho=compact(np.array(d),np.array(c),np.array(s),f,4)
  # independent Wootters via positive Hermitian square-root sandwich
  e,V=np.linalg.eigh(rho);R=(V*np.sqrt(np.clip(e,0,None)))@V.conj().T
  l=np.sqrt(np.clip(np.linalg.eigvalsh(R@Y2@rho.conj()@Y2@R),0,None))[::-1]
  con=max(0,l[0]-l[1:].sum())
  err.append([abs(eigen_entropy(RF)-r0['chi'][0,f-1]),abs(mi-r0['qmi'][0,f-1]),abs(nn-r0['negativity'][0,f-1]),abs(con-r0['concurrence'][0,f-1])])
 return [eprob,*np.max(err,axis=0)]
def free(r,phi,mode):
 if mode=='commuting':return r
 V=np.diag([np.exp(-.5j*phi),np.exp(.5j*phi)]) if mode=='actual_rotation' else I
 a=V@r@V.conj().T
 if mode=='pointer_dephased':
  q=U@a@U.conj().T;return U.conj().T@np.diag(np.diag(q))@U
 return (1+eta)/2*a+(1-eta)/2*Z@a@Z
def operators(m,r):
 p,q=probs(np.array(m),np.array(r))
 return np.array([U.conj().T@np.diag(np.sqrt([x,y]))@U for x,y in zip(p,q)])
def channel(K,r):return sum(k@r@k.conj().T for k in K)
def pair_record(K,L,r,phi,mode):
 # qutrit record pair: retain each selected record; trace memory after second.
 R=np.zeros((3,3,3,3),complex)
 for a,b in product(range(3),repeat=2):
  u=free(K[a]@r@K[b].conj().T,phi,mode)
  for c,d in product(range(3),repeat=2):R[a,c,b,d]=np.trace(L[c]@u@L[d].conj().T)
 R=R.reshape(9,9);assert abs(R-R.conj().T).max()<1e-12
 A=np.einsum('aibi->ab',R.reshape(3,3,3,3));B0=np.einsum('aiaj->ij',R.reshape(3,3,3,3))
 m=eigen_entropy(A)+eigen_entropy(B0)-eigen_entropy(R)
 p=np.diag(R).real.reshape(3,3);cl=H(p.sum(0))+H(p.sum(1))-H(p.ravel())
 return [float(m),float(cl),float(neg(R,3,3))]
def cross_job(arg):
 v,pars=arg;out=[]
 for mode in ('actual_rotation','no_rotation','commuting','pointer_dephased'):
  state=mat(v);w=[]
  for t,(m,r,phi) in enumerate(pars):
   if t:state=free(state,phi,mode)
   K=operators(m,r);d,c=pointer(np.array([np.trace(state@a).real for a in (X,Y,Z)]));w.append([float(d),float(abs(c)),float(eigen_entropy(state))])
   for _ in range(3):state=channel(K,state)
   if t+1<len(pars):
    m2,r2,phi2=pars[t+1];L=operators(m2,r2);w[-1]+=pair_record(K,L,state,phi2,mode)
   else:w[-1]+=[0.,0.,0.]
   state=channel(K,state)
  out.append(w)
 return out
def main():
 R=B/'mqft_restore65_20261126';reg=read(S/'REGISTRATION.json')
 assert sha(R/'GENERATOR65.npz')==reg['old_generator_sha'] and sha(R/'GENERATOR65.json')==reg['old_generator_receipt_sha']
 with np.load(R/'GENERATOR65.npz') as z:
  old=dict(v=z['first_bloch'].reshape(-1,3),p0=z['p0'].reshape(-1,3),p1=z['p1'].reshape(-1,3),y=z['counts'].reshape(-1,3),meta=np.repeat(z['meta'],32,axis=0))
 with np.load(S/'NATIVE_RECORDS69.npz') as z:
  nat=z['native'];meta=z['meta'];rp=z['rapfi'];rmeta=z['rapfi_meta']
 new=dict(v=nat[:,:,:3].reshape(-1,3),p0=nat[:,:,9:12].reshape(-1,3),p1=nat[:,:,12:15].reshape(-1,3),y=nat[:,:,6:9].reshape(-1,3),meta=np.repeat(meta,32,axis=0))
 assert len(old['v'])==49152 and len(new['v'])==8192
 datasets={'historical_R65':old,'fresh_R69':new}
 # separate 15x15 auxiliary state evolving on an immutable sequence of actual baseline scores
 vv=[]
 for pars in rp:
  state=mat([0,.8,0]);seq=[]
  for t,(m,r,phi) in enumerate(pars):
   if t:state=free(state,phi,'actual_rotation')
   seq.append([np.trace(state@a).real for a in (X,Y,Z)])
   K=operators(m,r)
   for _ in range(4):state=channel(K,state)
  vv.append(seq)
 p0,p1=probs(rp[:,:,0],rp[:,:,1])
 datasets['Rapfi_frozen_aux']=dict(v=np.array(vv).reshape(-1,3),p0=p0.reshape(-1,3),p1=p1.reshape(-1,3),meta=np.repeat(rmeta,32,axis=0))
 summaries=[];arr={};checks=[];rng=np.random.default_rng(690211)
 for name,data in datasets.items():
  v=data['v'];p0=data['p0'];p1=data['p1'];q=calc(v,p0,p1,data.get('y'))
  assert np.allclose(np.sum(p0,1),1) and np.allclose(np.sum(p1,1),1)
  for k,x in q.items():arr[name+'_'+k]=x
  arr[name+'_meta']=data['meta']
  groups=[('all',np.ones(len(v),bool))]
  if name!='Rapfi_frozen_aux':
   for g in (0,3):
    for arm in (0,1):groups.append((f'g{g}'+('+' if arm==0 else '-'),(data['meta'][:,0]==g)&(data['meta'][:,3]==arm)))
  for label,idx in groups:
   stats={k:np.mean(q[k][idx],axis=0).tolist() for k in ('Ssys','linear_entropy','coherence','chi','classical','qmi','discord_Z','negativity','concurrence','CHSH','record_pair_MI','chi_ratio','classical_ratio','dephased_qmi','H_pointer','initial_S','attenuation')}
   for k in ('chi','classical'):
    stats[k+'_eligible']=int(np.sum(idx&(q['H_pointer']>1e-10)))
    stats[k+'_achieved90']=int(np.sum(idx&(q[k+'_f90']>0)))
    stats[k+'_packing2']=int(np.sum(idx&(q[k+'_packing']>=2)))
   stats['CHSH_violation_counts']=(q['CHSH'][idx]>2+1e-10).sum(0).tolist()
   stats['entangled_counts']=(q['negativity'][idx]>1e-10).sum(0).tolist()
   stats['max_CHSH']=q['CHSH'][idx].max(0).tolist()
   if 'y' in data:stats['nll_exact_minus_iid']=float(np.mean((q['count_nll']-q['iid_nll'])[idx]))
   summaries.append(dict(dataset=name,group=label,n=int(idx.sum()),stats=stats))
  for i in rng.choice(len(v),size=32,replace=False):checks.append((v[i],p0[i],p1[i]))
 with cf.ProcessPoolExecutor(max_workers=12) as pool:errs=np.array(list(pool.map(direct_check,checks,chunksize=1)))
 assert errs[:,:4].max()<2e-10 and errs[:,4].max()<1e-7,errs.max(0)
 np.savez_compressed(S/'RECORD_STATES69.npz',**arr)
 # Actual observed scores held frozen per realized path: no adaptive quantum instrument
 # applied to alternate histories. 32 wave schedules; compare exact CP variants.
 jobs=[(nat[i,0,:3],nat[i,:,3:6]) for i in range(len(nat))]+[(np.array([0,.8,0]),x) for x in rp]
 with cf.ProcessPoolExecutor(max_workers=12) as pool:cross=np.array(list(pool.map(cross_job,jobs,chunksize=2)))
 assert cross.shape==(288,4,32,6)
 assert cross[:,:,:,5].max()<2e-10,cross[:,:,:,5].max()
 np.savez_compressed(S/'MULTITIME69.npz',cross=cross,meta=meta,rapfi_meta=rmeta,modes=np.array(['actual_rotation','no_rotation','commuting','pointer_dephased']))
 cs=[]
 for name,ii in [('g0',np.flatnonzero(meta[:,0]==0)),('g3',np.flatnonzero(meta[:,0]==3)),('Rapfi_frozen_aux',np.arange(len(nat),len(jobs)))]:
  for k,mode in enumerate(('actual_rotation','no_rotation','commuting','pointer_dephased')):
   x=cross[ii,k];cs.append(dict(dataset=name,mode=mode,n_schedules=len(ii),n_pairs=len(ii)*31,MI=float(x[:,:31,3].mean()),classical_MI=float(x[:,:31,4].mean()),max_negativity=float(x[:,:31,5].max()),memory_entropy=float(x[:,:,2].mean()),pointer_coherence=float((2*x[:,:,1]).mean())))
 result=dict(scope='R65 actual instrument isometrically dilated, conditional frozen waves; Rapfi auxiliary model driven by actual baseline score schedules; crosswave schedule frozen, not autonomous native process',old_waves=49152,fresh_waves=8192,rapfi_aux_waves=1024,independent_matrix_cases=len(checks),max_errors=dict(zip(['ordered_probability','record_entropy','QMI','negativity','concurrence'],errs.max(0).tolist())),record_array_sha=sha(S/'RECORD_STATES69.npz'),multitime_array_sha=sha(S/'MULTITIME69.npz'),summaries=summaries,crosswave=cs)
 dump(S/'PHYSICS_RESULTS.json',result)
 print('CHECKS',result['max_errors'],'cases',len(checks),flush=True)
 for x in summaries:
  if x['group']=='all':
   q=x['stats'];print('SUMMARY',x['dataset'],x['n'],'chi',q['chi'],'Icl',q['classical'],'MI',q['qmi'],'neg',q['negativity'],'con',q['concurrence'],'Ssys',q['Ssys'],'packing',q['chi_packing2'],q['classical_packing2'],'f90',q['chi_achieved90'],q['classical_achieved90'],'CHSH',q['CHSH_violation_counts'],q['max_CHSH'],flush=True)
 print('CROSS',cs,flush=True)
if __name__=='__main__':main()
