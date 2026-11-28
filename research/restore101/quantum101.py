from field101 import *
from instrument81 import *
import gzip
from scipy.linalg import expm
def choi(T):
 # normalized Choi, input tensor output order; unital Bloch map
 return (np.kron(I,I)+sum(np.kron(PAULI[j].T,np.einsum('i,ijk->jk',T[:,j],PAULI)) for j in range(3)))/4
def choi_kraus(ks):
 v=np.stack([a.T.reshape(-1) for a in ks]);return v.T@v.conj()/2
def cpcheck(e,C):
 ck(e,'Choi_hermitian',C,adj(C));ck(e,'Choi_TP',np.trace(C.reshape(2,2,2,2),axis1=1,axis2=3),I/2);ck(e,'Choi_trace',np.trace(C),1);x=np.linalg.eigvalsh(C);assert x.min()>-1e-10;return x
def norm(a):return float(np.linalg.norm(a))
def local_one(r):
 f=B/r['path'];assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256']
 with gzip.open(f,'rt') as z:d=json.load(z)
 es=d['trace']['events'];assert len(es)==128;mode=ARMS[r['arm']];assert d['input']['arm']==mode and d['input']['sources']==r['sources']
 err={};rows=[];T=np.eye(3);v=np.array([.8,0.,0.]);sing=0
 for t in range(32,44):
  e=es[t];l=e['root_selection']['law'];ph=l['phase65'];assert l['t']==t and ph['sources']==r['sources'];ax,lam,th,eta,erase=schedule(mode,t)
  scores=np.asarray(l['scores']);p=soft((10/3)*scores);acts=np.array(l['actions']);ia=int(np.flatnonzero(acts==r['sources'][0])[0]);ib=int(np.flatnonzero(acts==r['sources'][1])[0])
  m=float(p[ia]+p[ib]);rp=float(p[ia]/m);ck(err,'pair_mass',m,ph['pair_mass']);ck(err,'pair_population',rp,ph['population_a']);ck(err,'pre_state',v,ph['pre'])
  A,Bb,Oo,e0,e1=operators(m,rp,ax,lam);ks=[A,Bb,Oo];gamma=1-m+m*(np.sqrt(e0*e1)+np.sqrt((1-e0)*(1-e1)));assert 0<gamma<=1+1e-12
  gamma=min(gamma,1.);deta=0. if erase else eta
  U=np.diag([np.exp(-.5j*th),np.exp(.5j*th)]);preks=[np.sqrt((1+deta)/2)*U,np.sqrt((1-deta)/2)*Z@U];local=[k@q for k in ks for q in preks]
  ck(err,'Kraus_complete',sum(adj(k)@k for k in local),I)
  M=np.eye(3)*gamma;M[ax,ax]=1.;D=np.diag([deta,deta,1]);R=np.array([[np.cos(th),-np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1]])
  step=M@D@R;C=choi(step);ck(err,'Choi_independent',C,choi_kraus(local));eig=cpcheck(err,C);T=step@T;CE=choi(T);eg=cpcheck(err,CE)
  # individual fixed-axis dephasing GKSL segment, dimensionless unit segment, no physical time.
  rate=-.5*np.log(gamma);damp=np.exp(-2*rate);ck(err,'dephase_exponential',damp,gamma)
  if deta==0:sing+=1
  else:ck(err,'prep_exponential',np.exp(-2*(-.5*np.log(deta))),deta)
  # check noncommutation of rotation/dephase measurement versus reverse order
  comm=norm(M@D@R-D@R@M)
  pr=free(rho(v),th,eta,erase);pp=basep=p.copy();out=[x@pr@adj(x) for x in ks];prob=np.array([np.trace(y).real for y in out]);ck(err,'nonselective_Bloch',bloch(sum(out)),M@bloch(pr))
  pp[ia]=prob[0];pp[ib]=prob[1];ck(err,'native_probability',pp,l['p']);pick=l['picked'];assert acts[pick]==e['actions'][0]
  ind=0 if pick==ia else 1 if pick==ib else 2;v=bloch(out[ind]/prob[ind]);ck(err,'selected_state',v,ph['post']);ck(err,'prepared',bloch(pr),ph['prepared'])
  # same one-component visibility with pure unitary rotation; unit action dimensionless.
  missing=gamma>=1-1e-14;hstar=0. if missing else 1/math.acos(gamma)
  a=rho([gamma,0,0]);b=rho([gamma,np.sqrt(max(0.,1-gamma*gamma)),0]);td=.5*sum(abs(np.linalg.eigvalsh(a-b)));ck(err,'visibility_TD',td,np.sqrt(max(0.,1-gamma*gamma))/2);ck(err,'visibility_same_X',np.trace(a@X),np.trace(b@X))
  ck(err,'unitary_purity',np.trace(b@b),1);ck(err,'dephase_purity',np.trace(a@a),(1+gamma*gamma)/2)
  rows.append([gamma,rate,hstar,float(not missing),deta,float(erase),td,comm,eig.min(),eg.min(),float(np.linalg.det(step)),*np.linalg.svd(T,compute_uv=False),*v,np.linalg.norm(v[:2]),m,lam])
 return [r[k] for k in ['family','core','rep','arm']],rows,err,sing
def neg(M,d):
 # S is final factor; independent explicit index partial transpose on S.
 q=M.reshape(2**d,2,2**d,2).transpose(0,3,2,1).reshape(M.shape);z=np.linalg.eigvalsh(q);return float(-z[z<0].sum())
def record(src,err):
 p=pin(95,'JOINT95.npz',src);a=read(pin(95,'ENTANGLE95.json',src));assert sha(p)==a['joint_sha']
 old={(r['family'],r['core'],r['protocol'],r['mode'],r['depth'],r['xi']):r for r in a['rows']};assert len(old)==5184
 result=[];cnt=0
 with np.load(p) as z:
  meta=z['meta'];assert len(meta)==576
  for i,m in enumerate(meta):
   fam,core=int(m[0]),int(m[1]);protocol,mode=m[2:]
   for d in [1,2,3]:
    R=z[f'J{d}'][i];nr=2**d;bits=np.array([[int(x) for x in f'{j:0{d}b}'] for j in range(nr)]);hd=(bits[:,None]!=bits[None,:]).sum(-1);H=np.array([[1,1],[1,-1]])/np.sqrt(2);HH=H
    for j in range(d-1):HH=np.kron(HH,H)
    U=np.kron(HH,I);phase=np.repeat(np.exp(.5j*np.pi*bits[:,0]),2)
    for xi in [0.,.5,1.]:
     o=old[fam,core,protocol,mode,d,xi];M=R*np.repeat(np.repeat(xi**hd,2,axis=0),2,axis=1);M1=phase[:,None]*M*phase.conj()[None,:]
     ck(err,'joint_trace',np.trace(M),1);ck(err,'phase_diag',M.diagonal(),M1.diagonal());eig=np.linalg.eigvalsh(M);assert eig.min()>-1e-10;ck(err,'phase_spectrum',eig,np.linalg.eigvalsh(M1))
     cut=tuple(range(d));ix=o['cuts'].index(list(cut));nn=neg(M,d);ck(err,'R95_system_record_NPT',nn,o['cut_negativity'][ix]);ck(err,'phase_local_negativity',nn,neg(M1,d))
     q=np.diag(U@M@adj(U)).real.reshape(nr,2).sum(1);q1=np.diag(U@M1@adj(U)).real.reshape(nr,2).sum(1);ck(err,'Hadamard_norm',q.sum(),1);ck(err,'Hadamard_norm_tag',q1.sum(),1);assert min(q.min(),q1.min())>-1e-10
     # diagonal input -> uniform H outcome; verify using independent reduced matrix.
     rec=np.trace(M.reshape(nr,2,nr,2),axis1=1,axis2=3);ck(err,'Hadamard_reduce',q,np.diag(HH@rec@HH.T).real)
     tv=.5*abs(q-q1).sum();tv0=.5*abs(q-1/nr).sum();tv1=.5*abs(q1-1/nr).sum()
     if xi==0:ck(err,'tagged_dephased_TV',tv,0);ck(err,'tagged_dephased_NPT',nn,0)
     result.append(dict(family=fam,core=core,protocol=str(protocol),mode=str(mode),depth=d,xi=xi,success=o['success'],N=nn,diag_invariant=True,phase_TV=float(tv),weighted_phase_TV=float(o['success']*tv),Hadamard_TV=float(tv0),tag_Hadamard_TV=float(tv1),eigenvalues=eig.tolist()))
     cnt+=1
 assert cnt==5184
 return result
def main():
 t=time.monotonic();src={};err={};m94=read(R94/'RUN_MANIFEST94.json');m95=read(R95/'RUN_MANIFEST95.json')
 a=read(pin_input(R81/'AUDIT81.json',m94,src));p=pin(94,'NATIVE94.npz',src);pin(94,'NATIVE94.json',src);pin(94,'RECORD94.json',src);pin(95,'SUMMARY95.json',src);pin(97,'SUMMARY97.json',src)
 # exact code identity (copied fixed reference)
 f=pin(94,'instrument81.py',src);assert sha(S/'instrument81.py')==sha(f)
 raw=m94['prior_raw_receipts'].copy()
 for k,h in m95['prior_raw_receipts'].items():assert k not in raw or raw[k]==h;raw[k]=h
 assert len(a['raw_receipts'])==3456
 assert all(raw[r['path']]==r['sha256'] for r in a['raw_receipts'])
 with cf.ThreadPoolExecutor(max_workers=6) as pool:assert dict(pool.map(rehash,raw.items()))==raw
 print('QUANTUM RAW',len(raw),flush=True)
 with cf.ProcessPoolExecutor(max_workers=10) as pool:res=list(pool.map(local_one,a['raw_receipts'],chunksize=6))
 M=np.array([r[0] for r in res]);V=np.array([r[1] for r in res]);assert V.shape==(3456,12,20)
 for r in res:
  for k,v in r[2].items():err[k]=max(err.get(k,0.),v)
 with np.load(p) as z:
  idx={tuple(m):i for i,m in enumerate(z['meta'])};ii=[idx[tuple(m)] for m in M];ck(err,'R94_gamma',V[:,:,0],z['features'][ii,:,15]);ck(err,'R94_states',V[:,:,14:17],z['states'][ii]);ck(err,'R94_frozen_singular',V[:,-1,11:14],z['stability'][ii,:3])
 rr=record(src,err);groups=[]
 for g in range(3):
  for arm in range(6):
   z=np.stack([V[(M[:,0]==g)&(M[:,1]==c)&(M[:,3]==arm)].mean(0) for c in range(24)])
   # hbar finite only, separate mask and mean over finite cases
   val=V[(M[:,0]==g)&(M[:,3]==arm)];h=val[:,:,2];valid=val[:,:,3]>0
   groups.append(dict(family=g,arm=arm,ci=ci(z).tolist(),finite_hbar_count=valid.sum(0).tolist(),finite_hbar_mean=np.divide((h*valid).sum(0),valid.sum(0),out=np.zeros(12),where=valid.sum(0)>0).tolist()))
 assert len(groups)==18
 fields=['gamma','dimensionless_rate','hbar_proxy_or_zero','finite_hbar','eta_effective','erase','visibility_TD','order_commutator_norm','local_choi_min','cumulative_choi_min','local_det','cumulative_sv1','cumulative_sv2','cumulative_sv3','state_x','state_y','state_z','state_Cz','pair_mass','lambda']
 np.savez_compressed(S/'QUANTUM101.npz',meta=M,values=V)
 dump(S/'QUANTUM101.json',dict(status='PASS_SCOPED',paths=3456,events=3456*12,fields=fields,groups=groups,record_rows=rr,record_cases=len(rr),raw_receipts=raw,source_hashes=src,checks=err,array_sha=sha(S/'QUANTUM101.npz'),erasure_singular=sum(r[3] for r in res),elapsed_s=time.monotonic()-t,scope='Actual history-conditioned programmed qubit maps; frozen unselective comparison, explicitly declared complex R95 joint records, no physical hbar or native coherent observation.'))
 print('QUANTUM',len(raw),V.shape,len(rr),err,flush=True)
if __name__=='__main__':main()
