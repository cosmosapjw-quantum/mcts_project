from field101 import *
from itertools import product
def main():
 f=read(S/'FIELD101.json');q=read(S/'QUANTUM101.json');err={};src=f['source_hashes']|q['source_hashes'];raw=f['raw_receipts']|q['raw_receipts'];assert len(raw)==6408 and not set(f['raw_receipts'])&set(q['raw_receipts'])
 assert all(sha(B/k)==h for k,h in src.items()) and sha(S/'QUANTUM101.npz')==q['array_sha']
 with np.load(R74/'PULSE74.npz') as a:
  M=a['meta'];F=a['features'];lookup={tuple(x):i for i,x in enumerate(M)};assert len(lookup)==2304
 count=0;coarse=[]
 # Separate explicit long-double summation, fixed block averaging; exact source restriction.
 C=np.array([[.5,0,.5,0],[0,.5,0,.5]],dtype=np.longdouble)
 for row in f['rows']:
  g,c=row['family'],row['core'];v=F[[lookup[g,c,r,0] for r in range(16)]].astype(np.longdouble)
  O=np.c_[v[:,:8,:2].mean(1),v[:,8:16,:2].mean(1)];k=4*v[:,8,0];p=v[:,8,2];co=O@C.T;rr=[]
  for a in row['values']:
   h=np.longdouble(a['h']);J=np.asarray(a['J'],np.longdouble);lr=h*k-4*np.log(1+p*np.expm1(h));base=np.exp(lr);base/=base.sum();wt=base*np.exp(O@J);Z=wt.sum();wt/=Z;mu=wt@O;H=np.einsum('n,ni,nj->ij',wt,O,O)-np.outer(mu,mu)
   score=k-4*p*np.exp(h)/(1+p*np.expm1(h));resp=(wt*score)@O-(wt@score)*mu
   ck(err,'independent_W',np.log(Z),a['W']);ck(err,'independent_gradient',mu,a['mean']);ck(err,'independent_cov',H,a['Hessian']);ck(err,'independent_response',resp,a['response']);ck(err,'independent_dual',J@mu-np.log(Z),a['dual']);count+=1
   if a['h']==0 and not any(a['J']):
    for j in ([1.,0.],[0.,1.],[1.,-1.],[-1.,1.]):
     j=np.array(j,np.longdouble);ww=base*np.exp(co@j);zz=ww.sum();ww/=zz;mm=ww@co;hh=np.einsum('n,ni,nj->ij',ww,co,co)-np.outer(mm,mm)
     ff=base*np.exp(O@(C.T@j));z=ff.sum();ff/=z;mean=ff@O;cov=np.einsum('n,ni,nj->ij',ff,O,O)-np.outer(mean,mean)
     ck(err,'source_restriction_W',np.log(z),np.log(zz));ck(err,'source_restriction_mean',C@mean,mm);ck(err,'source_restriction_Hessian',C@cov@C.T,hh);ck(err,'source_restriction_dual',j@mm-np.log(zz),(C.T@j)@mean-np.log(z))
     rr.append(dict(J=j.astype(float).tolist(),W=float(np.log(zz)),dual=float(j@mm-np.log(zz))))
  coarse.append(dict(family=g,core=c,values=rr))
 assert count==2160 and len(coarse)==48 and sum(len(x['values']) for x in coarse)==192
 # Exact rational 2-wave causal-vs-correlation control.
 from fractions import Fraction as F0
 probs=[];vals=[]
 for a,b in product([0,1],repeat=2):
  p=F0(1,5) if a==0 else F0(4,5);probs.append(F0(1,2)*(p if b else 1-p));vals.append((F0(a),F0(b),F0(b)-p))
 E=lambda fn:sum(w*fn(*v) for w,v in zip(probs,vals))
 cov=E(lambda a,b,s:a*b)-E(lambda a,b,s:a)*E(lambda a,b,s:b);resp=E(lambda a,b,s:a*s)-E(lambda a,b,s:a)*E(lambda a,b,s:s)
 assert cov==F0(3,20) and resp==0
 r=q['record_rows'];assert len(r)==5184
 with np.load(S/'QUANTUM101.npz') as z:
  V=z['values'];M=z['meta'];valid=V[:,:,3]>0;assert V.shape==(3456,12,20)
  h=V[:,:,2];ck(err,'finite_visibility_reconstruction',np.cos(1/h[valid]),V[:,:,0][valid]);assert np.all(h[valid]>=2/np.pi)
  counts=dict(local_events=valid.size,finite_hbar=int(valid.sum()),singular_hbar=int(valid.size-valid.sum()),gamma_min=float(V[:,:,0].min()),finite_hbar_range=[float(h[valid].min()),float(h[valid].max())],max_visibility_TD=float(V[:,:,6].max()),noncommuting_events=int((V[:,:,7]>1e-10).sum()),max_commutator=float(V[:,:,7].max()),min_Choi=V[:,:,8:10].min((0,1)).tolist(),erase_singular=q['erasure_singular'],system_record_NPT=sum(a['N']>1e-10 for a in r),phase_change=sum(a['phase_TV']>1e-10 for a in r),max_phase_TV=max(a['phase_TV'] for a in r),max_weighted_phase_TV=max(a['weighted_phase_TV'] for a in r))
 # All recorded phase-tag cases: PSD complex Gram via square-root factor and local phase, independent amplitude check.
 with np.load(R95/'JOINT95.npz') as z:
  ix={(int(m[0]),int(m[1]),str(m[2]),str(m[3])):i for i,m in enumerate(z['meta'])}
  for a in r:
   d=a['depth'];i=ix[a['family'],a['core'],a['protocol'],a['mode']];rho=z[f'J{d}'][i];nr=2**d;b=np.array([[(j>>(d-1-k))&1 for k in range(d)] for j in range(nr)]);dist=np.sum(b[:,None]!=b[None,:],-1);rho=rho*np.repeat(np.repeat(a['xi']**dist,2,0),2,1)
   ev,U=np.linalg.eigh(rho);A=U*np.sqrt(np.maximum(ev,0));ck(err,'complex_Gram_amplitude',A@A.conj().T,rho);ck(err,'Gram_norm',np.sum(abs(A)**2),1)
   phase=np.repeat(np.exp(.5j*np.pi*b[:,0]),2);A1=phase[:,None]*A
   ck(err,'Gram_same_diag',np.sum(abs(A)**2,1),np.sum(abs(A1)**2,1))
   H=np.array([[1,1],[1,-1]])/np.sqrt(2);HH=H
   for j in range(d-1):HH=np.kron(HH,H)
   U=np.kron(HH,np.eye(2));p0=np.sum(abs(U@A)**2,1).reshape(nr,2).sum(1);p1=np.sum(abs(U@A1)**2,1).reshape(nr,2).sum(1)
   ck(err,'Gram_readout',.5*abs(p0-p1).sum(),a['phase_TV'])
 rep=[]
 for protocol in ['native_repeat','pair_diagnostic']:
  for xi0 in [0.,.5,1.]:
   a=[x for x in r if x['protocol']==protocol and x['xi']==xi0 and x['depth']==3 and x['mode']=='plus81'];assert len(a)==72
   rep.append(dict(protocol=protocol,xi=xi0,n=len(a),NPT=sum(x['N']>1e-10 for x in a),mean_phase_TV=float(np.mean([x['phase_TV'] for x in a])),mean_weighted_phase_TV=float(np.mean([x['weighted_phase_TV'] for x in a]))))
 dump(S/'SUMMARY101.json',dict(status='PASS_SCOPED',field_groups=f['groups'],ranks=f['rank_counts'],counts=counts,record_representative=rep,source_restriction=coarse,exact_control=dict(correlation=str(cov),response=str(resp)),raw_paths=len(raw),scope='Exact finite empirical source restriction and independent phase amplitude readout; no physical hbar or native coherent readout.'))
 checks={**{'field_'+k:v for k,v in f['checks'].items()},**{'quantum_'+k:v for k,v in q['checks'].items()},**err}
 dump(S/'INDEPENDENT101.json',dict(status='PASS_SCOPED',checks=checks,raw_receipts=raw,source_hashes=src,software=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),path_evaluations=count,local_events=counts['local_events'],record_cases=len(r),coarse_source_checks=192,independence='Long-double explicit weighted sums, exact rational response control, local Choi via Pauli and Kraus, full record Gram-amplitude readout vs density matrix, independent PT index and prior R95; same agent.'))
 print('INDEPENDENT',err,flush=True);print('SUMMARY',counts,rep,flush=True)
if __name__=='__main__':main()
