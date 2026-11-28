"""Exact finite declared feedback work CGF: direct triples vs diagonal-plus-rank-one reduction."""
from base61 import *
from scipy.special import logsumexp
O=B/'mqft_restore85_20261127';beta=10/3
def gibbs(e):return np.exp(-beta*e-logsumexp(-beta*e,axis=-1,keepdims=True))
def one(c,old):
 e=np.array(c['e0']);em=np.array(c['em']);g=np.array(c['group']);n=len(e);m=len(em);p=gibbs(e);pi=gibbs(em);eps=old['eps'];alpha=old['alpha']
 M=np.full((n,m),eps/m);M[np.arange(n),g]+=1-eps;pm=p@M;assert min(pm)>0
 sources=np.array([-beta,-1.,0.,1.,beta]);Z=np.zeros(len(sources));ZZ=Z.copy();mom=np.zeros(2);moment2=mom.copy();ift=support=0.
 for k in range(m):
  d=em[k]-e;f=(p*M[:,k])[:,None]*((1-alpha)*np.eye(n)+alpha*pi[k][None,:]);w=d[:,None]-d[None,:]
  mom+=np.array([(f*w).sum(),(f*w*w).sum()])
  a=p*M[:,k];b=pi[k];moment2[0]+=alpha*((a@d)-(a.sum())*(b@d))
  moment2[1]+=alpha*(a@(d*d)-2*(a@d)*(b@d)+a.sum()*(b@(d*d)))
  for j,t in enumerate(sources):
   Z[j]+=np.sum(f*np.exp(t*w));ZZ[j]+=(1-alpha)*a.sum()+alpha*(a@np.exp(t*d))*(b@np.exp(-t*d))
  mask=M[:,k]>0
  # info-corrected exponential work and independent reverse support mass.
  ift+=np.sum(f[mask]*np.exp(-beta*w[mask])*(pm[k]/M[mask,k])[:,None])
  reverse=pm[k]*p[None,:]*((1-alpha)*np.eye(n)+alpha*pi[k][:,None]);support+=reverse[mask].sum()
 assert abs(Z[2]-1)<1e-10 and abs(ZZ[2]-1)<1e-10 and abs(Z-ZZ).max()<1e-8 and abs(mom-moment2).max()<1e-10
 assert abs(mom[0]-old['mean_work'])<1e-10 and abs(mom[1]-(old['mean_work']**2+old['work_sd']**2))<1e-10
 assert abs(Z[0]-old['uncorrected_ift'])<1e-10 and abs(ift-old['exact_ift'])<1e-10 and abs(ift-support)<1e-10
 return dict(protocol=c['protocol'],family=c['family'],core=c['core'],eps=eps,alpha=alpha,source=sources.tolist(),logMGF=np.log(Z).tolist(),mean=float(mom[0]),variance=float(mom[1]-mom[0]**2),ift=float(ift),reverse_support=float(support),uncorrected_ift=float(Z[0]),max_mgf_error=float(abs(Z-ZZ).max()),moment_error=float(abs(mom-moment2).max()))
def main():
 A=read(O/'BRANCH_ENERGY85.json');R=read(O/'THERMAL85.json');assert sha(O/'BRANCH_ENERGY85.json')==R['energy_sha']
 assert len(A['contexts'])==104 and len(R['cases'])==832
 look={(x['protocol'],x['family'],x['core']):x for x in A['contexts']};args=[(look[x['protocol'],x['family'],x['core']],x) for x in R['cases']]
 with cf.ProcessPoolExecutor(max_workers=6) as pool:
  out=list(pool.map(one,*zip(*args),chunksize=8))
 dump(S/'WORK90.json',dict(status='PASS_SCOPED',cases=out,contexts=len(look),raw_receipts=A['raw_receipts'],source_sha=sha(O/'THERMAL85.json'),max_mgf_error=max(x['max_mgf_error'] for x in out),max_moment_error=max(x['moment_error'] for x in out),scope='Declared finite cyclic feedback Gibbs/heat-bath reference. Work CGF at five sources via direct enumeration and rank-one formula. NOT native heat/work or physical calibrated temperature.'))
 print('WORK COMPLETE',len(out),max(x['max_mgf_error'] for x in out),max(x['moment_error'] for x in out),flush=True)
if __name__=='__main__':main()
