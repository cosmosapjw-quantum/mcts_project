from scaling99 import *
from scipy.special import logsumexp
def main():
 q=read(S/'SUMMARY99.json');e=read(S/'INDEPENDENT99.json');err={};sources=np.array([-2.,-1.,0.,1.,2.]);data={};groups=[];n0=0
 for L in LS:
  p=R87/f'OBS_L{L}.npz';assert sha(p)==e['source_hashes'][str(p.relative_to(B))]
  with np.load(p) as a:
   M=a['meta'];out=[]
   for i,z in enumerate(a['A']):
    mask=z>0;p0=z[mask];n=len(p0);f=a['modes'][mask,:2].sum(1)/np.sqrt(2);f-=f.mean();j=p0.argmax();m=p0[j];sp=np.full(n,(1-m)/(n-1));sp[j]=m
    res=[]
    for p1 in (p0,sp):
     ll=np.log(p1);a0=ll[None]+sources[:,None]*f;W=logsumexp(a0,axis=1);t=np.exp(a0-W[:,None]);mu=t@f;var=np.sum(t*(f[None]-mu[:,None])**2,axis=1);gam=sources*mu-W
     direct=np.sum(t*np.log(t/p1),axis=1);ck(err,'action_KL_dual',gam,direct);assert var.min()>0 and gam.min()>-1e-10
     h=1e-4;plus=logsumexp(a0+h*f,axis=1);minus=logsumexp(a0-h*f,axis=1);ck(err,'action_first_derivative',(plus-minus)/(2*h),mu,1e-7);ck(err,'action_second_derivative',(plus+minus-2*W)/h**2,var,1e-6)
     ck(err,'action_zero',[W[2],gam[2]],[0,0]);assert np.min(np.diff(mu))>0
     res.append(np.stack([W,mu,var,gam],axis=1))
    out.append(res)
   V=np.array(out);data[f'M{L}']=M;data[f'V{L}']=V;n0+=len(M)
   for co,arm,bi,bud,h in sorted(set(tuple(v[[0,2,3,5,6]]) for v in M)):
    ii=np.all(M[:,[0,2,3,5,6]]==[co,arm,bi,bud,h],axis=1);root=[V[ii&(M[:,1]==k)].mean(0) for k in range(8)]
    groups.append(dict(L=L,cohort=int(co),arm=int(arm),bi=int(bi),budget=int(bud),h=int(h),ci=ci(root).tolist()))
 assert n0==10400 and len(groups)==400
 np.savez_compressed(S/'ACTION99.npz',**data)
 dump(S/'ACTION99.json',dict(status='PASS_SCOPED',source=sources.tolist(),fields=['W','mean','variance','Gamma'],distributions=['A','matched_peak'],evaluations=n0*2*5,groups=groups,array_sha=sha(S/'ACTION99.npz'),scope='Finite one-draw categorical source dual; not path-space or physical 1PI action. Extension after initial run.'))
 e['checks'].update(err);e['action_evaluations']=n0*2*5;dump(S/'INDEPENDENT99.json',e)
 print('ACTION',n0*2*5,err,flush=True)
if __name__=='__main__':main()
