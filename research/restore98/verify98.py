from rg98 import *
import itertools
def check_prefix(L,arg):
 x,a,v=arg;mask=a>0;legal=np.flatnonzero(mask);x=x[mask];a=a[mask];n=len(a);err={};qprev=uprev=tprev=phprev=it=None;idprev=None
 y,xx=np.divmod(legal,L)
 # Independent direct block index and weighted linear regression including an intercept.
 rawF=np.c_[np.cos(np.pi*(xx+.5)/L),np.cos(np.pi*(y+.5)/L)]
 for j,b in enumerate(BS):
  side=(L+b-1)//b;idx=(y//b)*side+xx//b
  mass=np.bincount(idx,minlength=side*side);ok=mass>0;u=mass[ok]/n
  q=np.bincount(idx,weights=a,minlength=len(mass))[ok];z=np.bincount(idx,weights=x,minlength=len(mass))[ok];f=np.stack([np.bincount(idx,weights=rawF[:,k],minlength=len(mass))[ok]/mass[ok] for k in range(2)],axis=1)
  f-=u@f;g=np.log(q/u);A=np.sqrt(u)[:,None]*np.c_[np.ones(len(u)),f]
  cf,_,rank,_=np.linalg.lstsq(A,np.sqrt(u)*g,rcond=1e-10);th=cf[1:];r=np.exp(f@th);ph=u*r/(u@r);a0=v[j]
  ck(err,'prefix_KL',float(kl(q,u)),a0[0]);ck(err,'prefix_X_KL',float(kl(z,u)),a0[1]);ck(err,'projection_theta',th,a0[8:10]);ck(err,'projection_rank',rank-1,a0[10]);ck(err,'projection_TV',tv(q,ph),a0[11]);ck(err,'prefix_entropy',[ent(q),ent(z)],a0[19:21])
  inv=np.searchsorted(np.flatnonzero(ok),idx)
  if j==0:it=ph
  else:
   # Expand previous block distribution to legal cells by conditional reference, then independently sum.
   def psh(prob):return np.bincount(inv,weights=prob[idprev]/(n*uprev[idprev]),minlength=len(u))
   p0=psh(phprev);q0=psh(qprev);ck(err,'nested_lift_push',q0,q)
   tp=np.linalg.lstsq(A,np.sqrt(u)*np.log(p0/u),rcond=1e-10)[0][1:]
   ck(err,'push_projection_TV',tv(p0,q),v[j-1,12]);ck(err,'theta_next_error',np.linalg.norm(tp-th),v[j-1,13]);ck(err,'beta_theta',(th-tprev)/np.log(2),v[j-1,22:24])
   field=(g[inv]-np.log(qprev/uprev)[idprev])/np.log(2)
   ck(err,'beta_field_rms',np.sqrt(np.mean(field**2)),v[j-1,6]);ck(err,'beta_field_mean',np.mean(field),v[j-1,7]);ck(err,'KL_step',kl(qprev,uprev)-kl(q,u),v[j-1,4])
   ii=psh(it);tt=np.linalg.lstsq(A,np.sqrt(u)*np.log(ii/u),rcond=1e-10)[0][1:];w=np.exp(f@tt);it=u*w/(u@w)
  ck(err,'iterated_TV',tv(q,it),a0[14]);qprev,uprev,tprev,phprev,idprev=q,u,th,ph,inv
 return err
def worker(arg):
 L,x,a,v=arg;return check_prefix(L,(x,a,v))
def main():
 u=read(S/'RG98.json');assert sha(S/'RG98.npz')==u['array_sha'];r=read(S/'INDEPENDENT98.json');source=r['source_hashes'];err={};total=0
 with np.load(S/'RG98.npz') as data:
  for L in LS:
   p=R87/f'OBS_L{L}.npz';assert sha(p)==source[str(p.relative_to(B))]
   with np.load(p) as obs:
    M=data[f'M{L}'];V=data[f'V{L}'];assert np.array_equal(M,obs['meta']);xs=obs['X'];aa=obs['A']
    with cf.ProcessPoolExecutor(max_workers=10) as pool:
     for e in pool.map(worker,[(L,x,a,v) for x,a,v in zip(xs,aa,V)],chunksize=8):
      for k,v in e.items():err[k]=max(err.get(k,0.),v)
      total+=1
    print('VERIFY',L,total,flush=True)
 assert total==10400
 r['independent_prefix_checks']=total;r['checks'].update({'independent_'+k:v for k,v in err.items()})
 r['independence']+='; separate weighted least-squares with intercept, direct original full-array bincount and lifted conditional reference push, all prefix rows'
 src=read(S/'ORIGINAL_SOURCE98.json');assert len(src['items'])==5 and len(src['helpers'])==2
 dump(S/'INDEPENDENT98.json',r);print('INDEPENDENT',err,flush=True)
if __name__=='__main__':main()
