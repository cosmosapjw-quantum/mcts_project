"""Finite model erasure of binary source-pair records, not CPU energy."""
from thermal66 import *
def bernH(p):
 p=np.asarray(p);return -np.sum(p*np.log(p),axis=-1)
def job(arg):
 i,d,K,n=arg;E0=np.asarray(d['energy']);pi,_=gibbs(E0);a=d['legal'].index(d['source'])
 # Choose other label from R65 prospectively registered source pair.
 b=d['legal'].index(d['source_b'])
 p0=pi[4,[a,b]];p0=p0/p0.sum();target=np.array([.05,.95])
 # state 0 = a; reset to b with probability .95. Energies at endpoints degenerate
 # outside the internally Gibbs path. Initial quench prepares Hamiltonian with
 # Gibbs law p0, not a fresh state.
 logit=np.linspace(np.log(p0[0]/p0[1]),np.log(target[0]/target[1]),K+1)
 p=np.stack([1/(1+np.exp(-logit)),1/(1+np.exp(logit))],-1)
 E=-np.log(p)/BETA
 x=sample(E,1.,n,6700000+i*16+2*int(np.log2(K)),False)
 y=sample(E,1.,n,6700001+i*16+2*int(np.log2(K)),True)
 seq,W,Q,lf,lr,err,bal=x;rev,WR,QR,lfr,lrr,er,br=y
 Wt=E[0,seq[:,0]]+W-E[-1,seq[:,-1]]
 Wtr=E[0,rev[:,0]]+WR-E[-1,rev[:,-1]]
 assert abs(Wt+Q).max()<1e-11 and abs(Wtr+QR).max()<1e-11
 sig=lf-lr;bound=float(bernH(p0)-bernH(target))
 diss=float(np.sum(p[:-1]*np.log(p[:-1]/p[1:])))
 mean_exact=float((bound+diss)/BETA)
 assert abs(exact(E,1.)['meanW']*BETA-diss)<1e-12
 ep=BETA*Wt+np.log(p0[seq[:,0]]/target[seq[:,-1]])
 assert abs(ep-sig).max()<1e-11
 return dict(generation=d['generation'],core=d['core'],steps=K,n=n,p0=p0.tolist(),target=target.tolist(),bound=bound,mean_work_exact=mean_exact,mean_work_sample=float(Wt.mean()),mean_heat_to_system=float(Q.mean()),mean_EP_exact=diss,mean_EP_sample=float(ep.mean()),jarzynski=weights_summary(-ep),reverse_jarzynski=weights_summary(lfr-lrr),final_error_rate=float((seq[:,-1]==0).mean()),mean_work_reverse=float(-Wtr.mean()),max_balance_error=max(float(abs(Wt+Q).max()),float(abs(Wtr+QR).max()))),seq,rev,Wt,Wtr
def main():
 ds=read(S/'THERMAL_SCHEDULES.json');reg=read(S/'REGISTRATION.json')
 for d in ds:d['source_b']=reg['cores'][d['core']]['sources'][1]
 n=2048;jobs=[(i,d,K,n) for i,d in enumerate(ds) for K in (1,4,16,64)]
 out=[];(S/'data/erasure').mkdir()
 with cf.ProcessPoolExecutor(max_workers=12) as pool:
  for (i,d,K,n),(r,seq,rev,W,WR) in zip(jobs,pool.map(job,jobs,chunksize=2)):
   p=S/'data/erasure'/f'g{d["generation"]}_c{d["core"]:02d}_k{K}.npz';np.savez_compressed(p,forward=seq,reverse_as_forward=rev,W=W,W_reverse=-WR)
   r.update(file=p.name,sha256=sha(p),bytes=p.stat().st_size);out.append(r)
 report=dict(scope='finite error erasure of a binary record conditioned on selected source pair from native pilot; prescribed degenerate memory, Gibbs quench then full refresh, final quench to zero; not measured device heat',paths=len(jobs)*n*2,transitions=sum(K*n*2 for _,_,K,n in jobs),results=out)
 dump(S/'ERASURE66.json',report)
 print('ERASURE',report['paths'],report['transitions'],max(x['max_balance_error'] for x in out))
if __name__=='__main__':main()
