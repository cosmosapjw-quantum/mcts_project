from measure103 import *
def main():
 src={};err={};a=read(pin(79,'AUDIT79.json',src));old=read(pin(79,'CALIBRATION79.json',src));raw={r['path']:r['sha256'] for r in a['raw_receipts']};assert len(raw)==960
 with cf.ThreadPoolExecutor(max_workers=6) as pool:assert dict(pool.map(rehash,raw.items()))==raw
 out=[];total=0;det=0;flat=0
 for family in ['g0','g3','Rapfi']:
  p=pin(79,f'CALIB_{family}.npz',src);assert sha(p)==a['arrays'][p.name]
  with np.load(p) as z:
   m=z['meta'];mask=z['mask'];sc=z['score'];q=z['Q'];P=z['p'];Y=z['y'];occ=z['occ'];Qend=z['Qcut']
  oo=next(x for x in old['results'] if x['family']==family);look={(x['core'],x['rep'],x['B']):x for x in oo['occupancy_fits']}
  vals=[];sl=[];scs=[]
  for i,meta in enumerate(m):
   mm=mask[i];s=sc[i][:,mm];p=P[i][:,mm];x=s-s.mean(1,keepdims=True);v=np.log(p);v-=v.mean(1,keepdims=True);ok=np.ptp(s,axis=1)>1e-12
   b=(x[ok]*v[ok]).sum(1)/(x[ok]**2).sum(1);ck(err,'R79_slope',b,10/3);det+=int(ok.sum());flat+=int((~ok).sum())
   for j,h in enumerate([4,8,16,32,64,128]):
    y=occ[i,j,mm];s=Qend[i,j,mm];vv,qq=fit(s,y,10/3,err,True)
    if meta[0]==1:
     r=look[int(meta[1]),int(meta[2]),h]
     if r['beta'] is not None:
      oldp=prob(s,r['beta']);ck(err,'R79_projection_probability',qq,oldp,1e-8);ck(err,'R79_projection_KL',vv[3],r['occupancy_KL'],1e-8)
      err['R79_beta_difference']=max(err.get('R79_beta_difference',0.),abs(vv[0]-r['beta']))
     else:assert int(vv[1])==0
    vals.append([int(meta[0]),int(meta[1]),int(meta[2]),h,*vv.tolist()]);total+=1
   if meta[0]==1:
    s=sc[i][:,mm];p=P[i][:,mm];y=Y[i][:,mm];nll=[];kl=[]
    for model in ('score','Q'):
     rr=next(x for x in oo['models'] if x['model']==model);xx=s if model=='score' else q[i][:,mm];logq=rr['beta']*xx;logq-=logsumexp(logq,axis=1)[:,None];nll.append(float(-np.mean(np.sum(y*logq,1))));kl.append(float(np.mean(np.sum(p*(np.log(p)-logq),1))))
    scs.append([*meta[1:].tolist(),*nll,*kl])
  v=np.array(vals);ss=np.array(scs);means=[]
  for k,mod in enumerate(('score','Q')):
   rr=next(x for x in oo['models'] if x['model']==mod);nl=ss[:,2+k].mean();kl=ss[:,4+k].mean();ck(err,'R79_heldout_nll',nl,rr['nll'][-1][0]);ck(err,'R79_heldout_KL',kl,rr['conditional_KL'][-1][0]);means.append(dict(model=mod,NLL=nl,KL=kl,beta=rr['beta']))
  out.append(dict(family=family,projection=v.tolist(),heldout=means,paths=len(m)))
 assert total==5760 and det+flat==30720
 dump(S/'PRIOR103.json',dict(status='PASS_SCOPED',paths=len(raw),waves=30720,identified=det,flat=flat,projections=total,families=out,raw_receipts=raw,source_hashes=src,checks=err,scope='Independent bisection on R79 arrays, existing heldout score/Q models re-evaluated, not new native experiment.'))
 print('PRIOR',total,det,flat,err,flush=True)
if __name__=='__main__':main()
