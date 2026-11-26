"""Historical Rapfi transfer of exact local frozen iid law; not trained-size scaling."""
from base61 import *
from scipy.special import gammaln
import gzip,itertools
R=B/'mqft_restore58_20261126'
def main():
 results=[];receipts=[]
 for L in (13,15,17,19,21):
  data=[]
  for bi in range(3):
   d=R/'data'/f'B32b{bi}-L{L}';an=R/'analysis'/d.name
   rep=read(next(d.glob('report-*.json')));man=read(an/'manifest.json');reg=read(R/f'REGISTRATION_L{L}.json');lock=read(d/'run_lock.json')
   assert len(rep['completed'])==12==man['completed']==man['expected'] and not rep['failures'] and not rep['unattempted']
   assert lock['backend_sha']==reg['backend_sha'] and lock['external']==reg['model'] and lock['plan_hash']==man['plan_hash']
   for rec in rep['completed']:
    p=d/rec['file'];assert sha(p)==rec['sha256'] and p.stat().st_size==rec['bytes'] and read(d/(rec['job_id']+'.receipt.json'))==rec
    with gzip.open(p,'rt') as f:
     hd=json.loads(next(f));rr=json.loads(next(f));end=json.loads(next(f));assert not f.read().strip()
    assert hd['backend_sha']==reg['backend_sha'] and hd['evaluator_runtime']==reg['model'] and end['backups']==32
    t=rr['trace'];root=t['root'];spec=t['spec'];assert root['board_size']==L and spec['budget']==32 and spec['mode']=='wave' and spec['workers']==4
    assert spec['root_law']['kind']=='iid' and spec['root_law']['logit_h']==0.
    ev=sorted(t['replicas'][0]['events'],key=lambda e:e['root_selection']['clock']);assert len(ev)==32
    legal=set(root['legal']);a=spec['source_action'];assert a in legal
    b=min(legal-{a},key=lambda b:((b//L-L//2)**2+(b%L-L//2)**2,b))
    C=np.zeros(3);res=[];covres=[];nll=[];occ=[];maxerr=0.;counts=collections.Counter();sums=collections.defaultdict(float);s2=collections.defaultdict(float)
    phases=[list(v) for _,v in itertools.groupby(ev,key=lambda e:e['phase'])];assert sum(map(len,phases))==32 and all(1<=len(v)<=4 for v in phases)
    for w in phases:
     width=len(w)
     pr=None;y=np.zeros(3)
     for e in w:
      r=e['root_selection'];law=r['law'];assert set(law['actions'])==legal and all(law['active'])
      p0=np.array(law['p']);sc=np.array(law['scores']);z=law['beta_p']*sc;z-=z.max();q=np.exp(z);q/=q.sum()
      maxerr=max(maxerr,float(abs(q-p0).max()));assert abs(q-p0).max()<2e-12
      if pr is None:
       ma=p0[law['actions'].index(a)];mb=p0[law['actions'].index(b)];pr=np.array([ma,mb,1-ma-mb])
      else:
       assert abs(pr[0]-p0[law['actions'].index(a)])<1e-13 and abs(pr[1]-p0[law['actions'].index(b)])<1e-13
      act=e['actions'][0];assert law['actions'][law['picked']]==act and e['root_value']==(-1)**len(e['actions'])*e['leaf_value']
      y[0 if act==a else 1 if act==b else 2]+=1;counts[act]+=1;sums[act]+=e['root_value'];s2[act]+=e['root_value']**2
     pmf=lambda q:gammaln(width+1)-gammaln(y+1).sum()+float(y@np.log(q))
     e=y-width*pr;v=width*(np.diag(pr)-np.outer(pr,pr));res.append(e);covres.append(np.outer(e,e)-v)
     nll.append(-pmf(pr));occ.append(-pmf((C+1)/(C.sum()+3)));C+=y
    for ed in t['replicas'][0]['cuts'][-1]['edges']:
     ac=ed['action'];assert counts[ac]==ed['n'] and abs(sums[ac]-ed['w'])<1e-8 and abs(s2[ac]-ed['s2'])<1e-8
    data.append(dict(waves=len(phases),beta=spec['root_law']['beta_p'][0],root=root['state_hash'],mean=np.mean(res,0).tolist(),cov=np.mean(covres,0).tolist(),nll=float(np.mean(nll)),occupancy_nll=float(np.mean(occ)),max_p_error=maxerr))
    receipts.append(dict(L=L,folder=d.name,job=rec['job_id'],sha256=rec['sha256'],bytes=rec['bytes']))
  assert len(data)==36
  results.append(dict(L=L,paths=len(data),waves=sum(x["waves"] for x in data),backups=len(data)*32,mean_nll=float(np.mean([x['nll'] for x in data])),occupancy_nll=float(np.mean([x['occupancy_nll'] for x in data])),mean_residual=np.mean([x['mean'] for x in data],0).tolist(),mean_covariance_residual=np.mean([x['cov'] for x in data],0).tolist(),max_p_error=max(x['max_p_error'] for x in data),rows=data))
 dump(S/'RAPFI_TRANSFER63.json',dict(scope='R58 historical Rapfi 13–21 native wave4 B32, three beta conditions; exact frozen count law with a=registered source and b=deterministic nearest-center legal alternative; not trained checkpoint scaling; correlated geometries and seeds',results=results,receipts=receipts,n_paths=len(receipts)))
 print('RAPFI',[(x['L'],x['paths'],x['mean_nll'],x['occupancy_nll'],x['mean_residual']) for x in results])
if __name__=='__main__':main()
