from markov102 import *
import gzip
from math import comb
LS=[13,15,17,19,21];LAGS=[1,2,4,8,16]
FIELDS=['raw_cov_root_clock','raw_rho_root_clock','clock_mix_cov','pooled_cov','innovation_raw_cross','innovation_cov_root_clock','raw_pair','pred_pair_product','mark_cov_root_clock','var_start','var_end']
def raw_one(arg):
 L,r=arg;f=B/r['path'];assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256']
 with gzip.open(f,'rt') as z:hd,row,end=map(json.loads,z)
 tr=row['trace'];sp=tr['spec'];h=r['budget'];beta=sp['root_law']['beta_p'][0];assert sp['budget']==h==end['backups'] and sp['seed']==r['seed'] and beta==[0.,10/3,8.,16.,math.log(L*L)][r['beta_index']]
 es=sorted(tr['replicas'][0]['events'],key=lambda e:e['root_selection']['clock']);assert len(es)==h
 legal=np.flatnonzero(np.array(tr['root']['board'])==0);yy,xx=np.divmod(legal,L);sec=np.digitize(np.maximum(abs(xx-L//2),abs(yy-L//2)),[2,4]);p=[];a=[];mark=[];err={}
 for j,e in enumerate(es):
  law=e['root_selection']['law'];ac=np.array(law['actions']);o=np.argsort(ac);q=np.array(law['p'])[o];s=np.array(law['scores'])[o];assert np.array_equal(ac[o],legal) and law['t']==j and all(law['active']) and law['logit_h']==0 and ac[law['picked']]==e['actions'][0]
  z=beta*s;z-=z.max();v=np.exp(z);v/=v.sum();ck(err,'raw_probability',q,v);ck(err,'normalization',q.sum(),1);assert q.min()>0
  pp=np.bincount(sec,weights=q,minlength=3);p.append(pp);a.append(sec[np.searchsorted(legal,e['actions'][0])]);mark.append(e['root_value']);assert e['root_value']==(-1)**len(e['actions'])*e['leaf_value']
 p=np.array(p);a=np.array(a);mark=np.array(mark);ck(err,'sector_norm',p.sum(1),1)
 ck(err,'startup_pair',p[2],p[3]);ck(err,'quiescent_wave',p[4:].reshape(-1,4,3),np.broadcast_to(p[4::4,None],(h//4-1,4,3)))
 if beta==0:ck(err,'uniform',p,np.broadcast_to(np.bincount(sec,minlength=3)/len(legal),p.shape))
 # exact conditional multinomial inner count, no first-four as a single wave.
 q=p[4::4,0];xx0=(a[4:]==0).reshape(-1,4);cnt=xx0.sum(1);pm=np.array([comb(4,k)*q**k*(1-q)**(4-k) for k in range(5)]).T;ck(err,'binomial_norm',pm.sum(1),1);ck(err,'binomial_mean',pm@np.arange(5),4*q);ck(err,'binomial_factorial',pm@(np.arange(5)*(np.arange(5)-1)),12*q*q)
 # analytic pair expectation conditional on wave law; empirical pairs kept separately.
 actual=cnt*(cnt-1)/12;pair=q*q;var=4*q*(1-q);score=-np.log(pm[np.arange(len(q)),cnt]);brier=((pm-np.eye(5)[cnt])**2).sum(1)
 d=(a==0)-p[:,0];defect=float(np.sum((p[:4]-p[:4].mean(0))**2));trace=float(np.sum(1-np.sum(p[:4]**2,1)));bad=4*(1-np.sum(p[:4].mean(0)**2));ck(err,'startup_trace_defect',bad-trace,defect);assert defect>=0
 if beta==0:ck(err,'uniform_startup_defect',defect,0)
 # per-wave predicted variance vs actual centered count residual, not forcing equality.
 v=np.column_stack([cnt,4*q,actual,pair,actual-pair,(cnt-4*q)**2,var,score,brier])
 return [r[k] for k in ['cohort','core','arm','beta_index','rep','budget']],p,a,mark,v,pm,cnt,defect,err
def rootlag(x,p,z,lag,e):
 # R x H; population covariance estimators across independent reps at each clock.
 R,H=x.shape;A=x[:,:-lag];B0=x[:,lag:];D0=(x-p)[:,:-lag];D1=(x-p)[:,lag:];A0=A-A.mean(0);B1=B0-B0.mean(0);u=D0-D0.mean(0);v=D1-D1.mean(0);va=np.mean(A0*A0)*R/(R-1);vb=np.mean(B1*B1)*R/(R-1);cv=np.mean(A0*B1)*R/(R-1);rho=cv/np.sqrt(va*vb) if va*vb>1e-16 else 0.
 # empirical pooled covariance decomposition (biased 1/R denominator); clock component separate.
 ac=A.mean(0);bc=B0.mean(0);clock=np.mean((ac-ac.mean())*(bc-bc.mean()));pooled=np.mean((A-A.mean())*(B0-B0.mean()));ck(e,'total_covariance',pooled,cv*(R-1)/R+clock)
 # independent covariance from all pairs of distinct replicates.
 ai=np.einsum('rt,st->t',A,B0)-np.sum(A*B0,axis=0);ind=np.mean(np.mean(A*B0,0)-ai/(R*(R-1)));ck(e,'independent_covariance',cv,ind)
 c=z[:,:-lag];b=z[:,lag:];mc=np.mean((c-c.mean(0))*(b-b.mean(0)))*R/(R-1)
 return [cv,rho,clock,pooled,float(np.mean(D0*D1)),float(np.mean(u*v)*R/(R-1)),float(np.mean(A*B0)),float(np.mean(p[:,:-lag]*p[:,lag:])),mc,va,vb]
def main():
 start=time.monotonic();src={};err={};m=read(B/'mqft_restore92_20261128'/'RUN_MANIFEST92.json');raw=m['prior_raw_receipts'];assert len(raw)==4320
 groups=[];recs={};arr={};total=0;waves=0;startup=0;strata=0
 for L in LS:
  ap=R87/f'AUDIT_L{L}.json';k=str(ap.relative_to(B));assert sha(ap)==m['inputs'][k];src[k]=sha(ap);au=read(ap);rs=au['raw_receipts'];assert len(rs)==864
  p0=R87/f'SECTOR_L{L}.npz';assert sha(p0)==au['sector_sha'];src[str(p0.relative_to(B))]=sha(p0)
  with np.load(p0) as z:oldm=z['meta'];oldp=z['p'];olda=z['selected'];lookup={tuple(x):i for i,x in enumerate(oldm)}
  with cf.ProcessPoolExecutor(max_workers=10) as pool:out=list(pool.map(raw_one,[(L,r) for r in rs],chunksize=4))
  M=np.array([x[0] for x in out]);defects=np.array([x[7] for x in out]);total+=sum(len(x[1]) for x in out);waves+=sum(len(x[4]) for x in out)
  for r,x in zip(rs,out):
   assert r['path'] not in recs and raw[r['path']]==r['sha256'];recs[r['path']]=r['sha256']
   for k,v in x[8].items():err[k]=max(err.get(k,0.),v)
   if r['budget']==64:
    i=lookup[tuple(x[0][:-1])];ck(err,'R87_sector_prob',x[1],oldp[i]);assert np.array_equal(x[2],olda[i])
    if r['cohort']<2 and r['beta_index'] in (1,3):startup=max(startup,x[7])
  for co,arm,bi,budget in sorted(set(tuple(x[[0,2,3,5]]) for x in M)):
   ii=(M[:,0]==co)&(M[:,2]==arm)&(M[:,3]==bi)&(M[:,5]==budget);ids=np.flatnonzero(ii);roots=[];lags=[];wvals=[];bins=[];hist=[]
   for core in range(8):
    ix=ids[M[ids,1]==core];assert len(ix)==(2 if co==2 else 4)
    P=np.stack([out[i][1] for i in ix]);A=np.stack([out[i][2] for i in ix]);Z=np.stack([out[i][3] for i in ix]);X=(A==0).astype(float);p=P[:,:,0]
    ls=[lag for lag in LAGS if lag<budget];lags.append([rootlag(X,p,Z,lag,err) for lag in ls]);V=np.stack([out[i][4] for i in ix]);wvals.append(V.mean((0,1)))
    pm=np.stack([out[i][5] for i in ix]);cnt=np.stack([out[i][6] for i in ix]);bins.append(np.stack([pm.mean((0,1)),np.eye(5)[cnt].mean((0,1))]));hist.append(defects[ix].mean());roots.append(core)
   rho=np.asarray(lags);mask=(rho[:,:,9]*rho[:,:,10]>1e-16);rho_ci=[]
   for j in range(len(ls)):
    v0=rho[mask[:,j],j,1];rho_ci.append(ci(v0).tolist() if len(v0)>1 else ([float(v0[0])]*3 if len(v0)==1 else None))
   groups.append(dict(rho_valid_roots=mask.sum(0).tolist(),rho_valid_ci=rho_ci,L=L,cohort=int(co),arm=int(arm),bi=int(bi),beta=[0.,10/3,8.,16.,math.log(L*L)][bi],budget=int(budget),roots=8,reps=int(ii.sum()/8),lags=ls,lag_ci=ci(lags).tolist(),lag_root=lags,wave_ci=ci(wvals).tolist(),wave_root=np.asarray(wvals).tolist(),bins_ci=ci(bins).tolist(),startup_defect_ci=ci(hist).tolist()));strata+=1
  # preserve variable budgets without padded fake observations
  for h in [16,64]:
   ids=np.flatnonzero(M[:,5]==h)
   arr[f'M{L}_{h}']=M[ids];arr[f'P{L}_{h}']=np.stack([out[i][1] for i in ids]);arr[f'A{L}_{h}']=np.stack([out[i][2] for i in ids]);arr[f'Z{L}_{h}']=np.stack([out[i][3] for i in ids]);arr[f'V{L}_{h}']=np.stack([out[i][4] for i in ids]);arr[f'D{L}_{h}']=defects[ids]
  print('LAG L',L,len(rs),total,waves,flush=True)
 assert recs==raw and total==215040 and waves==(215040-4320*4)//4 and len(groups)==160
 prior=read(B/'mqft_restore87_20261128'/'ERRATUM86.json');ck(err,'R87_startup_max',startup,prior['max_startup_trace_overestimate'])
 np.savez_compressed(S/'SELECTION102.npz',**arr)
 dump(S/'LAG102.json',dict(status='PASS_SCOPED',paths=len(raw),draws=total,quiescent_waves=waves,strata=strata,fields=FIELDS,wave_fields=['count','expected_count','pair_observed','pair_predicted','pair_residual','count_residual_square','conditional_variance','NLL','Brier'],groups=groups,checks=err,raw_receipts=raw,source_hashes=src,array_sha=sha(S/'SELECTION102.npz'),elapsed_s=time.monotonic()-start,scope='Inner Chebyshev sector; root-clock covariance across independent histories, raw innovation lag product; true quiescent waves exclude startup; not stationary physical ACF.'))
 print('LAG DONE',total,waves,strata,err,flush=True)
if __name__=='__main__':main()
