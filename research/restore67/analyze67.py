from base61 import *
def main():
 a=read(S/'AUDIT67.json');reg=read(S/'REGISTRATION.json');ext=read(S/'RAPFI67.json');assert sha(S/'NATIVE67.npz')==a['array_sha'] and sha(S/'REFERENCE67.npz')==a['reference_sha']
 with np.load(S/'NATIVE67.npz') as f:z={k:f[k] for k in f.files}
 with np.load(S/'REFERENCE67.npz') as f:r={k:f[k] for k in f.files}
 m=z['meta'];arms=list(z['arms']);fields=list(z['fields']);boot=np.random.default_rng(670301).integers(0,32,(10000,32))
 def stat(x):
  x=np.asarray(x,float);assert len(x)==32
  return dict(mean=x.mean(0).tolist(),ci95=np.quantile(x[boot].mean(1),[.025,.975],axis=0).tolist())
 def ratio(x,n):
  x=np.asarray(x);n=np.asarray(n);assert x.shape==n.shape==(32,);den=n[boot].sum(1);ok=den>0
  if n.sum()==0:return dict(mean=None,ci95=None,n=0)
  return dict(mean=float(x.sum()/n.sum()),ci95=np.quantile(x[boot].sum(1)[ok]/den[ok],[.025,.975]).tolist(),n=int(n.sum()))
 assert np.array_equal(r['meta'],np.array([[i,j] for i in range(32) for j in range(4)]))
 RN=r['counts'].reshape(32,4,N);RW=r['marks'].reshape(32,4,N);n=RN.sum(1);w=RW.sum(1);q=np.divide(w,n,out=np.zeros_like(w),where=n>0)
 assert np.all(RN.sum(2)==1024);ref=n/4096;top=np.argsort(-n,axis=1,kind='stable')[:,:5]
 h1=RN[:,:2].sum(1);h2=RN[:,2:].sum(1)
 htop=lambda x:np.argsort(-x,axis=1,kind='stable')[:,:5]
 stability=dict(top1=stat((h1.argmax(1)==h2.argmax(1)).astype(float)),top5_overlap=stat([len(set(x)&set(y))/5 for x,y in zip(htop(h1),htop(h2))]),policy_TV=stat(abs(h1/2048-h2/2048).sum(1)/2))
 extrows=ext['rows'];assert len(extrows)==96
 E={}
 for L,ms in ((9,150),(9,350),(15,350)):
  er=sorted([x for x in extrows if x['L']==L and x['ms']==ms],key=lambda x:x['core'])
  assert [x['core'] for x in er]==list(range(32))
  ee=np.array([x['result']['action'] if x['status']=='ok' else -1 for x in er])
  if L==15:
   yy=ee//15-3;xx=ee%15-3;ee=np.where((ee>=0)&(yy>=0)&(yy<9)&(xx>=0)&(xx<9),yy*9+xx,-1)
  E[L,ms]=ee
 st=(E[9,150]>=0)&(E[9,350]>=0);assert st.all()
 external=dict(valid_9x9_150=int(sum(E[9,150]>=0)),valid_9x9_350=int(sum(E[9,350]>=0)),same_board_time_agreement=stat((E[9,150]==E[9,350]).astype(float)),embedded_15x15_in_9_window=int(sum(E[15,350]>=0)),embedded_scope='different board; no same-game interpretation',external_vs_g3_top5=stat((E[9,350,None] if False else E[9,350][:,None]==top).any(1).astype(float)))
 rows=[];contrasts=[];by={}
 for g in (0,3):
  for arm in arms:
   ix=np.flatnonzero((m[:,0]==g)&(m[:,3]==arms.index(arm)));ix=ix[np.lexsort((m[ix,2],m[ix,1]))];assert len(ix)==256 and np.array_equal(m[ix,1],np.repeat(np.arange(32),8))
   f=z['faults'][ix].reshape(32,8,8);A=z['rows'][ix].reshape(32,8,6,-1);N0=z['counts'][ix].reshape(32,8,N);rec=N0.argmax(2);assert np.array_equal(rec,A[:,:,-1,fields.index('recommended')])
   sumf=f.sum(1);den=sumf[:,0];r0=dict(carrier_rate=ratio(sumf[:,1],3*den),first_rate=ratio(sumf[:,2],den),syndrome_rate=ratio(sumf[:,3],den),residual_rate=ratio(sumf[:,4],den),value_MAE=ratio(sumf[:,5],den),success_given_syndrome=ratio(sumf[:,6],sumf[:,3]),triple_rate=ratio(sumf[:,7],den),n=int(den.sum()))
   hit=(rec[:,:,None]==top[:,None,:]).any(2);pTV=abs(N0/128-ref[:,None,:]).sum(2)/2
   metrics=dict(top5=stat(hit.mean(1)),top1=stat((rec==n.argmax(1)[:,None]).mean(1)),policy_TV=stat(pTV.mean(1)),rapfi9=stat((rec==E[9,350][:,None]).mean(1)),rapfi9_150=stat((rec==E[9,150][:,None]).mean(1)),rapfi15_embedded=ratio(((rec==E[15,350][:,None])&(E[15,350][:,None]>=0)).sum(1),8*(E[15,350]>=0)),regret={})
   vv=np.take_along_axis(q,rec,axis=1);cn=np.take_along_axis(n,rec,axis=1)
   for t in (8,16,32):
    mask=cn>=t;qm=np.where(n>=t,q,-np.inf).max(1);assert np.isfinite(qm).all()
    metrics['regret'][str(t)]=ratio(np.where(mask,qm[:,None]-vv,0).sum(1),mask.sum(1))
   by[g,arm]=(f,A,N0,rec,hit,pTV,vv,cn)
   rows.append(dict(generation=g,arm=arm,fault=r0,utility=metrics,mark=stat((-A[:,:,-1,fields.index('E_Q')]).mean(1)),path_entropy=stat(A[:,:,-1,fields.index('Hvisit')].mean(1))))
  for left,right in [('rep_iid','single_iid'),('rep_mix','single_mix'),('rep_iid','clean'),('rep_mix','clean'),('rep_common','clean'),('rep_common','rep_iid')]:
   f,A,N0,rec,hit,tv,vv,cn=by[g,left];ff,AA,N1,rec1,hit1,tv1,vv1,cn1=by[g,right]
   d=dict(top5_diff=stat((hit.astype(float)-hit1).mean(1)),rapfi9_diff=stat(((rec==E[9,350][:,None]).astype(float)-(rec1==E[9,350][:,None])).mean(1)),policyTV_diff=stat((tv-tv1).mean(1)),recommendation_match=stat((rec==rec1).mean(1)),count_TV=stat(abs(N0-N1).sum(2).mean(1)/256),value_MAE_diff=stat((f[:,:,5]/f[:,:,0]-ff[:,:,5]/ff[:,:,0]).mean(1)),regret_diff={})
   for t in (8,16,32):
    mask=(cn>=t)&(cn1>=t);d['regret_diff'][str(t)]=ratio(np.where(mask,vv1-vv,0).sum(1),mask.sum(1))
   contrasts.append(dict(generation=g,left=left,right=right,metrics=d))
 out=dict(scope='new 32 roots, paired seed blocks; native sign fault before actual backup, external Rapfi9 fallback independent, 15 NNUE different board, g3 high-budget same evaluator separate seeds; no true game regret',reference=stability,external=external,rows=rows,contrasts=contrasts)
 dump(S/'PHYSICS_RESULTS.json',out)
 for x in rows:
  u=x['utility'];f=x['fault'];print('ARM',x['generation'],x['arm'],'n',f['n'],'err',f['residual_rate'],'MAE',f['value_MAE']['mean'],'detect',f['syndrome_rate']['mean'],'cond',f['success_given_syndrome']['mean'],'top5',u['top5']['mean'],'regret',u['regret']['16'],'Rapfi9',u['rapfi9']['mean'])
 for x in contrasts:
  if x['right'].startswith('single'):print('DIFF',x['generation'],x['left'],x['right'],x['metrics']['top5_diff'],x['metrics']['regret_diff']['16'],x['metrics']['rapfi9_diff'],x['metrics']['value_MAE_diff'])
 print('REFERENCE',stability,'EXTERNAL',external)
if __name__=='__main__':main()
