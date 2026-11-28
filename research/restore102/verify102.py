from lag102 import *
from functools import lru_cache
from itertools import permutations
@lru_cache(None)
def elog(N,a,b):
 den=comb(N,b);m=a*b/N
 return sum(comb(a,k)*comb(N-a,b-k)/den*k*math.log(k/m) for k in range(max(1,a+b-N),min(a,b)+1)) if m>0 else 0.
def null_cmi(p):
 p=np.asarray(p,int);ab=p.sum(2);bc=p.sum(0);n=p.sum();b=bc.sum(1)
 return sum(elog(int(b[j]),int(ab[i,j]),int(bc[j,k])) for j in range(D) if b[j] for i in range(D) if ab[i,j] for k in range(D) if bc[j,k])/n
def cmicheck(e,P,old):
 P=P/P.sum();ab=P.sum(2);bc=P.sum(0);b=bc.sum(1);ac=P.sum(1)
 q=np.einsum('ab,bc,b->abc',ab,bc,np.divide(1,b,out=np.zeros_like(b),where=b>0));v=.5*abs(P-q).sum()
 c=.5*abs(ac-q.sum(1)).sum();mi=0.
 for i,j,k in zip(*np.nonzero(P)):mi+=P[i,j,k]*math.log(P[i,j,k]*b[j]/(ab[i,j]*bc[j,k]))
 ck(e,'direct_CMI_TV_CK',[mi,v,c],old[:3]);assert c<=v+1e-10 and v<=np.sqrt(max(0,mi)/2)+1e-10
def main():
 t=time.monotonic();m=read(S/'MARKOV102.json');l=read(S/'LAG102.json');err={};src=m['source_hashes']|l['source_hashes'];raw=m['raw_receipts']|l['raw_receipts'];assert len(raw)==11808 and not set(m['raw_receipts'])&set(l['raw_receipts'])
 assert all(sha(B/k)==h for k,h in src.items()) and sha(S/'STATE102.npz')==m['array_sha'] and sha(S/'SELECTION102.npz')==l['array_sha']
 with np.load(S/'STATE102.npz') as z:M=z['meta'];Y=z['state'];R=z['root']
 controls=[]
 # Direct exact permutation baseline, tiny AB/BC margins with all permutations.
 x=np.array([0,0,1,1]);mid=np.zeros(4,int);y=np.array([0,0,0,1]);p=tab3(x,mid,y,np.ones(4));v=triple(p,err)[0];baseline=null_cmi(p);actual=np.mean([triple(tab3(x,mid,np.array(c),np.ones(4)),err)[0] for c in permutations(y)])
 ck(err,'hypergeom_permutation_exact',baseline,actual);controls.append(dict(tiny_CMI=v,exact_null=baseline))
 allbias=[];count=0;rootcount=0
 for g in m['groups']:
  f,co=g['family'],g['cohort'];ix=(M[:,1]==f)&((M[:,0]<3) if co==-1 else (M[:,0]==co));Z=Y[ix];r=R[ix];w=weights(r);u=np.unique(r);assert len(u)==g['roots']
  for row in g['triples']:
   i,j,k=[row[a] for a in ['i','j','k']];P=np.einsum('n,na,nb,nc->abc',w,np.eye(D)[Z[:,i]],np.eye(D)[Z[:,j]],np.eye(D)[Z[:,k]],optimize=True);cmicheck(err,P,row['metrics']);count+=1
   rr=[]
   for ri,root in enumerate(u):
    z=Z[r==root];p=tab3(z[:,i],z[:,j],z[:,k],np.ones(len(z)));v=triple(p,err)[0];ck(err,'root_CMI',v,row['root_values'][ri][0]);b=null_cmi(p);rr.append([v,b,v-b]);rootcount+=1
   allbias.append(dict(family=f,cohort=co,i=i,j=j,k=k,d=row['d'],root_ci=ci(rr).tolist(),root_values=rr))
 assert count==306 and rootcount==17952
 # independently reconstruct development model and full held-out scoring.
 predchecks=0
 for g in range(3):
  train=(M[:,1]==g)&(M[:,0]<3);Z=Y[train];w=weights(R[train])*len(Z);E=np.eye(D)[Z];h=np.einsum('n,nti,ntj->ij',w,E[:,:-1],E[:,1:]);marg=np.einsum('n,nti->ti',w,E);marg=(marg+8/D)/(marg.sum(1,keepdims=True)+8);K=[];K2=[]
  H=(h+8/D)/(h.sum(1,keepdims=True)+8)
  for t0 in range(15):
   c=E[:,t0].T@(w[:,None]*E[:,t0+1]);K.append((c+8*marg[t0+1])/(c.sum(1,keepdims=True)+8))
   if t0>=1:
    d=np.einsum('n,na,nb,nc->abc',w,E[:,t0-1],E[:,t0],E[:,t0+1],optimize=True);K2.append((d+8*K[t0][None,:,:])/(d.sum(2,keepdims=True)+8))
  K=np.array(K);K2=np.array(K2);old=m['models'][g];ck(err,'independent_homogeneous',H,old['homogeneous']);ck(err,'independent_clock',K,old['clock']);ck(err,'independent_lag2',K2,old['lag2'])
  for ev in [v for v in m['external'] if v['family']==g]:
   ix=(M[:,1]==g)&(M[:,0]==ev['cohort']);z=Y[ix];r=R[ix];scores=[]
   for t0 in range(1,15):
    a=z[:,t0-1];b=z[:,t0];c=z[:,t0+1];P=np.stack([H[b],K[t0,b],K2[t0-1,a,b]],axis=1);ls=-np.log(np.take_along_axis(P,c[:,None,None],axis=2)[:,:,0]);bs=(P*P).sum(2)-2*np.take_along_axis(P,c[:,None,None],axis=2)[:,:,0]+1;scores.append(np.stack([ls,bs],axis=2))
   rr=rootmeans(np.mean(scores,axis=0),r);ck(err,'one_step_score',rr,ev['one_step']['root_scores']);predchecks+=len(z)*14
   for row in ev['multi']:
    h=row['h'];scores=[]
    for i in range(16-h):
     j=i+h;b=z[:,i];a=z[:,j];pp=np.eye(D)
     for t0 in range(i,j):pp=pp@K[t0]
     c=E[:,i].T@(w[:,None]*E[:,j]);direct=(c+8*marg[j])/(c.sum(1,keepdims=True)+8);ck(err,'direct_kernel',direct,old['direct'][f'{i},{j}']);ck(err,'composed_kernel',pp,old['composed'][f'{i},{j}'])
     P=np.stack([np.linalg.matrix_power(H,h)[b],pp[b],direct[b]],axis=1);ls=-np.log(np.take_along_axis(P,a[:,None,None],axis=2)[:,:,0]);bs=(P*P).sum(2)-2*np.take_along_axis(P,a[:,None,None],axis=2)[:,:,0]+1;scores.append(np.stack([ls,bs],axis=2))
    ck(err,'multi_step_score',rootmeans(np.mean(scores,axis=0),r),row['root_scores']);predchecks+=len(z)*(16-h)
 # All per-selection sectors and quiescent binomial via polynomial products, root lag covariance via independent centered sums.
 nlag=0;undefined=0;waves=0;gindex={(x['L'],x['cohort'],x['arm'],x['bi'],x['budget']):x for x in l['groups']}
 with np.load(S/'SELECTION102.npz') as z:
  for L in LS:
   for h in [16,64]:
    mm=z[f'M{L}_{h}'];p=z[f'P{L}_{h}'];a=z[f'A{L}_{h}'];mark=z[f'Z{L}_{h}'];v=z[f'V{L}_{h}'];d0=z[f'D{L}_{h}']
    q=p[:,4::4,0];x=(a==0).astype(float);cnt=x[:,4:].reshape(len(x),-1,4).sum(2);ck(err,'wave_counts',cnt,v[:,:,0]);dist=np.ones(q.shape+(1,))
    for _ in range(4):
     n=np.zeros(q.shape+(dist.shape[-1]+1,));n[...,:-1]+=dist*(1-q)[...,None];n[...,1:]+=dist*q[...,None];dist=n
    ck(err,'binomial_nll_polynomial',-np.log(np.take_along_axis(dist,cnt.astype(int)[...,None],axis=-1)[...,0]),v[:,:,7]);ck(err,'binomial_Brier_polynomial',((dist-np.eye(5)[cnt.astype(int)])**2).sum(2),v[:,:,8])
    ck(err,'startup_defect',np.sum((p[:,:4]-p[:,:4].mean(1,keepdims=True))**2,(1,2)),d0);waves+=cnt.size
    for co,arm,bi in sorted(set(tuple(t[[0,2,3]]) for t in mm)):
     group=gindex[L,co,arm,bi,h];mask=np.all(mm[:,[0,2,3]]==[co,arm,bi],1)
     for core in range(8):
      ii=mask&(mm[:,1]==core);A=x[ii];P=p[ii,:,0];Z=mark[ii];R0=len(A);E=A-P
      ck(err,'wave_root_mean',v[ii].mean((0,1)),group['wave_root'][core])
      for j,k in enumerate(group['lags']):
       u=A[:,:-k];w0=A[:,k:];covs=[np.cov(u[:,s],w0[:,s],ddof=1)[0,1] for s in range(h-k)]
       v0=group['lag_root'][core][j];ck(err,'lag_root_cov',np.mean(covs),v0[0]);ck(err,'lag_innovation_cross',np.mean(E[:,:-k]*E[:,k:]),v0[4])
       valid=v0[9]*v0[10]>1e-16;undefined+=not valid;nlag+=1
       if valid:ck(err,'lag_rho',v0[0]/np.sqrt(v0[9]*v0[10]),v0[1])
 assert waves==49440
 summary=[]
 for g in m['groups']:
  rr=[x for x in allbias if (x['family'],x['cohort'])==(g['family'],g['cohort'])];x=np.array([v['metrics'] for v in g['triples']]);y=np.array([v['root_ci'][0] for v in rr]);summary.append(dict(family=g['family'],cohort=g['cohort'],mean=x.mean(0).tolist(),max=x.max(0).tolist(),root_CMI_bias_mean=y.mean(0).tolist()))
 dump(S/'CMI_NULL102.json',dict(status='PASS_SCOPED',triples=count,root_triples=rootcount,rows=allbias,groups=summary,controls=controls,scope='Exact conditional-permutation expectation of finite plug-in CMI, not corrected population estimate or significance test. Extension after initial calculation.'))
 fresh=[]
 for bi in [0,1,3]:
  a=[g for g in l['groups'] if g['cohort']==2 and g['arm']==1 and g['budget']==64 and g['bi']==bi];assert len(a)==5
  fresh.append(dict(bi=bi,lag=np.mean([g['lag_ci'][0] for g in a],axis=0).tolist(),wave=np.mean([g['wave_ci'][0] for g in a],axis=0).tolist()))
 counts=dict(raw_paths=len(raw),R86_paths=m['paths'],R87_paths=l['paths'],triples=count,root_triples=rootcount,lag_root_cases=nlag,undefined_rho=int(undefined),quiescent_waves=waves,prediction_cases=predchecks,one_step_logloss_positive=sum(v['one_step']['lag2_gain_ci'][1][0]>0 for v in m['external']),one_step_Brier_positive=sum(v['one_step']['lag2_gain_ci'][1][1]>0 for v in m['external']),multi_logloss_positive=sum(x['direct_gain_ci'][1][0]>0 for v in m['external'] for x in v['multi']),multi_Brier_positive=sum(x['direct_gain_ci'][1][1]>0 for v in m['external'] for x in v['multi']),pair_interval_exclusion=sum(g['wave_ci'][1][4]>0 or g['wave_ci'][2][4]<0 for g in l['groups']))
 dump(S/'SUMMARY102.json',dict(counts=counts,markov=summary,lag_fresh_dense=fresh,external=m['external'],control=m['control'],scope='Finite exact empirical projection identities and conditional independence null; separate heldout root and actual quiescent wave evidence. No physical time or global closure.'))
 dump(S/'INDEPENDENT102.json',dict(status='PASS_SCOPED',checks={**m['checks'],**l['checks'],**err},raw_receipts=raw,source_hashes=src,software=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),counts=counts,independence='Independent triple einsum and entropy/direct CMI, exact conditional permutation hypergeometric baseline, independent train matrix and heldout predictions, all polynomial count laws and root covariance via numpy.cov; same agent.'))
 print('INDEPENDENT',err,flush=True);print('COUNTS',counts,flush=True)
 for g in summary:print('CMI',g,flush=True)
 for g in fresh:print('FRESH',g,flush=True)
if __name__=='__main__':main()
