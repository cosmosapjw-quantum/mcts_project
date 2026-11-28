from measure103 import *
def main():
 m=read(S/'MEASURE103.json');q=read(S/'SUMMARY103.json');audit=read(S/'INDEPENDENT103.json');err={};rows=[];count=0;ordered=0
 with np.load(S/'MEASURE103.npz') as z:
  for L in LS:
   M=z[f'M{L}'];Z=z[f'Z{L}'];E=Z[:,1]-Z[:,0]
   for g in [a for a in m['groups'] if a['L']==L]:
    ii=np.all(M[:,[0,2,3,5,6]]==[g[k] for k in ['cohort','arm','bi','budget','h']],1);rr=[]
    for c in range(8):
     ix=ii&(M[:,1]==c);a=E[ix];f=Z[ix,2:];R=len(a);assert R in (2,4);count+=1;ordered+=R*(R-1)
     t=np.einsum('ri,ski->rsk',a,f);self0=np.einsum('rri->i',t)/R;cross=(t.sum((0,1))-R*self0)/(R*(R-1));cov=np.einsum('ri,rki->k',a-a.mean(0),f-f.mean(0))/(R-1)
     ck(err,'endogeneity_covariance',self0-cross,cov)
     rr.append(np.stack([self0,cross,self0-cross],1))
    rows.append(dict(L=L,cohort=g['cohort'],arm=g['arm'],bi=g['bi'],budget=g['budget'],h=g['h'],reps=g['reps'],root_values=np.asarray(rr).tolist(),ci=ci(rr).tolist()))
 assert count==3200
 # finite exact two-state counterexample with one observed action, actual uniform law.
 for sign,st in [(-1,4),(1,5)]:
  s=np.array([float(sign),0.]);X=np.array([1.,0.]);A=np.array([.5,.5]);v,_=fit(s,X,0.,err);v0,_=fit(s,A,0.,err);assert v[1]==st and abs(v0[0])<1e-10
  ck(err,'constructed_same_history_moment',(X-A)@s,sign/2)
 fresh=[]
 for bi in [0,1,3,4]:
  r=[a for a in rows if (a['cohort'],a['arm'],a['bi'],a['budget'],a['h'])==(2,1,bi,64,64)]
  fresh.append(dict(bi=bi,mean=np.mean([a['ci'][0] for a in r],0).tolist()))
 q['cross_history']=dict(root_strata=count,ordered_pairs=ordered,fresh=fresh,scope='descriptive cross-history moment; ordered pairs not independent');audit['checks'].update({k:max(v,audit['checks'].get(k,0.)) for k,v in err.items()});audit['cross_history_root_strata']=count;audit['cross_history_ordered_pairs']=ordered
 dump(S/'CROSS103.json',dict(status='PASS_SCOPED',root_strata=count,ordered_pairs=ordered,rows=rows,checks=err,scope='Post-initial-run diagnostic, same vs independent rep moments, no new acquisition.'))
 dump(S/'SUMMARY103.json',q);dump(S/'INDEPENDENT103.json',audit)
 print('CROSS',count,ordered,err,flush=True);print(fresh,flush=True)
if __name__=='__main__':main()
