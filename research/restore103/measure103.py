from base61 import *
from scipy.special import logsumexp,xlogy
from scipy.optimize import brentq
from scipy.stats import t as student
import gzip,time,sys,platform
LS=[13,15,17,19,21];HS=[16,32,64];GRID=np.array([-4,0,1,2,10/3,4,6,8,12,16,24,32.]);BO=32.
R87=B/'mqft_restore87_20261128';R79=B/'mqft_restore79_20261127'
STATUS=['flat','interior','lower_bound','upper_bound','lower_face','upper_face']
FIELDS=['beta','status','curvature','KL','TV','gradient','feature_range','control_KL','pythagorean_correction','mixture_gap','mean_conditional_KL']
def ck(e,k,a,b,tol=1e-9):
 d=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));e[k]=max(e.get(k,0.),d);assert d<tol,(k,d)
def ci(z):
 z=np.asarray(z);m=z.mean(0);w=student.ppf(.975,len(z)-1)*z.std(0,ddof=1)/np.sqrt(len(z));return np.stack([m,m-w,m+w])
def pin(stage,f,src):
 p=next(B.glob(f'mqft_restore{stage}_20*'));m=read(p/f'RUN_MANIFEST{stage}.json');k=str((p/f).relative_to(B));h=m['files'][f];assert sha(p/f)==h;src[k]=h;return p/f
def rehash(x):k,h=x;assert sha(B/k)==h,k;return k,h
def prob(s,b):
 z=b*s;z-=z.max();p=np.exp(z);return p/p.sum()
def KL(p,q):return float(np.sum(xlogy(p,p)-xlogy(p,q)))
def H(p):return -float(np.sum(xlogy(p,p)))
def fit(s,y,b0,e,ind=False):
 s=s-s.mean();span=float(np.ptp(s));target=float(y@s)
 def grad(b):
  q=prob(s,b);return float(q@s-target)
 if span<=1e-12:b=0.;st=0
 elif np.all(y[s>s.min()]>0)==False and float(y[s>s.min()].sum())==0:b=-BO;st=4
 elif float(y[s<s.max()].sum())==0:b=BO;st=5
 elif grad(-BO)>=0:b=-BO;st=2
 elif grad(BO)<=0:b=BO;st=3
 else:
  st=1
  if ind:
   lo,hi=-BO,BO
   for _ in range(70):
    mid=(lo+hi)/2
    if grad(mid)>0:hi=mid
    else:lo=mid
   b=(lo+hi)/2
  else:b=brentq(grad,-BO,BO,xtol=1e-12)
 q=prob(s,b);q0=prob(s,b0);mu=float(q@s);g=mu-target;v=float(q@((s-mu)**2));k=KL(y,q);k0=KL(y,q0);corr=(b0-b)*g
 ck(e,'projection_generalized_Pythagoras',k0-k,KL(q,q0)+corr)
 assert k>=-1e-10 and k0>=-1e-10
 if st==1:ck(e,'moment_matching',g,0)
 if st in (2,4):assert g>=-1e-10
 if st in (3,5):assert g<=1e-10
 return np.array([b,st,v,k,.5*abs(y-q).sum(),g,span,k0,corr]),q
def one(arg):
 L,r=arg;f=B/r['path'];assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256']
 with gzip.open(f,'rt') as z:hd,row,end=map(json.loads,z)
 tr=row['trace'];sp=tr['spec'];h=r['budget'];beta=sp['root_law']['beta_p'][0];assert sp['budget']==h==end['backups'] and sp['seed']==r['seed'] and beta==[0.,10/3,8.,16.,math.log(L*L)][r['beta_index']]
 rep=tr['replicas'][0];es=sorted(rep['events'],key=lambda e:e['root_selection']['clock']);done=sorted(rep['events'],key=lambda e:e['complete_seq']);assert len(es)==h
 legal=np.flatnonzero(np.array(tr['root']['board'])==0);n=len(legal);e={};P=[];Sc=[];Q=[];A=[];n0=np.zeros(n);w0=np.zeros(n);step=0;last=-1;sl=[];loss=[];fisher=[]
 # frozen blocks 1+1+2 then 4: all prior selections completed at block boundary
 for j,ev in enumerate(es):
  l=ev['root_selection']['law'];a=np.array(l['actions']);o=np.argsort(a);p=np.array(l['p'])[o];s=np.array(l['scores'])[o];assert np.array_equal(a[o],legal) and l['t']==j and all(l['active']) and l['logit_h']==0 and a[l['picked']]==ev['actions'][0]
  k=int(np.searchsorted(legal,ev['actions'][0]));assert legal[k]==ev['actions'][0] and p.min()>0
  q=prob(s,beta);ck(e,'actual_probability',q,p)
  ss=s-s.mean();lp=np.log(p);ll=lp-lp.mean();flat=np.ptp(s)<=1e-12
  b=0. if flat else float(ss@ll/(ss@ss));res=float(np.max(abs(ll-b*ss)))
  if flat:ck(e,'flat_law',p,1/n)
  else:ck(e,'score_slope',b,beta)
  ck(e,'log_slope_residual',res,0)
  ck(e,'affine_gauge',prob(2*s+7,beta/2),p)
  sl.append([b,float(not flat),res,np.ptp(s)]);mu=float(p@s);fisher.append(float(p@((s-mu)**2)))
  if j in (0,1,2) or j%4==0:
   while step<j:
    ev0=done[step];a0=int(np.searchsorted(legal,ev0['actions'][0]));n0[a0]+=1;w0[a0]+=ev0['root_value'];step+=1
   qpre=np.divide(w0,n0,out=np.zeros(n),where=n0>0);last=j
  P.append(p);Sc.append(s);Q.append(qpre.copy());A.append(k)
  # fixed candidate slopes, actual frozen-history prediction only
  yy=[]
  for x in (s,qpre):
   z=GRID[:,None]*x;logq=z-logsumexp(z,axis=1)[:,None]
   yy.append(np.stack([-logq[:,k],np.sum(p*lp)-logq@p],1))
  loss.append(yy)
  if j not in (0,1,2) and j%4!=0:ck(e,'frozen_scores',s,Sc[last])
 P=np.array(P);Sc=np.array(Sc);Q=np.array(Q);A=np.array(A);sl=np.array(sl);loss=np.array(loss);fisher=np.array(fisher)
 out=[];cuts={c['local_k']:c for c in rep['cuts']}
 for H0 in HS:
  if H0>h:continue
  cut=cuts[H0];counts=np.zeros(n);ws=np.zeros(n)
  for z in cut['edges']:
   idx=int(np.searchsorted(legal,z['action']));assert legal[idx]==z['action'] and z['pending']==0;counts[idx]=z['n'];ws[idx]=z['w']
  assert counts.sum()==H0;X=np.bincount(A[:H0],minlength=n)/H0;ck(e,'endpoint_counts',counts/H0,X)
  aa=P[:H0].mean(0);s0=np.divide(ws,counts,out=np.zeros(n),where=counts>0);feats=[s0,Sc[:H0].mean(0),Sc[H0-1]]
  gap=H(aa)-np.mean([H(p) for p in P[:H0]])
  ck(e,'mixture_gap',gap,np.mean([KL(p,aa) for p in P[:H0]]));assert gap>=-1e-10
  fitrows=[]
  for x in feats:
   fs=[]
   for yy in [aa,X]:
    v,q=fit(x,yy,beta,e);mix=float(np.mean([KL(p,q) for p in P[:H0]]));ck(e,'mixture_projection',mix,gap+KL(aa,q));fs.append(np.r_[v,gap,mix])
   fitrows.append(fs)
  # full array features/p and empirical sample counts for independent check; padded arrays stored only in npz.
  full=np.zeros((5,L*L));full[:,legal]=np.stack([aa,X,*feats])
  out.append((H0,np.asarray(fitrows),loss[:H0].mean(0),np.array([sl[:H0,1].sum(),sl[:H0,1].size-sl[:H0,1].sum(),fisher[:H0].mean(),gap,float(np.mean([H(p) for p in P[:H0]])),float(np.mean([-np.log(P[j,A[j]]) for j in range(H0)]))]),full))
 return [r[x] for x in ['cohort','core','arm','beta_index','rep','budget']],out,sl,e
def main():
 t=time.monotonic();src={};err={};prior=read(B/'mqft_restore98_20261128'/'RUN_MANIFEST98.json');raw=prior['prior_raw_receipts'];assert len(raw)==4320
 for n,f in [(79,'CALIBRATION79.json'),(91,'NATIVE91.json'),(99,'FITS99.json'),(100,'SUMMARY100.json'),(102,'SUMMARY102.json')]:pin(n,f,src)
 rg=np.load(pin(98,'RG98.npz',src));groups=[];arr={};recs={};allcount=0;flat=0;draws=0;status=np.zeros((3,2,6),int)
 for L in LS:
  p=R87/f'AUDIT_L{L}.json';k=str(p.relative_to(B));assert sha(p)==prior['inputs'][k];src[k]=sha(p);rs=read(p)['raw_receipts'];assert len(rs)==864
  p=R87/f'OBS_L{L}.npz';k=str(p.relative_to(B));assert sha(p)==prior['inputs'][k];src[k]=sha(p)
  with np.load(p) as z:old={k:z[k] for k in ['meta','A','X']}
  ix={tuple(m):i for i,m in enumerate(old['meta'])};rgi={tuple(m):i for i,m in enumerate(rg[f'M{L}'])}
  with cf.ProcessPoolExecutor(max_workers=10) as pool:out=list(pool.map(one,[(L,r) for r in rs],chunksize=4))
  M=[];V=[];C=[];F=[];Z=[];sm=[]
  for r,(m,rows,sl,e) in zip(rs,out):
   recs[r['path']]=r['sha256'];assert raw[r['path']]==r['sha256'];draws+=len(sl);flat+=int((sl[:,1]==0).sum());sm.append([m,int((sl[:,1]==0).sum())])
   for k,v in e.items():err[k]=max(err.get(k,0.),v)
   for h,v,c,f,z in rows:
    meta=m+[h];i=ix[tuple(meta)];j=rgi[tuple(meta)];ck(err,'R87_A',z[0],old['A'][i]);ck(err,'R87_X',z[1],old['X'][i]);ck(err,'R98_Fisher',f[2],rg[f'V{L}'][j,0,3]);M.append(meta);V.append(v);C.append(c);F.append(f);Z.append(z)
    for a in range(3):
     for b in range(2):status[a,b,int(v[a,b,1])]+=1
  M=np.array(M);V=np.array(V);C=np.array(C);F=np.array(F);Z=np.array(Z);assert len(M)==2080
  arr.update({f'M{L}':M,f'V{L}':V,f'C{L}':C,f'F{L}':F,f'Z{L}':Z})
  for co,arm,bi,bud,h in sorted(set(tuple(x[[0,2,3,5,6]]) for x in M)):
   mask=np.all(M[:,[0,2,3,5,6]]==[co,arm,bi,bud,h],1);roots=[];cl=[];fe=[];sts=np.zeros((3,2,6),int)
   for root in range(8):
    p0=mask&(M[:,1]==root);vv=V[p0];rv=vv.mean(0);rv[:,:,0]=np.divide((vv[:,:,:,0]*(vv[:,:,:,1]!=0)).sum(0),(vv[:,:,:,1]!=0).sum(0),out=np.zeros((3,2)),where=(vv[:,:,:,1]!=0).sum(0)>0);roots.append(rv);cl.append(C[p0].mean(0));fe.append(F[p0].mean(0))
    for a in range(3):
     for b in range(2):sts[a,b]+=np.bincount(vv[:,a,b,1].astype(int),minlength=6)
   # raw root means include explicit sentinel for no identifiable feature; separate masks for scaling below
   groups.append(dict(L=L,cohort=int(co),arm=int(arm),bi=int(bi),beta=[0.,10/3,8.,16.,math.log(L*L)][bi],budget=int(bud),h=int(h),roots=8,reps=int(mask.sum()/8),root_values=np.asarray(roots).tolist(),root_valid=np.stack([(V[mask&(M[:,1]==r),:,:,1]!=0).sum(0)>0 for r in range(8)]).tolist(),ci=ci(roots).tolist(),conditional_ci=ci(cl).tolist(),conditional_root=np.asarray(cl).tolist(),path_ci=ci(fe).tolist(),statuses=sts.tolist()))
  print('MEASURE',L,len(M),draws,flat,flush=True)
 assert recs==raw and draws==215040 and len(groups)==400 and status.sum()==10400*6
 np.savez_compressed(S/'MEASURE103.npz',**arr)
 dump(S/'MEASURE103.json',dict(status='PASS_SCOPED',paths=4320,draws=draws,flat_draws=flat,identified_draws=draws-flat,prefixes=10400,groups=groups,projection_status=STATUS,projection_fields=FIELDS,features=['endpoint_Q','mean_score','last_score'],targets=['A','X'],grid=GRID.tolist(),status_counts=status.tolist(),checks=err,raw_receipts=raw,source_hashes=src,array_sha=sha(S/'MEASURE103.npz'),elapsed_s=time.monotonic()-t,scope='Full-support algorithmic score and occupancy projection, no physical T; flat and bound masks explicit.'))
 print('MEASURE DONE',status.tolist(),err,flush=True)
if __name__=='__main__':main()
