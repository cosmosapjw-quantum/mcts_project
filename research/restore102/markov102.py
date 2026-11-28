from base61 import *
from scipy.special import xlogy
from scipy.stats import t as student
import time,sys,platform
R86=B/'mqft_restore86_20261127';R87=B/'mqft_restore87_20261128'
FAMS=['g0','g3','Rapfi'];D=9;DS=[1,2,4];ALPHA=8.
def ck(e,k,a,b,tol=1e-9):
 d=float(np.max(abs(np.asarray(a)-np.asarray(b))));e[k]=max(e.get(k,0.),d);assert d<tol,(k,d)
def ci(x):
 x=np.asarray(x);m=x.mean(0);w=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return np.stack([m,m-w,m+w])
def pin(stage,f,src):
 p=next(B.glob(f'mqft_restore{stage}_20*'));m=read(p/f'RUN_MANIFEST{stage}.json');k=str((p/f).relative_to(B));h=m['files'][f];assert sha(p/f)==h;src[k]=h;return p/f
def rehash(x):k,h=x;assert sha(B/k)==h,k;return k,h
def weights(root):return np.array([1/(len(set(root))*sum(root==i)) for i in root])
def rootmeans(x,root):return np.stack([x[root==k].mean(0) for k in np.unique(root)])
def entropy(x):return -float(np.sum(xlogy(x,x)))
def tab3(a,b,c,w):return np.bincount(a*D*D+b*D+c,weights=w,minlength=D**3).reshape(D,D,D)
def triple(P,e):
 P=P/P.sum();ab=P.sum(2);bc=P.sum(0);b=bc.sum(1);ac=P.sum(1);a=ab.sum(1)
 cb=np.divide(bc,b[:,None],out=np.zeros_like(bc),where=b[:,None]>0);Q=ab[:,:,None]*cb[None,:,:]
 ck(e,'markovization_mass',Q.sum(),1);ck(e,'markovization_ab',Q.sum(2),ab);ck(e,'markovization_bc',Q.sum(0),bc)
 nz=P>0;assert np.all(Q[nz]>0);I=float(np.sum(P[nz]*np.log(P[nz]/Q[nz])));IE=entropy(ab)+entropy(bc)-entropy(b)-entropy(P);ck(e,'CMI_entropy',I,IE)
 J=.5*abs(P-Q).sum();ckv=.5*abs(ac-Q.sum(1)).sum();G=np.divide(ab,a[:,None],out=np.zeros_like(ab),where=a[:,None]>0)
 ck(e,'CK_weighted',ckv,.5*np.sum(abs(ac-a[:,None]*(G@cb))))
 mix=.5*(P+Q);m0=P>0;m1=Q>0;js=.5*float(np.sum(P[m0]*np.log(P[m0]/mix[m0]))+np.sum(Q[m1]*np.log(Q[m1]/mix[m1])))
 assert ckv<=J+1e-10 and J<=np.sqrt(max(0,I)/2)+1e-10 and I>=-1e-10
 return [I,float(J),float(ckv),float(js),float(np.sqrt(max(0,I)/2))]
def pair(a,b,w):return np.bincount(a*D+b,weights=w,minlength=D*D).reshape(D,D)
def smooth(C,prior,alpha=ALPHA):
 p=(C+alpha*prior)/(C.sum(-1,keepdims=True)+alpha);assert np.max(abs(p.sum(-1)-1))<1e-12 and p.min()>0;return p
def losses(p,y):
 return np.column_stack([-np.log(p[np.arange(len(y)),y]),np.sum((p-np.eye(D)[y])**2,axis=1)])
def control(e):
 P=np.zeros((D,D,D))
 for a in range(2):
  for b in range(2):P[a,b,a^b]=.25
 v=triple(P,e);ck(e,'XOR',v[:3],[np.log(2),.5,0])
 K=np.array([[.9,.1],[.2,.8]]);L=np.array([[.1,.9],[.7,.3]]);p=np.zeros_like(P);p[:2,:2,:2]=np.array([.4,.6])[:,None,None]*K[:,:,None]*L[None,:,:];v2=triple(p,e);ck(e,'nonstationary_Markov',v2[:4],0)
 return dict(XOR=v,nonstationary_markov=v2,homogeneous_K_squared_TV=float(.5*abs(K@K-K@L).sum(1).mean()))
def main():
 t=time.monotonic();src={};err={};a=read(pin(86,'AUDIT86.json',src));p=pin(86,'WAVES86.npz',src);assert sha(p)==a['array_sha'];r90=read(pin(90,'PATH90.json',src))
 raw={x['path']:x['sha256'] for x in a['raw_receipts']};assert len(raw)==7488 and raw=={x['path']:x['sha256'] for x in r90['raw_receipts']}
 with cf.ThreadPoolExecutor(max_workers=6) as pool:assert dict(pool.map(rehash,raw.items()))==raw
 for n,f in [(71,'RESULTS_KO.md'),(72,'MEMORY72.json'),(77,'CLOSURE77.json'),(86,'COLLISION_WITNESSES86.json'),(86,'MEMORY86.json'),(87,'ERRATUM86.json'),(92,'SUMMARY92.json'),(101,'SUMMARY101.json')]:pin(n,f,src)
 with np.load(p) as z:
  M=z['meta'];c=np.cumsum(z['count'],axis=1)[:,1::2]/(8*np.arange(1,17));w=np.cumsum(z['mark'],axis=1)[:,1::2]/(8*np.arange(1,17))
 y=3*np.digitize(c,[1/16,1/4])+np.digitize(w,[-.2,.2]);root=M[:,2]+np.array([0,48,72,104,152])[M[:,0]]
 assert y.shape==(7488,16);groups=[];models=[];ev=[];checks=0
 for g in range(3):
  train=(M[:,1]==g)&(M[:,0]<3);Y=y[train];r=root[train];wt=weights(r);assert len(Y)==1344 and len(set(r))==104
  # Fixed clock model, full supported alphabet with smoothing; training weights scaled to number of histories.
  ww=wt*len(Y);un=np.ones(D)/D
  hist=np.array([np.bincount(Y[:,t],weights=ww,minlength=D) for t in range(16)])
  marg=(hist+ALPHA*un)/(hist.sum(1,keepdims=True)+ALPHA)
  C=np.array([pair(Y[:,t],Y[:,t+1],ww) for t in range(15)])
  H=smooth(C.sum(0),un);K=np.array([smooth(C[t],marg[t+1]) for t in range(15)])
  K2=np.array([smooth(tab3(Y[:,t-1],Y[:,t],Y[:,t+1],ww),K[t][None,:,:]) for t in range(1,15)])
  direct={};prodK={}
  for d in DS:
   for i in range(16-2*d):
    k=i+2*d;P=K[i].copy()
    for j in range(i+1,k):P=P@K[j]
    direct[i,k]=smooth(pair(Y[:,i],Y[:,k],ww),marg[k]);prodK[i,k]=P
    ck(err,'composed_normalization',P.sum(1),1)
  models.append(dict(family=g,homogeneous=H.tolist(),clock=K.tolist(),lag2=K2.tolist(),marginals=marg.tolist(),direct={f'{i},{j}':v.tolist() for (i,j),v in direct.items()},composed={f'{i},{j}':v.tolist() for (i,j),v in prodK.items()}))
  for co in [-1,3,4]:
   ix=train if co==-1 else (M[:,1]==g)&(M[:,0]==co);z=y[ix];rr=root[ix];w=weights(rr);tr=[]
   for d in DS:
    for i in range(16-2*d):
     j,k=i+d,i+2*d;P=tab3(z[:,i],z[:,j],z[:,k],w);v=triple(P,err)
     # Independent plain-loop cross-tab, root-balanced.
     check=np.zeros_like(P)
     for a0,b0,c0,w0 in zip(z[:,i],z[:,j],z[:,k],w):check[a0,b0,c0]+=w0
     ck(err,'independent_tabulation',P,check);checks+=1
     rv=[]
     for rid in np.unique(rr):
      s=z[rr==rid];p0=tab3(s[:,i],s[:,j],s[:,k],np.ones(len(s))/len(s));rv.append(triple(p0,err))
     tr.append(dict(i=i,j=j,k=k,d=d,metrics=v,root_ci=ci(rv).tolist(),root_values=rv))
   groups.append(dict(family=g,cohort=co,paths=len(z),roots=len(set(rr)),triples=tr))
   if co==-1:continue
   # one-step same clocks; root-wise losses, differences relative time-specific first order.
   pred=[]
   for t0 in range(1,15):
    b=z[:,t0];a0=z[:,t0-1];target=z[:,t0+1]
    pp=[H[b],K[t0,b],K2[t0-1,a0,b]]
    pred.append(np.stack([losses(p0,target) for p0 in pp],axis=1))
   v=np.stack(pred,axis=1);rm=rootmeans(v.mean(1),rr);one=dict(scores_ci=ci(rm).tolist(),lag2_gain_ci=ci(rm[:,1]-rm[:,2]).tolist(),clock_gain_ci=ci(rm[:,0]-rm[:,1]).tolist(),root_scores=rm.tolist())
   multi=[]
   for d in DS:
    x=[]
    for i in range(16-2*d):
     k=i+2*d;target=z[:,k];b=z[:,i];h=np.linalg.matrix_power(H,2*d)
     p0=[h[b],prodK[i,k][b],direct[i,k][b]]
     x.append(np.stack([losses(v,target) for v in p0],axis=1))
    rm=rootmeans(np.stack(x,axis=1).mean(1),rr)
    multi.append(dict(h=2*d,scores_ci=ci(rm).tolist(),direct_gain_ci=ci(rm[:,1]-rm[:,2]).tolist(),root_scores=rm.tolist()))
   ev.append(dict(family=g,cohort=co,roots=len(set(rr)),paths=len(z),one_step=one,multi=multi))
 assert checks==3*3*sum(16-2*d for d in DS) and len(ev)==6
 np.savez_compressed(S/'STATE102.npz',meta=M,state=y,root=root)
 dump(S/'MARKOV102.json',dict(status='PASS_SCOPED',paths=7488,backup_events=7488*128,root_contexts=3*176,groups=groups,models=models,external=ev,triples=checks,fields=['CMI','joint_TV','CK_TV','JS','Pinsker'],control=control(err),checks=err,raw_receipts=raw,source_hashes=src,array_sha=sha(S/'STATE102.npz'),elapsed_s=time.monotonic()-t,scope='Finite empirical triple Markovization, root/clock conditional CMI descriptive bias, independent root cohort forecasting. Not strong closure or stationary time.'))
 print('MARKOV',checks,err,flush=True)
 for x in ev:print('HELDOUT',x['family'],x['cohort'],x['one_step']['scores_ci'][0],x['one_step']['lag2_gain_ci'],flush=True)
if __name__=='__main__':main()
