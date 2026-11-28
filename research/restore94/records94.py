from instrument81 import *
from itertools import product
from scipy.stats import t as student
O=B/'mqft_restore83_20261127';O81=B/'mqft_restore81_20261127'
def norm(x):return float(abs(x).max())
def ent(p):
 p=np.asarray(p);p=p[p>0];return float(-p@np.log(p))
def vn(a):
 e=np.linalg.eigvalsh(a);assert e.min()>-1e-10
 return ent(e)
def sqrtm(a):
 e,v=np.linalg.eigh(a);assert e.min()>-1e-10
 return (v*np.sqrt(np.maximum(e,0)))@adj(v)
def ci(x):
 x=np.asarray(x);m=x.mean(0);h=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return np.stack([m,m-h,m+h]).tolist()
def evaluate(nodes,protocol,mode,errors):
 # each continuing history carries Kraus product for every common bath label;
 # abort flags discarded coherences by declared classical stopping operation.
 M={'':np.eye(2,dtype=complex)[None,:,:]};D={'':RHO0.copy()};ab=np.zeros((2,2),complex);rows=[];mats=[]
 def check(k,x,y,tol=1e-9):
  er=norm(np.asarray(x)-np.asarray(y));errors[k]=max(errors.get(k,0),er);assert er<tol,(k,er,protocol,mode)
 for depth in range(4):
  keys=sorted(M);A=np.stack([M[h] for h in keys]);n=len(keys);b=A.shape[1]
  # joint S, record; common bath + initial mixed state purifying reference traced out.
  J=np.einsum('hbik,kl,gbjl->higj',A,RHO0,A.conj()).reshape(n*2,n*2)
  success=float(np.trace(J).real);assert success>0
  J/=success;v=J.reshape(n,2,n,2);R=np.einsum('aibi->ab',v);Sys=np.einsum('aiaj->ij',v)
  # independent direct CP density recurrence and block-trace computation
  cp=sum(D.values());check('cp_system_relative',cp/success,Sys)
  check('joint_trace',np.trace(J),1);check('record_trace',np.trace(R),1);check('system_trace',np.trace(Sys),1)
  check('record_diagonal',R.diagonal().real,np.array([np.trace(D[h]).real/success for h in keys]))
  assert min(np.linalg.eigvalsh(J))>-1e-10 and min(np.linalg.eigvalsh(R))>-1e-10 and min(np.linalg.eigvalsh(Sys))>-1e-10
  stopped=cp+ab;check('stopped_trace',np.trace(stopped),1);assert min(np.linalg.eigvalsh(stopped))>-1e-10
  sj,sr,ss=vn(J),vn(R),vn(Sys);dd=ent(R.diagonal().real);mi=sr+ss-sj
  assert mi>=-1e-10 and dd>=sr-1e-10 and mi<=2*min(sr,ss)+1e-10
  # explicit per-depth partial trace of pure purification amplitude, independent flattening
  amp=np.einsum('hbik,kl->hibl',A,sqrtm(RHO0)).reshape(n*2,b*2)/np.sqrt(success)
  check('purification_reduced',amp@adj(amp),J)
  eig=np.linalg.svd(amp,compute_uv=False)**2;check('purification_entropy',ent(eig),sj)
  coh=float(abs(R-np.diag(R.diagonal())).sum());had=np.array([[1.,1.],[1.,-1.]])/np.sqrt(2);H=np.ones((1,1))
  for _ in range(depth):H=np.kron(H,had)
  readout=np.diag(H@R@H.T).real;tv=float(.5*abs(readout-1/n).sum())
  rows.append([success,ss,sr,sj,mi,dd,dd-sr,coh,tv,success*tv,vn(stopped),float(abs(Sys[0,1])*2),ent(Sys.diagonal().real)])
  mats.append(R)
  if depth==3:break
  new={};newD={}
  for h in keys:
   m,r=nodes[h];axis,lam,theta,eta,erase=schedule(mode,32+4*depth)
   if erase:eta=0
   U=np.diag([np.exp(-.5j*theta),np.exp(.5j*theta)])
   F=[np.sqrt((1+eta)/2)*U,np.sqrt((1-eta)/2)*Z@U]
   Ka,Kb,_,_,_=operators(m,r,axis,lam)
   L=[np.linalg.matrix_power(K,4) if protocol=='native_repeat' else K/np.sqrt(m) for K in (Ka,Kb)]
   fail=sqrtm(I-sum(adj(k)@k for k in L)) if protocol=='native_repeat' else np.zeros((2,2),complex)
   check('node_complete',sum(adj(k)@k for k in L+[fail]),I)
   fre=sum(f@D[h]@adj(f) for f in F);ab+=fail@fre@adj(fail)
   for i,k in enumerate(L):
    new[h+str(i)]=np.array([k@f@a for a in M[h] for f in F]);newD[h+str(i)]=k@fre@adj(k)
  M,D=new,newD
 return np.array(rows),mats
def main():
 T=read(O/'TREE83.json');a=read(O/'AUDIT83.json');r=read(O/'RECORD83.json');assert sha(O/'TREE83.json')==a['tree_sha']==r['tree_sha'] and sha(O/'RECORD_MATRICES83.npz')==r['matrix_sha']
 f=read(O/'FRAGMENTS83.json');assert len(f['contexts'])==144
 z=np.load(O/'RECORD_MATRICES83.npz');lab=list(z['labels']);errors={};rows=[];data=[];prefix=[];meta=[]
 for i,c in enumerate(T['contexts']):
  assert list(z['meta'][i])==[c['family'],c['core']];out={}
  for p in ['native_repeat','pair_diagnostic']:
   res={}
   for mode,tag in [('plus81','adaptive'),('minus81','minus'),('eraseplus81','erase'),('base81','base')]:
    x,rr=evaluate(c['nodes'],p,mode,errors);old=z['R'][i,lab.index(p+'_'+tag)]
    er=norm(rr[-1]-old);errors['final_record']=max(errors.get('final_record',0),er);assert er<1e-9
    if p=='native_repeat' and mode=='plus81':
     actual=np.array([l['probability'] for l in c['leaves']]);e=norm(rr[-1].diagonal().real*x[-1,0]/actual-1);errors['actual_leaf_relative']=max(errors.get('actual_leaf_relative',0),e);assert e<1e-9
    res[mode]=x.tolist();data.append(x);meta.append([c['family'],c['core'],p,mode]);prefix.append([m.tolist() for m in []]) # no complex JSON
   fr=next(q for q in f['contexts'] if q['family']==c['family'] and q['core']==c['core'] and q['protocol']==p)
   # independently compute measured first-bit, single-fragment Holevo from final matrix by explicit indices
   R=z['R'][i,lab.index(p+'_adaptive')];pv=np.array([np.trace(R[j*4:(j+1)*4,j*4:(j+1)*4]).real for j in range(2)]);q=[R[j*4:(j+1)*4,j*4:(j+1)*4]/pv[j] for j in range(2)]
   frac=[]
   for axis in range(2):
    mat=[]
    for x0 in q:
     u=x0.reshape(2,2,2,2);mat.append(np.trace(u,axis1=1-axis,axis2=3-axis))
    avg=sum(pv[j]*mat[j] for j in range(2));chi=vn(avg)-sum(pv[j]*vn(mat[j]) for j in range(2));frac.append(chi/ent(pv))
   e=norm(np.asarray(frac)-[v['holevo_fraction'] for v in fr['fragments'][:2]]);errors['fragment_Holevo']=max(errors.get('fragment_Holevo',0),e);assert e<1e-9
   res['fragments']=frac;res['redundant90']=fr['disjoint_90_Holevo'];out[p]=res
  rows.append(dict(family=c['family'],core=c['core'],protocols=out))
 fields=['success','S_system','S_record','S_joint_SR','I_SR','H_record_diagonal','Crel_record','Cl1_record','Hadamard_TV','success_weighted_TV','S_stopped_system','Cz_system','H_system_diagonal']
 out=[]
 for g in range(3):
  v=[x for x in rows if x['family']==g]
  for p in ['native_repeat','pair_diagnostic']:
   u=[x['protocols'][p] for x in v];xx=np.array([q['plus81'] for q in u]);assert len(u)==24
   out.append(dict(family=FAMS[g],protocol=p,plus_ci=ci(xx),success_median=np.median(xx[:,:,0],axis=0).tolist(),minus_ci=ci([q['minus81'] for q in u]),erase_ci=ci([q['eraseplus81'] for q in u]),base_ci=ci([q['base81'] for q in u]),fragment_ci=ci([q['fragments'] for q in u]),redundant90=sum(q['redundant90'] for q in u)))
 dump(S/'RECORD94.json',dict(status='PASS_SCOPED',contexts=rows,summary=out,fields=fields,errors=errors,source_contexts=72,prefix_cases=len(data)*4,scope='Declared coherent prefix survival at depth d, not marginal of final success; joint S-R reduced from bath and mixed input purification. Stopped system includes explicit complement abort channel. Not native coherent branch identification.'))
 print('RECORD DONE',len(rows),len(data)*4,errors,flush=True)
if __name__=='__main__':main()
