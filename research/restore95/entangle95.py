from instrument81 import *
from itertools import product,combinations
from scipy.stats import t as student
import gzip,time
O=B/'mqft_restore83_20261127';A=B/'mqft_restore81_20261127';P=B/'mqft_restore94_20261128'
MODES=['plus81','minus81','eraseplus81','base81'];PROTOCOLS=['native_repeat','pair_diagnostic'];XI=[0.,.5,1.]
FIELDS=['QMI','classical_Z_MI','negativity','concurrence','EoF_nats','CHSH_Pauli','s_times_negativity','s_times_concurrence','s_times_CHSH_excess']
YY=np.kron(Y,Y)
ERR=collections.defaultdict(float)
def norm(x):return float(np.max(abs(np.asarray(x))))
def check(k,a,b,tol=1e-9):
 e=norm(np.asarray(a)-np.asarray(b));ERR[k]=max(ERR[k],e);assert e<tol,(k,e)
def ent(p):
 p=np.asarray(p).real;p=p[p>0];return float(-p@np.log(p))
def vn(a):
 e=np.linalg.eigvalsh(a);assert e.min()>-1e-10;return ent(e)
def sqrtm(a):
 e,v=np.linalg.eigh(a);assert e.min()>-1e-10
 return (v*np.sqrt(np.maximum(e,0)))@adj(v)
def red(a,keep,n):
 rest=[i for i in range(n) if i not in keep];k=len(keep);p=list(keep)+rest
 v=a.reshape([2]*(2*n)).transpose(p+[i+n for i in p]).reshape(2**k,2**len(rest),2**k,2**len(rest))
 return np.trace(v,axis1=1,axis2=3)
def red_direct(a,keep,n):
 # Independent integer-index summation, preserving requested subsystem order.
 rem=[i for i in range(n) if i not in keep];d=1<<len(keep);z=np.zeros((d,d),complex)
 def pack(i,j):
  b=[0]*n
  for k,t in enumerate(keep):b[t]=(i>>(len(keep)-1-k))&1
  for k,t in enumerate(rem):b[t]=(j>>(len(rem)-1-k))&1
  return sum(x<<(n-1-k) for k,x in enumerate(b))
 for i,j in product(range(d),repeat=2):
  z[i,j]=sum(a[pack(i,b),pack(j,b)] for b in range(1<<len(rem)))
 return z
def pt(a,cut,n):
 p=list(range(2*n))
 for i in cut:p[i],p[n+i]=p[n+i],p[i]
 return a.reshape([2]*(2*n)).transpose(p).reshape(a.shape)
def neg(a,cut,n):
 x=pt(a,cut,n);e=np.linalg.eigvalsh(x);v=float(-e[e<0].sum())
 check('PT_trace_norm',v,(np.linalg.svd(x,compute_uv=False).sum()-1)/2)
 return v
def concurrence(a):
 # If a=F F^dagger, singular values of F^T YY F equal Wootters square roots.
 F=sqrtm(a);w=F.T@YY@F;s=np.linalg.svd(w,compute_uv=False);c=max(0.,float(s[0]-s[1:].sum()))
 # Alternative positive Hermitian sandwich sqrt(a) tilde(a) sqrt(a).
 h=F@YY@a.conj()@YY@F;h=(h+adj(h))/2;e=np.linalg.eigvalsh(h)
 assert e.min()>-1e-10
 d=np.sqrt(np.maximum(e,0))[::-1];c2=max(0.,float(d[0]-d[1:].sum()))
 check('concurrence_independent',c,c2,2e-7)
 return c
def chsh(a):
 T=np.array([[np.trace(a@np.kron(x,y)).real for y in PAULI] for x in PAULI])
 u,s,v=np.linalg.svd(T);b=float(2*np.linalg.norm(s[:2]));assert b<=2*np.sqrt(2)+1e-9
 if b>1e-14:
  a0=u[:,0];a1=u[:,1];b0=(s[0]*v[0]+s[1]*v[1])/(b/2);b1=(s[0]*v[0]-s[1]*v[1])/(b/2)
 else:a0,a1,b0,b1=np.eye(3)[0],np.eye(3)[1],np.eye(3)[0],np.eye(3)[1]
 dirs=np.stack([a0,a1,b0,b1]);check('CHSH_unit_norm',np.linalg.norm(dirs,axis=1),1)
 A0,A1,B0,B1=np.einsum('ki,ijl->kjl',dirs,PAULI)
 # explicit local projective Born probabilities, settings-specific no-signalling
 probs=np.array([[[[np.trace(a@np.kron((I+i*aa)/2,(I+j*bb)/2)).real for j in [1,-1]] for i in [1,-1]] for bb in [B0,B1]] for aa in [A0,A1]])
 assert probs.min()>-1e-10
 check('CHSH_Born_norm',probs.sum((-1,-2)),1)
 check('CHSH_nosignal_A',probs[:,0].sum(-1),probs[:,1].sum(-1))
 check('CHSH_nosignal_B',probs[0].sum(-2),probs[1].sum(-2))
 E=np.einsum('abij,i,j->ab',probs,[1,-1],[1,-1]);check('CHSH_explicit',E[0,0]+E[0,1]+E[1,0]-E[1,1],b)
 check('CHSH_operator',np.trace(a@(np.kron(A0,B0+B1)+np.kron(A1,B0-B1))).real,b)
 return b,dirs
def metrics(a,s):
 assert a.shape==(4,4) and np.linalg.eigvalsh(a).min()>-1e-10
 check('pair_trace',np.trace(a),1);u=red(a,[0],2);v=red(a,[1],2)
 q=vn(u)+vn(v)-vn(a);z=a.diagonal().real.reshape(2,2);cl=ent(z.sum(0))+ent(z.sum(1))-ent(z.ravel())
 assert q>=-1e-10 and cl>=-1e-10 and q>=cl-1e-10
 N=neg(a,[0],2);C=concurrence(a);B,dirs=chsh(a);assert N<=C/2+2e-7 and C<=1+1e-8
 assert not (N>1e-7 and C<1e-8)
 E=ent([(1+np.sqrt(max(0,1-C*C)))/2,(1-np.sqrt(max(0,1-C*C)))/2])
 return np.array([q,cl,N,C,E,B,s*N,s*C,s*max(B-2,0)]),dirs
def prefix(nodes,p,mode):
 M={'':np.eye(2,dtype=complex)[None,:,:]};out=[]
 for d in range(4):
  keys=sorted(M);A0=np.stack([M[h] for h in keys]);n=2**d;b=A0.shape[1]
  amp=np.einsum('hbik,kl->hibl',A0,sqrtm(RHO0)).reshape(2*n,2*b)
  J=amp@adj(amp);s=float(np.trace(J).real);assert s>0
  J/=s;check('joint_trace',np.trace(J),1);assert np.linalg.eigvalsh(J).min()>-1e-10
  out.append((s,J))
  if d==3:break
  new={}
  for h in keys:
   m,r=nodes[h];axis,lam,theta,eta,erase=schedule(mode,32+4*d);eta=0 if erase else eta
   U=np.diag([np.exp(-.5j*theta),np.exp(.5j*theta)]);F=[np.sqrt((1+eta)/2)*U,np.sqrt((1-eta)/2)*Z@U]
   Ka,Kb,_,_,_=operators(m,r,axis,lam)
   L=[np.linalg.matrix_power(k,4) if p=='native_repeat' else k/np.sqrt(m) for k in (Ka,Kb)]
   eff=sum(adj(k)@k for k in L);assert np.linalg.eigvalsh(I-eff).min()>-1e-10
   if p=='pair_diagnostic':check('pair_complete',eff,I)
   for i,k in enumerate(L):new[h+str(i)]=np.array([k@f@a for a in M[h] for f in F])
  M=new
 return out
def raw_receipt(arg):
 r=arg;f=B/r['path'];assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256']
 with gzip.open(f,'rt') as z:d=json.load(z)
 es=d['trace']['events'];assert len(es)==128
 a=np.array([e['actions'][0] for e in es]);q=[];l=[]
 for t in range(32,44):
  law=es[t]['root_selection']['law'];p=np.asarray(law['p']);ac=np.asarray(law['actions']);ph=law['phase65'];assert law['t']==t and ac[law['picked']]==a[t] and ph['sources']==r['sources']
  assert abs(p.sum()-1)<1e-10 and min(p)>0
  assert abs(np.log(p[law['picked']])-law['log_p'])<1e-10
  q.append([ph['pair_mass'],ph['population_a']]);l.append(law['log_p'])
 return r['path'],r['sha256'],a.tolist(),q,l
def controls():
 def dm(x):x=np.array(x,complex);return np.outer(x,x.conj())
 bell=dm(np.array([1,0,0,1])/np.sqrt(2));classic=np.diag([.5,0,0,.5]);ghz=dm(np.array([1,0,0,0,0,0,0,1])/np.sqrt(2));w=dm(np.array([0,1,1,0,1,0,0,0])/np.sqrt(3))
 out={}
 for name,a in [('Bell',bell),('classical',classic),('GHZ_pair',red(ghz,[0,1],3)),('W_pair',red(w,[0,1],3)),('product',dm([1,0,0,0]))]:
  m,_=metrics(a,1);out[name]=m.tolist()
 check('Bell',np.array(out['Bell'])[[2,3,5]],[.5,1,2*np.sqrt(2)])
 check('classic',np.array(out['classical'])[[0,1,2,3,5]],[np.log(2),np.log(2),0,0,2])
 check('GHZ_pair',out['GHZ_pair'],out['classical']);check('GHZ_cut',neg(ghz,[0],3),.5)
 check('W_pair_concurrence',out['W_pair'][3],2/3,2e-7)
 for p in [0,1/3,.5,1/np.sqrt(2),1]:
  a=p*bell+(1-p)*np.eye(4)/4;m,_=metrics(a,1);check('Werner',m[[2,3,5]],[max(0,(3*p-1)/4),max(0,(3*p-1)/2),2*np.sqrt(2)*p],2e-7);out['Werner_'+str(p)]=m.tolist()
 return out
def main():
 t0=time.monotonic();aa=read(O/'AUDIT83.json');a81=read(A/'AUDIT81.json');tree=read(O/'TREE83.json');old=read(P/'RECORD94.json');r83=read(O/'RECORD83.json')
 assert sha(O/'TREE83.json')==aa['tree_sha']==r83['tree_sha'] and sha(O/'RECORD_MATRICES83.npz')==r83['matrix_sha']
 src={x['path']:x for x in a81['raw_receipts']};srcs=[src[c['source']] for c in tree['contexts']]
 raw=aa['raw_receipts']+srcs;assert len(raw)==720 and len(set(x['path'] for x in raw))==720
 with cf.ProcessPoolExecutor(max_workers=10) as ex:raws=list(ex.map(raw_receipt,raw,chunksize=8))
 by={r[0]:r for r in raws};receipts={r[0]:r[1] for r in raws};assert len(by)==720
 for c in tree['contexts']:
  src0=by[c['source']];assert src0[1]==c['source_sha']
  for l in c['leaves']:
   v=by[l['path']];assert v[1]==l['sha256'] and v[2][:32]==src0[2][:32] and v[2][32:44]==l['root_actions']
   check('raw_logp',v[4],l['logp_active']);check('raw_leaf_relative',np.exp(sum(v[4]))/l['probability'],1)
   bits=''.join(map(str,l['bits']))
   for depth in range(3):check('raw_node_settings',v[3][4*depth],c['nodes'][bits[:depth]])
  ref=c['reference'];assert ref in by;ref0=by[ref];assert ref0[2:]==src0[2:]
  # R83 prior full semantic replay is inherited; raw references and source hashes checked.
 print('RAW VERIFIED',len(receipts),flush=True)
 refs=controls();rows=[];matrices={d:[] for d in range(4)};meta=[]
 z=np.load(O/'RECORD_MATRICES83.npz');labels=list(z['labels'])
 for k,c in enumerate(tree['contexts']):
  g=c['family'];core=c['core'];assert list(z['meta'][k])==[g,core];oldc=old['contexts'][k];assert [oldc['family'],oldc['core']]==[g,core]
  for p in PROTOCOLS:
   for mode,tag in zip(MODES,['adaptive','minus','erase','base']):
    frames=prefix(c['nodes'],p,mode);meta.append([g,core,p,mode])
    for d,(s,J) in enumerate(frames):
     matrices[d].append(J);n=d+1
     prev=np.array(oldc['protocols'][p][mode][d]);check('R94_success_relative',s/prev[0],1)
     check('R94_joint_entropy',vn(J),prev[3]);check('R94_system_entropy',vn(red(J,[d],n)),prev[1])
     if d==3:check('R83_final_record',red(J,[0,1,2],4),z['R'][k,labels.index(p+'_'+tag)])
     if d==0:continue
     bits=np.array(list(product(range(2),repeat=d)));dist=np.sum(bits[:,None,:]!=bits[None,:,:],axis=-1)
     pairs=list(combinations(range(n),2))
     # Unique bipartitions: choose representatives containing first qubit, exclude full set.
     cuts=[tuple([0]+list(cut)) for k0 in range(n-1) for cut in combinations(range(1,n),k0)]
     # include cut 0 + all but one: k0 up to n-2. yes.
     untag_diag=J.diagonal().real
     for xi in XI:
      # xi=0 means 0**0=1. Local record phase damping only; system unchanged.
      Jx=J*np.repeat(np.repeat(xi**dist,2,axis=0),2,axis=1);check('tag_diag',Jx.diagonal().real,untag_diag)
      assert np.linalg.eigvalsh(Jx).min()>-1e-10
      vals=[];dirs=[]
      for i,j in pairs:
       a=red(Jx,[i,j],n);check('partial_trace_independent',a,red_direct(Jx,[i,j],n))
       m,ds=metrics(a,s);vals.append(m);dirs.append(ds)
      cutsN=[neg(Jx,cut,n) for cut in cuts]
      # Fully tagged state is convex mixture of all record product basis projectors and conditional S.
      if xi==0:
       assert max(cutsN)<1e-9 and max(v[2] for v in vals)<1e-9 and max(v[3] for v in vals)<2e-7 and max(v[5] for v in vals)<=2+1e-9
      rows.append(dict(family=g,core=core,protocol=p,mode=mode,depth=d,xi=xi,success=s,labels=['R'+str(i+1) for i in range(d)]+['S'],pairs=[list(x) for x in pairs],metrics=np.array(vals).tolist(),directions=np.array(dirs).tolist(),cuts=[list(x) for x in cuts],cut_negativity=cutsN,weighted_cut_negativity=[s*v for v in cutsN]))
  if (k+1)%24==0:print('CONTEXT',k+1,'rows',len(rows),flush=True)
 assert len(rows)==72*2*4*3*3 and len(meta)==576
 np.savez_compressed(S/'JOINT95.npz',meta=np.asarray(meta),**{f'J{d}':np.array(x) for d,x in matrices.items()})
 dump(S/'ENTANGLE95.json',dict(status='PASS_SCOPED',raw_receipts=receipts,raw_paths=720,contexts=72,joint_prefixes=576*4,controlled_prefixes=len(rows),pair_evaluations=sum(len(x['pairs']) for x in rows),cut_evaluations=sum(len(x['cuts']) for x in rows),fields=FIELDS,rows=rows,controls=refs,errors=dict(ERR),joint_sha=sha(S/'JOINT95.npz'),elapsed_s=time.monotonic()-t0,scope='Explicit adaptive continuing-prefix states, not native observed coherent branches or Bell experiment. No new searches.'))
 print('DONE',len(rows),sum(len(x['pairs']) for x in rows),sum(len(x['cuts']) for x in rows),dict(ERR),flush=True)
if __name__=='__main__':main()
