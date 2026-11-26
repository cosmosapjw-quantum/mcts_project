"""Finite adaptive two-record instrument on enumerated actual native branches.
Classical feedback is diagonal; coherent control is an explicitly defined dilation.
"""
from base61 import *
import gzip
from audit_branch73 import audit_one
from scipy.special import xlogy
I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1.,-1.]).astype(complex)
U=np.cos(.35)*I-1j*np.sin(.35)*X
ETA=np.exp(-1.2*4/128)
def dag(x):return x.swapaxes(-1,-2).conj()
def rho(v):return (I+v[0]*X+v[1]*Y+v[2]*Z)/2
def bloch(r):return np.array([np.trace(r@a).real for a in (X,Y,Z)])
def openraw(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def getlaw(e):return e['root_selection']['law']
def ops(law,legal,src):
 a,b=src;h=law['phase65'];idx=np.argsort(law['actions']);assert np.array_equal(np.array(law['actions'])[idx],legal)
 sc=np.asarray(law['scores'])[idx];z=law['beta_p']*sc;z-=z.max();base=np.exp(z);base/=base.sum()
 ka,kb=np.searchsorted(legal,src);m=base[ka]+base[kb];r=base[ka]/m;e0=.35*r+.65;e1=.35*r
 assert max(abs(m-h['pair_mass']),abs(r-h['population_a']),abs(e0-h['e0']),abs(e1-h['e1']))<1e-11
 K=np.sqrt(base)[:,None,None]*I[None]
 K[ka]=np.sqrt(m)*dag(U)@np.diag(np.sqrt([e0,e1]))@U
 K[kb]=np.sqrt(m)*dag(U)@np.diag(np.sqrt([1-e0,1-e1]))@U
 assert np.max(abs(np.sum(dag(K)@K,axis=0)-I))<1e-11
 q=np.trace(K@rho(h['prepared'])@dag(K),axis1=-2,axis2=-1).real
 assert np.max(abs(q-np.asarray(law['p'])[idx]))<2e-11
 return K,q
def free(phi):
 V=np.diag([np.exp(-.5j*phi),np.exp(.5j*phi)])
 return np.array([np.sqrt((1+ETA)/2)*V,np.sqrt((1-ETA)/2)*Z@V])
def H(p):
 p=np.maximum(p,0);return float(-np.sum(xlogy(p,p)))
def ent(r):
 e=np.linalg.eigvalsh(r);assert e.min()>-1e-10 and abs(e.sum()-1)<1e-10
 return H(e)
def neg(r,da,db):
 q=r.reshape(da,db,da,db).transpose(0,3,2,1).reshape(da*db,da*db)
 return float(np.maximum(0,-np.linalg.eigvalsh(q)).sum())
def trace_distance(a,b):return float(.5*np.sum(abs(np.linalg.eigvalsh(a-b))))
def coarse(k,l,fs,r,grp,flag):
 # Amplitudes in A,B,S; bath and purification index shared; fine-rest labels retained in orthogonal flags.
 n=len(k);nf=int(flag.max())+1;v,Q=np.linalg.eigh(r);sq=(Q*np.sqrt(np.maximum(v,0)))@dag(Q)
 T=l[:,:,None]@fs[:,None]@k[:,None,None]@sq # i,j,bath,system,purification
 ii,jj,aa,ss,rr=np.indices(T.shape)
 row=((grp[ii]*3+grp[jj])*2+ss)
 col=((flag[ii]*nf+flag[jj])*2+aa)*2+rr
 V=np.zeros((18,nf*nf*4),complex);V[row.ravel(),col.ravel()]=T.ravel()
 q=V@dag(V);assert abs(np.trace(q)-1)<2e-10
 return q,T
def metrics(q):
 rr=q.reshape(9,2,9,2);R=np.einsum('asbs->ab',rr);s=np.einsum('asat->st',rr)
 a=np.einsum('aibi->ab',R.reshape(3,3,3,3));b=np.einsum('aiaj->ij',R.reshape(3,3,3,3))
 cl=np.diag(R).real.reshape(3,3);mi=ent(a)+ent(b)-ent(R);cmi=H(cl.sum(0))+H(cl.sum(1))-H(cl)
 de=q.reshape(3,6,3,6).copy()
 for i in range(3):
  for j in range(3):
   if i!=j:de[i,:,j,:]=0
 de=de.reshape(18,18);rd=np.einsum('asbs->ab',de.reshape(9,2,9,2))
 # Coherent record readout on pair a,b subspace with exterior outcome 0.
 x=np.zeros((3,3),complex);x[0,1]=x[1,0]=1;y=x.copy();y[0,1]=-1j;y[1,0]=1j
 zz=np.diag([1.,-1.,0.]);obs=[float(np.trace(R@np.kron(o,p)).real) for o,p in [(x,x),(x,y),(y,x),(y,y),(x,zz),(y,zz)]]
 return dict(record_MI=mi,record_classical_MI=cmi,record_neg=neg(R,3,3),system_record_neg=neg(q,9,2),system_entropy=ent(s),joint_entropy=ent(q),record_entropy=ent(R),phase_readout=obs,first_record_dephasing_TD=trace_distance(q,de),pair_dephasing_TD=trace_distance(R,rd),classical_feedback_record_neg=neg(rd,3,3),record_diagonal=cl.tolist(),record_density_real=R.real.tolist(),record_density_imag=R.imag.tolist(),system_record_density_real=q.real.tolist(),system_record_density_imag=q.imag.tolist())
def clean_events(x):
 return [{k:v for k,v in e.items() if k not in ('worker',)} for e in x]
def calc(c):
 j=c['job'];src=j['sources'];legal=np.asarray(c['legal']);n=len(legal);tag='branch'+str(j['generation'])
 p=B/c['source_path'];assert sha(p)==c['source_sha256'];original=openraw(p)['trace'];o=original['events'];law=getlaw(o[15]);K,P=ops(law,legal,src);r=rho(law['phase65']['prepared'])
 group=np.full(n,2,int);flag=np.arange(1,n+1);ia,ib=np.searchsorted(legal,src);group[ia]=0;group[ib]=1;flag[[ia,ib]]=0
 # All fine first actions retained before a fixed local compression, never coarse-grained before adaptive update.
 fs=[];l=[];ps=[];raw=[];errors=[];states=[];rot=[];rngmatch=0
 for i,a in enumerate(legal):
  bid=c['branch_ids'][i];qpath=S/'data'/tag/(bid+'.json.gz');d=openraw(qpath);bj=d['input']
  assert bj['id']==bid and bj['forced73']==c['prefix']+[int(a)] and all(bj[k]==v for k,v in j.items() if k!='id')
  audit=audit_one((tag,bj));errors.append([audit['max_probability'],audit['max_state'],audit['max_score'],audit['max_log']])
  tr=d['trace'];es=tr['events'];assert clean_events(es[:15])==clean_events(o[:15])
  e=getlaw(es[15]);assert e['actions']==law['actions'] and e['p']==law['p'] and e['scores']==law['scores'] and e['phase65']['prepared']==law['phase65']['prepared']
  assert es[15]['actions'][0]==a
  nextlaw=getlaw(es[16]);k,q=ops(nextlaw,legal,src);F=free(nextlaw['phase65']['rotation'])
  post=K[i]@r@dag(K[i])/P[i];prep=np.sum(F@post@dag(F),axis=0)
  er=max(np.max(abs(bloch(post)-np.array(e['phase65']['post']))),np.max(abs(bloch(prep)-np.array(nextlaw['phase65']['prepared']))))
  assert er<2e-11
  states.append(float(er));fs.append(F);l.append(k);ps.append(q);rot.append(nextlaw['phase65']['rotation'])
  assert audit['sha256']==sha(qpath)
  raw.append(dict(id=bid,path=str(qpath.relative_to(B)),sha256=audit['sha256'],bytes=audit['bytes'],forced_action=int(a),evaluations=audit['evaluations']))
  if a==c['observed']:
   assert clean_events(es)==clean_events(o[:32]) and tr['evaluations']==original['evaluations'][:len(tr['evaluations'])] and tr['cuts']==original['cuts'][:len(tr['cuts'])]
   rngmatch=1
 assert rngmatch
 fs=np.array(fs);l=np.array(l);ps=np.array(ps)
 C,T=coarse(K,l,fs,r,group,flag)
 joint=P[:,None]*ps;diag=np.sum(abs(T)**2,axis=(2,3,4));derr=float(abs(diag-joint).max());assert derr<2e-11 and abs(joint.sum()-1)<2e-11
 diag3=np.zeros((3,3))
 for i in range(n):
  for k in range(n):diag3[group[i],group[k]]+=joint[i,k]
 met=metrics(C);assert np.max(abs(diag3-met['record_diagonal']))<2e-11
 ix=int(np.searchsorted(legal,c['observed']));frozen,_=coarse(K,np.broadcast_to(l[ix],l.shape),np.broadcast_to(fs[ix],fs.shape),r,group,flag);fm=metrics(frozen)
 # All conditional branches use full Kraus channel; also compare measured first record.
 fr=np.asarray(fm['record_diagonal']);tv=float(.5*abs(diag3-fr).sum())
 # Independent operational measure: expected distance from actual observed-branch frozen conditional kernel with same prepared branch state.
 dist=[]
 for i in range(n):
  post=K[i]@r@dag(K[i])/P[i];prep=np.sum(fs[ix]@post@dag(fs[ix]),axis=0)
  q=np.trace(l[ix]@prep@dag(l[ix]),axis1=-2,axis2=-1).real
  dist.append(float(.5*abs(q-ps[i]).sum()))
 result=dict(generation=j['generation'],arm=j['arm'],core=j['core'],n_legal=n,source_pair=src,context_source=c['source_path'],context_source_sha=c['source_sha256'],first_pointer_bloch=law['phase65']['prepared'],rotation_range=[min(rot),max(rot)],adaptive=met,frozen=fm,adaptive_vs_frozen_TD=trace_distance(C,frozen),adaptive_vs_frozen_classical_TV=tv,expected_conditional_TV=float(P@np.array(dist)),max_conditional_TV=max(dist),max_native_replay=np.max(errors,axis=0).tolist(),max_prepared_state_error=max(states),diag_probability_error=derr,matched_original_32=True,raw_receipts=raw)
 return result
def main():
 R=read(S/'BRANCH_REGISTRATION73.json');assert sha(S/'mqft_branch73')==R['binary_sha'] and sha(S/'branch73.patch')==R['patch_sha']
 for p,q in R['plans'].items():assert sha(S/'plans'/q['path'])==q['sha256'] and len(list((S/'data'/('branch'+p)).glob('*.gz')))==q['count']
 rows=[]
 with cf.ProcessPoolExecutor(max_workers=12) as pool:
  for k,q in enumerate(pool.map(calc,R['contexts'],chunksize=1)):
   rows.append(q)
   if (k+1)%8==0:print('INSTRUMENT',k+1,flush=True)
 assert len(rows)==64 and sum(q['n_legal'] for q in rows)==4544
 result=dict(scope='exact two-record conditional probabilities under forced prefix, explicit coherent-control dilation (not native quantum engine); local a/b coherent compression with all rest flags traced; bath register shared by specified model; measured first record is actual diagonal feedback',contexts=rows,branch_paths=4544,branch_backups=4544*32,raw_bytes=sum(x['bytes'] for q in rows for x in q['raw_receipts']),max_replay=np.max([q['max_native_replay'] for q in rows],axis=0).tolist(),max_diagonal_error=max(q['diag_probability_error'] for q in rows),max_prepared_state_error=max(q['max_prepared_state_error'] for q in rows))
 dump(S/'INSTRUMENT73.json',result)
 for g in (0,3):
  for arm in ('persist_plus','persist_minus'):
   a=[q for q in rows if q['generation']==g and q['arm']==arm]
   print('RESULT',g,arm,'conditional TV',np.mean([q['expected_conditional_TV'] for q in a]),'joint TV',np.mean([q['adaptive_vs_frozen_classical_TV'] for q in a]),'pair negativity',np.mean([q['adaptive']['record_neg'] for q in a]),'frozen',np.mean([q['frozen']['record_neg'] for q in a]),'phase TD',np.mean([q['adaptive']['first_record_dephasing_TD'] for q in a]),flush=True)
 print('AUDIT', {k:v for k,v in result.items() if k!='contexts'},flush=True)
if __name__=='__main__':main()
