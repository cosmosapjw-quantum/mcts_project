from instrument81 import *
import gzip,time
from scipy.stats import t as student
O81=B/'mqft_restore81_20261127';O83=B/'mqft_restore83_20261127'
FIELDS=['Cz_pre','Cz_prepared','Cz_post','Cr_pre','Cr_prepared','Cr_post','S_pre','S_prepared','S_post','S_nonselective','mean_S_selective','Holevo_outcome','selective_info_gain','free_entropy_change','measurement_entropy_change','gamma','readout_TV','local_transverse_infidelity','actual_entropy_change','axis_alignment','phase_a_difference_from_base','purity_post']
PART=['conditional_PR','base_PR','occupancy_PR','legal_n','conditional_entropy','occupancy_entropy']
def ent(p):
 p=np.asarray(p);p=p[p>0];return float(-p@np.log(p))
def vn(v):
 r=float(np.linalg.norm(v));assert r<1+1e-10
 return ent([(1+min(r,1))/2,(1-min(r,1))/2])
def c1(v):return float(np.linalg.norm(v[:2]))
def cr(v):return vn(np.array([0.,0.,v[2]]))-vn(v)
def entmat(a):
 e=np.linalg.eigvalsh(a);assert e.min()>-1e-10
 return ent(e)
def norm(x):return float(abs(x).max())
def ci(x):
 x=np.asarray(x);m=x.mean(0);h=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return np.stack([m,m-h,m+h]).tolist()
def raw_one(arg):
 kind,r=arg;f=B/r['path'];assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256']
 with gzip.open(f,'rt') as z:d=json.load(z)
 j=d['input'];mode=ARMS[r['arm']];assert j['arm']==mode and j['sources']==r['sources'];es=d['trace']['events'];assert len(es)==128
 src=r['sources'];v=np.array([.8,0.,0.]);part=[];features=[];states=[];actions=[];marks=[];counts={};errors=collections.defaultdict(float);mat=np.eye(3);local=[];infos=[]
 def check(k,a,b,tol=1e-9):
  err=norm(np.asarray(a)-np.asarray(b));errors[k]=max(errors[k],err);assert err<tol,(k,err,f.name)
 for t,e in enumerate(es):
  l=e['root_selection']['law'];p=np.array(l['p']);s=np.array(l['scores']);a=np.array(l['actions']);pick=l['picked'];ac=e['actions'][0];assert a[pick]==ac and l['t']==t and all(l['active']) and min(p)>0 and abs(p.sum()-1)<1e-10
  assert e['root_value']==(-1)**len(e['actions'])*e['leaf_value'];actions.append(ac);marks.append(e['root_value'])
  base=np.exp((10/3)*(s-s.max()));base/=base.sum();ia=int(np.flatnonzero(a==src[0])[0]);ib=int(np.flatnonzero(a==src[1])[0])
  m=float(base[ia]+base[ib]);rr=float(base[ia]/m);ph=l['phase65'];assert ph['sources']==src
  check('pair_mass',m,ph['pair_mass']);check('pair_population',rr,ph['population_a']);check('pre',v,ph['pre'])
  counts[ac]=counts.get(ac,0)+1;x=np.array(list(counts.values()))/(t+1)
  part.append([1/(p@p),1/(base@base),1/(x@x),len(p),ent(p),ent(x)])
  if 32<=t<44:
   ax,lam,theta,eta,erase=schedule(mode,t);check('axis',np.eye(3)[ax],ph['axis81']);check('lam',lam,ph['lambda81'])
   r0=rho(v);pr=free(r0,theta,eta,erase);vp=bloch(pr);check('prepared',vp,ph['prepared'])
   Ka,Kb,Ko,e0,e1=operators(m,rr,ax,lam);ks=[Ka,Kb,Ko];out=[k@pr@adj(k) for k in ks];probs=np.array([np.trace(a0).real for a0 in out]);assert min(probs)>0;check('complete',sum(adj(k)@k for k in ks),I)
   non=sum(out);gamma=1-m+m*(np.sqrt(e0*e1)+np.sqrt((1-e0)*(1-e1)))
   pred=gamma*vp;pred[ax]=vp[ax];check('nonselective_bloch',pred,bloch(non))
   z=0 if pick==ia else 1 if pick==ib else 2
   post=out[z]/probs[z];q=base.copy();q[ia]=probs[0];q[ib]=probs[1];check('probability',q,p);check('state',bloch(post),ph['post'])
   qsc,vsc=scalar_step(mode,t,v.copy(),base,ia,ib,pick);check('scalar_p',qsc,p);check('scalar_post',vsc,bloch(post))
   v1=bloch(post);sp=vn(vp);sn=entmat(non);ss=sum(probs[k]*entmat(out[k]/probs[k]) for k in range(3));ch=sn-ss;info=sp-ss
   assert sn>=sp-1e-10 and info>=-1e-10 and ch>=-1e-10 and vn(vp)>=vn(v)-1e-10
   check('entropy_balance',sn-sp,ch-info);check('entropy_eigen',vn(v1),entmat(post));assert min(np.linalg.eigvalsh(post))>-1e-10 and abs(np.trace(post)-1)<1e-10
   ptr=rho(np.eye(3)[ax]);ptr_out=sum(k@ptr@adj(k) for k in ks);check('pointer_fixed',ptr,ptr_out)
   pp=np.array([m*e0,m*(1-e0),1-m]);pm=np.array([m*e1,m*(1-e1),1-m]);check('readout_TV',.5*abs(pp-pm).sum(),m*lam)
   # exact local mutual information of equiprobable pointer bit and grouped readout
   infos.append(ent((pp+pm)/2)-(ent(pp)+ent(pm))/2)
   Rz=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
   D=np.diag([0 if erase else eta,0 if erase else eta,1.]);M=np.eye(3)*gamma;M[ax,ax]=1.;mat=M@D@Rz@mat
   features.append([c1(v),c1(vp),c1(v1),cr(v),cr(vp),cr(v1),vn(v),sp,vn(v1),sn,ss,ch,info,sp-vn(v),sn-sp,gamma,m*lam,(1-gamma)/2,vn(v1)-vn(v),abs(v1[ax]),q[ia]-base[ia],(1+v1@v1)/2]);states.append(v1);v=v1
  else:
   check('inactive_probability',base,p);check('inactive_post',v,ph['post'])
   assert ph['lambda81']==0 and ph['eta81']==1
 # pure input frozen unselective map: right singular vector is entropy-minimizing at endpoint
 u,sig,Vh=np.linalg.svd(mat);assert sig[0]<=1+1e-10
 ax=Vh[0];lens=np.linalg.norm(mat,axis=0);check('svd',np.linalg.norm(mat@ax),sig[0])
 stab=np.r_[sig,ax*ax,lens,[vn(mat@ax)], [vn(mat@np.eye(3)[k]) for k in range(3)]]
 meta=[r[k] for k in ['family','core','rep','arm']]
 return meta,np.array(features),np.array(part),np.array(states),np.array(stab),np.array(infos),np.array(actions),np.array(marks),dict(errors),r['path'],r['sha256']
def main():
 begin=time.monotonic();a=read(O81/'AUDIT81.json');b=read(O83/'AUDIT83.json');assert sha(O81/'PATHS81.npz')==a['array_sha'];assert sha(O83/'TREE83.json')==b['tree_sha']
 allout=[];errs=collections.defaultdict(float);rec={}
 for kind,rows in [(81,a['raw_receipts']),(83,b['raw_receipts'])]:
  out=[]
  with cf.ProcessPoolExecutor(max_workers=10) as pool:
   for i,z in enumerate(pool.map(raw_one,[(kind,r) for r in rows],chunksize=8)):
    out.append(z)
    for k,v in z[8].items():errs[k]=max(errs[k],v)
    assert z[9] not in rec;rec[z[9]]=z[10]
    if (i+1)%576==0:print('RAW',kind,i+1,flush=True)
  allout.append(out)
 assert len(rec)==4104
 z=allout[0];meta,F,P,V,St,I0,Ac,Ma=[np.stack([r[i] for r in z]) for i in range(8)]
 with np.load(O81/'PATHS81.npz') as p:
  ix={tuple(m):i for i,m in enumerate(meta)};ii=[ix[tuple(m)] for m in p['meta']]
  assert np.array_equal(Ac[ii],p['actions']) and np.array_equal(Ma[ii],p['marks'])
  errs['R81_states']=norm(V[ii]-p['states']);assert errs['R81_states']<1e-9
 # 576 forced/replay data verified directly; compare tree leaf source and recorded product relative
 tree=read(O83/'TREE83.json');by={r[9]:r for r in allout[1]};old={r['path']:r for r in b['raw_receipts']}
 for c in tree['contexts']:
  for l in c['leaves']:
   z0=by[l['path']];assert old[l['path']]['sha256']==l['sha256'] and old[l['path']]['forced']
   assert np.array_equal(z0[6][32:44],l['root_actions'])
   # state_after is prior format inspected separately; actual probability checked from log in raw file.
   with gzip.open(B/l['path'],'rt') as f:tr=json.load(f)['trace']
   pp=np.array([e['root_selection']['law']['log_p'] for e in tr['events'][32:44]])
   assert norm(pp-np.asarray(l['logp_active']))<1e-10 and abs(np.exp(pp.sum())/l['probability']-1)<1e-9
 # use full native R81 actual trajectories for means; R83 forced is not sampled cohort.
 idx={tuple(m):i for i,m in enumerate(meta)};families=[]
 for g in range(3):
  d=np.array([[[idx[g,c,r,a] for a in range(6)] for r in range(8)] for c in range(24)])
  f=F[d];p=P[d];st=St[d];info=I0[d];act=Ac[d]
  s=V[d]
  assert np.array_equal(act[:,:,2],act[:,:,3]) and norm(s[:,:,2]-s[:,:,3])<1e-10
  # explicit paired phase-sensitivity at first draw and final recommendation; no performance claim
  source=np.array([a['raw_receipts'][idx[g,c,r,0]]['sources'] for c in range(24) for r in range(8)]).reshape(24,8,2)
  sig=(act==source[:,:,None,0,None]).astype(float)-(act==source[:,:,None,1,None]).astype(float)
  recommend=np.array([np.bincount(a,minlength=225).argmax() for a in act.reshape(-1,128)]).reshape(24,8,6)
  # mix over rep states first, then entropy: empirical mixed conditional ensemble and Jensen gap
  meanv=s.mean(1);mixed=np.array([[[vn(v) for v in arm] for arm in core] for core in meanv])
  meanS=f[:,:,:,:,8].mean(1);assert np.min(mixed-meanS)>-1e-10
  families.append(dict(family=FAMS[g],roots=24,reps=8,features_ci=ci(f.mean(1)),participation_ci=ci(p.mean(1)),frozen_stability_ci=ci(st.mean(1)),pointer_mi_ci=ci(info.mean(1)),empirical_mix_entropy_ci=ci(mixed),empirical_jensen_gap_ci=ci(mixed-meanS),phase_first_signed_ci=ci((sig[:,:,0,32]-sig[:,:,1,32]).mean(1)),phase_active_signed_ci=ci((sig[:,:,0,32:44]-sig[:,:,1,32:44]).mean((1,2))),recommend_disagreement=int(np.count_nonzero(recommend[:,:,0]!=recommend[:,:,1])),conditional_entropy_decreases=int(np.count_nonzero(f[...,18]<-1e-10)),conditional_entropy_increases=int(np.count_nonzero(f[...,18]>1e-10)),conditional_entropy_unchanged=int(np.count_nonzero(abs(f[...,18])<=1e-10))))
 np.savez_compressed(S/'NATIVE94.npz',meta=meta,features=F,participation=P,states=V,stability=St,pointer_mi=I0)
 dump(S/'NATIVE94.json',dict(status='PASS_SCOPED',paths=3456,forced_diagnostics=648,raw_receipts=rec,active_events=3456*12,total_events=3456*128,diagnostic_events=648*128,fields=FIELDS,participation_fields=PART,arms=ARMS,stability_fields=['s1','s2','s3','opt_x2','opt_y2','opt_z2','norm_X','norm_Y','norm_Z','S_opt','S_X','S_Y','S_Z'],families=families,errors=dict(errs),array_sha=sha(S/'NATIVE94.npz'),elapsed_s=time.monotonic()-begin,scope='Actual programmed state conditioned on recorded history; exact local nonselective counterfactual at fixed history, not global adaptive unselective law; frozen sequence pointer stability only.'))
 print('NATIVE DONE',len(rec),dict(errs),flush=True)
if __name__=='__main__':main()
