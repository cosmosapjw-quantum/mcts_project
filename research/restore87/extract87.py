"""Pinned raw path replay, full-action predictable covariance, and matched spatial modes."""
from base61 import *
import sys,gzip,subprocess
from mqft.campaign import verify_state,validate_receipt,load_plan
from mqft.path_law import replay
from mqft.rapfi import verify
L=int(sys.argv[1]);N=L*L;O=B/'mqft_restore49_20261126';P=B/'mqft_restore39_20261125/repo';R=read(S/f'REGISTRATION_L{L}.json')
assert sha(O/'runtime_verified'/f'mqft-L{L}')==R['backend_sha'] and verify(P/'.external/rapfi')==R['model']
assert sha(S/'THEORY_REGISTERED87.md')==R['contract_sha'] and sha(O/'family'/f'L{L}'/'SOURCE_MANIFEST.json')==R['source_manifest_sha']
helper=B/'mqft_restore46_20261126/runtime/mqft-rng-replay';assert sha(helper)==read(B/'mqft_restore46_20261126/PHYSICS_RESULTS.json')['rng_replay_sha256']
rp=subprocess.Popen([str(helper)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
xx=np.arange(N)%L;yy=np.arange(N)//L
F=np.stack([np.cos(np.pi*k*(a+.5)/L) for k in (1,2) for a in (xx,yy)],1)
ERR=collections.defaultdict(float);M=[];X=[];A=[];Q=[];NULL=[];FR=[];REC=[];FROZEN=[];SM=[];SP=[];SC=[];startup_max=0.;startup_changed=0;draws=samples=0
def error(k,a,b):
 ERR[k]=max(ERR[k],float(np.max(abs(np.asarray(a)-np.asarray(b)))))
def probs(beta,s):
 z=beta*s;z-=z.max();p=np.exp(z);return p/p.sum()
def stats(p,f):
 mu=p@f;v=p@(f*f)-mu*mu
 return np.r_[1-p@p,.5*(v[0]+v[1]),.5*(v[2]+v[3])]
try:
 for cohort,st in enumerate((76,77,87)):
  root=B/f'mqft_restore{st}_202611{28 if st==87 else 27}'
  r=read(root/f'REGISTRATION_L{L}.json');plan=load_plan(root/'plans'/f'L{L}_base.json')
  assert r['backend_sha']==R['backend_sha'] and r['model']==R['model'] and sha(root/'plans'/f'L{L}_base.json')==r['plans']['base']['sha256']
  jobs={j['id']:j for j in plan['jobs']};dd=root/'data'/f'L{L}'/'base';report=read(dd/'report.json')
  assert not report['failures'] and report['unattempted']==0 and set(jobs)=={x['job_id'] for x in report['completed']}
  old={x['id']:x for x in read(root/f'AUDIT_L{L}.json')['raw_receipts']} if st==76 else {x['id']:x for x in read(root/f'AUDIT_L{L}.json')['raw_receipts'] if x['cohort']==1} if st==77 else {}
  for rc in report['completed']:
   j=jobs[rc['job_id']];f=dd/rc['file'];assert read(dd/(j['id']+'.receipt.json'))==rc;validate_receipt(rc,f,j,plan['plan_hash'],R['backend_sha'])
   if st!=87:assert old[j['id']]['sha256']==rc['sha256'] and old[j['id']]['path']==str(f.relative_to(B))
   with gzip.open(f,'rt') as z:hd,row,end=map(json.loads,z)
   tr=row['trace'];sp=tr['spec'];k=sp['budget'];beta=sp['root_law']['beta_p'][0];rep=tr['replicas'][0];assert hd['job']==j and hd['backend_sha']==R['backend_sha'] and hd['evaluator_runtime']==R['model'] and end['backups']==k and end['status']=='root_complete'
   assert sp==dict(j['spec'],c_puct=float(np.float32(j['spec']['c_puct'])))
   verify_state(tr['root'],j['opening']);v=replay(rep,sp);ERR['score']=max(ERR['score'],v['root_score_max_error'])
   es=sorted(rep['events'],key=lambda e:e['root_selection']['clock']);assert len(es)==k and [e['root_selection']['law']['t'] for e in es]==list(range(k))
   if st==87:
    msg=dict(spec_seed=sp['seed'],replica_seed=rep['seed'],root_uniforms=[e['root_selection']['law']['u'] for e in es],evaluations=rep['evaluations'])
    rp.stdin.write(json.dumps(msg,separators=(',',':'))+'\n');rp.stdin.flush();q=json.loads(rp.stdout.readline());ERR['rng']=max(ERR['rng'],q['max_root_u_error']);ERR['sample_log']=max(ERR['sample_log'],q['max_sample_log_error']);draws+=q['root_draws'];samples+=q['sample_draws']
   a=np.array([e['actions'][0] for e in es]);z=np.array([e['root_value'] for e in es]);assert all(e['root_value']==(-1)**len(e['actions'])*e['leaf_value'] for e in es)
   board=np.array(tr['root']['board']);legal=np.flatnonzero(board==0);n=len(legal);f0=F[legal];pp=np.zeros((k,N));stat=[];spread=[];pmax=[];frozen=None
   for i,e in enumerate(es):
    law=e['root_selection']['law'];ac=np.array(law['actions']);s=np.array(law['scores']);p=np.array(law['p']);order=np.argsort(ac);s=s[order];p=p[order]
    assert np.array_equal(ac[order],legal) and all(law['active']) and law['logit_h']==0 and ac[law['picked']]==a[i]
    error('probability',p,probs(beta,s));assert min(p)>0 and abs(p.sum()-1)<1e-11
    pp[i,legal]=p;u=stats(p,f0);delta=float(np.ptp(s));top=float(p.max())
    bound=1/(1+(n-1)*np.exp(-beta*delta));assert top<=bound+1e-12 and u[0]<=1-1/n+1e-12 and min(u)>=-1e-12
    assert np.max(2*p*(1-p))<=.5+1e-15
    assert u[1]<=1+1e-12 and u[2]<=1+1e-12
    stat.append(u);spread.append(delta);pmax.append(top)
    if i==32 and st==87 and beta==10/3:
     frozen=[np.r_[stats(probs(b,s),f0),probs(b,s).max()] for b in (0.,10/3,8.,16.,math.log(L*L))]
   startup=float(abs(pp[:4]-pp[0]).max());startup_max=max(startup_max,startup);startup_changed+=int(startup>1e-12)
   error('startup_pair',pp[2:4],pp[2])
   for i in range(4,k,4):error('quiescent_wave',pp[i:i+4],pp[i])
   for cut in rep['cuts']:
    t=cut['local_k'];ee=sorted(rep['events'],key=lambda e:e['complete_seq'])[:t];nn=np.bincount([e['actions'][0] for e in ee],minlength=N);ww=np.bincount([e['actions'][0] for e in ee],weights=[e['root_value'] for e in ee],minlength=N);ss=np.bincount([e['actions'][0] for e in ee],weights=[e['root_value']**2 for e in ee],minlength=N)
    for ed in cut['edges']:
     ac=ed['action'];assert ed['n']==nn[ac] and ed['pending']==0;error('backup',ed['w'],ww[ac]);error('backup',ed['s2'],ss[ac])
   null=stats(np.ones(n)/n,f0)
   if beta==0:error('uniform',pp[:,legal],1/n);error('null',stat,np.tile(null,(k,1)))
   core=int(j['position_id'][1:3]);arm=int(j['position_id'].endswith('dense'));r0=j['replicate'];bi=(0.,10/3,8.,16.,math.log(L*L)).index(beta)
   if k==64:
    sec=np.digitize(np.maximum(abs(xx-L//2),abs(yy-L//2)),[2,4]);SM.append([cohort,core,arm,bi,r0]);SP.append(pp@np.eye(3)[sec]);SC.append(sec[a])
   for h in (16,32,64):
    if h>k:continue
    x=np.bincount(a[:h],minlength=N)/h;dr=pp[:h].mean(0);assert abs(x.sum()-1)<1e-12 and abs(dr.sum()-1)<1e-11
    M.append([cohort,core,arm,bi,r0,k,h]);X.append(x);A.append(dr);Q.append(np.mean(stat[:h],0));NULL.append(null);FR.append([np.mean(pmax[:h]),np.mean(spread[:h]),pmax[32] if k>32 else -1.])
   if frozen is not None:FROZEN.append(dict(core=core,arm=arm,rep=r0,values=np.array(frozen).tolist(),actual_H32_beta=beta))
   REC.append(dict(stage=st,cohort=cohort,core=core,arm=arm,beta_index=bi,rep=r0,budget=k,id=j['id'],path=str(f.relative_to(B)),sha256=rc['sha256'],bytes=rc['bytes'],seed=sp['seed']))
  print('AUDIT',L,st,len(REC),flush=True)
 rp.stdin.close();assert rp.wait(timeout=15)==0
finally:
 if rp.poll() is None:rp.kill();rp.wait()
assert len(REC)==864 and len(M)==2080 and draws==10240 and len(FROZEN)==32 and max(ERR.values())<1e-8,(len(REC),len(M),dict(ERR))
np.savez_compressed(S/f'OBS_L{L}.npz',meta=M,X=X,A=A,Q=Q,null=NULL,aux=FR,modes=F)
np.savez_compressed(S/f'SECTOR_L{L}.npz',meta=SM,p=SP,selected=SC)
dump(S/f'AUDIT_L{L}.json',dict(startup_max=startup_max,startup_changed=startup_changed,sector_sha=sha(S/f'SECTOR_L{L}.npz'),L=L,paths=len(REC),fresh_paths=160,fresh_backups=draws,prior_paths=704,raw_receipts=REC,array_sha=sha(S/f'OBS_L{L}.npz'),errors=dict(ERR),candidate_samples=samples,frozen=FROZEN,scope='Reauthenticated all base paths; fresh full native path/RNG checks, full legal conditional covariance trace, low/high cosine modes; B16/32/64 prefix and separate B16 budget distinguished.'))
print('AUDIT COMPLETE',L,len(REC),dict(ERR),flush=True)
