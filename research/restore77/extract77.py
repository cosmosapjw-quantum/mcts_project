"""Receipt-authenticated R76 training and R77 independent test trajectories."""
from base61 import *
import sys,gzip,subprocess
from mqft.campaign import verify_state,validate_receipt,load_plan
from mqft.path_law import replay
from mqft.rapfi import verify
L=int(sys.argv[1]); F=B/'mqft_restore49_20261126'; P=B/'mqft_restore39_20261125/repo'; OLD=B/'mqft_restore76_20261127'; R=read(S/f'REGISTRATION_L{L}.json')
assert sha(F/'runtime_verified'/f'mqft-L{L}')==R['backend_sha'] and verify(P/'.external/rapfi')==R['model'] and sha(F/'family'/f'L{L}'/'SOURCE_MANIFEST.json')==R['source_manifest_sha']
helper=B/'mqft_restore46_20261126/runtime/mqft-rng-replay';assert sha(helper)==read(B/'mqft_restore46_20261126/PHYSICS_RESULTS.json')['rng_replay_sha256']
rp=subprocess.Popen([str(helper)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
rg=np.digitize(np.maximum(abs(np.arange(L*L)//L-L//2),abs(np.arange(L*L)%L-L//2)),[2,4]);proj=np.eye(3)[rg]
M=[];C=[];Z=[];W=[];PP=[];Q=[];BO=[];PR=[];receipts=[];errs=collections.defaultdict(float);rngdraws=sampledraws=0
def err(k,a,b):errs[k]=max(errs[k],float(np.max(abs(np.asarray(a)-np.asarray(b)))))
try:
 for cohort,root in enumerate((OLD,S)):
  r=read(root/f'REGISTRATION_L{L}.json');plan=load_plan(root/'plans'/f'L{L}_base.json')
  assert r['backend_sha']==R['backend_sha'] and r['model']==R['model'] and sha(root/'plans'/f'L{L}_base.json')==r['plans']['base']['sha256']
  jobs={j['id']:j for j in plan['jobs']};dd=root/'data'/f'L{L}'/'base';report=read(dd/'report.json')
  assert not report['failures'] and report['unattempted']==0 and set(jobs)=={a['job_id'] for a in report['completed']}
  oldrec={x['id']:x for x in read(OLD/f'AUDIT_L{L}.json')['raw_receipts']} if cohort==0 else {}
  for rc in report['completed']:
   j=jobs[rc['job_id']];sp=j['spec'];beta=sp['root_law']['beta_p'][0]
   if sp['budget']!=64 or beta not in (0.,10/3,16.):continue
   f=dd/rc['file'];assert read(dd/(j['id']+'.receipt.json'))==rc;validate_receipt(rc,f,j,plan['plan_hash'],R['backend_sha'])
   if cohort==0:assert oldrec[j['id']]['sha256']==rc['sha256'] and oldrec[j['id']]['path']==str(f.relative_to(B))
   with gzip.open(f,'rt') as z:hd,row,end=map(json.loads,z)
   t=row['trace'];spec=t['spec'];assert hd['job']==j and hd['evaluator_runtime']==R['model'] and hd['backend_sha']==R['backend_sha'] and end['backups']==64 and end['status']=='root_complete'
   assert spec==dict(sp,c_puct=float(np.float32(sp['c_puct'])))
   verify_state(t['root'],j['opening']);rep=t['replicas'][0];a0=replay(rep,spec);errs['root_score']=max(errs['root_score'],a0['root_score_max_error'])
   es=sorted(rep['events'],key=lambda e:e['root_selection']['clock']);assert len(es)==64 and [e['root_selection']['law']['t'] for e in es]==list(range(64))
   msg=dict(spec_seed=spec['seed'],replica_seed=rep['seed'],root_uniforms=[e['root_selection']['law']['u'] for e in es],evaluations=rep['evaluations'])
   rp.stdin.write(json.dumps(msg,separators=(',',':'))+'\n');rp.stdin.flush();q=json.loads(rp.stdout.readline());errs['root_u']=max(errs['root_u'],q['max_root_u_error']);errs['sample_log']=max(errs['sample_log'],q['max_sample_log_error']);rngdraws+=q['root_draws'];sampledraws+=q['sample_draws']
   a=np.array([e['actions'][0] for e in es]);z=np.array([e['root_value'] for e in es]);assert np.all(abs(z)<=1+1e-7) and all(e['root_value']==(-1)**len(e['actions'])*e['leaf_value'] for e in es)
   board=np.array(t['root']['board']);legal=np.flatnonzero(board==0);pp=np.zeros((64,L*L));scores=np.zeros_like(pp)
   for i,e in enumerate(es):
    law=e['root_selection']['law'];ac=np.array(law['actions']);v=np.array(law['p']);sc=np.array(law['scores'])
    assert np.array_equal(np.sort(ac),legal) and all(law['active']) and ac[law['picked']]==a[i] and law['logit_h']==0.
    ll=beta*sc;ll-=ll.max();pred=np.exp(ll);pred/=pred.sum();err('probability',pred,v);pp[i,ac]=v;scores[i,ac]=sc
    if beta==0:err('uniform',v,1/len(legal))
   for cut in rep['cuts']:
    n=cut['local_k'];ev=sorted(rep['events'],key=lambda e:e['complete_seq'])[:n];nn=np.bincount([e['actions'][0] for e in ev],minlength=L*L);ww=np.bincount([e['actions'][0] for e in ev],weights=[e['root_value'] for e in ev],minlength=L*L);ss=np.bincount([e['actions'][0] for e in ev],weights=[e['root_value']**2 for e in ev],minlength=L*L)
    for e in cut['edges']:
     aa=e['action'];assert nn[aa]==e['n'] and e['pending']==0;err('backup',ww[aa],e['w']);err('backup',ss[aa],e['s2'])
   # Quiescent full waves begin at t=4. Verify actual equality of conditional laws within each wave.
   for k in range(4,64,4):err('wave_constant',pp[k:k+4],pp[k])
   c=proj[a].reshape(16,4,3).sum(1).astype(int);w=(z[:,None]*proj[a]).reshape(16,4,3).sum(1);p=(pp@proj).reshape(16,4,3).mean(1)
   qq=((pp*scores*scores).sum(1)-(pp*scores).sum(1)**2).reshape(16,4).mean(1)
   prior=np.zeros(L*L)
   for e in rep['cuts'][-1]['edges']:prior[e['action']]=e['prior']
   assert abs(prior.sum()-1)<2e-6;prior/=prior.sum()
   pos=j['position_id'];M.append([cohort,int(pos[1:3]),int(pos.endswith('dense')),(0.,10/3,16.).index(beta),j['replicate']]);C.append(c);Z.append(z.reshape(16,4).mean(1));W.append(w);PP.append(p);Q.append(qq);BO.append(board);PR.append(prior@proj)
   receipts.append(dict(cohort=cohort,id=j['id'],path=str(f.relative_to(B)),sha256=rc['sha256'],bytes=rc['bytes'],seed=spec['seed'],budget=64))
  print('AUDIT',L,cohort,len(M),flush=True)
 rp.stdin.close();assert rp.wait(timeout=15)==0
finally:
 if rp.poll() is None:rp.kill();rp.wait()
assert len(M)==384 and len(set(map(tuple,M)))==384 and rngdraws==24576 and max(errs.values())<1e-8
np.savez_compressed(S/f'PATHS_L{L}.npz',meta=M,counts=C,marks=Z,sector_marks=W,prob=PP,score_var=Q,board=BO,prior=PR)
dump(S/f'AUDIT_L{L}.json',dict(L=L,train_paths=192,fresh_paths=192,train_backups=12288,fresh_backups=12288,root_draws=rngdraws,candidate_samples=sampledraws,errors=dict(errs),train_bytes=sum(x['bytes'] for x in receipts if x['cohort']==0),fresh_bytes=sum(x['bytes'] for x in receipts if x['cohort']==1),array_sha256=sha(S/f'PATHS_L{L}.npz'),raw_receipts=receipts,scope='R76 independently authenticated selected B64 subset plus all R77 independent geometry paths, exact root/RNG/backup replay'))
print('DONE',L,dict(errs),flush=True)
