from base61 import *
import gzip,sys,subprocess
from mqft.campaign import verify_state,valid_moves
from mqft.path_law import replay
from mqft.rapfi import verify
L=int(sys.argv[1]);O=B/'mqft_restore49_20261126';P=B/'mqft_restore39_20261125/repo';R=read(S/f'REGISTRATION_L{L}.json');plan=read(S/'plans'/f'L{L}.json');d=S/'data'/f'L{L}'
assert sha(O/'runtime_verified'/f'mqft-L{L}')==R['backend_sha'] and verify(P/'.external/rapfi')==R['model']
assert sha(O/'family'/f'L{L}'/'SOURCE_MANIFEST.json')==R['source_manifest_sha']
lock=read(d/'run_lock.json');assert lock['backend_sha']==R['backend_sha'] and lock['external']==R['model'] and lock['plan_hash']==plan['plan_hash']==R['plan_hash']
reports=[read(x) for x in d.glob('report-*.json')];good=[x for x in reports if len(x['completed'])==64 and not x['failures'] and x['unattempted']==0];assert len(good)>=1
jobs={j['id']:j for j in plan['jobs']};receipts={q['job_id']:q for q in good[-1]['completed']};assert set(receipts)==set(jobs)
helper=B/'mqft_restore46_20261126/runtime/mqft-rng-replay';assert sha(helper)==read(B/'mqft_restore46_20261126/PHYSICS_RESULTS.json')['rng_replay_sha256']
p=subprocess.Popen([str(helper)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
rows=[];act=[];val=[];probs=[];priors=[];boards=[];meta=[];maxerr=maxu=maxlog=0.;draws=samples=0
try:
 for jid,j in sorted(jobs.items()):
  rc=receipts[jid];f=d/rc['file'];assert sha(f)==rc['sha256'] and f.stat().st_size==rc['bytes'] and read(d/(jid+'.receipt.json'))==rc
  with gzip.open(f,'rt') as z:hd,row,end=[json.loads(v) for v in z]
  assert hd['job']==j and hd['backend_sha']==R['backend_sha'] and hd['evaluator_runtime']==R['model'] and end['backups']==32 and end['status']=='root_complete'
  t=row['trace'];verify_state(t['root'],j['opening']);sp=t['spec'];rep=t['replicas'][0];assert sp['budget']==32 and sp['evaluator']=='rapfi_nnue_lookahead_v1' and len(rep['events'])==32
  audit=replay(rep,sp);maxerr=max(maxerr,audit['root_score_max_error'])
  es=sorted(rep['events'],key=lambda e:e['root_selection']['clock']);assert [e['root_selection']['law']['t'] for e in es]==list(range(32))
  msg=dict(spec_seed=sp['seed'],replica_seed=rep['seed'],root_uniforms=[e['root_selection']['law']['u'] for e in es],evaluations=rep['evaluations'])
  p.stdin.write(json.dumps(msg,separators=(',',':'))+'\n');p.stdin.flush();q=json.loads(p.stdout.readline());maxu=max(maxu,q['max_root_u_error']);maxlog=max(maxlog,q['max_sample_log_error']);draws+=q['root_draws'];samples+=q['sample_draws']
  a=np.array([e['actions'][0] for e in es]);v=np.array([e['root_value'] for e in es]);assert np.all(abs(v)<=1+1e-7) and all(e['root_value']==(-1)**len(e['actions'])*e['leaf_value'] for e in es)
  pp=np.zeros((32,L*L));prior=np.zeros(L*L);b=np.array(t['root']['board'])
  for i,e in enumerate(es):
   law=e['root_selection']['law']; pp[i,law['actions']]=law['p']
   assert abs(pp[i].sum()-1)<1e-11 and pp[i,a[i]]>0 and np.all(pp[i,b!=0]==0)
  for e in rep['cuts'][-1]['edges']:prior[e['action']]=e['prior']
  assert np.all(prior[b==0]>0) and abs(prior.sum()-1)<2e-6
  prior/=prior.sum()
  for cut in rep['cuts']:
   k=cut['local_k'];# completion sequence may differ from selection ordering within a wave, but cuts are quiescent
   ev=sorted(rep['events'],key=lambda e:e['complete_seq'])[:k];n=np.bincount([e['actions'][0] for e in ev],minlength=L*L)
   assert all(n[e['action']]==e['n'] for e in cut['edges'])
  pos=j['position_id'];core=int(pos[1:3]);arm=int(pos.endswith('dense'));meta.append([core,arm,j['replicate']]);act.append(a);val.append(v);probs.append(pp);priors.append(prior);boards.append(b)
  rows.append(dict(id=jid,path=str(f.relative_to(B)),sha256=rc['sha256'],bytes=rc['bytes'],seed=sp['seed'],root_hash=t['root']['state_hash'],wall_ns=t['wall_ns']))
 p.stdin.close();assert p.wait(timeout=15)==0
finally:
 if p.poll() is None:p.kill();p.wait()
assert len(rows)==64 and draws==2048 and len(set(map(tuple,meta)))==64
np.savez_compressed(S/f'SCALE_L{L}.npz',meta=meta,actions=act,marks=val,p=probs,prior=priors,board=boards)
out=dict(L=L,paths=64,backups=2048,root_draws=draws,candidate_samples=samples,max_root_score_error=maxerr,max_root_u_error=maxu,max_sample_log_error=maxlog,raw_bytes=sum(x['bytes'] for x in rows),array_sha256=sha(S/f'SCALE_L{L}.npz'),raw_receipts=rows,backend_sha=R['backend_sha'],plan_hash=R['plan_hash'],runtime_retries=sum(f.read_text().count('MQFT_RAPFI_RETRY') for f in (d/'diagnostics').glob('*.log')))
dump(S/f'AUDIT_L{L}.json',out);print('AUDIT',L,{k:v for k,v in out.items() if k!='raw_receipts'},flush=True)
