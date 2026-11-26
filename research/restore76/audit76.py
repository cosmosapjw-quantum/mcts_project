"""All native raw traces, exact root law, RNG and sufficient statistics."""
from base61 import *
import sys,gzip,subprocess
from mqft.campaign import verify_state,validate_receipt,load_plan
from mqft.client import digest
from mqft.path_law import replay
from mqft.rapfi import verify
L=int(sys.argv[1]);O=B/'mqft_restore49_20261126';P=B/'mqft_restore39_20261125/repo';R=read(S/f'REGISTRATION_L{L}.json');d=S/'data'/f'L{L}'
assert sha(O/'runtime_verified'/f'mqft-L{L}')==R['backend_sha'] and verify(P/'.external/rapfi')==R['model']
assert sha(O/'family'/f'L{L}'/'SOURCE_MANIFEST.json')==R['source_manifest_sha']
helper=B/'mqft_restore46_20261126/runtime/mqft-rng-replay';assert sha(helper)==read(B/'mqft_restore46_20261126/PHYSICS_RESULTS.json')['rng_replay_sha256']
rngproc=subprocess.Popen([str(helper)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
receipts=[];M=[];X=[];A=[];BO=[];PR=[];STAT=[];maxscore=maxu=maxlog=maxp=maxw=0.;draws=samples=0;prefix={}
def scrub(x):
 if isinstance(x,dict):return {k:scrub(v) for k,v in x.items() if k not in ('worker','elapsed_ns','wall_ns')}
 if isinstance(x,list):return [scrub(v) for v in x]
 return x
try:
 for name in ('base','cpuct','pulse'):
  p=S/'plans'/f'L{L}_{name}.json';plan=load_plan(p);assert sha(p)==R['plans'][name]['sha256']
  dd=d/name;lock=read(dd/'run_lock.json');assert lock==dict(plan_hash=plan['plan_hash'],backend_sha=R['backend_sha'],external=R['model'])
  report=read(dd/'report.json');jobs={j['id']:j for j in plan['jobs']};assert len(report['completed'])==len(jobs) and not report['failures'] and not report['unattempted']
  assert set(jobs)=={q['job_id'] for q in report['completed']}
  for rc in report['completed']:
   j=jobs[rc['job_id']];f=dd/rc['file'];assert read(dd/(j['id']+'.receipt.json'))==rc;validate_receipt(rc,f,j,plan['plan_hash'],R['backend_sha'])
   with gzip.open(f,'rt') as z:hd,row,end=map(json.loads,z)
   assert hd['job']==j and hd['backend_sha']==R['backend_sha'] and hd['evaluator_runtime']==R['model']
   t=row['trace'];sp=t['spec'];k=sp['budget'];rep=t['replicas'][0];assert sp==dict(j['spec'],c_puct=float(np.float32(j['spec']['c_puct']))) and end['backups']==k and end['status']=='root_complete'
   verify_state(t['root'],j['opening']);a0=replay(rep,sp);maxscore=max(maxscore,a0['root_score_max_error'])
   es=sorted(rep['events'],key=lambda e:e['root_selection']['clock']);assert len(es)==k and [e['root_selection']['law']['t'] for e in es]==list(range(k))
   msg=dict(spec_seed=sp['seed'],replica_seed=rep['seed'],root_uniforms=[e['root_selection']['law']['u'] for e in es],evaluations=rep['evaluations'])
   rngproc.stdin.write(json.dumps(msg,separators=(',',':'))+'\n');rngproc.stdin.flush();q=json.loads(rngproc.stdout.readline());maxu=max(maxu,q['max_root_u_error']);maxlog=max(maxlog,q['max_sample_log_error']);draws+=q['root_draws'];samples+=q['sample_draws']
   a=np.array([e['actions'][0] for e in es]);z=np.array([e['root_value'] for e in es]);assert np.all(abs(z)<=1+1e-7) and all(e['root_value']==(-1)**len(e['actions'])*e['leaf_value'] for e in es)
   board=np.array(t['root']['board']);legal=np.flatnonzero(board==0);pp=np.zeros((k,L*L));cap=[];H=[];prior=np.zeros(L*L);src=sp['source_action'];opp=R['opposite_action'];assert board[src]==board[opp]==0
   beta=sp['root_law']['beta_p'][0];h=sp['root_law'].get('logit_schedule',[0.])[-1];C=j['spec']['c_puct']
   for i,e in enumerate(es):
    law=e['root_selection']['law'];ac=np.array(law['actions']);score=np.asarray(law['scores']);v=np.asarray(law['p'])
    assert np.array_equal(np.sort(ac),legal) and all(law['active'])
    ll=beta*score+law['logit_h']*(ac==src);ll-=ll.max();pv=np.exp(ll);pv/=pv.sum();maxp=max(maxp,float(abs(pv-v).max()));assert maxp<1e-10
    pp[i,ac]=v;assert abs(pp[i].sum()-1)<1e-10 and v[law['picked']]>0 and ac[law['picked']]==a[i]
    # Conditional selector-score curvature at fixed scores. h contribution distinct.
    cap.append(beta**2*float(v@((score-v@score)**2)));H.append(entropy(v))
   for e in rep['cuts'][-1]['edges']:prior[e['action']]=e['prior']
   assert np.all(prior[legal]>0) and abs(prior.sum()-1)<2e-6;prior/=prior.sum()
   for cut in rep['cuts']:
    kk=cut['local_k'];ev=sorted(rep['events'],key=lambda e:e['complete_seq'])[:kk];nn=np.bincount([e['actions'][0] for e in ev],minlength=L*L);ww=np.bincount([e['actions'][0] for e in ev],weights=[e['root_value'] for e in ev],minlength=L*L);ss=np.bincount([e['actions'][0] for e in ev],weights=[e['root_value']**2 for e in ev],minlength=L*L)
    for e in cut['edges']:
     aa=e['action'];assert nn[aa]==e['n'] and e['pending']==0
     maxw=max(maxw,abs(ww[aa]-e['w']),abs(ss[aa]-e['s2']));assert maxw<1e-8
   x=np.bincount(a,minlength=L*L)/k;y=pp.mean(0);assert abs(x.sum()-1)<1e-12
   pair=sorted(legal,key=lambda i:(-prior[i],i))[:2]
   pos=j['position_id'];core=int(pos[1:3]);arm=int(pos.endswith('dense'));r=j['replicate']
   if beta==0 and h==0:assert np.max(abs(pp[:,legal]-1/len(legal)))<1e-15
   if k==16 and beta==10/3 and arm==1 and C==1.4:
    ids=[0]+[e['eval_id'] for e in es[:12] if e['eval_id'] is not None]
    pf=dict(events=es[:12],evaluations=[e for e in rep['evaluations'] if e['id'] in ids],cuts=[e for e in rep['cuts'] if e['local_k']<=12])
    prefix[core,r,h]=(sp['seed'],digest(scrub(pf)))
    if h==0:assert abs(pp[12:,src]-pp[12,src]).max()<1e-11
   # recorded marks and last-wave score response; source rate uses source, not top pair.
   stat=[x[src]-x[opp],x[pair[0]]-x[pair[1]],prior[pair[0]]-prior[pair[1]],float(x.max()),entropy(x),float(z.mean()),float(np.mean(cap)),float(np.mean(H)),float(np.sum(1-(pp*pp).sum(1))/k**2),float(np.mean(a[-4:]==src)),float(np.mean(z[-4:])),float(pp[-4,src]),float(np.sum((a==src)-pp[:,src])),float(np.sum((a[-4:]==src)-pp[-4:,src]))]
   M.append([core,arm,k,beta,C,h,r]);X.append(x);A.append(y);BO.append(board);PR.append(prior);STAT.append(stat)
   receipts.append(dict(id=j['id'],path=str(f.relative_to(B)),sha256=rc['sha256'],bytes=rc['bytes'],seed=sp['seed'],budget=k,wall_ns=t['wall_ns'],campaign=name))
  print('AUDIT',L,name,len(report['completed']),flush=True)
 rngproc.stdin.close();assert rngproc.wait(timeout=15)==0
finally:
 if rngproc.poll() is None:rngproc.kill();rngproc.wait()
assert len(M)==640 and draws==sum(x[2] for x in M) and len(set(map(tuple,M)))==640
assert len(prefix)==96
for i in range(8):
 for r in range(4):assert len({prefix[i,r,h] for h in (0.,-.35,.35)})==1
np.savez_compressed(S/f'SCALE_L{L}.npz',meta=M,X=X,A=A,board=BO,prior=PR,stat=STAT)
out=dict(L=L,paths=len(M),backups=draws,root_draws=draws,candidate_samples=samples,max_root_score_error=maxscore,max_root_u_error=maxu,max_sample_log_error=maxlog,max_probability_error=maxp,max_backup_moment_error=maxw,raw_bytes=sum(r['bytes'] for r in receipts),array_sha256=sha(S/f'SCALE_L{L}.npz'),raw_receipts=receipts,backend_sha=R['backend_sha'],paired_pulse_triples=32,stat_names=['fixed_pair_m','top_prior_pair_m','prior_gap','max_occupancy','occupancy_entropy','root_mark','conditional_score_curvature','conditional_entropy','bracket','last_source_fraction','last_mark','last_source_p','path_score','last_wave_score'])
dump(S/f'AUDIT_L{L}.json',out);print('DONE AUDIT',L,{k:v for k,v in out.items() if k!='raw_receipts'},flush=True)
