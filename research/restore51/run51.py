"""Receipt-checked native reduction for registered RESTORE51 cohorts.
Run under generated L-scoped Python package. Reuse unchanged v3 RNG checker."""
from pathlib import Path
import json,gzip,sys,subprocess,math,statistics as st
from collections import defaultdict,Counter
import numpy as np
from mqft.client import sha256
from mqft.campaign import atomic_json,verify_state,valid_moves,choose
from mqft.path_law import replay,softmax
import mqft
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore49_20261126'
mqft.__path__.append(str(P/'python/mqft'))
from mqft.physics49 import root_observed,spatial_structure,actual_schedule_ratio
from mqft.physics50 import pulse_label,retarded,projected_action,coherent_records,sector_blocks
from mqft.physics51 import selection_diagnostics
from mqft.observables import entropy
read=lambda p:json.loads(p.read_text())
def scrub(x):
 if isinstance(x,dict):return {k:scrub(v) for k,v in x.items() if k not in ('worker','elapsed_ns','wall_ns')}
 if isinstance(x,list):return [scrub(v) for v in x]
 return x
def audit(L,name,p,reg):
 stem=f'{name}-L{L}';d=S/'data'/stem;a=S/'analysis'/stem;pl=read(S/'plans'/f'{stem}.json')
 m=read(a/'manifest.json');lock=read(d/'run_lock.json');rp=read(next(d.glob('report-*.json')))
 assert m['completed']==m['expected']==len(pl['jobs'])==len(rp['completed']) and not rp['failures'] and not rp['unattempted']
 assert lock['backend_sha']==reg['backend_sha'] and lock['external']==reg['model'] and lock['plan_hash']==pl['plan_hash']==reg['plans'][name]['hash']
 assert m['plan_hash']==pl['plan_hash'] and all(sha256(a/f)==h for f,h in m['outputs'].items())
 jobs={j['id']:j for j in pl['jobs']};cnt=Counter();native=[];games=[];prefix={};maxu=maxlog=maxscore=0.;maxbalance=0.
 for rc in rp['completed']:
  j=jobs[rc['job_id']];f=d/rc['file'];assert sha256(f)==rc['sha256'] and f.stat().st_size==rc['bytes'] and read(d/(j['id']+'.receipt.json'))==rc
  cnt['jobs']+=1;cnt['compressed_bytes']+=rc['bytes']
  with gzip.open(f,'rt') as g:
   hd=json.loads(next(g));assert hd['job']==j and hd['evaluator_runtime']==reg['model'] and hd['backend']['board_size']==L
   moves=j['opening'].copy();nr=nb=0;end=None
   for line in g:
    row=json.loads(line)
    if row['type']=='end':end=row;break
    t=row['trace'];sp=t['spec'];rep=t['replicas'][0];ev=sorted(rep['events'],key=lambda x:x['root_selection']['clock'])
    assert t['root']['board_size']==L and sp['evaluator']=='rapfi_nnue_lookahead_v1' and sp['samples_per_eval']==4 and len(ev)==sp['budget']
    verify_state(t['root'],moves);q=replay(rep,sp);maxscore=max(maxscore,q['root_score_max_error'])
    u=[e['root_selection']['law']['u'] for e in ev] if sp['root_law'] else []
    msg=dict(spec_seed=sp['seed'],replica_seed=rep['seed'],root_uniforms=u,evaluations=rep['evaluations'])
    p.stdin.write(json.dumps(msg,separators=(',',':'))+'\n');p.stdin.flush();l=p.stdout.readline()
    if not l:raise RuntimeError('RNG helper failed')
    v=json.loads(l);cnt['rng_root']+=v['root_draws'];cnt['rng_samples']+=v['sample_draws'];maxu=max(maxu,v['max_root_u_error']);maxlog=max(maxlog,v['max_sample_log_error'])
    cnt['searches']+=1;nb+=sp['budget'];nr+=1;cnt['parent_queries']+=len(rep['evaluations'])
    cnt['candidate_samples']+=sum(len(e['samples']) for e in rep['evaluations']);cnt['child_queries']+=sum(not z['terminal'] for e in rep['evaluations'] for z in e['samples']);cnt['cuts']+=len(rep['cuts'])
    for cut in rep['cuts']:
     vv=defaultdict(list)
     for e in rep['events'][:cut['local_k']]:
      assert e['root_value']==(-1)**len(e['actions'])*e['leaf_value'];vv[e['actions'][0]].append(e['root_value'])
     for e in cut['edges']:
      z=vv[e['action']];assert len(z)==e['n'] and e['pending']==0 and abs(math.fsum(z)-e['w'])<1e-8 and abs(math.fsum(x*x for x in z)-e['s2'])<1e-8
    if name=='game':
     assert row['action'] in t['root']['legal'];moves.append(row['action']);continue
    z=root_observed(t,L,sp['source_action']);assert abs(sum(z['G'])-q['score_p'])<1e-9
    lab=pulse_label(z['h']);z.update(position=j['position_id'],job=j['id'],shard_sha=rc['sha256'],campaign=name,opening=j['opening'],density=len(j['opening'])/L**2,beta=sp['root_law']['beta_p'][0],schedule=lab[0],amplitude=lab[1],block=lab[2],replicate=j['replicate'])
    target=[0.]*sp['budget'] if lab[0]=='pulse' else [.2]*sp['budget']
    lr=actual_schedule_ratio(z,target);lp=0.
    for i,e in enumerate(ev):
     x=e['root_selection']['law'];pr=softmax(x['scores'],x['beta_p'],x['active'],x['actions'],x['source'],target[i]);lp+=math.log(pr[x['picked']])
    assert abs(lp-z['root_log_prob']+lr['sigma'])<1e-8 and abs(lr['balance'])<1e-10
    maxbalance=max(maxbalance,abs(lr['balance']));z['protocol']=lr
    native.append(z)
    if name.startswith('pulse'):
     for k in (4*(sp['budget']//8),sp['budget']-4):
      if lab[0]=='base' or lab[2]*4==k:
       early=ev[:k];ids=[0]+[e['eval_id'] for e in early if e['eval_id'] is not None]
       obj=dict(events=early,evaluations=[e for e in rep['evaluations'] if e['id'] in ids],cuts=[e for e in rep['cuts'] if e['local_k']<=k])
       prefix[(j['position_id'],z['law'],z['seed'],lab,k)]=scrub(obj)
   assert end and end['moves']==moves and nb==end['backups']==rc['backups'] and nr==rc['records']
   cnt['backups']+=nb
   if name=='game':
    assert end['status'] in ('natural_terminal','truncated')
    games.append(dict(L=L,job=j['id'],plies=len(moves),searched=nr,winner=end['winner'],status=end['status'],moves=moves,opening=j['opening']))
 if prefix:
  for z in native:
   if z['schedule']!='pulse' or z['block']==0:continue
   k=z['block']*4;key=(z['position'],z['law'],z['seed'])
   assert prefix[key+(('pulse',z['amplitude'],z['block']),k)]==prefix[key+(('base',0.,None),k)]
   cnt['exact_prefix_pairs']+=1;cnt['exact_prefix_draws']+=k
 diag=[f.read_text() for f in (d/'diagnostics').glob('*.log') if f.stat().st_size]
 return dict(**dict(cnt),max_root_u_error=maxu,max_sample_log_error=maxlog,max_score_error=maxscore,max_ratio_balance=maxbalance,runtime_retries=sum(x.count('MQFT_RAPFI_RETRY') for x in diag),runtime_diagnostics=diag),native,games
def one(L):
 reg=read(S/f'REGISTRATION_L{L}.json');exe=O/'runtime_verified'/f'mqft-L{L}';assert sha256(exe)==reg['backend_sha']
 old=B/'mqft_restore46_20261126';helper=old/'runtime/mqft-rng-replay';assert sha256(helper)==read(old/'PHYSICS_RESULTS.json')['rng_replay_sha256']
 p=subprocess.Popen([str(helper)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
 counts={};native=[];games=[]
 try:
  for n in reg['plans']:
   stem=f'{n}-L{L}';cache=S/'audits'/f'{stem}.json'
   if cache.exists():
    x=read(cache)
    assert x['registration_sha']==sha256(S/f'REGISTRATION_L{L}.json') and x['reducer_sha']==sha256(S/'run51.py')
    assert x['report_sha']==sha256(next((S/'data'/stem).glob('report-*.json'))) and x['analysis_sha']==sha256(S/'analysis'/stem/'manifest.json')
    c,a,g=x['counts'],x['native'],x['games']
   else:c,a,g=audit(L,n,p,reg)
   counts[n]=c;native.extend(a);games.extend(g);print('AUDITED',L,n,c['jobs'],flush=True)
  p.stdin.close();assert p.wait(timeout=15)==0
 finally:
  if p.poll() is None:p.kill();p.wait()
 assert len(games)==1
 groups=defaultdict(list);pulse=defaultdict(lambda:defaultdict(list))
 for r in native:
  if r['schedule']=='base' and r['campaign']=='confirm':groups[r['position'],r['law']].append(r)
  if r['campaign'].startswith('pulse'):pulse[r['campaign'],r['position'],r['law']][r['schedule'],r['amplitude'],r['block']].append(r)
 spatial=[];actions=[];temporal=[];diagnostics=[]
 for (pos,law),rr in sorted(groups.items()):
  assert len(rr)==8
  r0=rr[0];q=spatial_structure(rr)
  q.update(position=pos,density=r0['density'],mark=st.mean(r['mark'] for r in rr),mean_parent=st.mean(r['parent_queries'] for r in rr),mean_child=st.mean(r['child_queries'] for r in rr))
  spatial.append(q)
  actions.append(dict(L=L,position=pos,law=law,**projected_action(rr)))
  d=selection_diagnostics(rr);sel=reg['source_selection'][pos]['strata']['high']
  assert r0['source']==sel['action']
  diagnostics.append(dict(L=L,position=pos,law=law,campaign='confirm',stratum='high',pilot=sel,**d))
 for (name,pos,law),rr in sorted(pulse.items()):
  n=4 if name=='pulse32_high' else 8
  assert len(rr)==7 and all(len(a)==n for a in rr.values())
  base=rr['base',0.,None];tag=name.split('_')[-1];sel=reg['source_selection'][pos]['strata'][tag]
  assert base[0]['source']==sel['action']
  diagnostics.append(dict(L=L,position=pos,law=law,campaign=name,stratum=tag,pilot=sel,**selection_diagnostics(base)))
  temporal.extend(dict(L=L,position=pos,law=law,campaign=name,stratum=tag,B=base[0]['B'],**x) for x in retarded(rr))
 tmp=[];ids={r['job']:r for r in native if r['schedule']=='base' and r['campaign']=='confirm'}
 with (S/'analysis'/f'confirm-L{L}'/'local.jsonl').open() as f:
  for line in f:
   x=json.loads(line)
   if x['job'] not in ids:continue
   a=ids[x['job']];tmp.append(dict(L=L,position=a['position'],law=a['law'],k=x['k'],H=x['H'],beta_status=x['beta_status']))
 out=dict(L=L,counts=counts,native=native,games=games,spatial=spatial,actions=actions,temporal=temporal,diagnostics=diagnostics,temperature=tmp,
          registration_sha=sha256(S/f'REGISTRATION_L{L}.json'),rng_helper_sha=sha256(helper))
 atomic_json(S/f'PHYSICS_L{L}.json',out);print('SIZE COMPLETE',L,len(native),len(spatial),len(temporal),games[0]['status'],flush=True)
if __name__=='__main__':one(int(sys.argv[1]))
