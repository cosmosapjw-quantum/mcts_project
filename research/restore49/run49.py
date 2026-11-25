"""Independent raw-data reduction for one specialized size, or cross-size synthesis.
Use generated package for size-specific trace check; use common physics49 module.
No synthetic MCTS samples, no mixing sizes without explicit metadata."""
from pathlib import Path
import sys,json,gzip,subprocess,statistics as st,math
from collections import defaultdict,Counter
import numpy as np
from mqft.client import sha256
from mqft.campaign import atomic_json,verify_state,choose,valid_moves,win_at
from mqft.path_law import replay,softmax
from mqft.analysis import check_trace
import mqft
mqft.__path__.append(str(Path(__file__).resolve().parent.parent/"mqft_restore39_20261125/repo/python/mqft"))
from mqft.physics49 import root_observed,spatial_structure,schedule_label,scheduled_response,actual_schedule_ratio,spatial_record_model
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo'
def read(p):return json.loads(p.read_text())
def mean(v):return float(np.mean(v))
def scrub(x):
 if isinstance(x,dict):return {k:scrub(v) for k,v in x.items() if k not in ('worker','elapsed_ns','wall_ns')}
 if isinstance(x,list):return [scrub(v) for v in x]
 return x
def audit(L,name,helper,reg):
 d=S/'data'/f'{name}-L{L}';a=S/'analysis'/f'{name}-L{L}';pl=read(S/'plans'/f'{name}-L{L}.json')
 m=read(a/'manifest.json');lock=read(d/'run_lock.json');rp=read(next(d.glob('report-*.json')))
 assert m['completed']==m['expected']==len(pl['jobs'])==len(rp['completed']) and not rp['failures'] and not rp['unattempted']
 assert lock['backend_sha']==reg['backend_sha'] and lock['external']==reg['model'] and lock['plan_hash']==pl['plan_hash']==reg['plans'][name]
 assert m['plan_hash']==pl['plan_hash'] and all(sha256(a/f)==h for f,h in m['outputs'].items())
 jobs={j['id']:j for j in pl['jobs']};cnt=Counter();records=[];games=[];prefix={};maxu=maxlog=0.;maxscore=0.
 for r in rp['completed']:
  j=jobs[r['job_id']];f=d/r['file'];assert sha256(f)==r['sha256'] and f.stat().st_size==r['bytes'] and read(d/(j['id']+'.receipt.json'))==r
  cnt['jobs']+=1;cnt['compressed_bytes']+=r['bytes']
  with gzip.open(f,'rt') as g:
   hd=json.loads(next(g));assert hd['job']==j and hd['evaluator_runtime']==reg['model'] and hd['backend']['board_size']==L
   moves=j['opening'].copy();nr=nb=0;end=None
   for line in g:
    row=json.loads(line)
    if row['type']=='end':end=row;break
    t=row['trace'];sp=t['spec'];rep=t['replicas'][0];ev=sorted(rep['events'],key=lambda x:x['root_selection']['clock'])
    assert t['root']['board_size']==L and sp['evaluator']=='rapfi_nnue_lookahead_v1' and sp['samples_per_eval']==4 and len(ev)==sp['budget']
    verify_state(t['root'],moves);a0=replay(rep,sp);maxscore=max(maxscore,a0['root_score_max_error'])
    # Additional independent RNG replay: fixed-size helper depends only on
    # seed, recorded NNUE scores and leaf hashes, not board size.
    u=[e['root_selection']['law']['u'] for e in ev] if sp['root_law'] else []
    msg=dict(spec_seed=sp['seed'],replica_seed=rep['seed'],root_uniforms=u,evaluations=rep['evaluations'])
    helper.stdin.write(json.dumps(msg,separators=(',',':'))+'\n');helper.stdin.flush();l=helper.stdout.readline()
    if not l:raise RuntimeError("RNG helper failed")
    v=json.loads(l);cnt['rng_root']+=v['root_draws'];cnt['rng_samples']+=v['sample_draws'];maxu=max(maxu,v['max_root_u_error']);maxlog=max(maxlog,v['max_sample_log_error'])
    cnt['searches']+=1;nb+=sp['budget'];nr+=1
    cnt['parent_queries']+=len(rep['evaluations']);cnt['candidate_samples']+=sum(len(a['samples']) for a in rep['evaluations'])
    cnt['child_queries']+=sum(not z['terminal'] for a in rep['evaluations'] for z in a['samples']);cnt['cuts']+=len(rep['cuts'])
    # Independently verify per-cut raw moments from exact events.
    for cut in rep['cuts']:
     vv=defaultdict(list)
     for e in rep['events'][:cut['local_k']]:
      assert e['root_value']==(-1)**len(e['actions'])*e['leaf_value'];vv[e['actions'][0]].append(e['root_value'])
     for e in cut['edges']:
      q=vv[e['action']];assert len(q)==e['n'] and e['pending']==0 and abs(math.fsum(q)-e['w'])<1e-8 and abs(math.fsum(x*x for x in q)-e['s2'])<1e-8
    if name=='game':
     assert row['action'] in t['root']['legal'];moves.append(row['action']);continue
    z=root_observed(t,L,sp['source_action']);assert abs(sum(z['G'])-a0['score_p'])<1e-9
    lab=schedule_label(z['h']);z.update(position=j['position_id'],job=j['id'],shard_sha=r['sha256'],schedule=lab[0],amplitude=lab[1],replicate=j['replicate'])
    # Exact alternative source-schedule likelihood on same native history.
    protocols={}
    for kind,start in [('constant',None),('early',0),('late',12)]:
     if kind!='constant' and L not in (15,21):continue
     for h in (-.2,0.,.2):
      if h==0 and kind!='constant':continue
      target=[h if start is None or start<=i<start+4 else 0. for i in range(16)]
      q=actual_schedule_ratio(z,target);lp=0.
      for i,e in enumerate(ev):
       x=e['root_selection']['law'];p=softmax(x['scores'],x['beta_p'],x['active'],x['actions'],x['source'],target[i]);lp+=math.log(p[x['picked']])
      assert abs(lp-z['root_log_prob']+q['sigma'])<1e-8 and abs(q['balance'])<1e-10
      protocols[f'{kind}:{h}']=q
    z['protocols']=protocols
    records.append(z)
    if L in (15,21) and (lab==('constant',0.) or lab[0]=='late'):
     # Full early scientific prefix before the late pulse: root selection,
     # all backup events, linked evaluator marks, and completed cut fields.
     early=ev[:12];ids=[0]+[e['eval_id'] for e in early if e['eval_id'] is not None]
     obj=dict(events=early,evaluations=[e for e in rep['evaluations'] if e['id'] in ids],cuts=[x for x in rep['cuts'] if x['local_k']<=12])
     prefix[(j['position_id'],sp['root_law']['kind'],sp['seed'],lab)]=scrub(obj)
   assert end and end['moves']==moves and nb==end['backups']==r['backups'] and nr==r['records']
   cnt['backups']+=nb
   if name=='game':
    assert end['status'] in ('natural_terminal','truncated')
    games.append(dict(L=L,job=j['id'],plies=len(moves),searched=nr,winner=end['winner'],status=end['status'],moves=moves))
 if prefix:
  for x in records:
   if x['schedule']!='late':continue
   k=(x['position'],x['law'],x['seed'])
   assert prefix[k+((x['schedule'],x['amplitude']),)]==prefix[k+(('constant',0.),)],"late pulse changed pre-intervention trajectory"
   cnt['exact_late_prefix_pairs']+=1
 diag=[f.read_text() for f in (d/'diagnostics').glob('*.log') if f.stat().st_size]
 return dict(**dict(cnt),max_root_u_error=maxu,max_sample_log_error=maxlog,max_score_error=maxscore,runtime_retries=sum(x.count("MQFT_RAPFI_RETRY") for x in diag),runtime_diagnostics=diag),records,games

def one(L):
 reg=read(S/f'REGISTRATION_L{L}.json');exe=S/'runtime_verified'/f'mqft-L{L}';assert sha256(exe)==reg['backend_sha']
 old=B/'mqft_restore46_20261126';helper=old/'runtime/mqft-rng-replay';assert sha256(helper)==read(old/'PHYSICS_RESULTS.json')['rng_replay_sha256']
 p=subprocess.Popen([str(helper)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
 try:
  c,a,g=audit(L,'roots',p,reg);d,b,h=audit(L,'game',p,reg)
  assert not g and not b and len(h)==1
  p.stdin.close();assert p.wait(timeout=15)==0
 finally:
  if p.poll() is None:p.kill();p.wait()
 grouped=defaultdict(lambda:defaultdict(list))
 for r in a:grouped[(r['position'],r['law'])][(r['schedule'],r['amplitude'])].append(r)
 assert len(grouped)==6 and all(all(len(rr)==4 for rr in v.values()) for v in grouped.values())
 physics=[];responses=[];records=[]
 for (pos,law),rr in sorted(grouped.items()):
  base=rr['constant',0.]
  physics.append(dict(position=pos,**spatial_structure(base)))
  responses.extend(dict(L=L,position=pos,law=law,**z) for z in scheduled_response(rr))
  records.append(dict(position=pos,**spatial_record_model(base)))
 out=dict(L=L,counts=dict(roots=c,game=d),native=a,games=h,spatial=physics,responses=responses,conditional_records=records,registration_sha=sha256(S/f'REGISTRATION_L{L}.json'),rng_helper_sha=sha256(helper))
 atomic_json(S/f'PHYSICS_L{L}.json',out)
 print('L',L,'root jobs',c['jobs'],'game',h[0]['status'],h[0]['plies'],'late prefix',c.get('exact_late_prefix_pairs',0),flush=True)
def combined():
 a=[read(S/f'PHYSICS_L{L}.json') for L in (13,15,17,19,21)];sp=sum([x['spatial'] for x in a],[]);rs=sum([x['responses'] for x in a],[]);g=sum([x['games'] for x in a],[]);qr=sum([x['conditional_records'] for x in a],[])
 assert len(sp)==30 and len(rs)==54 and len(g)==5
 assert sum(x['counts']['roots']['jobs'] for x in a)==552
 assert sum(x['counts']['roots'].get('exact_late_prefix_pairs',0) for x in a)==96
 # Finite-size summary equal weight two geometries per law; descriptive.
 summary=[]
 for L in (13,15,17,19,21):
  for law in ('iid','pl','reserve'):
   x=[r for r in sp if r['L']==L and r['law']==law];assert len(x)==2
   y=[r for r in rs if r['L']==L and r['law']==law and r['kind']=='constant'];assert len(y)==2
   summary.append(dict(L=L,law=law,n_roots=2,**{k:mean([r[k] for r in x]) for k in ('cov_trace','S_low','S_diag','H','prior_H','H_over_log_min','central_mass','central_variance','source_cov_response')},finite_response=np.mean([r['values'] for r in y],axis=(0,2)).tolist()))
 out=dict(sizes=[x['L'] for x in a],counts={f"L{x['L']}":x['counts'] for x in a},games=g,spatial=sp,responses=rs,conditional_records=qr,summary=summary,
          size_results_sha={f"L{x['L']}":sha256(S/f"PHYSICS_L{x['L']}.json") for x in a},
          scope="same pinned NNUE across size-specialized Quartz; two common absolute geometries; 4 seed groups, B16; no population scaling/exponent; conditional quantum record isometry only")
 atomic_json(S/'PHYSICS_RESULTS.json',out)
 print('Combined',len(sp),len(rs),len(g),len(qr))
if __name__=='__main__':
 if sys.argv[1]=='combined':combined()
 else:one(int(sys.argv[1]))
