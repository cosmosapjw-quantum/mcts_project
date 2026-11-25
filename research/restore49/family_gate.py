from pathlib import Path
import json,gzip,copy,sys,math,subprocess
from mqft.client import Backend,sha256
from mqft.analysis import check_trace
from mqft.campaign import atomic_json,valid_moves,verify_state
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo'
L=int(sys.argv[1]);exe=S/'runtime_verified'/f'mqft-L{L}';d=P/'.external/rapfi'
def scrub(x):
 if isinstance(x,dict):return {k:scrub(v) for k,v in x.items() if k not in ('wall_ns','elapsed_ns','schema','board_size','worker')}
 if isinstance(x,list):return [scrub(v) for v in x]
 return x
out=[]
with Backend(exe,90,d,4,diagnostic_path=S/'logs'/f'gate-L{L}.log') as be:
 assert be.identity['board_size']==L and be.identity['schema']==f'mqft.quartz.trace.v3.L{L}'
 # D4 boundary and terminal tests on all sizes. Freestyle overline >=5.
 c=L//2
 for axis in ((0,1),(1,0),(1,1),(1,-1)):
  dy,dx=axis
  pts=[(c+i*dy,c+i*dx) for i in (-2,-1,0,1,2)]
  others=[(0,i*2) for i in range(4)]
  moves=[]
  for i,(y,x) in enumerate(pts):
   moves.append(y*L+x)
   if i<4:moves.append(others[i][0]*L+others[i][1])
  st=be.state(moves);verify_state(st,moves);assert st['terminal'] and st['winner']==1
  try:be.state(moves+[L*L-1]);raise AssertionError("move after terminal accepted")
  except ValueError:pass
  out.append(dict(type='terminal',direction=axis,hash=st['state_hash']))
 for moves in ([L*L],[0,0],[-1],):
  try:be.state(moves);raise AssertionError("illegal move accepted")
  except ValueError:pass
 # Test legal support, both sides, D4 geometric orbit and hash identity.
 moves=[c*L+c,c*L+c+1,(c-1)*L+c,(c+1)*L+c-1]
 for m in ([],moves,moves+[0],[(L-1-(a//L))*L+a%L for a in moves]):
  st=be.state(m);verify_state(st,m);assert len(st['legal'])==L*L-len(m)
  out.append(dict(type='state',moves=m,hash=st['state_hash'],legal=len(st['legal'])))
 if L==15:
  # Compare every non-timing trace field at actual archived v3 seeds.
  old=B/'mqft_restore48_20261126';pl=json.loads((old/'plans/scale32a.json').read_text())
  chosen=[]
  for law in ('iid','pl','reserve'):
   j=next(x for x in pl['jobs'] if x['spec']['root_law']['kind']==law and x['spec']['root_law']['logit_h']==0.)
   chosen.append(j)
  for j in chosen:
   f=old/'data/scale32a'/(j['id']+'.jsonl.gz');rc=json.loads((f.parent/(j['id']+'.receipt.json')).read_text());assert sha256(f)==rc['sha256']
   with gzip.open(f,'rt') as g:head,x,end=[json.loads(l) for l in g]
   oldt=x['trace'];new=be.search(j['opening'],j['spec']);check_trace(new,j['opening'],j['spec'])
   # semantic comparison must not conflate timing/build/schema differences.
   a,b=scrub(oldt),scrub(new)
   if a!=b:
    def first(a,b,path=''):
     if type(a)!=type(b):return path,a,b
     if isinstance(a,dict):
      if set(a)!=set(b):return path,set(a)-set(b),set(b)-set(a)
      for k in a:
       if a[k]!=b[k]:return first(a[k],b[k],path+'/'+k)
     elif isinstance(a,list):
      if len(a)!=len(b):return path,len(a),len(b)
      for i,(x,y) in enumerate(zip(a,b)):
       if x!=y:return first(x,y,path+'/'+str(i))
     else:return path,a,b
    print('MISMATCH',first(a,b),flush=True)
    atomic_json(S/f'gate-L{L}-mismatch-{j["spec"]["root_law"]["kind"]}.json',dict(new=new,old=oldt))
    raise AssertionError("L15 semantic mismatch")
   out.append(dict(type='v3_trace_equivalence',law=j['spec']['root_law']['kind'],job=j['id'],sha=rc['sha256'],backups=j['spec']['budget']))
 else:
  spec=dict(mode='wave',workers=4,budget=8,cuts=[1,2,4,8],seed=49000+L,c_puct=1.4,rollout_depth=1,samples_per_eval=4,prior='local',source_action=(c-1)*L+c+1,source_h=0.,root_law=dict(kind='reserve',beta_p=[1/.3],beta_q=[1/.3],proposal='p',logit_h=0.),record_rollouts=True,evaluator='rapfi_nnue_lookahead_v1')
  t=be.search(moves,spec);check_trace(t,moves,spec);out.append(dict(type='actual_nnue_search',backups=8))
atomic_json(S/f'GATE_L{L}.json',dict(L=L,backend_sha=sha256(exe),tests=out))
print('GATE',L,len(out),flush=True)
