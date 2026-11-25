from pathlib import Path
import json,sys,math
from mqft.campaign import make_plan,atomic_json,valid_moves
from mqft.client import sha256
from mqft.rapfi import verify
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo'
L=int(sys.argv[1]);c=L//2
p47=B/'mqft_restore47_20261126'
old=json.loads((p47/'plans/scale32.json').read_text())['config']['positions']
# Predeclared first anchor from each older cohort; translate same absolute
# central coordinates (no density matching or outcomes in selection).
anchors=[sorted([x for x in old if not x['id'].startswith('r45_')],key=lambda x:x['id'])[0],
         sorted([x for x in old if x['id'].startswith('r45_')],key=lambda x:x['id'])[0]]
def map_a(a):
 y,x=divmod(a,15);y+=c-7;x+=c-7
 assert 0<=y<L and 0<=x<L
 return y*L+x
positions=[dict(id=f"L{L}_"+x['id'],moves=[map_a(a) for a in x['moves']],source_action=map_a(x['source_action'])) for x in anchors]
for x in positions:assert len(x['moves'])==8 and valid_moves(x['moves'])[x['source_action']]==0
# The first anchor's four prescribed moves are also a common legal self-game opening.
cfg=dict(kind='roots',seed=49000+L,replicates=4,budgets=[16],c_puct=[1.4],modes=[],positions=positions,sources=[0.],rollout_depth=1,prior='local',evaluator='rapfi_nnue_lookahead_v1',record_rollouts=True,paired_schedule_seed=True,jobs=1,cpu_limit=8,max_output_bytes=2**29,search_timeout=90,max_plies=L*L)
for kind in ('iid','pl','reserve'):
 for h in (-.2,0.,.2):
  cfg['modes'].append(dict(mode='wave',workers=4,samples_per_eval=4,root_law=dict(kind=kind,beta_p=[1/.3],beta_q=[1/.3],proposal='p',logit_h=h)))
 if L in (15,21):
  for t in (0,12):
   for h in (-.2,.2):
    sched=[h if t<=k<t+4 else 0. for k in range(16)]
    cfg['modes'].append(dict(mode='wave',workers=4,samples_per_eval=4,root_law=dict(kind=kind,beta_p=[1/.3],beta_q=[1/.3],proposal='p',logit_h=0.,logit_schedule=sched)))
a=make_plan(cfg);n=168 if L in (15,21) else 72
assert len(a['jobs'])==n
groups={}
for j in a['jobs']:
 k=(j['position_id'],j['replicate'],j['spec']['root_law']['kind']);groups.setdefault(k,[]).append(j)
assert len(groups)==24 and all(len({j['spec']['seed'] for j in g})==1 for g in groups.values())
gcfg=dict(kind='games',seed=49100+L,replicates=1,budgets=[16],c_puct=[1.4],modes=[dict(mode='serial',workers=1,samples_per_eval=4)],openings=[positions[0]['moves'][:4]],opponents=['self'],rollout_depth=1,prior='local',evaluator='rapfi_nnue_lookahead_v1',record_rollouts=True,move_temperature=0.,max_plies=L*L,jobs=1,cpu_limit=2,max_output_bytes=2**28,search_timeout=90)
b=make_plan(gcfg)
atomic_json(S/'plans'/f'roots-L{L}.json',a);atomic_json(S/'plans'/f'game-L{L}.json',b)
atomic_json(S/f'REGISTRATION_L{L}.json',dict(L=L,backend_sha=sha256(S/'runtime_verified'/f'mqft-L{L}'),source_manifest_sha=sha256(S/'family'/f'L{L}'/'SOURCE_MANIFEST.json'),model=verify(P/'.external/rapfi'),anchor_source_sha=sha256(p47/'plans/scale32.json'),anchors=positions,plans=dict(roots=a['plan_hash'],game=b['plan_hash']),scope='same absolute geometry two actual old p8 roots; fixed B16, four seeds/arm, h=+/- .2 and zero; L15/L21 additionally first/last block source pulses; one actual serial game per L; no density matched comparison'))
print('REGISTERED',L,n,positions)
