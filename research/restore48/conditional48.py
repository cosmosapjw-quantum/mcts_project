"""Actual previous registered drive records, independent conditional calculations."""
import json,subprocess,math,statistics as st
from pathlib import Path
import numpy as np
from mqft.client import sha256
from mqft.campaign import atomic_json
from mqft.physics48 import tomography_trials,duration_cycle
S=Path(__file__).resolve().parent;prev=S.parent/'mqft_restore47_20261126'
P=S.parent/'mqft_restore39_20261125/repo'
def read(p):return json.loads(p.read_text())
r=read(prev/'PHYSICS_RESULTS.json');reg=read(S/'REGISTRATION.json')
assert sha256(prev/'PHYSICS_RESULTS.json')==reg['previous_physics_sha']
a=tomography_trials(r['probes']);assert len(a)==12*3*32*3
# Thermal reference: same fixed three knots as previous cycle, close cycle in log odds.
cycles=[]
for x in r['thermal_cycles']:
 if x['eta']!=.5:continue
 for T in (1.,4.,16.,64.):
  for m in (8,32,128):
   cycles.append(dict(position=x['position'],law=x['law'],seed=x['seed'],job=x['job'],**duration_cycle(x['probabilities'],T,m)))
assert len(cycles)==576
# Direct Rapfi NNUE size support probe only. No MCTS at different sizes.
d=P/'.external/rapfi';exe=d/'pbrain-rapfi-linux-clang-avx2';sizes=[]
for L in (9,11,13,15):
 c=L//2;coords=[(c,c,1),(c+1,c,2),(c,c-1,1),(c-1,c+1,2)]
 cmd=f'START {L}\nINFO rule 0\nINFO thread_num 1\nYXBOARD\n'+''.join(f'{x},{y},{z}\n' for x,y,z in coords)+'DONE\nTRACEBOARD\nTRACESEARCH\nABOUT\nEND\n'
 try:
  q=subprocess.run([str(exe)],cwd=d,input=cmd,text=True,capture_output=True,timeout=15)
  (S/'logs'/f'rapfi-size{L}.log').write_text(q.stdout+'\nSTDERR\n'+q.stderr)
  lines=q.stdout.splitlines()
  if any('no compatible weight config found' in l for l in lines):
   sizes.append(dict(L=L,status='nnue_disabled_incompatible_weight',error='no compatible weight config found; heuristic fallback excluded',log_sha=sha256(S/'logs'/f'rapfi-size{L}.log')));continue
  loaded=any('nnue: load weight from mix9svqfreestyle_bsmix.bin.lz4' in l for l in lines)
  k=next(i for i,l in enumerate(lines) if 'Policy------Self' in l);grid=lines[k+1:k+1+L]
  score=[];board={(x,y):z for x,y,z in coords}
  for y,l in enumerate(grid):
   a0=l.split();assert a0[0]=='MESSAGE' and len(a0)==L+2 and int(a0[-1])==y+1
   for x,v in enumerate(a0[1:-1]):
    if (x,y) in board:assert v==('[X]' if board[x,y]==1 else '[O]')
    else:score.append(int(v))
  w=next(l.split() for l in lines if l.startswith('MESSAGE Evaluator Eval[Black]:'))
  wdl=[float(w[w.index(x)+1]) for x in ('WR:','LR:','DR:')]
  assert loaded and len(score)==L*L-4 and abs(sum(wdl)-1)<.001 and q.returncode==0
  assert any(l.startswith('MESSAGE Ply: ') and l.endswith('4') for l in lines)
  assert any(l.startswith('MESSAGE SideToMove: ') and l.endswith('Black') for l in lines)
  sizes.append(dict(L=L,status='direct_evaluator_verified',legal_scores=len(score),wdl=wdl,score_range=[min(score),max(score)],log_sha=sha256(S/'logs'/f'rapfi-size{L}.log')))
 except Exception as e:sizes.append(dict(L=L,status='blocked_or_failed',error=repr(e)))
atomic_json(S/'CONDITIONAL_RESULTS.json',dict(previous_physics_sha=reg['previous_physics_sha'],tomography=a,thermal=cycles,board_size_probe=sizes,scope='conditional simulated shots, declared finite-duration thermal protocol, direct Rapfi evaluator only; not native multi-size MCTS'))
print('TOMO',len(a),'THERMAL',len(cycles),'SIZE',sizes)
