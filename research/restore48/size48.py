import json,struct,subprocess
from pathlib import Path
import lz4.frame
from mqft.client import sha256
from mqft.campaign import atomic_json
S=Path(__file__).resolve().parent;P=S.parent/'mqft_restore39_20261125/repo'
d=P/'.external/rapfi';w=d/'mix9svqfreestyle_bsmix.bin.lz4';exe=d/'pbrain-rapfi-linux-clang-avx2'
with lz4.frame.open(w,'rb') as f:magic,arch,rule,mask,n=struct.unpack('<5I',f.read(20))
assert magic==0xacd8cc6a and rule&1
sizes=[i+1 for i in range(32) if mask>>i&1]
print('weight board mask',hex(mask),sizes)
out=[]
for L in sizes:
 # one direct reference board per declared supported size; no config mutation.
 c=L//2;coords=[(c,c,1),(c+1,c,2),(c,c-1,1),(c-1,c+1,2)]
 cmd=f'START {L}\nINFO rule 0\nINFO thread_num 1\nYXBOARD\n'+''.join(f'{x},{y},{z}\n' for x,y,z in coords)+'DONE\nTRACEBOARD\nTRACESEARCH\nABOUT\nEND\n'
 try:
  q=subprocess.run([str(exe)],cwd=d,input=cmd,text=True,capture_output=True,timeout=15)
  log=S/'logs'/f'rapfi-supported-size{L}.log';log.write_text(q.stdout+'\nSTDERR\n'+q.stderr)
  lines=q.stdout.splitlines();assert q.returncode==0 and any('nnue: load weight from mix9svqfreestyle_bsmix.bin.lz4' in l for l in lines)
  assert any(l.startswith('MESSAGE Ply: ') and l.endswith('4') for l in lines) and any(l.startswith('MESSAGE SideToMove: ') and l.endswith('Black') for l in lines)
  k=next(i for i,l in enumerate(lines) if 'Policy------Self' in l);scores=[];board={(x,y):z for x,y,z in coords}
  for y,l in enumerate(lines[k+1:k+1+L]):
   a=l.split();assert len(a)==L+2 and int(a[-1])==y+1
   for x,v in enumerate(a[1:-1]):
    if (x,y) in board:assert v==('[X]' if board[x,y]==1 else '[O]')
    else:scores.append(int(v))
  a=next(l.split() for l in lines if l.startswith('MESSAGE Evaluator Eval[Black]:'));wdl=[float(a[a.index(x)+1]) for x in ('WR:','LR:','DR:')]
  assert len(scores)==L*L-4 and abs(sum(wdl)-1)<.001
  out.append(dict(L=L,status='direct_evaluator_verified',legal_scores=len(scores),wdl=wdl,log_sha=sha256(log)))
 except Exception as e:out.append(dict(L=L,status='failed',error=repr(e)))
atomic_json(S/'SIZE_SUPPORT.json',dict(weight_sha=sha256(w),magic=hex(magic),architecture=hex(arch),rule_mask=hex(rule),board_mask=hex(mask),header_sizes=sizes,actual_probes=out,scope='direct NNUE only, production Quartz Gomoku15 unchanged; never override header or use fallback'))
print(out)
