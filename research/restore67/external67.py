"""Pinned Rapfi executable and NNUE; independent same-board move reference."""
from base61 import *
import sys,re
P=B/'mqft_restore39_20261125/repo'
sys.path.insert(0,str(P/'python'))
import mqft.rapfi as rapfi
def adapter(L):
 def protocol(moves,timeout_ms,threads):
  assert len(moves)==len(set(moves)) and all(0<=a<L*L for a in moves)
  side=len(moves)%2;rows=[f'START {L}','INFO rule 0',f'INFO timeout_turn {timeout_ms}',f'INFO thread_num {threads}','BOARD']
  rows.extend(f'{a%L},{a//L},{1 if i%2==side else 2}' for i,a in enumerate(moves))
  return '\n'.join(rows+['DONE'])+'\n'
 def parse(output,moves):
  lines=output.splitlines()
  if not any(x.strip()=='OK' for x in lines) or any(x.strip().upper().startswith(('ERROR','UNKNOWN')) for x in lines):raise ValueError('protocol')
  low=output.lower()
  if L==9:
   if 'evaluator mix9svq disabled: no compatible weight config found.' not in low:raise ValueError('9x9 fallback not explicitly confirmed')
  elif f'nnue: load weight from {rapfi.WEIGHT}' not in low or any(re.search(r'(failed|error).*(nnue|weight)|(nnue|weight).*(failed|error)',x,re.I) for x in lines):raise ValueError('NNUE load')
  v=re.findall(r'(?m)^[ \t]*(\d{1,2}),(\d{1,2})[ \t\r]*$',output)
  if len(v)!=1:raise ValueError('expected completed move')
  x,y=map(int,v[0]);a=y*L+x
  if not(0<=x<L and 0<=y<L) or a in moves:raise ValueError('invalid move')
  return a
 rapfi.protocol=protocol;rapfi.parse=parse
def one(arg):
 L,c,ms=arg;adapter(L);moves=c['moves'] if L==9 else [(a//9+3)*15+(a%9+3) for a in c['moves']]
 engine=rapfi.Rapfi(P/'.external/rapfi',timeout_ms=ms,threads=1,wall_timeout=15)
 try:
  z=engine.best(moves)
  assert rapfi.parse(z['stdout'],moves)==z['action']
  return dict(L=L,core=c['id'],moves=moves,ms=ms,status='ok',evaluator_mode='verified_fallback_no_NNUE' if L==9 else 'verified_NNUE',result=z)
 except Exception as e:return dict(L=L,core=c['id'],moves=moves,ms=ms,status='blocked',error=str(e))
def main():
 reg=read(S/'REGISTRATION.json');assert rapfi.verify(P/'.external/rapfi')==reg['external']
 jobs=[(L,c,ms) for L,ms in ((9,150),(9,350),(15,350)) for c in reg['cores']]
 with cf.ProcessPoolExecutor(max_workers=8) as pool:out=list(pool.map(one,jobs,chunksize=1))
 dump(S/'RAPFI67.json',dict(scope='pinned external NNUE engine; 9x9 explicit non-NNUE fallback, 15x15 center embedding verified NNUE separate; finite time top1 proxy, not exact minimax',identity=reg['external'],rows=out))
 print('RAPFI',[(L,ms,sum(x['status']=='ok' for x in out if (x['L'],x['ms'])==(L,ms))) for L,ms in ((9,150),(9,350),(15,350))],flush=True)
if __name__=='__main__':
 if '--smoke' in sys.argv:
  print(one((9,dict(id=0,moves=[40,41,31,32]),150)))
 else:main()
