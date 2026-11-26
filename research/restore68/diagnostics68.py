from arena68 import *
import re
def req(x):
 j,m,old=x;b=board(m);side=1 if len(m)%2==0 else -1
 black=np.flatnonzero(b==1);white=np.flatnonzero(b==2)
 seq=[]
 for i,a in enumerate(black):
  seq.append(f'{a%15},{a//15},{1 if side==1 else 2}')
  if i<len(white):
   a=white[i];seq.append(f'{a%15},{a//15},{1 if side==-1 else 2}')
 cmd='START 15\nINFO rule 0\nINFO thread_num 1\nABOUT\nYXBOARD\n'+'\n'.join(seq)+'\nDONE\nTRACEBOARD\nTRACESEARCH\nABOUT\nEND\n'
 exe=P/'.external/rapfi/pbrain-rapfi-linux-clang-avx2'
 t=subprocess.run([str(exe)],cwd=exe.parent,input=cmd,text=True,capture_output=True,timeout=25)
 assert t.returncode==0 and 'nnue: load weight from mix9svqfreestyle_bsmix.bin.lz4' in t.stdout.lower()
 lines=t.stdout.splitlines();line=next(l for l in lines if l.startswith('MESSAGE Evaluator Eval[Black]:'))
 wdl=[float(re.search(key+r':\s*([0-9.]+)',line)[1]) for key in ('WR','LR','DR')]
 ix=next(i for i,l in enumerate(lines) if 'Policy------Self' in l)
 scores=[]
 for y in range(15):
  q=lines[ix+y+1].split();assert q[0]=='MESSAGE' and len(q)==17 and int(q[-1])==y+1
  for x in range(15):
   a=y*15+x
   if b[a]==0:scores.append([a,int(q[x+1])])
   else:assert q[x+1]==('[X]' if b[a]==1 else '[O]')
 assert scores==old['scores'] and wdl==old['wdl'],(j,wdl,old['wdl'])
 return dict(id=j,moves=m,side=side,wdl=wdl,scores_equal=True,stdout=t.stdout,stderr=t.stderr)
def main():
 a=read(S/'CAMPAIGN68.json');jobs=[]
 sel=[x for x in a if x['status']=='complete' and x['job']['match'].startswith('ext_') and x['job']['color']==1][:12]
 for x in sel:
  with gzip.open(S/'data'/(x['id']+'.json.gz'),'rt') as f:d=json.load(f)
  row=next(y for y in d['rows'] if y['engine']=='native');r=row['search'];m=r['moves'];jobs.append((x['id']+'_root',m,r['evaluations'][0]))
  e=next(y for y in r['events'] if y['eval_id'] is not None);jobs.append((x['id']+'_leaf',m+e['actions'],r['evaluations'][e['eval_id']]))
 with cf.ProcessPoolExecutor(max_workers=4) as pool:out=list(pool.map(req,jobs))
 failed=[]
 for x in a:
  if x['status']=='complete':continue
  with gzip.open(S/'data'/(x['id']+'.json.gz'),'rt') as f:d=json.load(f)
  m=d['moves'];assert (1 if len(m)%2==0 else -1)!=x['job']['color']
  for k in range(3):
   try:r=Rapfi(P/'.external/rapfi',timeout_ms=x['job']['external_ms'],threads=1,wall_timeout=15).best(m);failed.append(dict(id=x['id'],rep=k,moves=m,status='ok',result=r))
   except Exception as e:failed.append(dict(id=x['id'],rep=k,moves=m,status='failed',error=repr(e)))
 dump(S/'DIAGNOSTICS68.json',dict(scope='separate raw protocol exact WDL and policy requery, plus diagnostic retry of original external failures only; not new games and no substitution of failed outcomes',nnue_queries=out,failed_state_requeries=failed))
 print('DIAGNOSTICS',len(out),'exact NNUE queries',[(x['id'],x['rep'],x['status']) for x in failed],flush=True)
if __name__=='__main__':main()
