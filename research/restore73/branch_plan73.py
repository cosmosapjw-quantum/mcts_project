from base61 import *
import gzip,subprocess
R=read(S/'REGISTRATION.json');assert sha(S/'mqft_train')==R['binary_sha']
contexts=[];alljobs={}
for g in (0,3):
 tag='phase'+str(g);jobs=[]
 for j in read(S/'plans'/f'{tag}.json'):
  if j['rep']!=0 or j['arm']=='persist_dephased':continue
  p=S/'data'/tag/(j['id']+'.json.gz')
  with gzip.open(p,'rt') as f:q=json.load(f)
  assert q['input']==j
  es=q['trace']['events'];prefix=[e['actions'][0] for e in es[:15]]
  law=es[15]['root_selection']['law'];legal=sorted(law['actions']);assert legal==np.flatnonzero(board(j['moves'])==0).tolist()
  c=dict(tag=tag,job=j,source_path=str(p.relative_to(B)),source_sha256=sha(p),prefix=prefix,observed=es[15]['actions'][0],legal=legal,branch_ids=[])
  for a in legal:
   k=dict(j);k.update(id=f'{j["id"]}_a{a:02d}',forced73=prefix+[a]);jobs.append(k);c['branch_ids'].append(k['id'])
  contexts.append(c)
 f=S/'plans'/f'branch{g}.json';dump(f,jobs);alljobs[str(g)]=dict(path=f.name,sha256=sha(f),count=len(jobs))
assert len(contexts)==64 and sum(x['count'] for x in alljobs.values())==4544
dump(S/'BRANCH_REGISTRATION73.json',dict(binary_sha=sha(S/'mqft_branch73'),patch_sha=sha(S/'branch73.patch'),contexts=contexts,plans=alljobs,scope='every legal 16th choice (t15, last selection of fourth wave); prior 15 actual records fixed; 16 forced actions, complete 32 backups with original 128 clock; alternative conditional next action t16. Forced branch replays are NOT independent random histories.'))
print('BRANCH PLAN',alljobs,flush=True)
for g in (0,3):
 f=S/'plans'/f'branch{g}.json';out=S/'data'/f'branch{g}';out.mkdir()
 cmd=[str(S/'mqft_branch73'),'phase65',str(B/R['weights'][str(g)]['path']),str(f),str(out),'12']
 with (S/'logs'/f'branch{g}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert len(list(out.glob('*.json.gz')))==alljobs[str(g)]['count']
 print('BRANCH COMPLETE',g,alljobs[str(g)]['count'],flush=True)
