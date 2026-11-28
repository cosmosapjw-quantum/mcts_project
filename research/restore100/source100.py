from base61 import *
import csv,zipfile
O=B/'mqft_restore99_20261129'
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
orig=read(O/'ORIGINAL_SOURCE99.json')['archives'];assert all(sha(p)==orig[k] for k,p in paths.items())
qm={x['id']:x for x in csv.DictReader(open(O/'QUALITY_MATRIX99.csv'))};items=[]
with zipfile.ZipFile(paths['quantum.zip']) as z:
 for k in read(S/'REGISTRATION.json')['targets']:
  x=qm[k];code=x['source_member'];b=z.read(code);h=hashlib.sha256(b).hexdigest();ls=b.decode().splitlines();assert h==x['source_sha'];l,r=map(int,x['source_span'].split('-'));assert x['source_function'] in ls[l-1]
  items.append(dict(id=k,archive='quantum.zip',member=code,member_sha=h,span=[l,r],source_function=x['source_function'],source_excerpt=ls[l-1:r],acceptance=x['final_acceptance']))
study=[]
with zipfile.ZipFile(paths['mcts_study.zip']) as z:
 assert z.testzip() is None
 old={x['member']:x['sha256'] for x in read(O/'ORIGINAL_SOURCE99.json')['study_selected']}
 names=list(old)+[k for k in z.namelist() if k.endswith('.md') and 'thermodynamic-mcts-' in k]
 for k in names:
  b=z.read(k);h=hashlib.sha256(b).hexdigest();ls=b.decode().splitlines()
  if k in old:assert h==old[k]
  match=[(i+1,s) for i,s in enumerate(ls) if any(w in s.lower() for w in ['jarzynski','crooks','thermodynam','entropy production','heat','work','detailed balance','temperature','carnot','otto'])]
  study.append(dict(member=k,sha256=h,lines=len(ls),relevant_line_count=len(match),excerpts=match[:50]))
dump(S/'ORIGINAL_SOURCE100.json',dict(archives=orig,items=items,study_selected=study,authority='R69 exact archive spans; original mcts_study thermal proposals/guides and prior study docs'))
print('SOURCE',len(items),len(study),flush=True)
