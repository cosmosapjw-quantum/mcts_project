from base61 import *
import csv,zipfile
O=B/'mqft_restore98_20261128';R77=B/'mqft_restore77_20261127'
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
orig=read(O/'ORIGINAL_SOURCE98.json')['archives'];assert all(sha(p)==orig[k] for k,p in paths.items())
qm={x['id']:x for x in csv.DictReader(open(O/'QUALITY_MATRIX98.csv'))};items=[];helper=[]
with zipfile.ZipFile(paths['quantum.zip']) as z:
 for k in read(S/'REGISTRATION.json')['targets']:
  x=qm[k];code=x['source_member'];b=z.read(code);h=hashlib.sha256(b).hexdigest();ls=b.decode().splitlines();assert h==x['source_sha'];l,r=map(int,x['source_span'].split('-'));assert x['source_function'] in ls[l-1]
  items.append(dict(id=k,archive='quantum.zip',member=code,member_sha=h,span=[l,r],source_function=x['source_function'],source_excerpt=ls[l-1:r],acceptance=x['final_acceptance']))
 for code,l,r in [('quantum/research/authentic_mcts_physics_extractor.py',982,1010),('quantum/research/authentic_mcts_physics_extractor.py',1771,1905)]:
  b=z.read(code);ls=b.decode().splitlines();helper.append(dict(member=code,sha256=hashlib.sha256(b).hexdigest(),span=[l,r],excerpt=ls[l-1:r]))
study=[]
with zipfile.ZipFile(paths['mcts_study.zip']) as z:
 assert z.testzip() is None
 for prior in read(O/'ORIGINAL_SOURCE98.json')['study_selected']:
  k=prior['member'];b=z.read(k);h=hashlib.sha256(b).hexdigest();ls=b.decode().splitlines();assert h==prior['sha256']
  match=[(i+1,s) for i,s in enumerate(ls) if any(w in s.lower() for w in ['critical','scaling','susceptib','binder','universality','correlation length','heat capacity'])]
  study.append(dict(member=k,sha256=h,lines=len(ls),relevant_line_count=len(match),excerpts=match[:50]))
dump(S/'ORIGINAL_SOURCE99.json',dict(archives=orig,items=items,helpers=helper,study_selected=study,authority='R69 exact originals; archived mcts_study; not edited'))
print('SOURCE',[(x['id'],x['span']) for x in items],flush=True)
