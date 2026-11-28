from base61 import *
import zipfile,csv
O=B/'mqft_restore97_20261128';R77=B/'mqft_restore77_20261127'
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
orig=read(O/'ORIGINAL_SOURCE97.json')['archives'];assert all(sha(p)==orig[k] for k,p in paths.items())
qm={x['id']:x for x in csv.DictReader(open(O/'QUALITY_MATRIX97.csv'))};items=[];helper=[];code='quantum/research/visualization/plot_rg_flow_phase_diagrams.py'
with zipfile.ZipFile(paths['quantum.zip']) as z:
 b=z.read(code);h=hashlib.sha256(b).hexdigest();ls=b.decode().splitlines()
 for k in read(S/'REGISTRATION.json')['targets']:
  x=qm[k];assert h==x['source_sha'];l,r=map(int,x['source_span'].split('-'));assert x['source_function'] in ls[l-1]
  items.append(dict(id=k,archive='quantum.zip',member=code,member_sha=h,span=[l,r],source_function=x['source_function'],source_excerpt=ls[l-1:r],acceptance=x['final_acceptance']))
 for code,l,r in [('quantum/research/authentic_mcts_physics_extractor.py',957,1010),('quantum/research/rg_flow.py',163,259)]:
  b=z.read(code);ls=b.decode().splitlines();helper.append(dict(member=code,sha256=hashlib.sha256(b).hexdigest(),span=[l,r],excerpt=ls[l-1:r]))
study=[]
with zipfile.ZipFile(paths['mcts_study.zip']) as z:
 assert z.testzip() is None
 for k in ['mcts_study/v3.0/qft-mcts-statistical.md','mcts_study/v3.0/qft-mcts-foundations.md','mcts_study/v5.0/COMPREHENSIVE_PLOT_DOCUMENTATION.md','mcts_study/v6.0/quantum_mcts_foundation.md','mcts_study/v6.0/quantum_mcts_validation.md']:
  b=z.read(k);h=hashlib.sha256(b).hexdigest();ls=b.decode().splitlines()
  if 'statistical' in k:assert h==read(R77/'ORIGINAL_RG_AUDIT77.json')['source_member']['sha256']
  match=[(i+1,s) for i,s in enumerate(ls) if any(w in s.lower() for w in ['renormaliz','fixed point','wilson','beta function','coarse-grain','scaling','critical'])]
  study.append(dict(member=k,sha256=h,lines=len(ls),relevant_line_count=len(match),excerpts=match[:40]))
dump(S/'ORIGINAL_SOURCE98.json',dict(archives=orig,items=items,helpers=helper,study_selected=study,authority='R69 original archives; precise source spans; not edited'))
print('SOURCE',[(x['id'],x['span']) for x in items],[(x['member'],x['relevant_line_count']) for x in study])
