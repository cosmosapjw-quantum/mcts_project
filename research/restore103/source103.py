from base61 import *
import csv,zipfile
O=B/'mqft_restore102_20261129'
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
orig=read(O/'ORIGINAL_SOURCE102.json')['archives'];assert all(sha(p)==orig[k] for k,p in paths.items())
qm={x['id']:x for x in csv.DictReader(open(O/'QUALITY_MATRIX102.csv'))};items=[]
for k in read(S/'REGISTRATION.json')['targets']:
 x=qm[k];code=x['source_member'];ar='quantum.zip' if k.startswith('Q') else 'omoknuni_quantum-main.zip'
 with zipfile.ZipFile(paths[ar]) as z:b=z.read(code)
 h=hashlib.sha256(b).hexdigest();ls=b.decode().splitlines();assert h==x['source_sha'];l,r=map(int,x['source_span'].split('-'));assert x['source_function'] in ls[l-1]
 items.append(dict(id=k,archive=ar,member=code,member_sha=h,span=[l,r],source_function=x['source_function'],source_excerpt=ls[l-1:r],acceptance=x['final_acceptance']))
study=[]
with zipfile.ZipFile(paths['mcts_study.zip']) as z:
 assert z.testzip() is None
 old={x['member']:x['sha256'] for x in read(O/'ORIGINAL_SOURCE102.json')['study_selected']}
 names=[k for k in z.namelist() if k.endswith('.md') and any(w in k.lower() for w in ['qft-mcts-statistical','qft-mcts-decoherence','qft-mcts-foundations','quantum_mcts_foundation','quantum_mcts_validation','comprehensive_plot_documentation','qft-mcts-math-foundations','quantum-theory-foundations'])]
 for k in names:
  b=z.read(k);h=hashlib.sha256(b).hexdigest();ls=b.decode().splitlines()
  if k in old:assert h==old[k]
  match=[(i+1,s) for i,s in enumerate(ls) if any(w in s.lower() for w in ['temperature','boltzmann','scaling','measurement','entropy','conditional','thermal','energy'])]
  study.append(dict(member=k,sha256=h,lines=len(ls),relevant_line_count=len(match),excerpts=match[:45]))
helpers=[]
with zipfile.ZipFile(paths['omoknuni_quantum-main.zip']) as z:
 k='omoknuni_quantum-main/python/mcts/quantum/analysis/authentic_physics_extractor.py';b=z.read(k);ls=b.decode().splitlines()
 for l,r in [(101,284),(421,530)]:
  helpers.append(dict(member=k,sha256=hashlib.sha256(b).hexdigest(),span=[l,r],excerpt=ls[l-1:r]))
dump(S/'ORIGINAL_SOURCE103.json',dict(archives=orig,items=items,helpers=helpers,study_selected=study,authority='R69 archived source spans and temperature/measurement study documents'))
print('SOURCE',[(x['id'],x['span']) for x in items],len(study),flush=True)
