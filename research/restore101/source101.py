from base61 import *
import csv,zipfile
O=B/'mqft_restore100_20261129'
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
orig=read(O/'ORIGINAL_SOURCE100.json')['archives'];assert all(sha(p)==orig[k] for k,p in paths.items())
qm={x['id']:x for x in csv.DictReader(open(O/'QUALITY_MATRIX100.csv'))};items=[]
for k in read(S/'REGISTRATION.json')['targets']:
 x=qm[k];code=x['source_member'];ar='quantum.zip' if k.startswith('Q') else 'omoknuni_quantum-main.zip'
 with zipfile.ZipFile(paths[ar]) as z:b=z.read(code)
 h=hashlib.sha256(b).hexdigest();ls=b.decode().splitlines();assert h==x['source_sha'];l,r=map(int,x['source_span'].split('-'));assert x['source_function'] in ls[l-1]
 items.append(dict(id=k,archive=ar,member=code,member_sha=h,span=[l,r],source_function=x['source_function'],source_excerpt=ls[l-1:r],acceptance=x['final_acceptance']))
study=[]
with zipfile.ZipFile(paths['mcts_study.zip']) as z:
 assert z.testzip() is None
 old={x['member']:x['sha256'] for x in read(O/'ORIGINAL_SOURCE100.json')['study_selected']}
 names=[k for k in z.namelist() if k.endswith('.md') and any(w in k.lower() for w in ['qft-mcts-decoherence','qft-mcts-foundations','quantum_mcts_foundation','quantum_mcts_validation','comprehensive_plot_documentation','qft-mcts-math-foundations','quantum-theory-foundations'])]
 for k in names:
  b=z.read(k);h=hashlib.sha256(b).hexdigest();ls=b.decode().splitlines()
  if k in old:assert h==old[k]
  match=[(i+1,s) for i,s in enumerate(ls) if any(w in s.lower() for w in ['lindblad','hbar','path integral','action','decoherence','classical limit','kraus','entangle','coherence'])]
  study.append(dict(member=k,sha256=h,lines=len(ls),relevant_line_count=len(match),excerpts=match[:45]))
dump(S/'ORIGINAL_SOURCE101.json',dict(archives=orig,items=items,study_selected=study,authority='R69 archived source spans and quantum/field study documents'))
print('SOURCE',[(x['id'],x['span']) for x in items],len(study),flush=True)
