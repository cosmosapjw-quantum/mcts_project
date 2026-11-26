from base61 import *
import csv,shutil,subprocess,sys,zipfile
from PIL import Image
O=B/'mqft_restore74_20261126';A=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo';H=B/'mqft_restore68_20261126'
parent='11aaf8483fb3a8ab18c6c557039891d9140a8c46'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain')
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:
 assert sha(S/f)==sha(O/f)==sha(A/f),f
original=read(O/'ORIGINAL_SOURCE74.json')
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert all(sha(paths[k])==h for k,h in original['archives'].items())
a=read(S/'AUDIT75.json');r=read(S/'FIRST_RESULTS75.json');q=read(S/'TRANSPORT75.json');g=read(S/'GAMES75.json');ind=read(S/'INDEPENDENT75.json');pm=read(S/'PLOT_MANIFEST75.json');reg=read(S/'REGISTRATION.json');p74=read(O/'REGISTRATION.json')
assert a['paths']==3072 and a['backups']==393216 and a['eligible']==3071 and a['contexts']==96 and g['source_games']==16 and ind['status']=='PASS_SCOPED'
assert all(sha(S/k)==h for k,h in pm['sources'].items())
assert sha(S/'FIRST75.npz')==a['array_sha256']==r['array_sha256'] and sha(S/'FIRST_RESULTS75.json')==q['source_sha']
assert sha(S/'mqft_pulse74')==reg['pulse_binary_sha']==p74['pulse_binary_sha'] and sha(S/'pulse74.patch')==reg['pulse_patch_sha']==p74['pulse_patch_sha'] and sha(S/'mqft_train')==reg['binary_sha']
assert reg['external']==p74['external'] and all(sha(B/x['path'])==x['sha256'] for x in reg['weights'].values())
assert sha(H/'CAMPAIGN68.json')==g['campaign_sha'] and sha(H/'REGISTRATION.json')==g['reg_sha']
scopes={
 'L10':'Actual full risk-set first subsequent target hit and historical same-player value drawdown; separate declared coherent reference.',
 'G19':'Observed hit, censor and non-entry with actual compensator and separate historical episodes.',
 'G20':'Frozen native legal graph minimax height/width and exact threshold certificate; no native spatial motion constraint.',
 'G21':'Exact graph and path transmission, lattice evanescent action and continuum convergence, explicit Hamiltonian, leads and units; no native physical WKB identification.',
 'G22':'Scoped native risk/compensator, coherent phase-sensitive transport and dephased controls, negative evidence; no physical quantum tunneling or full final certification.'
}
ids={i for f in pm['figures']+pm['animations'] for i in f['ids']};assert ids==set(scopes)
catalog={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};members={};guides=[]
for i in sorted(ids):
 x=catalog[i];archive=x['archive'];member=x['member']
 if archive=='guide_only':guides.append(dict(id=i,member=member,authority='inherited original attachment receipt and R69 catalog, not local source member bytes'));continue
 with zipfile.ZipFile(paths[archive]) as z:b=z.read(member)
 h=hashlib.sha256(b).hexdigest();assert h==x['member_sha256'];members[archive,member]=dict(archive=archive,member=member,sha256=h)
with zipfile.ZipFile(paths['omoknuni_quantum-main.zip']) as z:
 member='omoknuni_quantum-main/python/mcts/quantum/phenomena/tunneling.py';b=z.read(member);assert b'np.sqrt(barrier.duration)' in b;members['producer',member]=dict(archive='omoknuni_quantum-main.zip',member=member,sha256=hashlib.sha256(b).hexdigest())
with zipfile.ZipFile(paths['mcts_study.zip']) as z:
 for member in ['mcts_study/v6.0/quantum_mcts_foundation.md','mcts_study/effective-action-mcts.md']:
  b=z.read(member);members['study',member]=dict(archive='mcts_study.zip',member=member,sha256=hashlib.sha256(b).hexdigest())
dump(S/'ORIGINAL_SOURCE75.json',dict(archives=original['archives'],verified_members=list(members.values()),guide_only=guides,original_attachment_receipt=original['original_attachment_receipt'],scope='Three original local archive hashes and relevant actual source members. Guide-only labels inherited; no nonexistent member bytes claimed.'))
prev=read(O/'PROGRESS74.json');inv=cr(O/'PLOT_INVENTORY74.csv');assert len(inv)==595 and sha(O/'PLOT_INVENTORY74.csv')==prev['inventory_sha256']
assert len(pm['figures'])==12 and not pm['animations']
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha'] and pdf.read_bytes()[:4]==b'%PDF'
 with Image.open(png) as im:im.verify()
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==607
cw(S/'PLOT_INVENTORY75.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS74.csv');assert len(rows)==121
ev=[]
for x in rows:
 k=x['id'];x['R75_evidence']=scopes.get(k,'');x['R75_scope_status']='SCOPED_BARRIER_TRANSPORT_PANEL' if k in scopes else ''
 if k in scopes:
  y=catalog[k];z=[f['figure'] for f in pm['figures'] if k in f['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(z),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_BARRIER_TRANSPORT_PANEL',scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE75.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS75.csv',rows)
prior={x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,75))}
union={x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,76))}
assert len(prior)==112 and len(union)==114 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
prev.update(figure_pairs=607,distinct_png_sha=607,files_paired=1214,current_pairs=12,prior_pairs=595,current_animations=0,animation_scope='No new animation; historical animations retained.',inventory_sha256=sha(S/'PLOT_INVENTORY75.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE75.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS75.csv'))
prev['stages'][S.name]=12;prev['original_output_evidence']=dict(current=len(ids),R55_R75_union=len(union),new_scoped_ids=sorted(union-prior),baseline_blocked_with_scoped_evidence=sorted(k for k in union if catalog[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='R55-R75 scoped evidence, not final completion.')
dump(S/'PROGRESS75.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for n in dag['nodes']:
 if n['id']=='X-BARRIER':n['restore75_followup']=dict(evidence='3072 fresh actual native full-support paths, 3071 risk-eligible geometries, exact minimax and predictable hazard, 95%+ nonneighbor hits; 96 declared score graph models, 288 energies, full/path exact scattering, WKB/continuum and gauge/phase/detector controls; 16 historical same-side game episodes.',remaining='Native physical coordinate, kinetic action, energy and phase not identified; finite graph model not physical tunneling observation; historical game censoring potentially informative.',artifact=S.name+'/{FIRST_RESULTS75.json,TRANSPORT75.json,GAMES75.json}')
 if n['id']=='X-REP':n['restore75_followup']=dict(evidence='Complex Hermitian graph dynamics, lead matching, plaquette phase, same-H coherent/dephased detector comparison.',remaining='Actual operational coherent phase readout, not native stochastic path quantum identification.',artifact=S.name+'/TRANSPORT75.json')
dag.update(revision='RESTORE75_20261127',scope='Actual full-risk native first passage and declared score-graph coherent transport',previous_entry=dag['entry'],entry='X-BARRIER',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE76-INDEPENDENT-FINITE-SIZE-CRITICALITY',mainline_priorities=['R76 independent common evaluator size/budget/beta criticality and null','R77 autonomous RG fixed point and augmented closure','R78 structural gauge and R79 original quality report; final original scientific requirements'])
files=['REGISTRATION.json','AUDIT75.json','FIRST_RESULTS75.json','TRANSPORT75.json','GAMES75.json','INDEPENDENT75.json','PLOT_MANIFEST75.json','PLOT_INVENTORY75.csv','ORIGINAL_OUTPUT_EVIDENCE75.csv','ORIGINAL_OUTPUT_PROGRESS75.csv','ORIGINAL_SOURCE75.json','PROGRESS75.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']
rec=dict(status='scoped_native_first_passage_graph_transport',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=3072,cumulative_fixed_root_histories=76472,fresh_backups=393216,eligible_paths=3071,graph_contexts=96,stationary_energies_contexts=288,prior_games=16,raw_bytes=a['raw_bytes'],new_games=0,optimizer_updates=0,figures=12,animations=0,retained_pairs=607,original_evidence_items=len(ids),original_evidence_union_R55_R75=len(union),additional_plot_nodes=0,original_obligations=121,native_subagent='NOT_RUN',remaining='physical native phase, coordinate/kinetic scale, population; independent criticality and RG, structure and final certification')
rec.update({f+'_sha256':sha(S/f) for f in files});dag['restore75']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore75']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=reg['external'],pulse_binary_sha=reg['pulse_binary_sha'],original_binary_sha=reg['binary_sha']);dump(S/'PROVENANCE.json',prov)
(S/'execute75.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd "$(dirname "$0")"\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\n# Existing registered raw data only; do not overwrite native data.\npython3 first75.py\npython3 transport75.py\npython3 games75.py\npython3 verify75.py\npython3 plot75.py\n')
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','arena68.py','audit_native74.py','audit_rapfi74.py','pulse74.patch','plan75.py','generate75.py','first75.py','transport75.py','games75.py','verify75.py','plot75.py','publish75.py','backup75.py','execute75.sh','ROOT_PREPARATION.json','RUN_SHARDS.json','RAPFI_RECEIPTS75.json','RESULTS_KO.md','STAGE_LEDGER.md']
prior_raw=[dict(path=x['source_path'],sha256=x['source_sha']) for x in g['results'][0]['rows']];assert len(prior_raw)==16
inputs={str((H/'CAMPAIGN68.json').relative_to(B)):g['campaign_sha'],str((H/'REGISTRATION.json').relative_to(B)):g['reg_sha'],str((O/'REGISTRATION.json').relative_to(B)):sha(O/'REGISTRATION.json'),str((O/'PULSE_PATCH74.json').relative_to(B)):sha(O/'PULSE_PATCH74.json')}
manifest=dict(schema='mqft_restore75_scoped_run_v1',parent=parent,status='completed_scoped',command='bash execute75.sh (registered raw); fresh plan75.py then generate75.py with pinned R74 zero-pulse backend',files={f:sha(S/f) for f in files+sources+['DAG.json','PROVENANCE.json','FIRST75.npz','mqft_train','mqft_pulse74']},raw_receipts=a['raw_receipts'],prior_raw_receipts=prior_raw,weights=reg['weights'],external=reg['external'],plan_hashes=reg['plans'],inputs=inputs,software={'python':sys.version,'numpy':np.__version__},limits='Same-agent separate implementation, no full native regression; native and model transport distinct; prior game censoring; guide-only labels inherited')
dump(S/'RUN_MANIFEST75.json',manifest)
dst=P/'research/restore75';assert not dst.exists();dst.mkdir()
for f in files+sources+['DAG.json','PROVENANCE.json','RUN_MANIFEST75.json']:
 p=dst/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,p)
shutil.copytree(S/'plans',dst/'plans')
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS75.csv','ORIGINAL_OUTPUT_EVIDENCE75.csv','PLOT_INVENTORY75.csv','PROGRESS75.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE75_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE75_RESULTS.md')
git('add','research/restore75','docs');git('diff','--cached','--check','--','.',':(exclude)research/restore75/pulse74.patch');git('commit','-m','RESTORE75: native first passage and coherent score graph transport')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE75.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE75.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE75_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST75.json'),source_zip_sha256=sha(S/'MQFT_RESTORE75_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE75.git.bundle'),progress_sha256=sha(S/'PROGRESS75.json'),fresh_paths=3072,fresh_backups=393216,figures=12,cumulative_retained_pairs=607,original_union=114,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
