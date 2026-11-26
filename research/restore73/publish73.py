from base61 import *
import csv,shutil,subprocess,sys,zipfile
from PIL import Image
O=B/'mqft_restore72_20261126'; A=B/'mqft_restore69_20261126'; P=B/'mqft_restore39_20261125/repo'; R=B/'mqft_restore71_20261126'
parent='31277d04b65d37931ad35686c5ae660fb573563c'
def git(*args):return subprocess.check_output(['git','-C',str(P),*args],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain')
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:
 assert sha(S/f)==sha(O/f)==sha(A/f),f
original=read(O/'ORIGINAL_SOURCE72.json')
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert all(sha(paths[k])==h for k,h in original['archives'].items())
a=read(S/'AUDIT73.json');r=read(S/'INSTRUMENT73.json');dec=read(S/'DECISION_RESULTS73.json');ind=read(S/'INDEPENDENT73.json');pm=read(S/'PLOT_MANIFEST73.json');reg=read(S/'REGISTRATION.json');br=read(S/'BRANCH_REGISTRATION73.json')
assert a['paths']==896 and a['backups']==114688 and a['reused_paths']==9216 and r['branch_paths']==4544 and r['branch_backups']==145408 and ind['status']=='PASS_SCOPED' and ind['contexts']==64
assert all(sha(S/k)==h for k,h in pm['sources'].items())
assert sha(S/'DECISION73.npz')==a['array_sha256']==dec['array_sha256'] and sha(R/'AUDIT71.json')==dec['source_R71_audit_sha']
assert sha(S/'mqft_branch73')==br['binary_sha'] and sha(S/'branch73.patch')==br['patch_sha']
assert sha(S/'mqft_train')==reg['binary_sha'] and all(sha(B/x['path'])==x['sha256'] for x in reg['weights'].values())
assert all(q['matched_original_32'] for q in r['contexts'])
assert '1 passed; 0 failed' in (S/'logs/branch_tests.log').read_text()
assert sum(q['adaptive']['record_neg']>1e-10 for q in r['contexts'])==63
scopes={'L05':'Typed native and forced-diagnostic evidence counts plus cross-validation; no physical temperature/critical-point count.', 'L34':'Scoped measurement-quality summary and explicit quantum model versus native observation classification.', 'L15':'Actual adaptive branch-aware records, separate disjoint final-decision predictive fragments; no Darwinism plateau.', 'L30':'Disjoint fragment predictive log-loss gains and within-root shuffle controls; not redundancy certification.', 'Q24':'Actual two-time branch-dependent complex density, system entropy and measured-record dephasing; no full autonomous 128-step coherence.', 'Q25':'Record quantum and classical MI and independent root decision predictive fragments; no objectivity claim.', 'Q26':'Pointer and decision distinguished, specified dephasing and decision-class distribution.', 'Q28':'Scoped actual feedback and decoherence control report; no physical quantum hardware.', 'Q35':'Complex two-record density, adaptive coherent model negativity and measured/frozen null; no physical Bell claim.', 'Q37':'Actual conditional feedback and separate final-decision prediction information flow, no full physical information network.', 'Q40':'Scoped entropy/information/negativity and measurement audit report, not thermodynamic/area-law/criticality certification.'}
ids={i for q in pm['figures'] for i in q['ids']};assert ids==set(scopes)
catalog={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')}
members={}
for i in sorted(ids):
 q=catalog[i];archive=q['archive'];member=q['member'];key=(archive,member)
 with zipfile.ZipFile(paths[archive]) as z:b=z.read(member)
 h=hashlib.sha256(b).hexdigest();assert h==q['member_sha256']
 members[key]=dict(archive=archive,member=member,sha256=h)
with zipfile.ZipFile(paths['mcts_study.zip']) as z:
 for member in z.namelist():
  if member.endswith('.md') and any(k in member.lower() for k in ('decoherence','darwinism','quantum-classical')):
   b=z.read(member);members['study',member]=dict(archive='mcts_study.zip',member=member,sha256=hashlib.sha256(b).hexdigest())
dump(S/'ORIGINAL_SOURCE73.json',dict(archives=original['archives'],verified_members=list(members.values()),original_attachment_receipt=original['original_attachment_receipt'],scope='Three original local archives and relevant source members; original supplied attachment receipt inherited.'))
prev=read(O/'PROGRESS72.json');inv=cr(O/'PLOT_INVENTORY72.csv');assert len(inv)==574 and sha(O/'PLOT_INVENTORY72.csv')==prev['inventory_sha256']
assert len(pm['figures'])==10
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha'] and pdf.read_bytes()[:4]==b'%PDF'
 with Image.open(png) as im:im.verify()
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==584
cw(S/'PLOT_INVENTORY73.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS72.csv');assert len(rows)==121
ev=[]
for x in rows:
 k=x['id'];x['R73_evidence']=scopes.get(k,'');x['R73_scope_status']='SCOPED_ADAPTIVE_RECORD_PANEL' if k in scopes else ''
 if k in scopes:
  y=catalog[k];z=[f['figure'] for f in pm['figures'] if k in f['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(z),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_ADAPTIVE_RECORD_PANEL',scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE73.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS73.csv',rows)
prior={x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,73))}
union={x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,74))}
assert len(prior)==101 and len(union)==104 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
prev.update(figure_pairs=584,distinct_png_sha=584,files_paired=1168,current_pairs=10,prior_pairs=574,current_animations=0,animation_scope='No new animation; historical retained figures unchanged.',inventory_sha256=sha(S/'PLOT_INVENTORY73.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE73.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS73.csv'))
prev['stages'][S.name]=10;prev['original_output_evidence']=dict(current=len(ids),R55_R73_union=len(union),new_scoped_ids=sorted(union-prior),baseline_blocked_with_scoped_evidence=sorted(k for k in union if catalog[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='R55-R73 scoped evidence, not final completion.')
dump(S/'PROGRESS73.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for n in dag['nodes']:
 if n['id']=='REPLAN-05':n['restore73_followup']=dict(evidence='64 actual conditional contexts, 4544 exhaustive native forced branches; diagonal law exact; explicit coherent-control two-record density negativity positive 63/64; measured/frozen zero; fresh disjoint decision predictive controls.',remaining='Full process instrument and operational coherent feedback, independent quantum readout and final-decision redundancy; no Darwinism plateau.',artifact=S.name+'/{THEORY.md,INSTRUMENT73.json,DECISION_RESULTS73.json}')
 if n['id']=='X-REP':n['restore73_followup']=dict(evidence='Explicit complex positive-Kraus branch-dependent model, shared bath and local record flags; independent density construction.',remaining='Three or more time points, actual coherent intervention identifiability and non-unique phase extension.',artifact=S.name+'/INDEPENDENT73.json')
 if n['id']=='REPLAN-01':n['restore73_followup']=dict(evidence='Measurement quality explicit typed audit and original L05/L34 scope.',remaining='Complete original measurement-quality report including physical thermal/critical claims not certified.',artifact=S.name+'/plots/measurement_quality.png')
 if n['id']=='X-MEMORY':n['restore73_followup']=dict(evidence='R71-trained disjoint temporal fragment decision predictors tested on all 16 fresh R73 roots; Rapfi transfer negative.',remaining='Fresh classifier calibration, root-conditioned information and autonomous closure; no universal memory sufficiency.',artifact=S.name+'/DECISION_RESULTS73.json')
dag.update(revision='RESTORE73_20261126',scope='Actual finite adaptive two-record branch law and explicit coherent extension; independent decision information',previous_entry=dag['entry'],entry='REPLAN-05',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE74-DRIVEN-REVERSE-THERMAL',mainline_priorities=['R74 driven finite open-system reverse law and thermal identification','Multi-time coherent control, final-decision instrument and Rapfi calibration','Augmented coarse state and independent population/size/criticality'])
files=['REGISTRATION.json','BRANCH_REGISTRATION73.json','BRANCH_PATCH73.json','AUDIT73.json','INSTRUMENT73.json','DECISION_RESULTS73.json','INDEPENDENT73.json','PLOT_MANIFEST73.json','PLOT_INVENTORY73.csv','ORIGINAL_OUTPUT_EVIDENCE73.csv','ORIGINAL_OUTPUT_PROGRESS73.csv','ORIGINAL_SOURCE73.json','PROGRESS73.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']
rec=dict(status='scoped_actual_branch_law_coherent_model_nonunique',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=896,cumulative_fixed_root_histories=71096,fresh_backups=114688,forced_branch_replays=4544,forced_branch_backups=145408,conditional_contexts=64,raw_bytes=a['raw_bytes']+r['raw_bytes'],reused_R71_paths=9216,reused_R71_backups=1179648,positive_model_record_negativity=63,new_games=0,optimizer_updates=0,figures=10,animations=0,retained_pairs=584,original_evidence_items=len(ids),original_evidence_union_R55_R73=len(union),additional_plot_nodes=0,original_obligations=121,native_subagent='NOT_RUN',remaining='coherent operational identification, process tensor, decision redundancy, thermal reverse law')
rec.update({f+'_sha256':sha(S/f) for f in files});dag['restore73']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore73']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=reg['external'],branch_binary_sha=br['binary_sha'],original_binary_sha=reg['binary_sha']);dump(S/'PROVENANCE.json',prov)
(S/'execute73.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd "$(dirname "$0")"\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\n# Registered raw data already exist. No overwrite of raw data.\npython3 decision73.py\npython3 instrument73.py\npython3 verify73.py\npython3 plot73.py\n')
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','arena68.py','audit65.py','audit68.py','channel64.py','audit_branch73.py','plan73.py','generate73.py','build_branch73.py','branch73.patch','branch_plan73.py','decision73.py','instrument73.py','verify73.py','plot73.py','publish73.py','backup73.py','execute73.sh','ROOT_PREPARATION.json','RUN_SHARDS.json','RAPFI_RECEIPTS73.json','RESULTS_KO.md','STAGE_LEDGER.md','logs/branch_tests.log']
raw=a['raw_receipts']+[x for q in r['contexts'] for x in q['raw_receipts']]
manifest=dict(schema='mqft_restore73_scoped_run_v1',parent=parent,status='completed_scoped',command='bash execute73.sh (registered raw data); fresh generation: plan73.py then generate73.py; diagnostic branch build and branch_plan73.py',files={f:sha(S/f) for f in files+sources+['DAG.json','PROVENANCE.json','DECISION73.npz','mqft_train','mqft_branch73']},raw_receipts=raw,prior_raw_receipts=[],weights=reg['weights'],external=reg['external'],plan_hashes=reg['plans']|{f'branch{k}':{'sha256':v['sha256'],'n':v['count']} for k,v in br['plans'].items()},inputs={str((R/'AUDIT71.json').relative_to(B)):sha(R/'AUDIT71.json'),str((R/'JOINT71.npz').relative_to(B)):sha(R/'JOINT71.npz')},software={'python':sys.version,'numpy':np.__version__},limits='Same-agent independent projector check. Only focused Rust phase65 test; no full regression. Native branch interventions not iid; no physical quantum inference.')
dump(S/'RUN_MANIFEST73.json',manifest)
dst=P/'research/restore73';assert not dst.exists();dst.mkdir()
for f in files+sources+['DAG.json','PROVENANCE.json','RUN_MANIFEST73.json']:
 p=dst/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,p)
shutil.copytree(S/'plans',dst/'plans')
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS73.csv','ORIGINAL_OUTPUT_EVIDENCE73.csv','PLOT_INVENTORY73.csv','PROGRESS73.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE73_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE73_RESULTS.md')
git('apply','--check',str(S/'branch73.patch'));git('add','research/restore73','docs');git('diff','--cached','--check','--','.',':(exclude)research/restore73/branch73.patch',':(exclude)research/restore73/logs/branch_tests.log');git('commit','-m','RESTORE73: native adaptive record branches and coherent-control model')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain')
print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE73.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE73.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE73_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST73.json'),source_zip_sha256=sha(S/'MQFT_RESTORE73_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE73.git.bundle'),progress_sha256=sha(S/'PROGRESS73.json'),fresh_paths=896,fresh_backups=114688,forced_branch_paths=4544,forced_branch_backups=145408,figures=10,cumulative_retained_pairs=584,original_union=104,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
