from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore70_20261126'; A=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
parent='798e68fb1d79e9c3515069608921ca8be3fbe14c'
def git(*args):return subprocess.check_output(['git','-C',str(P),*args],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain')
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:
 assert sha(S/f)==sha(O/f)==sha(A/f),f
original=read(O/'ORIGINAL_SOURCE70.json')
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert all(sha(paths[k])==h for k,h in original['archives'].items())
import zipfile,hashlib
for q in original['verified_members']:
 with zipfile.ZipFile(paths[q['archive']]) as z:assert hashlib.sha256(z.read(q['member'])).hexdigest()==q['sha256']
original['scope']='All three original archives and four relevant original member identities reverified; supplied attachment receipt inherited from R69.'
dump(S/'ORIGINAL_SOURCE71.json',original)
r=read(S/'PHYSICS_RESULTS.json');ind=read(S/'INDEPENDENT71.json');pm=read(S/'PLOT_MANIFEST71.json');audit=read(S/'AUDIT71.json');reg=read(S/'REGISTRATION.json')
assert ind['status']=='PASS_SCOPED' and ind['forward_cases']==15 and len(pm['figures'])==10 and audit['paths']==9216 and audit['backups']==1179648
assert all(sha(S/k)==h for k,h in pm['sources'].items())
prev=read(O/'PROGRESS70.json');inv=cr(O/'PLOT_INVENTORY70.csv');assert len(inv)==553 and sha(O/'PLOT_INVENTORY70.csv')==prev['inventory_sha256']
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha'] and pdf.read_bytes()[:4]==b'%PDF'
 with Image.open(png) as im:im.verify()
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==563
cw(S/'PLOT_INVENTORY71.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS70.csv');catalog={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};assert len(rows)==121
scopes={'L16':'Autonomous finite marked donor transition, exact fitted-model generating functional and fourth directional cumulants, disjoint geometry tests; population bias remains.', 'Q21':'Autonomous marked stochastic path-law approximation, finite operator products; no physical heat/work, quantum amplitude or Jarzynski identity.'}
ev=[]
for x in rows:
 k=x['id'];x['R71_evidence']=scopes.get(k,'');x['R71_scope_status']='SCOPED_NATIVE_STATISTICAL_PANEL' if k in scopes else ''
 if k in scopes:
  y=catalog[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(z['figure'] for z in pm['figures'] if k in z['ids']),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_NATIVE_STATISTICAL_PANEL',scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE71.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS71.csv',rows)
prior={x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,71))}
union={x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,72))}
assert len(prior)==len(union)==90 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
prev.update(figure_pairs=563,distinct_png_sha=563,files_paired=1126,current_pairs=10,prior_pairs=553,current_animations=0,animation_scope='No fresh animation; historical retained figures unchanged.',inventory_sha256=sha(S/'PLOT_INVENTORY71.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE71.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS71.csv'))
prev['stages'][S.name]=10;prev['original_output_evidence']=dict(current=2,R55_R71_union=len(union),new_scoped_ids=sorted(union-prior),baseline_blocked_with_scoped_evidence=sorted(k for k in union if catalog[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='Explicit R55-R71 scoped evidence overlays, not final certification.')
dump(S/'PROGRESS71.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for n in dag['nodes']:
 if n['id']=='REPLAN-03':
  n['restore71_followup']=dict(evidence='9216 new actual searches, autonomous empirical joint transition count/mark/next-state, 15 fitted family/model combinations, exact within-model W and fourth directional moments, independent forward verification.',remaining='Population closure remains incomplete: residual support, Rapfi and g3 source bias; full joint cumulant tensor and physical hbar/continuum not certified.',artifact=S.name+'/{THEORY.md,PHYSICS_RESULTS.json,INDEPENDENT71.json}')
 if n['id']=='X-MEMORY':
  n['restore71_followup']=dict(evidence='One-lag autonomous state improves W RMSE on all five evaluator strata, but joint Brier degrades g3 and Rapfi vs state-only; root-level residual persists.',remaining='History/sufficiency, geometry covariates, support smoothing and out-of-cohort calibration.',artifact=S.name+'/PHYSICS_RESULTS.json')
dag.update(revision='RESTORE71_20261126',scope='Autonomous empirical marked joint transition and heldout large geometry cohort',previous_entry=dag['entry'],entry='REPLAN-03',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE72-COARSE-MAPS-SIZE-FAMILY',mainline_priorities=['R70-P residual support and population calibration follow-up','R72 five-size coarse maps and conditional closure','R73 adaptive multi-time complex records and decision information'])
files=['REGISTRATION.json','AUDIT71.json','PHYSICS_RESULTS.json','INDEPENDENT71.json','PLOT_MANIFEST71.json','PLOT_INVENTORY71.csv','ORIGINAL_OUTPUT_EVIDENCE71.csv','ORIGINAL_OUTPUT_PROGRESS71.csv','ORIGINAL_SOURCE71.json','PROGRESS71.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']
rec=dict(status='scoped_autonomous_joint_transition_population_bias_remaining',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=9216,cumulative_fixed_root_histories=69880,fresh_backups=1179648,fresh_waves=294912,raw_bytes=audit['raw_bytes'],full_replay_n=320,train_blocks=48,test_blocks=16,new_games=0,optimizer_updates=0,figures=10,retained_pairs=563,original_evidence_items=2,original_evidence_union_R55_R71=len(union),additional_plot_nodes=0,original_obligations=121,native_subagent='NOT_RUN',remaining='support, geometry and history dependence; external heldout bias; scale and complex instruments')
rec.update({f+'_sha256':sha(S/f) for f in files});dag['restore71']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore71']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=reg['external']);dump(S/'PROVENANCE.json',prov)
(S/'execute71.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd "$(dirname "$0")"\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\n# Existing registered raw data are read, not overwritten.\npython3 extract71.py\npython3 closure71.py\npython3 verify71.py\npython3 plot71.py\n')
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','arena68.py','audit65.py','audit68.py','channel64.py','marked70.py','plan71.py','generate71.py','extract71.py','closure71.py','verify71.py','plot71.py','publish71.py','backup71.py','execute71.sh','ROOT_PREPARATION.json','RUN_SHARDS.json','RAPFI_RECEIPTS71.json','RESULTS_KO.md','STAGE_LEDGER.md']
manifest=dict(schema='mqft_restore71_scoped_run_v1',parent=parent,status='completed_scoped',command='bash execute71.sh (registered raw data); fresh generation: python3 plan71.py then python3 generate71.py',files={f:sha(S/f) for f in files+sources+['DAG.json','PROVENANCE.json','JOINT71.npz','mqft_train']},raw_receipts=audit['raw_receipts'],prior_raw_receipts=[],weights=reg['weights'],external=reg['external'],plan_hashes=reg['plans'],software={'python':sys.version,'numpy':np.__version__},limits='No full legacy regression or native subagent; 320 native full replay; 16 test geometry blocks and fixed-fit pointwise uncertainty; population closure not certified')
dump(S/'RUN_MANIFEST71.json',manifest)
dst=P/'research/restore71';assert not dst.exists();dst.mkdir()
for f in files+sources+['DAG.json','PROVENANCE.json','RUN_MANIFEST71.json']:shutil.copy2(S/f,dst/f)
shutil.copytree(S/'plans',dst/'plans')
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS71.csv','ORIGINAL_OUTPUT_EVIDENCE71.csv','PLOT_INVENTORY71.csv','PROGRESS71.json']:
 shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE71_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE71_RESULTS.md')
git('add','research/restore71','docs');git('diff','--cached','--check');git('commit','-m','RESTORE71: large heldout autonomous marked joint transition')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain')
print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE71.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE71.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE71_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST71.json'),source_zip_sha256=sha(S/'MQFT_RESTORE71_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE71.git.bundle'),progress_sha256=sha(S/'PROGRESS71.json'),fresh_paths=9216,fresh_backups=1179648,figures=10,cumulative_retained_pairs=563,original_union=len(union),clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
