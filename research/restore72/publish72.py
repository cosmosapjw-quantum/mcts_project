from base61 import *
import csv,shutil,subprocess,sys,zipfile,hashlib
from PIL import Image
O=B/'mqft_restore71_20261126'; A=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'; F=B/'mqft_restore49_20261126'
parent='72f54ef26446876cc2d1ac747589b9deec4fc735'
LS=[13,15,17,19,21]
def git(*args):return subprocess.check_output(['git','-C',str(P),*args],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain')
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:
 assert sha(S/f)==sha(O/f)==sha(A/f),f
original=read(O/'ORIGINAL_SOURCE71.json')
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert all(sha(paths[k])==h for k,h in original['archives'].items())
r=read(S/'SCALE_RESULTS72.json');mem=read(S/'MEMORY72.json');ind=read(S/'INDEPENDENT72.json');pm=read(S/'PLOT_MANIFEST72.json')
assert ind['status']=='PASS_SCOPED' and ind['raw_witness_checks']==112 and ind['quiescent_witnesses']==112 and len(pm['figures'])==11 and len(pm['animations'])==2 and r['fresh_paths']==320
assert sha(O/'JOINT71.npz')==mem['data_sha']==read(O/'AUDIT71.json')['array_sha256']
assert all(sha(S/k)==h for k,h in pm['sources'].items())
assert all(sha(B/k)==h for k,h in r['prior_R58_source_hashes'].items())
scopes={'L13':'Exact nested finite crop map, covariance/source lift and count-only nonclosure; not autonomous RG.','L14':'Five-size finite-budget covariance and low-mode diagnostics, historical beta cohort; no critical transition.','L16':'Exact source lift and nonclosure, R71 reused block-memory generating functional.','L27':'Finite size/budget and historic beta dependence; no phase boundary.','L28':'Negative critical-exponent identification via competing finite-size descriptions; no exponent admitted.','Q09':'Finite exterior-mass differences across sizes, not physical beta functions.','Q10':'Separate R58 finite beta scans; no phase diagram certification.','Q11':'Finite spatial mass flow, no autonomous RG trajectory.','Q12':'Actual finite geometric pushforward animation; no RG dynamics.','Q13':'Scoped cross-size/closure report, no RG fixed point.','Q14':'Finite-size model alternatives and held-size-out errors; not critical exponents.','Q15':'Finite-budget covariance rescaling diagnostic, not original critical 3D collapse.','Q16':'Finite normalized occupancy covariance and martingale/drift scaling.','Q17':'Finite prefix scaling animation, not critical animation certification.','Q18':'Scoped finite-size diagnostic report; no universal criticality.','Q21':'Marked block generating functional and source-error separation; no physical path amplitude, heat or Jarzynski.','Q30':'Finite-budget covariance rescaling; no critical collapse.'}
ids={i for q in pm['figures']+pm['animations'] for i in q['ids']};assert ids==set(scopes)
catalog={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')}
members={}
for i in sorted(ids):
 q=catalog[i];archive=q['archive'];member=q['member'];key=(archive,member)
 with zipfile.ZipFile(paths[archive]) as z:b=z.read(member)
 h=hashlib.sha256(b).hexdigest();assert h==q['member_sha256']
 if key not in members:members[key]=dict(archive=archive,member=member,sha256=h)
with zipfile.ZipFile(paths['mcts_study.zip']) as z:
 key='mcts_study/v3.0/qft-mcts-statistical.md';b=z.read(key);h=hashlib.sha256(b).hexdigest()
 lines=b.decode().splitlines();assert 'Renormalization Group Analysis' in '\n'.join(lines) and 'Critical Phenomena' in '\n'.join(lines)
 members['study',key]=dict(archive='mcts_study.zip',member=key,sha256=h,verified_sections=['2','3','4'])
dump(S/'ORIGINAL_SOURCE72.json',dict(archives=original['archives'],verified_members=list(members.values()),original_attachment_receipt=original['original_attachment_receipt'],scope='All three local original archives and relevant source members; attachment receipt inherited.'))
prev=read(O/'PROGRESS71.json');inv=cr(O/'PLOT_INVENTORY71.csv');assert len(inv)==563 and sha(O/'PLOT_INVENTORY71.csv')==prev['inventory_sha256']
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha'] and pdf.read_bytes()[:4]==b'%PDF'
 with Image.open(png) as im:im.verify()
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
for x in pm['animations']:
 p=S/'plots'/x['file'];assert sha(p)==x['sha256']
 with Image.open(p) as im:
  assert im.n_frames==x['frames']
  for k in range(im.n_frames):im.seek(k);im.load()
assert len(inv)==len({x['png_sha'] for x in inv})==574
cw(S/'PLOT_INVENTORY72.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS71.csv');assert len(rows)==121
ev=[]
for x in rows:
 k=x['id'];x['R72_evidence']=scopes.get(k,'');x['R72_scope_status']='SCOPED_FINITE_DIAGNOSTIC_PANEL' if k in scopes else ''
 if k in scopes:
  y=catalog[k];z=[f['figure'] for f in pm['figures'] if k in f['ids']]+[f['file'] for f in pm['animations'] if k in f['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(z),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_FINITE_DIAGNOSTIC_PANEL',scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE72.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS72.csv',rows)
prior={x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,72))}
union={x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,73))}
assert len(prior)==90 and len(union)==101 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
prev.update(figure_pairs=574,distinct_png_sha=574,files_paired=1148,current_pairs=11,prior_pairs=563,current_animations=2,animation_scope='Two new finite flow animations; separate from cumulative PNG/PDF pair inventory.',inventory_sha256=sha(S/'PLOT_INVENTORY72.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE72.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS72.csv'))
prev['stages'][S.name]=11;prev['original_output_evidence']=dict(current=len(ids),R55_R72_union=len(union),new_scoped_ids=sorted(union-prior),baseline_blocked_with_scoped_evidence=sorted(k for k in union if catalog[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='R55-R72 scoped evidence, includes negative diagnostics, not final completion.')
dump(S/'PROGRESS72.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for n in dag['nodes']:
 if n['id']=='REPLAN-04.R':n['restore72_followup']=dict(evidence='80 exact nested map compositions; source lift, entropy/KL and covariance; 112 same-root count-only nonlumpability raw witnesses.',remaining='Augmented state and inter-size dynamical closure; no physical beta function.',artifact=S.name+'/{THEORY.md,SCALE_RESULTS72.json,INDEPENDENT72.json}')
 if n['id']=='REPLAN-04.C':n['restore72_followup']=dict(evidence='Five genuine common NNUE sizes, 320 paths, fixed count/density nested geometry, low-mode covariance and finite alternatives.',remaining='Fresh beta/c_puct grid, dimensionless crossings, larger sizes; critical point/exponents open.',artifact=S.name+'/SCALE_RESULTS72.json')
 if n['id'] in ('REPLAN-03','X-MEMORY'):n['restore72_followup']=dict(evidence='R71 block-memory donor diagnostic on existing large cohort; fixed 18 variants; count/mark source errors separate.',remaining='Fresh root holdout, full joint marks/next-state coupling and support; no exact closure.',artifact=S.name+'/MEMORY72.json')
dag.update(revision='RESTORE72_20261126',scope='Exact finite spatial pushforward, raw nonlumpability witnesses, finite-size comparisons and R71 memory diagnostics',previous_entry=dag['entry'],entry='REPLAN-04.R',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE73-ADAPTIVE-COMPLEX-RECORDS',mainline_priorities=['R73 actual adaptive multi-time complex records and final decision','Augmented coarse state and size/criticality grid','Population memory/support and independent calibration'])
files=['SCALE_RESULTS72.json','MEMORY72.json','INDEPENDENT72.json','PLOT_MANIFEST72.json','PLOT_INVENTORY72.csv','ORIGINAL_OUTPUT_EVIDENCE72.csv','ORIGINAL_OUTPUT_PROGRESS72.csv','ORIGINAL_SOURCE72.json','PROGRESS72.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']+[f'REGISTRATION_L{L}.json' for L in LS]+[f'AUDIT_L{L}.json' for L in LS]
rec=dict(status='scoped_exact_pushforward_count_only_closure_refuted',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=320,cumulative_fixed_root_histories=70200,fresh_backups=10240,fresh_candidate_samples=42020,raw_bytes=r['raw_bytes'],reused_R71_paths=9216,reused_R71_backups=1179648,full_root_replay_n=320,composition_checks=80,count_collision_witnesses=112,new_games=0,optimizer_updates=0,figures=11,animations=2,retained_pairs=574,original_evidence_items=len(ids),original_evidence_union_R55_R72=len(union),additional_plot_nodes=0,original_obligations=121,native_subagent='NOT_RUN',remaining='augmented closure, physical criticality, complex multi-time records')
rec.update({f+'_sha256':sha(S/f) for f in files});dag['restore72']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');regs=[read(S/f'REGISTRATION_L{L}.json') for L in LS];prov['restore72']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=regs[0]['model'],backends={str(L):q['backend_sha'] for L,q in zip(LS,regs)});dump(S/'PROVENANCE.json',prov)
(S/'execute72.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd "$(dirname "$0")"\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\nB="$(dirname "$PWD")"; O="$B/mqft_restore49_20261126"; P="$B/mqft_restore39_20261125/repo"\nfor L in 13 15 17 19 21; do PYTHONPATH="$O/family/L$L/python" "$P/.venv/bin/python" audit72.py "$L"; done\npython3 memory72.py\npython3 scale72.py\npython3 verify72.py\npython3 plot72.py\n')
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','plan72.py','audit72.py','memory72.py','scale72.py','verify72.py','plot72.py','publish72.py','backup72.py','execute72.sh','RESULTS_KO.md','STAGE_LEDGER.md']
raw=[x for L in LS for x in read(S/f'AUDIT_L{L}.json')['raw_receipts']]
manifest=dict(schema='mqft_restore72_scoped_run_v1',parent=parent,status='completed_scoped',command='bash execute72.sh (registered raw data); plan72.py L and size-specific mqft.cli generate',files={f:sha(S/f) for f in files+sources+['DAG.json','PROVENANCE.json']+[f'SCALE_L{L}.npz' for L in LS]},raw_receipts=raw,prior_raw_receipts=[],weights={},external=regs[0]['model'],plan_hashes={f'L{L}':{'sha256':sha(S/'plans'/f'L{L}.json')} for L in LS},inputs={str((F/'runtime_verified'/f'mqft-L{L}').relative_to(B)):q['backend_sha'] for L,q in zip(LS,regs)}|{str((O/'JOINT71.npz').relative_to(B)):mem['data_sha']}|r['prior_R58_source_hashes'],software={'python':sys.version,'numpy':np.__version__},limits='No full legacy regression or native subagent; root replay all; R71 split reused, no fresh holdout for block models; no critical limit')
dump(S/'RUN_MANIFEST72.json',manifest)
dst=P/'research/restore72';assert not dst.exists();dst.mkdir()
for f in files+sources+['DAG.json','PROVENANCE.json','RUN_MANIFEST72.json']:shutil.copy2(S/f,dst/f)
shutil.copytree(S/'plans',dst/'plans')
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS72.csv','ORIGINAL_OUTPUT_EVIDENCE72.csv','PLOT_INVENTORY72.csv','PROGRESS72.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE72_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE72_RESULTS.md')
git('add','research/restore72','docs');git('diff','--cached','--check');git('commit','-m','RESTORE72: five-size exact pushforward and count-closure witnesses')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain')
print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE72.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE72.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE72_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST72.json'),source_zip_sha256=sha(S/'MQFT_RESTORE72_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE72.git.bundle'),progress_sha256=sha(S/'PROGRESS72.json'),fresh_paths=320,fresh_backups=10240,figures=11,animations=2,cumulative_retained_pairs=574,original_union=len(union),clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
