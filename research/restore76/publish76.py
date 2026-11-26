from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore75_20261127'; A=B/'mqft_restore69_20261126'; P=B/'mqft_restore39_20261125/repo'; F=B/'mqft_restore49_20261126'; H=B/'mqft_restore46_20261126'
parent='e8f6b803cff2bd8d1046243aac99f55a9253b63d'; LS=[13,15,17,19,21]
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain')
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A/f),f
old=read(O/'ORIGINAL_SOURCE75.json');src=read(S/'ORIGINAL_SOURCE76.json')
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert old['archives']==src['archives'] and all(sha(paths[k])==v for k,v in src['archives'].items())
q=read(S/'CRITICAL76.json');v=read(S/'INDEPENDENT76.json');pm=read(S/'PLOT_MANIFEST76.json')
aud=[read(S/f'AUDIT_L{L}.json') for L in LS];reg=[read(S/f'REGISTRATION_L{L}.json') for L in LS];gen=[read(S/f'GEN_L{L}.json') for L in LS]
assert q['paths']==3200 and q['backups']==112640 and len(q['groups'])==80 and v['status']=='PASS_SCOPED' and v['root_group_checks']==640 and v['raw_paths_independently_reduced']==100
assert all(sha(S/k)==h for k,h in pm['sources'].items())
assert all(a['paths']==640 and a['backups']==22528 and a['paired_pulse_triples']==32 and a['raw_bytes']==g['raw_bytes'] and sha(S/f'SCALE_L{L}.npz')==a['array_sha256'] for L,a,g in zip(LS,aud,gen))
assert all(r['model']['files']==reg[0]['model']['files'] and r['cores']==reg[0]['cores'] for r in reg)
assert all(sha(F/'runtime_verified'/f'mqft-L{L}')==r['backend_sha'] and sha(F/'family'/f'L{L}'/'SOURCE_MANIFEST.json')==r['source_manifest_sha'] for L,r in zip(LS,reg))
assert all(read(S/f'PILOT_L{L}.json')['status']==read(S/f'FRESH_L{L}.json')['status']=='exact_science_fields_equal' and len(read(S/f'FRESH_L{L}.json')['cases'])==2 for L in LS)
assert len({r['model']['files']['mix9svqfreestyle_bsmix.bin.lz4'] for r in reg})==1
# Verify genuine independent budgets and cross-beta common random seed design.
for L in LS:
 jobs=read(S/'plans'/f'L{L}_base.json')['jobs'];assert len(jobs)==512
 by=collections.defaultdict(set)
 for j in jobs:by[(j['position_id'],j['replicate'],j['spec']['budget'])].add(j['spec']['seed'])
 assert len(by)==128 and all(len(s)==1 for s in by.values())
 for k,s in by.items():assert not s&by[(k[0],k[1],64 if k[2]==16 else 16)]
scopes={'L01':'Actual c_puct and beta grid, independent full B16/64, five common pretrained-evaluator sizes, pair order and fluctuation diagnostics; no critical point or exponent admitted.',
'L14':'Finite-budget covariance, martingale/drift decomposition and legal radial correlations with exact uniform null.',
'L17':'Actual finite selector-beta scan, conditional score curvature, dimensionless crossing diagnostics and uncertainty; no physical temperature or phase labels.',
'L28':'Original 3D Ising mismatch and held-size-out critical-type vs crossover/analytic prediction; no universal exponent.',
'Q02':'Native finite-size beta response, score curvature, Binder-type and mode ratios, source control; no inferred thermodynamic phase boundary.',
'Q16':'Independent-budget scaling and finite size covariance alternatives; exact multinomial null.',
'Q18':'Held-size-out predictions and sampled-grid peaks, not a critical exponent certification.',
'Q34':'Comprehensive scoped statistical report with analytic null, observed correlations, order and susceptibility diagnostics, paired source response and negative/limited critical evidence.'}
ids={i for f in pm['figures'] for i in f['ids']};assert ids==set(scopes)
catalog={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')}
prev=read(O/'PROGRESS75.json');inv=cr(O/'PLOT_INVENTORY75.csv');assert len(inv)==607 and sha(O/'PLOT_INVENTORY75.csv')==prev['inventory_sha256']
assert len(pm['figures'])==14 and not pm['animations']
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha'] and pdf.read_bytes()[:4]==b'%PDF'
 with Image.open(png) as im:im.verify()
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==621
cw(S/'PLOT_INVENTORY76.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS75.csv');assert len(rows)==121
ev=[]
for x in rows:
 k=x['id'];x['R76_evidence']=scopes.get(k,'');x['R76_scope_status']='SCOPED_INDEPENDENT_FINITE_SIZE_PANEL' if k in scopes else ''
 if k in scopes:
  y=catalog[k];z=[f['figure'] for f in pm['figures'] if k in f['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(z),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_INDEPENDENT_FINITE_SIZE_PANEL',scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE76.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS76.csv',rows)
prior={x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,76))}
union={x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,77))}
assert len(prior)==114 and len(union)==118 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
prev.update(figure_pairs=621,distinct_png_sha=621,files_paired=1242,current_pairs=14,prior_pairs=607,current_animations=0,animation_scope='No new animation; historical animations retained.',inventory_sha256=sha(S/'PLOT_INVENTORY76.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE76.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS76.csv'))
prev['stages'][S.name]=14;prev['original_output_evidence']=dict(current=len(ids),R55_R76_union=len(union),new_scoped_ids=sorted(union-prior),baseline_blocked_with_scoped_evidence=sorted(k for k in union if catalog[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='R55-R76 scoped evidence, not final completion.')
dump(S/'PROGRESS76.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for n in dag['nodes']:
 if n['id']=='REPLAN-04.C':n['restore76_followup']=dict(evidence='3200 fresh actual common MixNet size-family runs, five sizes, independent B16/64, four beta and c_puct/paired source controls; 640 root-group reductions; exact beta0 multinomial null and finite cohort control; Binder-type/mode ratio crossing bootstrap, held-size-out alternatives; original Ising mismatch.',remaining='No unique critical point, universality or physical correlation length admitted; broader size, budget-density and beta resolution, root ensemble and finite-size ansatz needed.',artifact=S.name+'/{THEORY.md,CRITICAL76.json,INDEPENDENT76.json,ORIGINAL_CLAIM_AUDIT76.json}')
 if n['id']=='REPLAN-02':n['restore76_followup']=dict(evidence='Finite conditional score curvature, occupancy entropy and fluctuation susceptibility measured across five sizes and independent budgets; selector beta not physical temperature.',remaining='Physical thermodynamic identification remains open.',artifact=S.name+'/CRITICAL76.json')
 if n['id']=='REPLAN-03':n['restore76_followup']=dict(evidence='Exact fixed-total covariance, martingale/drift/cross, paired actual final-wave source pulse with exact conditional response.',remaining='Population joint closure and autonomous generating function unresolved.',artifact=S.name+'/CRITICAL76.json')
dag.update(revision='RESTORE76_20261127',scope='Independent common-evaluator finite-size/budget statistical criticality diagnostics',previous_entry=dag['entry'],entry='REPLAN-04.C',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE77-AUTONOMOUS-RG-FIXED-POINT',mainline_priorities=['R77 augmented state and autonomous RG fixed point with actual path validation','R78 structural gauge/symmetry, R79 original quality report','Final scientific certification of all 121 original outputs and remaining physical identification'])
files=['SUMMARY76.json','PULSE_EXACT76.json','CRITICAL76.json','INDEPENDENT76.json','ORIGINAL_CLAIM_AUDIT76.json','ORIGINAL_SOURCE76.json','PLOT_MANIFEST76.json','PLOT_INVENTORY76.csv','ORIGINAL_OUTPUT_EVIDENCE76.csv','ORIGINAL_OUTPUT_PROGRESS76.csv','PROGRESS76.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']+[f'{k}_L{L}.json' for L in LS for k in ('REGISTRATION','GEN','AUDIT','PILOT','FRESH')]
rec=dict(status='scoped_independent_finite_size_and_null',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=q['paths'],cumulative_fixed_root_histories=79672,fresh_backups=q['backups'],fresh_candidate_samples=q['candidate_samples'],raw_bytes=q['raw_bytes'],root_groups=640,paired_pulse_triples=160,independent_budget_grid=[16,64],beta_grid=[0.,10/3,8.,16.],sizes=LS,synthetic_null_cohorts_per_design=2000,new_games=0,optimizer_updates=0,figures=14,animations=0,retained_pairs=621,original_evidence_items=len(ids),original_evidence_union_R55_R76=len(union),original_obligations=121,native_subagent='NOT_RUN',remaining='Physical criticality/temperature, controlled scaling limit, autonomous RG and final certification.')
rec.update({f+'_sha256':sha(S/f) for f in files});dag['restore76']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore76']=dict(rec,dag_sha256=sha(S/'DAG.json'),external_by_size={str(L):r['model'] for L,r in zip(LS,reg)},backends={str(L):r['backend_sha'] for L,r in zip(LS,reg)});dump(S/'PROVENANCE.json',prov)
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','plan76.py','generate76.py','audit76.py','critical76.py','fresh76.py','verify76.py','pulse76.py','plot76.py','summary76.py','publish76.py','backup76.py','execute76.sh','RESULTS_KO.md','STAGE_LEDGER.md']
raw=[x for a in aud for x in a['raw_receipts']];assert len(raw)==3200
inputs={str((F/'runtime_verified'/f'mqft-L{L}').relative_to(B)):r['backend_sha'] for L,r in zip(LS,reg)}
inputs.update({str((F/'family'/f'L{L}'/'SOURCE_MANIFEST.json').relative_to(B)):r['source_manifest_sha'] for L,r in zip(LS,reg)})
helper=H/'runtime/mqft-rng-replay';inputs[str(helper.relative_to(B))]=sha(helper)
manifest=dict(schema='mqft_restore76_scoped_run_v1',parent=parent,status='completed_scoped',command='bash execute76.sh (registered raw only); plan76.py and generate76.py using pinned size-specific environment',files={f:sha(S/f) for f in files+sources+['DAG.json','PROVENANCE.json']+[f'SCALE_L{L}.npz' for L in LS]},raw_receipts=raw,prior_raw_receipts=[],weights={},external=reg[0]['model'],plan_hashes={f'L{L}_{name}':x for L,r in zip(LS,reg) for name,x in r['plans'].items()},inputs=inputs,software={'python':sys.version,'numpy':np.__version__},limits='Same-agent separate implementation; full legacy regression not run. Finite geometry and sparse beta grid; no thermodynamic phase/exponent certification.')
dump(S/'RUN_MANIFEST76.json',manifest)
dst=P/'research/restore76';assert not dst.exists();dst.mkdir()
for f in files+sources+['DAG.json','PROVENANCE.json','RUN_MANIFEST76.json']:shutil.copy2(S/f,dst/f)
shutil.copytree(S/'plans',dst/'plans')
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS76.csv','ORIGINAL_OUTPUT_EVIDENCE76.csv','PLOT_INVENTORY76.csv','PROGRESS76.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE76_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE76_RESULTS.md')
git('add','research/restore76','docs');git('diff','--cached','--check');git('commit','-m','RESTORE76: independent five-size criticality and exact finite-budget null')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE76.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE76.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE76_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST76.json'),source_zip_sha256=sha(S/'MQFT_RESTORE76_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE76.git.bundle'),progress_sha256=sha(S/'PROGRESS76.json'),fresh_paths=3200,fresh_backups=q['backups'],figures=14,cumulative_retained_pairs=621,original_union=118,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
