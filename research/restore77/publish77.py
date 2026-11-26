from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore76_20261127'; A=B/'mqft_restore69_20261126'; P=B/'mqft_restore39_20261125/repo'; F=B/'mqft_restore49_20261126'; H=B/'mqft_restore46_20261126'
parent='a55241eae9df6a58e4433044106a08e33a45fba0'; LS=[13,15,17,19,21]
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain')
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A/f),f
old=read(O/'ORIGINAL_SOURCE76.json');src=read(S/'ORIGINAL_SOURCE77.json')
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert old['archives']==src['archives'] and all(sha(paths[k])==v for k,v in src['archives'].items())
q=read(S/'SUMMARY77.json');v=read(S/'INDEPENDENT77.json');pm=read(S/'PLOT_MANIFEST77.json')
aud=[read(S/f'AUDIT_L{L}.json') for L in LS];reg=[read(S/f'REGISTRATION_L{L}.json') for L in LS];gen=[read(S/f'GEN_L{L}.json') for L in LS]
assert q['fresh_paths']==q['train_paths']==960 and q['fresh_backups']==q['train_backups']==61440 and v['status']=='PASS_SCOPED' and v['models_checked']==300 and v['raw_paths']==60
assert all(sha(S/k)==h for k,h in pm['sources'].items())
assert all(a['fresh_paths']==a['train_paths']==192 and a['fresh_backups']==a['train_backups']==12288 and a['fresh_bytes']==g['raw_bytes'] and sha(S/f'PATHS_L{L}.npz')==a['array_sha256'] for L,a,g in zip(LS,aud,gen))
assert all(r['model']['files']==reg[0]['model']['files'] and r['cores']==reg[0]['cores'] for r in reg)
assert all(sha(F/'runtime_verified'/f'mqft-L{L}')==r['backend_sha'] and sha(F/'family'/f'L{L}'/'SOURCE_MANIFEST.json')==r['source_manifest_sha'] for L,r in zip(LS,reg))
assert all(read(S/f'PILOT_L{L}.json')['status']=='exact_science_fields_equal' for L in LS)
assert len({r['model']['files']['mix9svqfreestyle_bsmix.bin.lz4'] for r in reg})==1
for L in LS:
 jobs=read(S/'plans'/f'L{L}_base.json')['jobs'];assert len(jobs)==192
 oldjobs=read(O/'plans'/f'L{L}_base.json')['jobs']
 assert not ({j['id'] for j in jobs}&{j['id'] for j in oldjobs})
 assert not ({j['spec']['seed'] for j in jobs}&{j['spec']['seed'] for j in oldjobs})
scopes={'L13':'Actual count/mark/probability/lag projection collision witnesses and independent new geometry.', 'L29':'Specified finite donor kernel, direct vs composed transition laws and finite predictive errors; no native autonomous RG.', 'L35':'Exact conditional original coupling audit and finite temporal-kernel flow; native scale flow remains open.', 'L36':'Finite supported Markov kernel stationary projector and contraction; not a physical fixed point.', 'L37':'Original Wilson-Fisher candidate contradicts original beta function; actual frozen-tau unstable solution, exact probability RG, and fitted finite kernel stationary reference with native counterexamples.', 'L38':'Finite block cumulant and CGF deviations, separate geometry/clock mixing and mark dependence.', 'Q09':'Original beta functions checked algebraically, Gaussian singularity and finite frozen-tau eigenvalues.', 'Q11':'Explicit independent-sum RG and auxiliary matrix squaring; heldout native deviations retained.', 'Q13':'Scoped original fixed-point audit, exact block controls and finite model prediction, no physical universality.'}
ids={i for f in pm['figures'] for i in f['ids']};assert ids==set(scopes)
catalog={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')}
prev=read(O/'PROGRESS76.json');inv=cr(O/'PLOT_INVENTORY76.csv');assert len(inv)==621 and sha(O/'PLOT_INVENTORY76.csv')==prev['inventory_sha256']
assert len(pm['figures'])==10 and not pm['animations']
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha'] and pdf.read_bytes()[:4]==b'%PDF'
 with Image.open(png) as im:im.verify()
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==631
cw(S/'PLOT_INVENTORY77.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS76.csv');assert len(rows)==121
ev=[]
for x in rows:
 k=x['id'];x['R77_evidence']=scopes.get(k,'');x['R77_scope_status']='SCOPED_RG_FIXED_POINT_AND_CLOSURE' if k in scopes else ''
 if k in scopes:
  y=catalog[k];z=[f['figure'] for f in pm['figures'] if k in f['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(z),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_RG_FIXED_POINT_AND_CLOSURE',scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE77.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS77.csv',rows)
prior={x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,77))}
union={x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,78))}
assert len(prior)==118 and len(union)==119 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
prev.update(figure_pairs=631,distinct_png_sha=631,files_paired=1262,current_pairs=10,prior_pairs=621,current_animations=0,animation_scope='No new animation; historical animations retained.',inventory_sha256=sha(S/'PLOT_INVENTORY77.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE77.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS77.csv'))
prev['stages'][S.name]=10;prev['original_output_evidence']=dict(current=len(ids),R55_R77_union=len(union),new_scoped_ids=sorted(union-prior),baseline_blocked_with_scoped_evidence=sorted(k for k in union if catalog[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='R55-R77 scoped evidence, not final completion.')
dump(S/'PROGRESS77.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for n in dag['nodes']:
 if n['id']=='REPLAN-04.R':n['restore77_followup']=dict(evidence='Original proposed Wilson-Fisher point fails original beta_g; frozen-tau stationary solution unstable; exact IID probability RG and 150 auxiliary homogeneous kernel stationary projectors. 960 new common MixNet paths and 960 authenticated R76 training paths; 300 heldout candidate fits, fresh nonlumpability and CK defects; 112 R72 witnesses rechecked.',remaining='Exact autonomous closure fails for tested binned states. No native physical coupling RG, fixed point, exponent or universal scaling admitted; finite kernel is reference only.',artifact=S.name+'/{THEORY.md,ORIGINAL_RG_AUDIT77.json,CLOSURE77.json,RG77.json,INDEPENDENT77.json}')
 if n['id']=='X-MEMORY':n['restore77_followup']=dict(evidence='Independent count/mark/probability/lag state and homogeneous/clock-conditioned joint donor comparison, multiwave prediction, exact collision witnesses and dependent block law.',remaining='Learn continuous belief or finite-memory path law on fresh geometries; distinguish within-root from mixture effects.',artifact=S.name+'/SUMMARY77.json')
dag.update(revision='RESTORE77_20261127',scope='Original RG fixed-point algebra, auxiliary normalized probability/Markov scale maps and independent native closure tests',previous_entry=dag['entry'],entry='REPLAN-04.R',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE78-STRUCTURAL-GAUGE-SYMMETRY',mainline_priorities=['R78 structural gauge/symmetry (L22)','R79 original quality report (L31)','Final scientific certification of all 121 original outputs and remaining physical identification'])
files=['SUMMARY77.json','CLOSURE77.json','RG77.json','INDEPENDENT77.json','ORIGINAL_RG_AUDIT77.json','ORIGINAL_SOURCE77.json','PLOT_MANIFEST77.json','PLOT_INVENTORY77.csv','ORIGINAL_OUTPUT_EVIDENCE77.csv','ORIGINAL_OUTPUT_PROGRESS77.csv','PROGRESS77.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']+[f'{k}_L{L}.json' for L in LS for k in ('REGISTRATION','GEN','AUDIT','PILOT')]
rec=dict(status='scoped_rg_fixed_point_and_independent_closure',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=960,cumulative_fixed_root_histories=80632,fresh_backups=61440,prior_training_histories=960,prior_training_backups=61440,raw_bytes=q['fresh_bytes'],model_strata=30,model_combinations=300,homogeneous_stationary_projectors=150,sizes=LS,beta_grid=[0.,10/3,16.],new_games=0,optimizer_updates=0,figures=10,animations=0,retained_pairs=631,original_evidence_items=len(ids),original_evidence_union_R55_R77=len(union),original_obligations=121,native_subagent='NOT_RUN',remaining='Native autonomous RG, physical fixed-point identification, structural gauge, quality report and final scientific certification.')
rec.update({f+'_sha256':sha(S/f) for f in files});dag['restore77']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore77']=dict(rec,dag_sha256=sha(S/'DAG.json'),external_by_size={str(L):r['model'] for L,r in zip(LS,reg)},backends={str(L):r['backend_sha'] for L,r in zip(LS,reg)});dump(S/'PROVENANCE.json',prov)
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','plan77.py','generate77.py','extract77.py','closure77.py','rg77.py','original77.py','verify77.py','summary77.py','plot77.py','publish77.py','backup77.py','execute77.sh','RESULTS_KO.md','STAGE_LEDGER.md']
raw=[x for a in aud for x in a['raw_receipts'] if x['cohort']==1];prior_raw=[x for a in aud for x in a['raw_receipts'] if x['cohort']==0];assert len(raw)==len(prior_raw)==960
inputs={str((F/'runtime_verified'/f'mqft-L{L}').relative_to(B)):r['backend_sha'] for L,r in zip(LS,reg)}
inputs.update({str((F/'family'/f'L{L}'/'SOURCE_MANIFEST.json').relative_to(B)):r['source_manifest_sha'] for L,r in zip(LS,reg)})
helper=H/'runtime/mqft-rng-replay';inputs[str(helper.relative_to(B))]=sha(helper)
inputs.update(read(S/'RG77.json')['old_witnesses']['inputs'])
for L in LS:
 for f in (O/f'AUDIT_L{L}.json',O/f'REGISTRATION_L{L}.json',O/'plans'/f'L{L}_base.json'):
  inputs[str(f.relative_to(B))]=sha(f)
manifest=dict(schema='mqft_restore77_scoped_run_v1',parent=parent,status='completed_scoped',command='bash execute77.sh (registered raw only); plan77.py and generate77.py using pinned size-specific environment',files={f:sha(S/f) for f in files+sources+['DAG.json','PROVENANCE.json']+[f'PATHS_L{L}.npz' for L in LS]},raw_receipts=raw,prior_raw_receipts=prior_raw,weights={},external=reg[0]['model'],plan_hashes={f'L{L}_{name}':x for L,r in zip(LS,reg) for name,x in r['plans'].items()},inputs=inputs,software={'python':sys.version,'numpy':np.__version__},limits='Same-agent separate implementation; full legacy regression not run; finite donor model and independent-sum probability RG are not native physical scale flow.')
dump(S/'RUN_MANIFEST77.json',manifest)
dst=P/'research/restore77';assert not dst.exists();dst.mkdir()
for f in files+sources+['DAG.json','PROVENANCE.json','RUN_MANIFEST77.json']:shutil.copy2(S/f,dst/f)
shutil.copytree(S/'plans',dst/'plans')
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS77.csv','ORIGINAL_OUTPUT_EVIDENCE77.csv','PLOT_INVENTORY77.csv','PROGRESS77.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE77_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE77_RESULTS.md')
git('add','research/restore77','docs');git('diff','--cached','--check');git('commit','-m','RESTORE77: original RG fixed-point audit and independent native closure')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE77.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE77.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE77_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST77.json'),source_zip_sha256=sha(S/'MQFT_RESTORE77_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE77.git.bundle'),progress_sha256=sha(S/'PROGRESS77.json'),fresh_paths=960,fresh_backups=61440,figures=10,cumulative_retained_pairs=631,original_union=119,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
