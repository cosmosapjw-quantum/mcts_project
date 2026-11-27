from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore77_20261127';A=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo';R75=B/'mqft_restore75_20261127'
parent='888da8f4bbeb0b4aa07b3c4b6124bfec8caaa2ff'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A/f),f
old=read(O/'ORIGINAL_SOURCE77.json');src=read(S/'ORIGINAL_SOURCE78.json')
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert old['archives']==src['archives'] and all(sha(paths[k])==v for k,v in src['archives'].items())
a=read(S/'AUDIT78.json');n=read(S/'NATIVE78.json');q=read(S/'GAUGE78.json');v=read(S/'INDEPENDENT78.json');pm=read(S/'PLOT_MANIFEST78.json');reg=read(S/'REGISTRATION.json')
assert a['paths']==1536 and a['backups']==196608 and len(n['contexts'])==len(q['contexts'])==48 and len(n['comparisons'])==1344 and v['status']=='PASS_SCOPED' and v['raw_paths']==1536 and v['graph_contexts']==48
assert q['source_sha']==sha(S/'NATIVE78.json') and all(sha(S/k)==h for k,h in pm['sources'].items())
r75=read(R75/'REGISTRATION.json');assert reg['external']==r75['external'] and reg['weights']==r75['weights']
assert sha(S/'mqft_pulse74')==reg['pulse_binary_sha']==r75['pulse_binary_sha'] and sha(S/'mqft_train')==reg['binary_sha']==r75['binary_sha'] and sha(S/'pulse74.patch')==reg['pulse_patch_sha']
assert all(sha(B/x['path'])==x['sha256'] for x in reg['weights'].values()) and all(sha(P/'.external/rapfi'/k)==h for k,h in reg['external']['files'].items())
raw=a['raw_receipts'];assert len(raw)==len({r['path'] for r in raw})==1536
paths_by_id={Path(r['path']).name[:-8]:r for r in raw};assert len(paths_by_id)==1536
for tag,r in reg['plans'].items():
 p=S/'plans'/f'{tag}.json';assert sha(p)==r['sha256'];js=read(p);assert len(js)==r['n']==512
 for j in js:assert paths_by_id[j['id']]['seed']==j['seed']
for x in read(S/'RUN_SHARDS.json'):assert sha(S/'plans'/f"{x['shard']}.json")==x['plan_sha256']
for x in read(S/'RAPFI_RECEIPTS78.json'):assert paths_by_id[x['job']['id']]['sha256']==x['sha256']
assert sum(x['cycles'] for x in q['contexts'])==4005 and sum(x['quotient_cycles'] for x in q['contexts'])==19
scopes={'L16':'Actual D4 orientation-dependent native path/cut statistics, exact group-averaged prior control and signed mark maps; not original full quantum path integral.', 'L21':'Actual conditioned finite U1 and SU2 connection with true unitary links, loop and coherent propagation/readout controls; declared model not physical native gauge.', 'L22':'Original gauge-policy three-panel intent reconstructed: normalized Wilson loop distribution, gauge-covariant density-field policy stability, actual graph topology; original malformed links and virtual cycles audited; no native gauge identification.', 'L24':'Actual prefix tree, visited state quotient and legal-grid graph cycle counts; no ungrounded native physical topology claim.'}
ids={i for f in pm['figures'] for i in f['ids']};assert ids==set(scopes)
cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')}
prev=read(O/'PROGRESS77.json');inv=cr(O/'PLOT_INVENTORY77.csv');assert len(inv)==631 and sha(O/'PLOT_INVENTORY77.csv')==prev['inventory_sha256']
assert len(pm['figures'])==9 and not pm['animations'];plotfiles=[]
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha'] and pdf.read_bytes()[:4]==b'%PDF'
 with Image.open(png) as im:im.verify()
 subprocess.run(['pdfinfo',str(pdf)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
 plotfiles.extend(str(f.relative_to(S)) for f in (png,pdf))
assert len(inv)==len({x['png_sha'] for x in inv})==640
cw(S/'PLOT_INVENTORY78.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS77.csv');assert len(rows)==121;ev=[]
for x in rows:
 k=x['id'];x['R78_evidence']=scopes.get(k,'');x['R78_scope_status']='SCOPED_STRUCTURAL_GAUGE_PANEL' if k in scopes else ''
 if k in scopes:
  y=cat[k];z=[f['figure'] for f in pm['figures'] if k in f['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(z),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_STRUCTURAL_GAUGE_PANEL',scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE78.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS78.csv',rows)
prior={x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,78))}
union={x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,79))}
assert len(prior)==119 and len(union)==120 and union-prior=={'L22'} and set(cat)-union=={'L31'} and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
prev.update(figure_pairs=640,distinct_png_sha=640,files_paired=1280,current_pairs=9,prior_pairs=631,current_animations=0,animation_scope='No new animation; historical animations retained.',inventory_sha256=sha(S/'PLOT_INVENTORY78.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE78.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS78.csv'))
prev['stages'][S.name]=9;prev['original_output_evidence']=dict(current=len(ids),R55_R78_union=len(union),new_scoped_ids=sorted(union-prior),baseline_blocked_with_scoped_evidence=sorted(k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='R55-R78 scoped evidence, not final completion.')
dump(S/'PROGRESS78.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for node in dag['nodes']:
 if node['id']=='X-STRUCT':node['restore78_followup']=dict(evidence='1536 actual raw orientation paths; separate D4 residual and exact group average; 48 actual prefix-tree/quotient/legal graphs, 4005 fundamental loops, explicit U1 and SU2 unitary links, covariant density-field update, independent numerical checks; original gauge producer defects and L22 triptych reconstructed.',remaining='Physical native local gauge/phase and operational phase-sensitive intervention not identified; assigned connection and model policy not production native gauge; no final certification.',artifact=S.name+'/{THEORY.md,NATIVE78.json,GAUGE78.json,ORIGINAL_GAUGE_AUDIT78.json,INDEPENDENT78.json,plots/L22_gauge_policy_analysis.pdf}')
 if node['id']=='X-REP':node['restore78_followup']=dict(evidence='Unitary U1/SU2 connections, loop holonomy covariance, true noncommuting based loops, gauge-covariant density update and phase-sensitive fixed-probe controls.',remaining='Native operational phase identification and physical gauge dynamics.',artifact=S.name+'/GAUGE78.json')
dag.update(revision='RESTORE78_20261127',scope='Native D4 orientation audit, actual graph topology and declared U1/SU2 covariant policy model',previous_entry=dag['entry'],entry='X-STRUCT',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE79-ORIGINAL-QUALITY-REPORT',mainline_priorities=['R79 original honest summary quality report L31, all 121 typed acceptance requirements','Final scientific certification and corrected original outputs; physical gauge, thermal and RG identification remain'])
files=['REGISTRATION.json','REGISTRATION_before_serialization_fix.json','ROOT_PREPARATION.json','RUN_SHARDS.json','RAPFI_RECEIPTS78.json','AUDIT78.json','NATIVE78.json','GAUGE78.json','INDEPENDENT78.json','ORIGINAL_SOURCE78.json','ORIGINAL_GAUGE_AUDIT78.json','PLOT_MANIFEST78.json','PLOT_INVENTORY78.csv','ORIGINAL_OUTPUT_EVIDENCE78.csv','ORIGINAL_OUTPUT_PROGRESS78.csv','PROGRESS78.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']
rec=dict(status='scoped_native_symmetry_declared_gauge_policy',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=1536,cumulative_fixed_root_histories=82168,fresh_backups=196608,raw_bytes=a['raw_bytes'],native_orientation_comparisons=1344,graph_contexts=48,fundamental_cycles=4005,quotient_cycles=19,new_games=0,optimizer_updates=0,figures=9,animations=0,retained_pairs=640,original_evidence_items=len(ids),original_evidence_union_R55_R78=len(union),original_obligations=121,native_subagent='NOT_RUN',remaining='Native physical gauge/phase, structural and physical identification, L31 honest quality report and final individual scientific certification.')
rec.update({f+'_sha256':sha(S/f) for f in files});dag['restore78']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore78']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=reg['external'],weights=reg['weights'],pulse_binary_sha=reg['pulse_binary_sha']);dump(S/'PROVENANCE.json',prov)
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','arena68.py','audit_native74.py','audit_rapfi74.py','pulse74.patch','plan78.py','generate78.py','native78.py','gauge78.py','verify78.py','original78.py','plot78.py','publish78.py','backup78.py','execute78.sh','RESULTS_KO.md','STAGE_LEDGER.md']
inputs={str((R75/'REGISTRATION.json').relative_to(B)):sha(R75/'REGISTRATION.json')}
manifest=dict(schema='mqft_restore78_scoped_run_v1',parent=parent,status='completed_scoped',command='bash execute78.sh (registered raw paths); plan78.py then generate78.py with pinned binaries',files={f:sha(S/f) for f in files+sources+plotfiles+['DAG.json','PROVENANCE.json','mqft_train','mqft_pulse74']},raw_receipts=raw,prior_raw_receipts=[],weights=reg['weights'],external=reg['external'],plan_hashes=reg['plans'],inputs=inputs,software={'python':sys.version,'numpy':np.__version__},limits='Same-agent independent numerical implementation. No full regression or independent seed-to-stream replay. Local gauge model not native physical gauge. Original attachment receipt inherited.')
dump(S/'RUN_MANIFEST78.json',manifest)
dst=P/'research/restore78';assert not dst.exists();dst.mkdir()
for f in files+sources+plotfiles+['DAG.json','PROVENANCE.json','RUN_MANIFEST78.json']:
 z=dst/f;z.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,z)
shutil.copytree(S/'plans',dst/'plans')
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS78.csv','ORIGINAL_OUTPUT_EVIDENCE78.csv','PLOT_INVENTORY78.csv','PROGRESS78.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE78_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE78_RESULTS.md')
git('add','research/restore78','docs');git('diff','--cached','--check','--','.',':(exclude)research/restore78/pulse74.patch');git('commit','-m','RESTORE78: native D4 and covariant finite gauge policy')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE78.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE78.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE78_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST78.json'),source_zip_sha256=sha(S/'MQFT_RESTORE78_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE78.git.bundle'),progress_sha256=sha(S/'PROGRESS78.json'),fresh_paths=1536,fresh_backups=196608,figures=9,cumulative_retained_pairs=640,original_union=120,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
