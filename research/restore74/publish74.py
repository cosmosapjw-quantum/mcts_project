from base61 import *
import csv,shutil,subprocess,sys,zipfile
from PIL import Image
O=B/'mqft_restore73_20261126';A=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
parent='d929a233b26f45669242cf5ee10d8c279203692b'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain')
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:
 assert sha(S/f)==sha(O/f)==sha(A/f),f
original=read(O/'ORIGINAL_SOURCE73.json')
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert all(sha(paths[k])==h for k,h in original['archives'].items())
a=read(S/'AUDIT74.json');r=read(S/'RESPONSE74.json');q=read(S/'PETZ74.json');ind=read(S/'INDEPENDENT74.json');pm=read(S/'PLOT_MANIFEST74.json');reg=read(S/'REGISTRATION.json')
assert a['paths']==2304 and a['backups']==294912 and a['matched_prepulse_triples']==768 and q['branch_hashes_checked']==4544 and q['blocks']==323904 and len(q['contexts'])==64 and ind['status']=='PASS_SCOPED' and ind['reference_cases']==192
assert all(sha(S/k)==h for k,h in pm['sources'].items())
assert sha(S/'PULSE74.npz')==a['array_sha256']==r['array_sha256']
assert all(sha(B/k)==h for k,h in q['input_hashes'].items())
assert sha(S/'mqft_pulse74')==reg['pulse_binary_sha'] and sha(S/'pulse74.patch')==reg['pulse_patch_sha'] and sha(S/'mqft_train')==reg['binary_sha']
assert all(sha(B/x['path'])==x['sha256'] for x in reg['weights'].values())
assert all(t['quantum_rel_loss']>0 and abs(t['TPM_IFT']-1)<1e-10 for x in q['contexts'] for t in x['reference'])
scopes={
 'G04':'Dimensionless conditional score Gibbs capacity and explicitly declared qubit capacity; no physical heat capacity.',
 'G05':'Scoped native score ledger, quantum reference energy/entropy summary; no calibrated thermodynamics.',
 'G10':'Time-resolved causal score-response identity and direct finite pulse checks, not equilibrium FDT.',
 'G11':'Real pulse source response curves, matched blocks, predictable covariance correction; not physical FDT.',
 'G12':'Scoped direct versus reweighted and score response summary; no universal FDT.',
 'G15':'Exact conditional score-law participation along native histories; not a thermodynamic phase transition.',
 'L18':'Native likelihood response, reference Petz entropy loss and switched-H ledger; no physical flux-force identification.',
 'Q05':'Reference-specific quantum relative entropy loss, TPM fluctuation identity and reverse audit; not physical entropy flux.',
 'Q06':'Actual-conditioned six-stage switched-H animation and identity-free control; state not closed cycle, no Otto/Carnot claim.',
 'Q07':'Scoped comprehensive response, reverse, capacity, work ledger and explicit limitations; original report not final.',
 'Q08':'Exact conditional model energy balance and reference relative-entropy loss, no unsupported absolute entropy derivative.',
 'Q38':'Quantum relative entropy and conditional entropies, declared reference heat capacity, no measured bath or physical temperature.'
}
ids={i for f in pm['figures']+pm['animations'] for i in f['ids']};assert ids==set(scopes)
catalog={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')}
members={};guides=[]
for i in sorted(ids):
 x=catalog[i];archive=x['archive'];member=x['member']
 if archive=='guide_only':
  guides.append(dict(id=i,member=member,authority='inherited original attachment receipt and R69 catalog, not local source member bytes'))
  continue
 with zipfile.ZipFile(paths[archive]) as z:b=z.read(member)
 h=hashlib.sha256(b).hexdigest();assert h==x['member_sha256']
 members[archive,member]=dict(archive=archive,member=member,sha256=h)
with zipfile.ZipFile(paths['mcts_study.zip']) as z:
 for member in z.namelist():
  if member.endswith('.md') and 'thermodynamic-mcts' in member:
   b=z.read(member);members['study',member]=dict(archive='mcts_study.zip',member=member,sha256=hashlib.sha256(b).hexdigest())
dump(S/'ORIGINAL_SOURCE74.json',dict(archives=original['archives'],verified_members=list(members.values()),guide_only=guides,original_attachment_receipt=original['original_attachment_receipt'],scope='Three original local archive hashes and relevant actual source member hashes. Guide-only labels inherited, no nonexistent member bytes claimed.'))
prev=read(O/'PROGRESS73.json');inv=cr(O/'PLOT_INVENTORY73.csv');assert len(inv)==584 and sha(O/'PLOT_INVENTORY73.csv')==prev['inventory_sha256']
assert len(pm['figures'])==11 and len(pm['animations'])==1
for x in pm['figures']:
 base=S/'plots'/x['figure'];png=Path(str(base)+'.png');pdf=Path(str(base)+'.pdf')
 assert sha(png)==x['png_sha'] and sha(pdf)==x['pdf_sha'] and pdf.read_bytes()[:4]==b'%PDF'
 with Image.open(png) as im:im.verify()
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=png.stat().st_size,pdf_bytes=pdf.stat().st_size))
for x in pm['animations']:
 p=S/'plots'/x['file'];assert sha(p)==x['sha256']
 with Image.open(p) as im:assert im.n_frames==x['frames']
assert len(inv)==len({x['png_sha'] for x in inv})==595
cw(S/'PLOT_INVENTORY74.csv',inv)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS73.csv');assert len(rows)==121
ev=[]
for x in rows:
 k=x['id'];x['R74_evidence']=scopes.get(k,'');x['R74_scope_status']='SCOPED_RESPONSE_REVERSE_PANEL' if k in scopes else ''
 if k in scopes:
  y=catalog[k];z=[f.get('figure',f.get('file')) for f in pm['figures']+pm['animations'] if k in f['ids']]
  ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(z),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_RESPONSE_REVERSE_PANEL',scope=scopes[k]))
cw(S/'ORIGINAL_OUTPUT_EVIDENCE74.csv',ev);cw(S/'ORIGINAL_OUTPUT_PROGRESS74.csv',rows)
prior={x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,74))}
union={x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,75))}
assert len(prior)==104 and len(union)==112 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows)
prev.update(figure_pairs=595,distinct_png_sha=595,files_paired=1190,current_pairs=11,prior_pairs=584,current_animations=1,animation_scope='One new six-frame explicit model switched-H protocol; not physical cycle.',inventory_sha256=sha(S/'PLOT_INVENTORY74.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE74.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS74.csv'))
prev['stages'][S.name]=11;prev['original_output_evidence']=dict(current=len(ids),R55_R74_union=len(union),new_scoped_ids=sorted(union-prior),baseline_blocked_with_scoped_evidence=sorted(k for k in union if catalog[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),final_certified=0,scope='R55-R74 scoped evidence, not final completion.')
dump(S/'PROGRESS74.json',prev)
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for n in dag['nodes']:
 if n['id']=='REPLAN-06':n['restore74_followup']=dict(evidence='2304 fresh matched native source pulse paths; causal score response and predictable correction; 64 actual fine two-time channel contexts, Petz 192 reference cases and exact measured IFT; switched Hamiltonian first-law ledger and declared capacity.',remaining='Physical Hamiltonian/bath/time calibration and microscopic reverse protocol, thermodynamic cycle, physical heat and FDT not identified.',artifact=S.name+'/{THEORY.md,RESPONSE74.json,PETZ74.json}')
 if n['id']=='REPLAN-02':n['restore74_followup']=dict(evidence='Conditional Gibbs score capacity, entropy, participation and exact score-energy ledger across three evaluator families.',remaining='Physical temperature, heat capacity, criticality not inferred.',artifact=S.name+'/RESPONSE74.json')
 if n['id']=='X-REP':n['restore74_followup']=dict(evidence='Full-rank reference Petz recovery, separate measured TPM fluctuation relation and independently verified finite complex matrix functions.',remaining='Reference dependence, no unique physical time reversal, operational coherent multi-time channel.',artifact=S.name+'/INDEPENDENT74.json')
dag.update(revision='RESTORE74_20261126',scope='Actual causal native pulse response and explicit reference-dependent quantum reverse/energy model',previous_entry=dag['entry'],entry='REPLAN-06',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE75-BARRIER-FIRST-PASSAGE-WKB',mainline_priorities=['R75 actual native barrier/first passage and explicit coherent transport with controlled WKB','Independent size criticality, population and gauge/fixed-point checks','Operational coherent multi-time and final-decision redundancy; physically calibrated thermal bath only if identifiable'])
files=['REGISTRATION.json','PULSE_PATCH74.json','AUDIT74.json','RESPONSE74.json','PETZ74.json','INDEPENDENT74.json','PLOT_MANIFEST74.json','PLOT_INVENTORY74.csv','ORIGINAL_OUTPUT_EVIDENCE74.csv','ORIGINAL_OUTPUT_PROGRESS74.csv','ORIGINAL_SOURCE74.json','PROGRESS74.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']
rec=dict(status='scoped_native_response_reference_reverse_model',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=2304,cumulative_fixed_root_histories=73400,fresh_backups=294912,matched_prepulse_triples=768,reused_R73_branch_paths=4544,reused_R73_branch_backups=145408,conditional_contexts=64,fine_outcome_blocks=323904,reference_cases=192,raw_bytes=a['raw_bytes'],new_games=0,optimizer_updates=0,figures=11,animations=1,retained_pairs=595,original_evidence_items=len(ids),original_evidence_union_R55_R74=len(union),additional_plot_nodes=0,original_obligations=121,native_subagent='NOT_RUN',remaining='physical bath and energy identification, operational coherence, actual microscopic reverse, critical scaling and remaining original targets')
rec.update({f+'_sha256':sha(S/f) for f in files});dag['restore74']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore74']=dict(rec,dag_sha256=sha(S/'DAG.json'),external=reg['external'],pulse_binary_sha=reg['pulse_binary_sha'],original_binary_sha=reg['binary_sha']);dump(S/'PROVENANCE.json',prov)
(S/'execute74.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd "$(dirname "$0")"\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\n# Reanalyze registered immutable raw histories; fresh generation separate.\npython3 response74.py\npython3 petz74.py\npython3 verify74.py\npython3 plot74.py\n')
shutil.copy2(B/'CURRENT_RESEARCH_PLAN.md',S/'CURRENT_RESEARCH_PLAN_before.md')
sources=['base61.py','arena68.py','audit65.py','audit66.py','audit68.py','channel64.py','audit_native74.py','audit_rapfi74.py','build74.py','pulse74.patch','plan74.py','generate74.py','response74.py','petz74.py','verify74.py','plot74.py','publish74.py','backup74.py','execute74.sh','ROOT_PREPARATION.json','RUN_SHARDS.json','RAPFI_RECEIPTS74.json','RESULTS_KO.md','STAGE_LEDGER.md']
prior_raw=[x for c in read(O/'INSTRUMENT73.json')['contexts'] for x in c['raw_receipts']]
assert len(prior_raw)==4544
manifest=dict(schema='mqft_restore74_scoped_run_v1',parent=parent,status='completed_scoped',command='bash execute74.sh (registered raw data); fresh build74.py, plan74.py, generate74.py',files={f:sha(S/f) for f in files+sources+['DAG.json','PROVENANCE.json','PULSE74.npz','mqft_train','mqft_pulse74']},raw_receipts=a['raw_receipts'],prior_raw_receipts=prior_raw,weights=reg['weights'],external=reg['external'],plan_hashes=reg['plans'],inputs=q['input_hashes'],software={'python':sys.version,'numpy':np.__version__},limits='Same-agent separate implementation. No full native regression. Guide-only labels inherited. Native pulse not equilibrium physical FDT; Petz reverse is reference-specific and not inverse tree.')
dump(S/'RUN_MANIFEST74.json',manifest)
dst=P/'research/restore74';assert not dst.exists();dst.mkdir()
for f in files+sources+['DAG.json','PROVENANCE.json','RUN_MANIFEST74.json']:
 p=dst/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,p)
shutil.copytree(S/'plans',dst/'plans')
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS74.csv','ORIGINAL_OUTPUT_EVIDENCE74.csv','PLOT_INVENTORY74.csv','PROGRESS74.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE74_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE74_RESULTS.md')
git('apply','--check',str(S/'pulse74.patch'));git('add','research/restore74','docs');git('diff','--cached','--check','--','.',':(exclude)research/restore74/pulse74.patch');git('commit','-m','RESTORE74: native pulse response and reference Petz reverse')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain')
print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE74.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE74.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE74_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST74.json'),source_zip_sha256=sha(S/'MQFT_RESTORE74_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE74.git.bundle'),progress_sha256=sha(S/'PROGRESS74.json'),fresh_paths=2304,fresh_backups=294912,reused_branch_paths=4544,figures=11,animations=1,cumulative_retained_pairs=595,original_union=112,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
