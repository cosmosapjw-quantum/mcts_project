from base61 import *
import csv,shutil,subprocess,sys
from PIL import Image
O=B/'mqft_restore90_20261128';A69=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
def cr(p):
 with p.open() as f:return list(csv.DictReader(f))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
parent='846c3b91f6ba802249dbfabb69439c9e154c3576'
assert git('rev-parse','HEAD')==parent and not git('status','--porcelain') and git('ls-remote','origin','refs/heads/main').split()[0]==parent
r=read(S/'REGISTRATION.json');assert r['contract_sha']==sha(S/'THEORY_REGISTERED91.md') and r['parent_dag_sha']==sha(O/'DAG.json')
n=read(S/'NATIVE91.json');t=read(S/'THERMAL91.json');v=read(S/'INDEPENDENT91.json');pm=read(S/'PLOT_MANIFEST91.json');src=read(S/'ORIGINAL_SOURCE91.json')
assert n['status']==t['status']==v['status']=='PASS_SCOPED' and len(pm['figures'])==8
scopes={'G01':'Actual signed score energy at six cuts and actual game progression, no calibrated energy.','G02':'Budget proxy, conditional logit scale and endpoint KL projection distinguished, no physical T.','G04':'Fixed declared H Gibbs C/k_B and centered derivative under theta scan; no native heat capacity.','G05':'Native score and separately declared signed energy/entropy/work/heat ledger with residual.','Q05':'Exact nonnegative conditional entropy production and signed temporal increments for declared reference.','Q08':'Signed marginal entropy minus information change minus beta heat, independent repeated kernel; no native bath.','Q19':'Partial: actual historical F/R auxiliary MC and support-aware feedback IFT; source-specific full panel not complete.','Q20':'Partial: finite declared work mean and SD and MC estimates; full work distribution panels still open.'}
acc=['G01','G02','G04','G05','Q05','Q08'];partial=['Q19','Q20']
fmap={f['ids'][0]:f for f in pm['figures']};sm={f['id']:f for f in src['items']}
prev=read(O/'PROGRESS90.json');inv=cr(O/'PLOT_INVENTORY90.csv');assert len(inv)==685 and sha(O/'PLOT_INVENTORY90.csv')==prev['inventory_sha256']
for f in pm['figures']:
 p=S/'plots'/f['figure'];a=Path(str(p)+'.png');b=Path(str(p)+'.pdf');assert sha(a)==f['png_sha'] and sha(b)==f['pdf_sha']
 with Image.open(a) as im:im.verify()
 subprocess.run(['pdfinfo',str(b)],stdout=subprocess.DEVNULL,check=True)
 inv.append(dict(stage=S.name,plot='plots/'+f['figure'],png_sha=f['png_sha'],pdf_sha=f['pdf_sha'],png_bytes=a.stat().st_size,pdf_bytes=b.stat().st_size))
assert len(inv)==len({x['png_sha'] for x in inv})==693;cw(S/'PLOT_INVENTORY91.csv',inv)
c=read(O/'CONVERGENCE90.json');old=c['corrected_report_artifacts_accepted'];assert len(old)==16 and not set(acc)&set(old)
newclaim=dict(id='THERMO-SPLIT-06',stage=91,items=['G04','G05','Q05','Q08'],claim='For the specified Gibbs-start finite branch feedback and heat-bath refresh, beta <W>+I0 = Sigma_rel + D(r||p)+I1, with nonnegative conditional relaxation loss; signed step balance is marginal entropy increment minus mutual information increment minus beta heat. Independent measurement does not force zero final branch-state information.',status='ACCEPTED_DERIVED_AND_CHECKED',evidence=[S.name+'/'+p for p in ['THEORY.md','THERMAL91.json','INDEPENDENT91.json']],limit='Explicit declared finite reference; not native calibrated bath, physical entropy or controller reset cost. Same-agent algorithmic cross-check, 104 contexts.')
assert not any(x['id']==newclaim['id'] for x in c['claims']);c['claims'].append(newclaim)
items=[]
for k,s in scopes.items():
 f=fmap[k];items.append(dict(id=k,decision='ACCEPTED_CORRECTED_SCOPED_ARTIFACT' if k in acc else 'PARTIAL_CORRECTED_MODEL_ARTIFACT',original_source=sm[k],acceptance_contract=sm[k]['acceptance'],method=s,figures=[f['figure']],png_sha=f['png_sha'],pdf_sha=f['pdf_sha'],original_science='NOT_CERTIFIED',remaining_limit='Native physical identification remains open; declared model and actual native statistics separate.'))
dump(S/'ITEM_RECONSTRUCTION91.json',dict(status='SCOPED_CONVERGENCE',items=items,prior_report_artifacts=old,corrected_artifacts_total=22,original_scientific_certifications=0))
c.update(status='SCOPED_CONVERGENCE',accepted_subclaims=26,new_accepted_subclaims=1,new_corrected_artifacts=6,corrected_report_artifacts_accepted=old+acc,artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION91.json'),scope='Six corrected scoped artifacts; Q19/Q20 partial; six open science claims unchanged.')
assert sum(x['status'].startswith('OPEN') for x in c['claims'])==6 and sum(x['status'].startswith('ACCEPTED') for x in c['claims'])==26
dump(S/'CONVERGENCE91.json',c)
rows=cr(O/'ORIGINAL_OUTPUT_PROGRESS90.csv');cat={x['id']:x for x in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};ev=[]
for x in rows:
 k=x['id'];x['R91_evidence']=scopes.get(k,'');x['R91_scope_status']=next((i['decision'] for i in items if i['id']==k),'')
 if k in scopes:
  y=cat[k];ev.append(dict(id=k,primary_node=x['owner'],evidence_figures=fmap[k]['figure'],source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence=x['R91_scope_status'],scope=scopes[k]))
assert len(rows)==121 and all(x['final_completion']=='NOT_CERTIFIED' for x in rows);cw(S/'ORIGINAL_OUTPUT_PROGRESS91.csv',rows);cw(S/'ORIGINAL_OUTPUT_EVIDENCE91.csv',ev)
qm=cr(O/'QUALITY_MATRIX90.csv')
for x in qm:
 k=x['id'];x['R91_convergence_subclaims']=newclaim['id'] if k in newclaim['items'] else '';x['R91_open_subclaims']='';x['R91_scope']=scopes.get(k,'')
 if k in scopes:x['latest_stage']='91';x['latest_scope']=scopes[k];x['corrected_output_state']=next(i['decision'] for i in items if i['id']==k)+'_R91'
cw(S/'QUALITY_MATRIX91.csv',qm)
dump(S/'ITEM_CONVERGENCE91.json',dict(items=[dict(id=x['id'],accepted=x['R91_convergence_subclaims'],scope=x['R91_scope'],remaining_final_acceptance=x['final_acceptance']) for x in qm if x['R91_scope']],claims=[newclaim],parent_quality_matrix_sha256=sha(O/'QUALITY_MATRIX90.csv'),parent_convergence_sha256=sha(O/'CONVERGENCE90.json'),artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION91.json')))
prev.update(figure_pairs=693,distinct_png_sha=693,files_paired=1386,current_pairs=8,prior_pairs=685,current_animations=0,inventory_sha256=sha(S/'PLOT_INVENTORY91.csv'),original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE91.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS91.csv'),quality_matrix_sha256=sha(S/'QUALITY_MATRIX91.csv'),corrected_report_artifact_accepted=old+acc,accepted_corrected_subclaims=26,final_scientific_certified=0)
prev['stages'][S.name]=8;prev['original_output_evidence'].update(current=8,R55_R91_union=121,scope='121 scoped coverage; 22 corrected artifact acceptances,26 scoped claims; Q19/Q20 partial.')
dump(S/'PROGRESS91.json',prev)
(S/'ORIGINAL_OUTPUT_REPORT91.md').write_text('# RESTORE91 original output crosswalk\n\n'+ '\n'.join(f"- {k}: {scopes[k]} ({fmap[k]['figure']})" for k in scopes)+'\n\nFull scientific certification remains 0/121.\n')
dag=read(O/'DAG.json');assert len(dag['nodes'])==len(dag['edges'])==13
for node in dag['nodes']:
 if node['id'] in ('REPLAN-02','REPLAN-06'):
  node['restore91_followup']=dict(evidence='960 raw histories, 5760 cuts, 832 declared feedback cases, 3328 signed steps',accepted_subclaims=[newclaim['id']],corrected_artifacts=acc,partial=partial,remaining='Native physical temperature/heat/reverse not identified.',artifact=S.name+'/{ORIGINAL_OUTPUT_REPORT91.md,ITEM_RECONSTRUCTION91.json,CONVERGENCE91.json}')
dag.update(revision='RESTORE91_20261128',scope='Convergence: corrected statistical mechanics and declared thermal entropy outputs',previous_entry=dag['entry'],entry='REPLAN-06',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE92-FINITE-SIZE-ORDER-PARAMETER-OUTPUTS',mainline_priorities=['R92 G06/G08/G09 and Q02/Q29/Q31 corrected finite-size/order parameter outputs','Direct original reconstruction; no repetitive diagnostic sweep','Six open science claims, Q19/Q20 partial'],research_phase='CONVERGENCE',quality_matrix='QUALITY_MATRIX91.csv',convergence_ledger='CONVERGENCE91.json')
rec=dict(status='six_corrected_original_artifacts_one_new_scoped_claim_two_partial',parent_commit=parent,parent_dag_sha256=sha(O/'DAG.json'),fresh_fixed_root_histories=0,cumulative_fixed_root_histories=93336,fresh_backups=0,reused_fixed_root_paths=960,branch_source_receipts=2448,figures=8,animations=0,retained_pairs=693,original_evidence_items=8,original_evidence_union_R55_R91=121,original_obligations=121,corrected_report_artifact_accepted=22,new_corrected_artifacts=6,accepted_corrected_subclaims=26,new_accepted_subclaims=1,final_scientific_certified=0,native_subagent='NOT_RUN',remaining='99 corrected artifact obligations; six open science claims, Q19/Q20 partial; G06 next.')
for f in ['REGISTRATION.json','THEORY_REGISTERED91.md','NATIVE91.json','THERMAL91.json','INDEPENDENT91.json','ORIGINAL_SOURCE91.json','CONVERGENCE91.json','ITEM_RECONSTRUCTION91.json','ORIGINAL_OUTPUT_REPORT91.md','ITEM_CONVERGENCE91.json','QUALITY_MATRIX91.csv','PLOT_MANIFEST91.json','PLOT_INVENTORY91.csv','ORIGINAL_OUTPUT_EVIDENCE91.csv','ORIGINAL_OUTPUT_PROGRESS91.csv','PROGRESS91.json','THEORY.md','DRIFT_WATCHDOG.md','NEXT_DYNAMICS.md']:rec[f+'_sha256']=sha(S/f)
dag['restore91']=rec;dump(S/'DAG.json',dag)
prov=read(O/'PROVENANCE.json');prov['restore91']=dict(rec,dag_sha256=sha(S/'DAG.json'),input_hashes=v['source_hashes']);dump(S/'PROVENANCE.json',prov)
# Capture source + output identities; 2D plot files and logs included, skip ephemeral pycache.
files=[str(x.relative_to(S)) for x in S.rglob('*') if x.is_file() and '__pycache__' not in x.parts and not x.name.endswith('.part') and x.name not in ('publish91.log','backup91.log','RUN_MANIFEST91.json')]
a79=read(B/'mqft_restore79_20261127'/'AUDIT79.json');b85=read(B/'mqft_restore85_20261127'/'BRANCH_ENERGY85.json')
raw={x['path']:x['sha256'] for x in a79['raw_receipts']+b85['raw_receipts']};assert len(raw)<=3408
manifest=dict(schema='mqft_restore91_convergence_run_v1',parent=parent,status='completed_scoped',command='bash execute91.sh',files={f:sha(S/f) for f in sorted(files)},inputs=v['source_hashes'],prior_raw_receipts=raw,software=v['software'],limits='Same-agent independent methods; no native physical T/heat; historical R61 reduction reauthenticated only.',backup_policy='Dropbox only')
dump(S/'RUN_MANIFEST91.json',manifest)
dst=P/'research/restore91';assert not dst.exists();dst.mkdir()
for f in files+['RUN_MANIFEST91.json']:
 if f=='CURRENT_RESEARCH_PLAN_before.md':continue
 p=dst/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/f,p)
for f in ['DAG.json','PROVENANCE.json','NEXT_DYNAMICS.md','DRIFT_WATCHDOG.md','STAGE_LEDGER.md','ORIGINAL_OUTPUT_PROGRESS91.csv','ORIGINAL_OUTPUT_EVIDENCE91.csv','PLOT_INVENTORY91.csv','PROGRESS91.json','QUALITY_MATRIX91.csv','CONVERGENCE91.json','ITEM_RECONSTRUCTION91.json']:shutil.copy2(S/f,P/'docs'/f)
shutil.copy2(S/'THEORY.md',P/'docs/RESTORE91_THEORY.md');shutil.copy2(S/'RESULTS_KO.md',P/'docs/RESTORE91_RESULTS.md')
git('add','research/restore91','docs')
git('diff','--cached','--check','--','.',':(exclude)research/restore91/logs')
git('commit','-m','RESTORE91: reconstruct statistical mechanics and signed feedback entropy outputs')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
# Retain full source + Git bundle in Dropbox delta, no lasting duplicate outside it.
git('bundle','create',str(S/'MQFT_RESTORE91.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE91.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE91_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST91.json'),source_zip_sha256=sha(S/'MQFT_RESTORE91_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE91.git.bundle'),progress_sha256=sha(S/'PROGRESS91.json'),fresh_paths=0,fresh_backups=0,figures=8,cumulative_retained_pairs=693,original_union=121,accepted_corrected_subclaims=26,corrected_report_artifact_accepted=22,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
