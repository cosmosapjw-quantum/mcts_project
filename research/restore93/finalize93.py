from base61 import *
import subprocess,shutil
P=B/'mqft_restore39_20261125/repo';parent='c51e57021376661a2c8c042058f14778d86727d4'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and git('ls-remote','origin','refs/heads/main').split()[0]==parent
dst=P/'research/restore93'
for f in ['CORE_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md']:
 assert sha(S/f)==sha(B/'mqft_restore69_20261126'/f)==sha(dst/f)
assert sha(S/'THEORY_REGISTERED93.md')==read(S/'REGISTRATION.json')['contract_sha']
git('add','research/restore93','docs')
git('diff','--cached','--check','--','.',':(exclude)research/restore93/logs',':(exclude)research/restore93/CORE_POLICY.md',':(exclude)research/restore93/QUANTUM_REPRESENTATION_POLICY.md')
git('commit','-m','RESTORE93: reconstruct causal response and finite FDT diagnostics')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE93.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE93.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE93_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST93.json'),source_zip_sha256=sha(S/'MQFT_RESTORE93_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE93.git.bundle'),progress_sha256=sha(S/'PROGRESS93.json'),fresh_paths=0,fresh_backups=0,figures=5,cumulative_retained_pairs=705,original_union=121,accepted_corrected_subclaims=30,corrected_report_artifact_accepted=29,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
