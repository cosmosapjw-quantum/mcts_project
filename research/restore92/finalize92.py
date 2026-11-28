from base61 import *
import subprocess,shutil
P=B/'mqft_restore39_20261125/repo';parent='e35626561951a04494ea73945d07db8df1affae7'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and git('ls-remote','origin','refs/heads/main').split()[0]==parent
dst=P/'research/restore92'
for f in ['CORE_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md']:
 assert sha(S/f)==sha(B/'mqft_restore69_20261126'/f)==sha(dst/f)
# Preserve byte-locked pre-run contract; its pre-existing whitespace is not a scientific change.
f='THEORY_REGISTERED92.md';assert sha(S/f)==sha(dst/f)==read(S/'REGISTRATION.json')['contract_sha']
git('add','research/restore92','docs')
git('diff','--cached','--check','--','.',':(exclude)research/restore92/logs',':(exclude)research/restore92/CORE_POLICY.md',':(exclude)research/restore92/QUANTUM_REPRESENTATION_POLICY.md',':(exclude)research/restore92/THEORY_REGISTERED92.md')
git('commit','-m','RESTORE92: reconstruct finite-size concentration and signed spatial-temporal covariance')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE92.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE92.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE92_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST92.json'),source_zip_sha256=sha(S/'MQFT_RESTORE92_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE92.git.bundle'),progress_sha256=sha(S/'PROGRESS92.json'),fresh_paths=0,fresh_backups=0,figures=7,cumulative_retained_pairs=700,original_union=121,accepted_corrected_subclaims=28,corrected_report_artifact_accepted=26,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
