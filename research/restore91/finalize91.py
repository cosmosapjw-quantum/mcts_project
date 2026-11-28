from base61 import *
import subprocess
P=B/'mqft_restore39_20261125/repo';parent='846c3b91f6ba802249dbfabb69439c9e154c3576'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and git('ls-remote','origin','refs/heads/main').split()[0]==parent
# These two immutable files are byte-locked by R69; no whitespace normalization allowed.
for f in ['CORE_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md']:assert sha(S/f)==sha(B/'mqft_restore69_20261126'/f)==sha(P/'research/restore91'/f)
git('add','research/restore91','docs')
git('diff','--cached','--check','--','.',':(exclude)research/restore91/logs',':(exclude)research/restore91/CORE_POLICY.md',':(exclude)research/restore91/QUANTUM_REPRESENTATION_POLICY.md')
git('commit','-m','RESTORE91: reconstruct statistical mechanics and signed feedback entropy outputs')
commit=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==parent and not git('status','--porcelain');print('LOCAL COMMIT',commit,flush=True)
git('push','origin','main');remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
git('bundle','create',str(S/'MQFT_RESTORE91.git.bundle'),'--all');git('bundle','verify',str(S/'MQFT_RESTORE91.git.bundle'))
git('archive','--format=zip','--output='+str(S/'MQFT_RESTORE91_SOURCE.zip'),'HEAD')
pub=dict(status='PUSH_VERIFIED',commit=commit,parent=parent,remote_main=remote,tree=git('rev-parse','HEAD^{tree}'),dag_sha256=sha(S/'DAG.json'),run_manifest_sha256=sha(S/'RUN_MANIFEST91.json'),source_zip_sha256=sha(S/'MQFT_RESTORE91_SOURCE.zip'),git_bundle_sha256=sha(S/'MQFT_RESTORE91.git.bundle'),progress_sha256=sha(S/'PROGRESS91.json'),fresh_paths=0,fresh_backups=0,figures=8,cumulative_retained_pairs=693,original_union=121,accepted_corrected_subclaims=26,corrected_report_artifact_accepted=22,final_scientific_certified=0,clean_worktree=True)
dump(S/'PUBLICATION_RECEIPT.json',pub);print('PUBLISHED',pub,flush=True)
