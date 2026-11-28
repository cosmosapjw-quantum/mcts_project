from base61 import *
import shutil,subprocess
P=B/'mqft_restore39_20261125/repo';dst=P/'research/restore101';parent='3a2a5b702f3e61cb9eb21ece5de94664d8f4aff2'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and git('ls-remote','origin','refs/heads/main').split()[0]==parent
assert not git('diff','--name-only') and not git('ls-files','--others','--exclude-standard')
paths=git('diff','--cached','--name-only').splitlines();assert paths and all(p.startswith(('research/restore101/','docs/')) for p in paths)
r=read(S/'REGISTRATION.json');assert sha(S/'THEORY_REGISTERED101.md')==r['contract_sha']==sha(dst/'THEORY_REGISTERED101.md')
m=read(S/'RUN_MANIFEST101.json');assert all(sha(S/f)==h for f,h in m['files'].items() if f!='publish101.py')
for f in ['publish101.py','resume_publication101.py','logs/publish101_initial_failure.log','PUBLICATION_REPAIR101.md']:
 m['files'][f]=sha(S/f);shutil.copy2(S/f,dst/f)
dump(S/'RUN_MANIFEST101.json',m);shutil.copy2(S/'RUN_MANIFEST101.json',dst/'RUN_MANIFEST101.json')
git('add','research/restore101','docs')
assert all(sha(dst/f)==h for f,h in m['files'].items() if f!='CURRENT_RESEARCH_PLAN_before.md')
assert sha(P/'docs/DAG.json')==sha(S/'DAG.json')
git('diff','--cached','--check','--','.',':(exclude)research/restore101/logs',':(exclude)research/restore101/CORE_POLICY.md',':(exclude)research/restore101/QUANTUM_REPRESENTATION_POLICY.md',':(exclude)research/restore101/THEORY_REGISTERED101.md')
print('STAGED CHECK VERIFIED',len(m['files']),flush=True)
s=(S/'publish101.py').read_text();exec(s[s.index("git('commit','-m',"):],globals())
