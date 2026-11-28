from base61 import *
import shutil,subprocess
P=B/'mqft_restore39_20261125/repo';dst=P/'research/restore99';parent='aac6e5afef6662781143856a7a0e7b64cf3e41c8'
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==parent and git('ls-remote','origin','refs/heads/main').split()[0]==parent
assert not git('diff','--name-only') and not git('ls-files','--others','--exclude-standard')
paths=git('diff','--cached','--name-only').splitlines();assert paths and all(p.startswith(('research/restore99/','docs/')) for p in paths)
r=read(S/'REGISTRATION.json');assert sha(S/'THEORY_REGISTERED99.md')==r['contract_sha']==sha(dst/'THEORY_REGISTERED99.md')
m=read(S/'RUN_MANIFEST99.json');assert all(sha(S/f)==h for f,h in m['files'].items() if f!='publish99.py')
for f in ['publish99.py','resume_publication99.py','logs/publish99_initial_failure.log','PUBLICATION_REPAIR99.md']:
 m['files'][f]=sha(S/f);shutil.copy2(S/f,dst/f)
dump(S/'RUN_MANIFEST99.json',m);shutil.copy2(S/'RUN_MANIFEST99.json',dst/'RUN_MANIFEST99.json')
git('add','research/restore99','docs')
assert all(sha(dst/f)==h for f,h in m['files'].items() if f!='CURRENT_RESEARCH_PLAN_before.md')
assert sha(P/'docs/DAG.json')==sha(S/'DAG.json')
git('diff','--cached','--check','--','.',':(exclude)research/restore99/logs',':(exclude)research/restore99/CORE_POLICY.md',':(exclude)research/restore99/QUANTUM_REPRESENTATION_POLICY.md',':(exclude)research/restore99/THEORY_REGISTERED99.md')
print('STAGED CHECK VERIFIED',len(m['files']),flush=True)
# Continue only the publication suffix, using the exact stage code, after verified staged checkpoint.
s=(S/'publish99.py').read_text();suffix=s[s.index("git('commit','-m',"):];exec(suffix,globals())
