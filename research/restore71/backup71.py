"""Create verified R71 delta, then retire R70 standalone paths by atomic relocation."""
from base61 import *
import zipfile,shutil,subprocess,os
O=B/'mqft_restore70_20261126';P=B/'mqft_restore39_20261125/repo';D=Path.home()/'Dropbox/personal_projects/mqft_backups'
pub=read(S/'PUBLICATION_RECEIPT.json');oldrec=read(O/'BACKUP_RECEIPT.json');commit=pub['commit']
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==commit and git('rev-parse','HEAD^')==pub['parent'] and not git('status','--porcelain')
assert sha(P/'docs/DAG.json')==pub['dag_sha256']==sha(S/'DAG.json')
m=read(S/'RUN_MANIFEST71.json');assert all(sha(S/f)==v for f,v in m['files'].items())
assert all(sha(B/r['path'])==r['sha256'] for r in m['raw_receipts']+m['prior_raw_receipts'])
assert all(sha(B/r['path'])==r['sha256'] for r in m['weights'].values())
assert all(sha(P/'.external/rapfi'/f)==h for f,h in m['external']['files'].items())
assert all(sha(S/'plans'/f'{k}.json')==v['sha256'] for k,v in m['plan_hashes'].items())
def ident(p):
 s=p.stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns
def syncdir(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def fsync(p):
 with p.open('rb') as f:os.fsync(f.fileno())
def verify_tree(root,h,rel=''):
 p=root/'MANIFEST.json';assert root.is_dir() and not root.is_symlink() and sha(p)==h
 z={rel+'MANIFEST.json':ident(p)}
 for c in read(p)['components']:
  r=c['relative_path'];q=root/r
  if c.get('kind')=='snapshot_directory':z.update(verify_tree(q,c['manifest_sha256'],rel+r+'/'))
  else:
   assert q.is_file() and not q.is_symlink() and q.stat().st_size==c['bytes'];s=ident(q);assert sha(q)==c['sha256'] and ident(q)==s;z[rel+r]=s
 return z
old=[Path(x) for x in oldrec['snapshot_dirs']];assert old==[B/'MQFT_RESTORE70_COMPLETE_20261126',D/'MQFT_RESTORE70_COMPLETE_20261126']
new=[B/'MQFT_RESTORE71_COMPLETE_20261126',D/'MQFT_RESTORE71_COMPLETE_20261126'];tmp=[x.with_name(x.name+'.part') for x in new]
assert all(not x.exists() for x in new+tmp)
# Verify all predecessor components against already-published manifest; retain identities.
old_ids=[]
for p in old:
 assert read(p/'MANIFEST.json')['components']==oldrec['components']
 old_ids.append(verify_tree(p,oldrec['manifest_sha256']));print('PRIOR VERIFIED',p,len(old_ids[-1]),flush=True)
for p in tmp:p.mkdir()
entries={}
def add(p,n):
 assert p.is_file() and not p.is_symlink() and n not in entries;entries[n]={'path':p,'bytes':p.stat().st_size,'sha256':sha(p)}
for n in ['BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json','PUBLICATION_RECEIPT.json']:add(O/n,'prior_receipts/R70/'+n)
for p in sorted(S.rglob('*')):
 if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and not p.name.endswith('.part') and p.name not in ('BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json','backup71.log'):
  add(p,S.name+'/'+p.relative_to(S).as_posix())
dm=dict(schema='mqft_stage_delta_v2',stage=S.name,commit=commit,prior_snapshot_manifest_sha256=oldrec['manifest_sha256'],entries={k:{a:v[a] for a in ('bytes','sha256')} for k,v in entries.items()})
dn='MQFT_RESTORE71_DELTA_20261126.zip';a,b=[p/dn for p in tmp]
with zipfile.ZipFile(a,'x',allowZip64=True) as z:
 for k,v in entries.items():
  p=v['path'];typ=zipfile.ZIP_STORED if p.suffix in ('.zip','.gz','.png','.pdf','.npz','.bundle','.pt','.bin') else zipfile.ZIP_DEFLATED
  z.write(p,k,compress_type=typ,compresslevel=3 if typ==zipfile.ZIP_DEFLATED else None)
 z.writestr('MANIFEST.json',json.dumps(dm,indent=2,sort_keys=True),compress_type=zipfile.ZIP_DEFLATED)
fsync(a)
with zipfile.ZipFile(a) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(entries)+1 and set(z.namelist())==set(entries)|{'MANIFEST.json'} and json.loads(z.read('MANIFEST.json'))==dm
 for k,v in entries.items():
  h=hashlib.sha256();size=0
  with z.open(k) as f:
   for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk);size+=len(chunk)
  assert h.hexdigest()==v['sha256'] and size==v['bytes'],k
digest=sha(a);size=a.stat().st_size;shutil.copyfile(a,b);fsync(b);assert b.stat().st_size==size and sha(b)==digest
print('DELTA VERIFIED x2',size,digest,len(entries),flush=True)
bm=dict(schema='mqft_verified_incremental_snapshot_v2',stage=S.name,commit=commit,dag_sha256=pub['dag_sha256'],components=[dict(role='immutable_predecessor_snapshot',kind='snapshot_directory',relative_path='prior_snapshot',manifest_sha256=oldrec['manifest_sha256'],component_bytes=oldrec['bytes_per_snapshot_components']),dict(role='new_stage_delta',kind='zip_archive',relative_path=dn,bytes=size,sha256=digest,entries=len(entries))],restore='Recursively restore prior_snapshot then extract this delta to the same scratch root; both components required.')
for p in tmp:
 dump(p/'MANIFEST.json',bm);fsync(p/'MANIFEST.json')
 (p/'RESTORE.md').write_text('# RESTORE71 restore\nVerify recursively prior_snapshot/MANIFEST.json and all components, restore prior_snapshot according to its own RESTORE.md, then extract verified R71 delta to same scratch root. Source Git bundle and source ZIP included. Full extraction drill not run. Both components required.\n');fsync(p/'RESTORE.md');syncdir(p)
assert sha(tmp[0]/'MANIFEST.json')==sha(tmp[1]/'MANIFEST.json')
# Atomic same-device relocation, no unique data deletion; restore old paths if failure.
try:
 for i,p in enumerate(tmp):
  assert p.stat().st_dev==old[i].stat().st_dev and all(ident(old[i]/k)==v for k,v in old_ids[i].items())
  os.rename(old[i],p/'prior_snapshot');syncdir(p);syncdir(old[i].parent)
 for i,p in enumerate(tmp):os.rename(p,new[i]);syncdir(new[i].parent)
 for i,p in enumerate(new):
  assert all(ident(p/'prior_snapshot'/k)==v for k,v in old_ids[i].items()) and sha(p/'prior_snapshot'/'MANIFEST.json')==oldrec['manifest_sha256'] and sha(p/dn)==digest and read(p/'MANIFEST.json')==bm
 assert all(not p.exists() for p in old)
except Exception:
 for i in range(2):
  for root in (new[i],tmp[i]):
   p=root/'prior_snapshot'
   if p.exists() and not old[i].exists():os.rename(p,old[i]);syncdir(old[i].parent)
 raise
mh=sha(new[0]/'MANIFEST.json');assert sha(new[1]/'MANIFEST.json')==mh
receipt=dict(status='VERIFIED_TWO_LOCAL_INCREMENTAL_SNAPSHOT_COPIES',snapshot_dirs=list(map(str,new)),manifest_sha256=mh,components=bm['components'],bytes_per_snapshot_components=oldrec['bytes_per_snapshot_components']+size,delta_manifest_entries=len(entries),repo_commit=commit,dag_sha256=pub['dag_sha256'],prior_components_verified_before_atomic_relocation=True,delta_verification='all member hashes first copy; second byte identity; both post-relocation digests',cloud_sync='not independently verified',physical_failure_domain='same device possible',full_extraction_restore_drill='not run')
cleanup=dict(status='RETIRED_TWO_PREDECESSOR_STANDALONE_SNAPSHOTS_BY_ATOMIC_RELOCATION',old_paths=list(map(str,old)),new_required_snapshot_paths=[str(p/'prior_snapshot') for p in new],retained_prior_manifest_sha256=oldrec['manifest_sha256'],base_bytes_reused=2*oldrec['bytes_per_snapshot_components'],physically_freed_bytes=0,delta_bytes_added=2*size,originals_and_unique_data='preserved',note='No separately redundant complete predecessor directories remain. Their contents were moved, not deleted. Other old stage working artifacts retained because manifests reference them.')
for path,obj in [(S/'BACKUP_RECEIPT.json',receipt),(S/'CLEANUP_RECEIPT.json',cleanup),(D/'RESTORE71_BACKUP_RECEIPT.json',receipt),(D/'RESTORE71_CLEANUP_RECEIPT.json',cleanup)]:dump(path,obj);fsync(path)
plan=B/'CURRENT_RESEARCH_PLAN.md';assert plan.read_bytes()==(S/'CURRENT_RESEARCH_PLAN_before.md').read_bytes()
head=f'''# MQFT current entry: RESTORE71 (2026-11-26 KST)
Active repository: https://github.com/cosmosapjw-quantum/mcts_project
main commit: {commit}
Local checkout: {P}
DAG: {S.name}/DAG.json (sha256 {pub['dag_sha256']})
Results: {S.name}/RESULTS_KO.md
Physics: {S.name}/PHYSICS_RESULTS.json
Audit: {S.name}/AUDIT71.json and INDEPENDENT71.json
Progress: {S.name}/PROGRESS71.json
Plot inventory: {S.name}/PLOT_INVENTORY71.csv
Original item progress: {S.name}/ORIGINAL_OUTPUT_PROGRESS71.csv
Next: {S.name}/NEXT_DYNAMICS.md
Backup: {new[0].name}/MANIFEST.json (sha256 {mh}); two verified local incremental snapshots.
9,216 fresh fixed-root searches, 1,179,648 backups; 69,880 cumulative fixed-root searches (historical census plus fresh). No games/training.
563 distinct plot pairs (+10), R55-R71 union 90/121; final certified 0.
13 nodes with evidence (1 scope-complete, 7 scoped, 5 partial).
R71 autonomous empirical joint closure evaluated; population bias remains. R72 coarse maps/five-size family, R73 multi-time records.
Prior plan follows.

'''
t=plan.with_suffix('.md.part')
with t.open('xb') as f:f.write(head.encode()+plan.read_bytes());f.flush();os.fsync(f.fileno())
t.replace(plan);syncdir(B)
print('BACKUP',receipt,flush=True);print('CLEANUP',cleanup,flush=True)
