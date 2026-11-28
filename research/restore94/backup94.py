"""Dropbox-only verified snapshot, remove only byte-proven redundant backup copies."""
from base61 import *
import os,shutil,subprocess,zipfile,struct,re,io
P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore93_20261128';D=Path.home()/'Dropbox/personal_projects/mqft_backups'
pub=read(S/'PUBLICATION_RECEIPT.json');oldrec=read(O/'BACKUP_RECEIPT.json');m=read(S/'RUN_MANIFEST94.json')
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==pub['commit']==git('ls-remote','origin','refs/heads/main').split()[0] and not git('status','--porcelain')
assert sha(S/'DAG.json')==pub['dag_sha256']==sha(P/'docs/DAG.json')
assert all(sha(S/k)==h for k,h in m['files'].items())
assert all(sha(B/k)==h for k,h in m['prior_raw_receipts'].items())
assert all(sha(B/k)==h for k,h in m['inputs'].items())
def fsync(p):
 with p.open('rb') as f:os.fsync(f.fileno())
def syncdir(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def store(p,obj):
 dump(p,obj);fsync(p)
def ident(p):
 s=p.stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns
def tree(root,expected):
 assert root.is_dir() and not root.is_symlink() and sha(root/'MANIFEST.json')==expected
 for c in read(root/'MANIFEST.json')['components']:
  q=root/c['relative_path']
  if c.get('kind')=='snapshot_directory':tree(q,c['manifest_sha256'])
  else:assert q.is_file() and not q.is_symlink() and q.stat().st_size==c['bytes'] and sha(q)==c['sha256']
def file_index(root):
 out={}
 for p in root.rglob('*'):
  assert not p.is_symlink(),str(p)
  if p.is_file():
   i=ident(p);h=sha(p);assert ident(p)==i
   out[p.relative_to(root).as_posix()]=(i,h)
 return out
old=D/'MQFT_RESTORE93_COMPLETE_20261128';new=D/'MQFT_RESTORE94_COMPLETE_20261128';tmp=D/(new.name+'.part')
assert [str(old)]==oldrec['snapshot_dirs'] and not new.exists() and not tmp.exists()
assert sha(old/'MANIFEST.json')==oldrec['manifest_sha256']
di=file_index(old)
def check_index(root,expected):
 key=(root/'MANIFEST.json').relative_to(old).as_posix();assert di[key][1]==expected
 for c in read(root/'MANIFEST.json')['components']:
  p=root/c['relative_path'];k=p.relative_to(old).as_posix()
  if c.get('kind')=='snapshot_directory':check_index(p,c['manifest_sha256'])
  else:assert di[k][1]==c['sha256'] and di[k][0][2]==c['bytes']
check_index(old,oldrec['manifest_sha256'])
print('PRIOR DROPBOX TREE VERIFIED',len(di),flush=True)
tmp.mkdir();entries={}
def add(p,k):
 assert p.is_file() and not p.is_symlink() and k not in entries
 entries[k]={'path':p,'bytes':p.stat().st_size,'sha256':sha(p)}
for f in ('BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json','PUBLICATION_RECEIPT.json'):add(O/f,'prior_receipts/R93/'+f)
for p in sorted(S.rglob('*')):
 if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and not p.name.endswith('.part') and p.name not in ('BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json','backup94.log'):
  add(p,S.name+'/'+p.relative_to(S).as_posix())
dm=dict(schema='mqft_stage_delta_v2',stage=S.name,commit=pub['commit'],prior_snapshot_manifest_sha256=oldrec['manifest_sha256'],entries={k:{a:v[a] for a in ('bytes','sha256')} for k,v in entries.items()})
dn='MQFT_RESTORE94_DELTA_20261128.zip';delta=tmp/dn
with zipfile.ZipFile(delta,'x',allowZip64=True) as z:
 for k,v in entries.items():
  p=v['path'];typ=zipfile.ZIP_STORED if p.suffix in ('.zip','.gz','.png','.pdf','.npz','.bundle','.pt','.bin') else zipfile.ZIP_DEFLATED
  z.write(p,k,compress_type=typ,compresslevel=3 if typ==zipfile.ZIP_DEFLATED else None)
 z.writestr('MANIFEST.json',json.dumps(dm,indent=2,sort_keys=True),compress_type=zipfile.ZIP_DEFLATED)
fsync(delta)
def member_sha(z,k):
 h=hashlib.sha256();n=0
 with z.open(k) as f:
  for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk);n+=len(chunk)
 return h.hexdigest(),n
with zipfile.ZipFile(delta) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(entries)+1 and set(z.namelist())==set(entries)|{'MANIFEST.json'} and json.loads(z.read('MANIFEST.json'))==dm
 for k,v in entries.items():assert member_sha(z,k)==(v['sha256'],v['bytes']),k
digest=sha(delta);size=delta.stat().st_size
bm=dict(schema='mqft_verified_incremental_snapshot_v2',stage=S.name,commit=pub['commit'],dag_sha256=pub['dag_sha256'],components=[dict(role='immutable_predecessor_snapshot',kind='snapshot_directory',relative_path='prior_snapshot',manifest_sha256=oldrec['manifest_sha256'],component_bytes=oldrec['bytes_per_snapshot_components']),dict(role='new_stage_delta',kind='zip_archive',relative_path=dn,bytes=size,sha256=digest,entries=len(entries))],restore='Recursively restore prior_snapshot then extract delta to same scratch root. Both components required.',backup_policy='DROPBOX_ONLY')
store(tmp/'MANIFEST.json',bm);(tmp/'RESTORE.md').write_text('# RESTORE94 Dropbox-only snapshot\nVerify recursively prior_snapshot/MANIFEST.json and all components. Restore prior_snapshot, then extract the new verified delta to same scratch root. Full extraction drill not run.\n');fsync(tmp/'RESTORE.md');syncdir(tmp)
assert old.stat().st_dev==tmp.stat().st_dev and all(ident(old/k)==v[0] for k,v in di.items())
try:
 os.rename(old,tmp/'prior_snapshot');syncdir(tmp);os.rename(tmp,new);syncdir(D)
 assert all(ident(new/'prior_snapshot'/k)==v[0] for k,v in di.items()) and sha(new/dn)==digest and read(new/'MANIFEST.json')==bm
except Exception:
 for root in (new,tmp):
  q=root/'prior_snapshot'
  if q.exists() and not old.exists():os.rename(q,old);syncdir(D)
 raise
mh=sha(new/'MANIFEST.json')
receipt=dict(status='VERIFIED_DROPBOX_ONLY_SNAPSHOT',snapshot_dirs=[str(new)],manifest_sha256=mh,components=bm['components'],bytes_per_snapshot_components=oldrec['bytes_per_snapshot_components']+size,delta_manifest_entries=len(entries),repo_commit=pub['commit'],dag_sha256=pub['dag_sha256'],prior_components_verified_before_atomic_relocation=True,delta_verification='all member hashes; post-relocation byte digest',cloud_sync='pending separate daemon query; local Dropbox bytes verified',full_extraction_restore_drill='not run')
for p in (S/'BACKUP_RECEIPT.json',D/'RESTORE94_BACKUP_RECEIPT.json'):store(p,receipt)
print('DROPBOX DELTA VERIFIED',size,digest,len(entries),flush=True)
# Verify the two transient standalone archive copies against members of the new delta.
candidates=[]
for st in B.glob('mqft_restore*'):
 if not st.is_dir() or st.is_symlink() or not re.fullmatch(r'mqft_restore\d+_\d{8}',st.name):continue
 for p in st.iterdir():
  if p.is_file() and not p.is_symlink() and re.fullmatch(r'MQFT_RESTORE\d+(?:\.git\.bundle|_SOURCE\.zip)',p.name):candidates.append(p)
matches=[];unmatched=[]
with zipfile.ZipFile(new/dn) as z:
 for p in sorted(candidates):
  k=str(p.relative_to(B));i=ident(p);h=sha(p)
  if k in entries and member_sha(z,k)==(h,i[2]):matches.append(dict(path=str(p),member=k,sha256=h,bytes=i[2],identity=list(i)))
  else:unmatched.append(str(p))
assert len(matches)==2 and {Path(x['path']).name for x in matches}=={'MQFT_RESTORE94.git.bundle','MQFT_RESTORE94_SOURCE.zip'}
local_complete=[str(p) for p in B.glob('MQFT_RESTORE*_COMPLETE_*') if p.exists()]
plan=dict(status='PREPARED_EXACT_DUPLICATE_REMOVAL',retained_snapshot=str(new),retained_manifest_sha256=mh,standalone_matches=matches,unmatched=unmatched,local_complete_candidates=local_complete,originals_and_active_raw='preserved')
for p in (S/'CLEANUP_RECEIPT.json',D/'RESTORE94_CLEANUP_RECEIPT.json'):store(p,plan)
before=shutil.disk_usage(B).free
for x in matches:
 p=Path(x['path']);assert ident(p)==tuple(x['identity']) and sha(p)==x['sha256'];p.unlink();assert not p.exists();x['status']='REMOVED_VERIFIED_DUPLICATE'
after=shutil.disk_usage(B).free
plan.update(status='DROPBOX_ONLY_DUPLICATES_REMOVED',deleted_standalone=len(matches),deleted_standalone_logical_bytes=sum(x['bytes'] for x in matches),disk_free_before=before,disk_free_after=after,disk_free_delta=after-before)
for p in (S/'CLEANUP_RECEIPT.json',D/'RESTORE94_CLEANUP_RECEIPT.json'):store(p,plan)
# Update current plan only once complete; keep old bytes as history.
p=B/'CURRENT_RESEARCH_PLAN.md';assert p.read_bytes()==(S/'CURRENT_RESEARCH_PLAN_before.md').read_bytes()
head=f"""# MQFT current entry: RESTORE94 (2026-11-28 KST)
Active repository: https://github.com/cosmosapjw-quantum/mcts_project
main commit: {pub['commit']}
Local checkout: {P}
DAG: {S.name}/DAG.json (sha256 {pub['dag_sha256']})
Results: {S.name}/RESULTS_KO.md
Physics: {S.name}/ORIGINAL_OUTPUT_REPORT94.md, ITEM_RECONSTRUCTION94.json, NATIVE94.json, RECORD94.json, CONVERGENCE94.json
Audit: {S.name}/INDEPENDENT94.json
Progress: {S.name}/PROGRESS94.json
Plot inventory: {S.name}/PLOT_INVENTORY94.csv
Original item progress: {S.name}/ORIGINAL_OUTPUT_PROGRESS94.csv
Next: {S.name}/NEXT_DYNAMICS.md
Backup: {new}/MANIFEST.json (sha256 {mh}); verified Dropbox-only snapshot. No redundant local complete backup.
0 fresh fixed-root searches; 93,336 cumulative. Reauthenticated 3,456 stochastic phase histories and 648 forced/replay diagnostics; 2,304 declared prefix states.
712 distinct plot pairs (+7), 121/121 evidence coverage,34/121 corrected artifacts (+5),32 scoped claims (+2),6 open science claims; final original science certification0.
13 nodes with evidence (1 scope-complete,7 scoped,5 partial).
R94 complex state, coherence, entropy and pointer outputs. Next R95 entanglement/QMI/concurrence/Bell.
Prior plan follows.

"""
q=p.with_suffix('.md.part')
with q.open('xb') as f:f.write(head.encode()+p.read_bytes());f.flush();os.fsync(f.fileno())
q.replace(p);syncdir(B)
print('BACKUP',receipt,flush=True);print('CLEANUP', {k:v for k,v in plan.items() if k not in ('local_file_hashes','standalone_matches')},flush=True)
