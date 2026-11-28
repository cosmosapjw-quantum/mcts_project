"""Dropbox-only verified snapshot, remove only byte-proven redundant backup copies."""
from base61 import *
import os,shutil,subprocess,zipfile,struct,re,io
P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore88_20261128';D=Path.home()/'Dropbox/personal_projects/mqft_backups'
pub=read(S/'PUBLICATION_RECEIPT.json');oldrec=read(O/'BACKUP_RECEIPT.json');m=read(S/'RUN_MANIFEST89.json')
def git(*a):return subprocess.check_output(['git','-C',str(P),*a],text=True).strip()
assert git('rev-parse','HEAD')==pub['commit']==git('ls-remote','origin','refs/heads/main').split()[0] and not git('status','--porcelain')
assert sha(S/'DAG.json')==pub['dag_sha256']==sha(P/'docs/DAG.json')
assert all(sha(S/k)==h for k,h in m['files'].items())
assert all(sha(B/r['path'])==r['sha256'] for r in m['prior_raw_receipts'])
assert all(sha(B/x['path'])==x['sha256'] for x in m['weights'].values())
assert all(sha(B/k)==h for k,h in m['inputs'].items())
assert all(sha(P/'.external/rapfi'/f)==h for f,h in m['external']['files'].items())
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
local=B/'MQFT_RESTORE88_COMPLETE_20261128';old=D/local.name;new=D/'MQFT_RESTORE89_COMPLETE_20261128';tmp=D/(new.name+'.part')
assert [str(local),str(old)]==oldrec['snapshot_dirs'] and not new.exists() and not tmp.exists()
assert sha(local/'MANIFEST.json')==sha(old/'MANIFEST.json')==oldrec['manifest_sha256']
# Verify exact full byte identity, including files outside recursive manifest components.
li=file_index(local);di=file_index(old)
assert li.keys()==di.keys() and all(li[k][0][2]==di[k][0][2] and li[k][1]==di[k][1] for k in li)
# Validate the old surviving recursive manifest against already measured digests.
def check_index(root,expected):
 key=(root/'MANIFEST.json').relative_to(old).as_posix();assert di[key][1]==expected
 for c in read(root/'MANIFEST.json')['components']:
  p=root/c['relative_path'];k=p.relative_to(old).as_posix()
  if c.get('kind')=='snapshot_directory':check_index(p,c['manifest_sha256'])
  else:assert di[k][1]==c['sha256'] and di[k][0][2]==c['bytes']
check_index(old,oldrec['manifest_sha256'])
print('PRIOR DROPBOX AND LOCAL TREE IDENTICAL',len(li),sum(x[0][2] for x in li.values()),flush=True)
tmp.mkdir();entries={}
def add(p,k):
 assert p.is_file() and not p.is_symlink() and k not in entries
 entries[k]={'path':p,'bytes':p.stat().st_size,'sha256':sha(p)}
for f in ('BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json','PUBLICATION_RECEIPT.json','DUPLICATE_ARCHIVE_CLEANUP88.json'):add(O/f,'prior_receipts/R88/'+f)
for p in sorted(S.rglob('*')):
 if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and not p.name.endswith('.part') and p.name not in ('BACKUP_RECEIPT.json','CLEANUP_RECEIPT.json','backup89.log'):
  add(p,S.name+'/'+p.relative_to(S).as_posix())
dm=dict(schema='mqft_stage_delta_v2',stage=S.name,commit=pub['commit'],prior_snapshot_manifest_sha256=oldrec['manifest_sha256'],entries={k:{a:v[a] for a in ('bytes','sha256')} for k,v in entries.items()})
dn='MQFT_RESTORE89_DELTA_20261128.zip';delta=tmp/dn
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
store(tmp/'MANIFEST.json',bm);(tmp/'RESTORE.md').write_text('# RESTORE89 Dropbox-only snapshot\nVerify recursively prior_snapshot/MANIFEST.json and all components. Restore prior_snapshot, then extract the new verified delta to same scratch root. Full extraction drill not run.\n');fsync(tmp/'RESTORE.md');syncdir(tmp)
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
for p in (S/'BACKUP_RECEIPT.json',D/'RESTORE89_BACKUP_RECEIPT.json'):store(p,receipt)
print('DROPBOX DELTA VERIFIED',size,digest,len(entries),flush=True)
# Index the exact generated archive candidates. Only these named backup files in stage directories are eligible.
candidates={}
for st in B.glob('mqft_restore*'):
 if not st.is_dir() or st.is_symlink() or not re.fullmatch(r'mqft_restore\d+_\d{8}',st.name):continue
 for p in st.iterdir():
  if p.is_file() and not p.is_symlink() and re.fullmatch(r'MQFT_RESTORE\d+(?:\.git\.bundle|_SOURCE\.zip)',p.name):
   k=p.relative_to(B).as_posix();candidates[k]=(p,ident(p),sha(p))
print('STANDALONE CANDIDATES',len(candidates),flush=True)
# Read stored nested ZIPs as bounded views of their containing file. No expansion to disk.
class Slice(io.RawIOBase):
 def __init__(self,path,offset,size):
  self.f=path.open('rb');self.base=offset;self.size=size;self.pos=0
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,off,whence=0):
  p=off if whence==0 else self.pos+off if whence==1 else self.size+off
  if not 0<=p<=self.size:raise OSError('seek outside archive slice')
  self.pos=p;return p
 def read(self,n=-1):
  n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
  self.f.seek(self.base+self.pos);v=self.f.read(n);self.pos+=len(v);return v
 def close(self):
  if not self.closed:self.f.close()
  super().close()
matches={};skipped_nested=[];scanned=[]
def scan(path,offset,size,chain):
 with Slice(path,offset,size) as raw,zipfile.ZipFile(raw) as z:
  scanned.append(chain)
  infos={i.filename:i for i in z.infolist()}
  for k in sorted(set(candidates)&infos.keys()):
   if k in matches:continue
   p,st,h=candidates[k]
   if infos[k].file_size==st[2] and member_sha(z,k)==(h,st[2]):
    matches[k]=dict(path=str(p),relative=k,bytes=st[2],sha256=h,identity=list(st),archive=str(path),nested_chain=chain,member=k)
  for n,i in infos.items():
   if n.startswith('prior_archive/') and n.endswith('.zip'):
    if i.compress_type!=zipfile.ZIP_STORED:skipped_nested.append(chain+[n]);continue
    raw.seek(i.header_offset);head=raw.read(30);assert head[:4]==b'PK\x03\x04'
    a,b=struct.unpack_from('<HH',head,26);start=offset+i.header_offset+30+a+b
    assert start+ i.file_size<=offset+size
    scan(path,start,i.file_size,chain+[n])
# Only authenticated Dropbox archive components, not arbitrary external ZIPs.
def walk(root):
 for c in read(root/'MANIFEST.json')['components']:
  p=root/c['relative_path']
  if c.get('kind')=='snapshot_directory':walk(p)
  elif p.suffix=='.zip':scan(p,0,p.stat().st_size,[str(p.relative_to(new))])
walk(new)
print('VERIFIED STANDALONE DUPLICATES',len(matches),'unmatched',len(candidates)-len(matches),'archives',len(scanned),flush=True)
# Record intent durably, recheck identity before deleting exact scope only.
plan=dict(status='PREPARED_EXACT_DUPLICATE_REMOVAL',retained_snapshot=str(new),retained_manifest_sha256=mh,local_duplicate_tree=str(local),local_files=len(li),local_logical_bytes=sum(v[0][2] for v in li.values()),local_file_hashes={k:v[1] for k,v in li.items()},standalone_matches=list(matches.values()),unmatched=sorted(set(candidates)-matches.keys()),skipped_nested=skipped_nested,archives_scanned=len(scanned),originals_and_active_raw='preserved')
for p in (S/'CLEANUP_RECEIPT.json',D/'RESTORE89_CLEANUP_RECEIPT.json'):store(p,plan)
assert local.parent==B and local.name=='MQFT_RESTORE88_COMPLETE_20261128' and not local.is_symlink()
assert all(ident(local/k)==v[0] and ident(new/'prior_snapshot'/k)==di[k][0] for k,v in li.items())
assert {p.relative_to(local).as_posix() for p in local.rglob('*') if p.is_file()}==set(li)
assert all(ident(Path(x['path']))==tuple(x['identity']) for x in matches.values())
before=shutil.disk_usage(B).free
shutil.rmtree(local);assert not local.exists()
for x in matches.values():
 p=Path(x['path']);assert ident(p)==tuple(x['identity']);p.unlink();assert not p.exists();x['status']='REMOVED_VERIFIED_DUPLICATE'
after=shutil.disk_usage(B).free
plan.update(status='DROPBOX_ONLY_DUPLICATES_REMOVED',deleted_standalone=len(matches),deleted_standalone_logical_bytes=sum(x['bytes'] for x in matches.values()),disk_free_before=before,disk_free_after=after,disk_free_delta=after-before,local_duplicate_tree_removed=True)
for p in (S/'CLEANUP_RECEIPT.json',D/'RESTORE89_CLEANUP_RECEIPT.json'):store(p,plan)
# Update current plan only once complete; keep old bytes as history.
p=B/'CURRENT_RESEARCH_PLAN.md';assert p.read_bytes()==(S/'CURRENT_RESEARCH_PLAN_before.md').read_bytes()
head=f"""# MQFT current entry: RESTORE89 (2026-11-28 KST)
Active repository: https://github.com/cosmosapjw-quantum/mcts_project
main commit: {pub['commit']}
Local checkout: {P}
DAG: {S.name}/DAG.json (sha256 {pub['dag_sha256']})
Results: {S.name}/RESULTS_KO.md
Physics: {S.name}/ORIGINAL_OUTPUT_REPORT89.md, ITEM_RECONSTRUCTION89.json, SUMMARY89.json, CONVERGENCE89.json
Audit: {S.name}/INDEPENDENT89.json
Progress: {S.name}/PROGRESS89.json
Plot inventory: {S.name}/PLOT_INVENTORY89.csv
Original item progress: {S.name}/ORIGINAL_OUTPUT_PROGRESS89.csv
Next: {S.name}/NEXT_DYNAMICS.md
Backup: {new}/MANIFEST.json (sha256 {mh}); verified Dropbox-only snapshot. No redundant local complete backup.
0 fresh fixed-root searches; 93,336 cumulative. Reauthenticated 3,200 prior fixed-root paths,80 actual registered games and579,026 eval records.
678 distinct plot pairs (+2), 121/121 evidence coverage,11/121 corrected artifacts (+2),23 scoped claims (+2),6 open science claims; final original science certification0.
13 nodes with evidence (1 scope-complete,7 scoped,5 partial).
R89 QEC and integrated advanced report. Next R90 statistical action outputs.
Prior plan follows.

"""
q=p.with_suffix('.md.part')
with q.open('xb') as f:f.write(head.encode()+p.read_bytes());f.flush();os.fsync(f.fileno())
q.replace(p);syncdir(B)
print('BACKUP',receipt,flush=True);print('CLEANUP', {k:v for k,v in plan.items() if k not in ('local_file_hashes','standalone_matches')},flush=True)
