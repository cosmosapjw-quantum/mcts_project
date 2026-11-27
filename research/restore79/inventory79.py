"""Reauthenticate archive source spans, publication and retained plot inventory; preserve discrepancies."""
from base61 import *
import csv,zipfile,subprocess,ast,re
from PIL import Image
O=B/'mqft_restore78_20261127';A=B/'mqft_restore69_20261126';P=B/'mqft_restore39_20261125/repo'
def cr(p):return list(csv.DictReader(p.open()))
def cw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A/f),f
prev=read(O/'ORIGINAL_SOURCE78.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert all(sha(paths[k])==h for k,h in prev['archives'].items())
cat={r['id']:r for r in cr(S/'ORIGINAL_TARGET_CATALOG.csv')};old={r['id']:r for r in cr(O/'ORIGINAL_OUTPUT_PROGRESS78.csv')};assert len(cat)==len(old)==121 and set(cat)==set(old)
issues=[];members={};spans=[];source_text=[]
def issue(kind,**kw):issues.append(dict(kind=kind,**kw))
for k,x in sorted(cat.items()):
 if x['archive']=='guide_only':
  spans.append(dict(id=k,archive='guide_only',member=x['member'],member_sha='',span_sha='',start='',end='',source_status='GUIDE_LABEL_ONLY',function='',panel=x['panel'],observable_fields=[]))
  continue
 key=(x['archive'],x['member'])
 if key not in members:
  with zipfile.ZipFile(paths[key[0]]) as z:b=z.read(key[1])
  assert hashlib.sha256(b).hexdigest()==x['member_sha256'];members[key]=b
 b=members[key];lines=b.decode().splitlines();s,e=int(x['start']),int(x['end']);assert 1<=s<=e<=len(lines)
 a='\n'.join(lines[s-1:e]);assert re.search(r'\bdef\s+'+re.escape(x['function'])+r'\s*\(',a),k
 fields=sorted(set(re.findall(r"\[['\"]([^'\"]+)['\"]\]|\.get\(['\"]([^'\"]+)['\"]",a)))
 fields=sorted({p or q for p,q in fields})
 spans.append(dict(id=k,archive=key[0],member=key[1],member_sha=x['member_sha256'],span_sha=hashlib.sha256(a.encode()).hexdigest(),start=s,end=e,source_status='EXACT_SOURCE_SPAN',function=x['function'],panel=x['panel'],observable_fields=fields))
 source_text.extend([f"\n## {k} {key[1]}:{s}-{e}\n",a])
# Guide source itself can be verified, even when catalog references original output filename only.
guides=['omoknuni_quantum-main/docs/quantum_physics/'+x+'_ANALYSIS_GUIDE.md' for x in ('STATISTICAL_MECHANICS','QUANTUM_PHENOMENA')]
with zipfile.ZipFile(paths['omoknuni_quantum-main.zip']) as z:
 for guide in guides:members[('omoknuni_quantum-main.zip',guide)]=z.read(guide)
for x in spans:
 if x['archive']=='guide_only':
  found=[]
  for guide in guides:
   b=members['omoknuni_quantum-main.zip',guide];hits=[i+1 for i,l in enumerate(b.decode().splitlines()) if x['member'] in l]
   if hits:found.append((guide,hashlib.sha256(b).hexdigest(),hits))
  assert len(found)==1,x
  x['guide_source_member'],x['guide_source_sha'],x['guide_lines']=found[0]
# Actual study note identities and semantic anchors.
with zipfile.ZipFile(paths['mcts_study.zip']) as z:
 for m in ['mcts_study/v3.0/qft-mcts-statistical.md','mcts_study/effective-action-mcts.md','mcts_study/v5.0/new_quantum_mcts.md','mcts_study/v3.0/qft-mcts-decoherence.md']:
  members['mcts_study.zip',m]=z.read(m)
dump(S/'ORIGINAL_SOURCE79.json',dict(archives=prev['archives'],verified_members=[dict(archive=k[0],member=k[1],sha256=hashlib.sha256(b).hexdigest()) for k,b in members.items()],spans=spans,original_attachment_receipt=prev['original_attachment_receipt'],scope='Exact 94 source function spans, 27 guide-only obligations, local original archives. Original upload receipt inherited, no guide-only PNG bytes claimed.'))
(S/'ORIGINAL_SOURCE_SPANS79.md').write_text('# Archived original source spans, read-only transcription\n'+'\n'.join(source_text)+'\n')
# All retained plot pairs; exact byte identity, including size and format.
inv=cr(O/'PLOT_INVENTORY78.csv');assert len(inv)==640 and len({x['png_sha'] for x in inv})==640 and sha(O/'PLOT_INVENTORY78.csv')==read(O/'PROGRESS78.json')['inventory_sha256']
def check_pair(x):
 p=B/x['stage']/x['plot'];a=Path(str(p)+'.png');b=Path(str(p)+'.pdf');err=[]
 for f,k in [(a,'png'),(b,'pdf')]:
  if not f.is_file():err.append(k+':missing');continue
  if sha(f)!=x[k+'_sha']:err.append(k+':sha_mismatch')
  if f.stat().st_size!=int(x[k+'_bytes']):err.append(k+':size_mismatch')
 try:
  with Image.open(a) as im:im.verify()
 except Exception as e:err.append('png:'+type(e).__name__)
 if b.is_file():
  if b.open('rb').read(4)!=b'%PDF':err.append('pdf:signature')
 return dict(stage=x['stage'],plot=x['plot'],status='VERIFIED' if not err else 'DISCREPANCY',errors=err,png_sha=x['png_sha'],pdf_sha=x['pdf_sha'])
with cf.ThreadPoolExecutor(max_workers=8) as pool:plotchecks=list(pool.map(check_pair,inv))
for x in plotchecks:
 if x['errors']:issue('inventory_pair_failure',**x)
# Publications and source result file identities from R55--78. File audit only; do not claim full rerun.
stages={};refs=collections.defaultdict(list);manifest_audit=[]
for i in range(55,79):
 d=next(B.glob(f'mqft_restore{i}_*'));pub=read(d/'PUBLICATION_RECEIPT.json');assert pub['commit']==pub['remote_main']
 stage=dict(stage=i,dir=d.name,commit=pub['commit'],published_status=pub['status'],publication_sha=sha(d/'PUBLICATION_RECEIPT.json'))
 p=d/f'ORIGINAL_OUTPUT_EVIDENCE{i}.csv';er=cr(p);stage['evidence_sha']=sha(p);stage['evidence_rows']=len(er)
 m=d/f'RUN_MANIFEST{i}.json';pf=d/f'PLOT_MANIFEST{i}.json';mf=read(m) if m.exists() else None;pm=read(pf) if pf.exists() else None
 if mf:
  stage['manifest_sha']=sha(m);assert stage['manifest_sha']==pub['run_manifest_sha256'];stage['manifest_path']=str(m.relative_to(B))
  checks=[]
  for rel,h in mf.get('files',{}).items():
   if not isinstance(h,str) or len(h)!=64:continue
   f=d/rel;ok=f.is_file() and sha(f)==h
   checks.append(dict(file=str(f.relative_to(B)),sha=h,status='VERIFIED' if ok else 'MISSING' if not f.is_file() else 'SHA_MISMATCH'))
   if not ok:issue('stage_manifest_file_discrepancy',stage=i,file=rel,status=checks[-1]['status'])
  stage['manifest_files_checked']=len(checks);stage['manifest_files_verified']=sum(x['status']=='VERIFIED' for x in checks);manifest_audit.extend(checks)
 else:
  f=d/'PHYSICS_RESULTS.json';assert f.is_file();stage['legacy_physics_sha']=sha(f)
  if pub.get('physics_results_sha256')!=stage['legacy_physics_sha']:issue('legacy_physics_sha_mismatch',stage=i)
 if pm:
  stage['plot_manifest_sha']=sha(pf);stage['plot_manifest_path']=str(pf.relative_to(B))
  for x in pm['figures']:
   f=d/'plots'/x['figure'];a=Path(str(f)+'.png');b=Path(str(f)+'.pdf')
   if not a.is_file() or not b.is_file() or sha(a)!=x['png_sha'] or sha(b)!=x['pdf_sha']:issue('stage_figure_mismatch',stage=i,figure=x['figure'])
  for x in pm.get('animations',[]):
   name=x.get('file',x.get('name'));assert name;f=d/'plots'/name
   if not f.is_file() or sha(f)!=x['sha256']:issue('animation_mismatch',stage=i,file=name)
   else:
    with Image.open(f) as im:
     if im.n_frames!=x['frames']:issue('animation_frame_count',stage=i,file=name)
 for x in er:
  k=x['id'];assert k in cat
  r=dict(stage=i,commit=pub['commit'],evidence_csv=str(p.relative_to(B)),scope=x['scope'],type=x.get('current_evidence',''),figures=x.get('evidence_figures',''),figure_links=[],source_files=[])
  # Exact evidence row links; some legacy rows only node/group references, not named plots.
  if pm:
   for z in pm['figures']:
    if k in z.get('ids',[]):r['figure_links'].append(dict(plot=str((d/'plots'/z['figure']).relative_to(B)),png_sha=z['png_sha'],pdf_sha=z['pdf_sha']))
   for rel,h in pm.get('sources',{}).items():
    f=d/rel
    r['source_files'].append(dict(path=str(f.relative_to(B)),sha=h,status='VERIFIED' if f.is_file() and sha(f)==h else 'UNVERIFIED'))
  if pm:
   r['animation_links']=[dict(path=str((d/'plots'/z.get('file',z.get('name'))).relative_to(B)),sha=z['sha256'],frames=z['frames']) for z in pm.get('animations',[]) if k in z.get('ids',[])]
  if not r['figure_links'] and r['figures']:
   for name in r['figures'].split(';'):
    cand=[v for v in inv if v['stage']==d.name and Path(v['plot']).name==Path(name).stem]
    for v in cand:r['figure_links'].append(dict(plot=d.name+'/'+v['plot'],png_sha=v['png_sha'],pdf_sha=v['pdf_sha']))
  if not r['figure_links'] and not pm:
   r['source_files'].append(dict(path=str((d/'PHYSICS_RESULTS.json').relative_to(B)),sha=sha(d/'PHYSICS_RESULTS.json'),status='VERIFIED'))
  refs[k].append(r)
 stages[i]=stage
 # Compare historical scope row with final overlay; R55/56 have legacy wording differences.
 for x in er:
  if not old[x['id']].get(f'R{i}_evidence'):issue('overlay_missing_historical_evidence',stage=i,id=x['id'])
for k,x in old.items():
 ev=[i for i in range(55,79) if x.get(f'R{i}_evidence')]
 if set(ev)!={r['stage'] for r in refs[k]}:issue('overlay_evidence_stage_mismatch',id=k,overlay=ev,actual=[r['stage'] for r in refs[k]])
 if k!='L31' and not refs[k]:issue('no_evidence_csv_row',id=k)
assert sum(bool(x) for x in refs.values())==120
# Exact direct source files needed by L31: bind to stage run manifest or existing published result.
inputs={}
for i,f in [(66,'FEEDBACK66.json'),(66,'THERMAL66.json'),(66,'ERASURE66.json'),(69,'PHYSICS_RESULTS.json'),(73,'DECISION_RESULTS73.json'),(74,'PETZ74.json'),(76,'SUMMARY76.json'),(76,'ORIGINAL_CLAIM_AUDIT76.json'),(77,'SUMMARY77.json'),(77,'ORIGINAL_RG_AUDIT77.json'),(78,'GAUGE78.json')]:
 d=next(B.glob(f'mqft_restore{i}_*'));m=read(d/f'RUN_MANIFEST{i}.json');p=d/f;h=sha(p);assert m['files'][f]==h,(i,f)
 inputs[str(p.relative_to(B))]=h
dump(S/'EVIDENCE_LINKS79.json',dict(stages=stages,items=dict(refs),scope='All R55-R78 evidence rows and authenticated direct plot links; older grouped rows may have no unique plot identity; no implication of scientific acceptance.'))
dump(S/'INVENTORY_AUDIT79.json',dict(status='CHECKED_WITH_RECORDED_DISCREPANCIES' if issues else 'PASS_SCOPED',plot_pairs=len(plotchecks),verified_pairs=sum(x['status']=='VERIFIED' for x in plotchecks),plot_checks=plotchecks,stage_publications=len(stages),stage_manifest_checks=manifest_audit,issues=issues,source_spans=len(spans),function_spans=sum(x['source_status']=='EXACT_SOURCE_SPAN' for x in spans),guide_labels=27,guide_literal_matches=sum(bool(x.get('guide_lines')) for x in spans if x['archive']=='guide_only'),summary_inputs=inputs,scope='Byte identity and mapping audit, not all historical raw paths re-executed or plots visually inspected.'))
print('AUDIT INVENTORY',len(plotchecks),sum(x['status']=='VERIFIED' for x in plotchecks),'source',len(spans),'manifest files',len(manifest_audit),'issues',collections.Counter(x['kind'] for x in issues),flush=True)
for x in issues[:25]:print(x,flush=True)
