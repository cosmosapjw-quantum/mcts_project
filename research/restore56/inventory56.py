from pathlib import Path
import hashlib,json,csv,collections
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];unpaired=[]
for s in sorted(B.glob('mqft_restore*')):
 if not s.is_dir():continue
 a=list(s.glob('*.png'))+list((s/'plots').rglob('*.png'))
 for p in sorted(a):
  if {'qa','failed_attempts','first_render','analysis_failed'}&set(p.relative_to(s).parts):continue
  if not p.is_file() or p.is_symlink():continue
  q=p.with_suffix('.pdf')
  if not q.is_file():unpaired.append(str(p.relative_to(B)));continue
  rows.append(dict(stage=s.name,plot=str(p.relative_to(s).with_suffix('')),png_sha=sha(p),pdf_sha=sha(q),png_bytes=p.stat().st_size,pdf_bytes=q.stat().st_size))
with (S/'PLOT_INVENTORY56.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
d=json.loads((P/'docs/DAG.json').read_text());cat=list(csv.DictReader((S/'ORIGINAL_TARGET_CATALOG.csv').open()))
c=collections.Counter(x['execution_status'].split('_')[0] for x in d['nodes'])
stages=collections.Counter(x['stage'] for x in rows)
h=collections.defaultdict(list)
for x in rows:h[x['png_sha']].append(x['stage']+'/'+x['plot'])
current=[x for x in rows if x['stage']==S.name]
out=dict(scope='retained standalone RESTORE stage-root or plots/ paired PNG/PDF figures, excluding QA, first_render and failed attempts; not exhaustive archive-only historical generation',
          figure_pairs=len(rows),distinct_png_sha=len(h),duplicate_png_occurrences=len(rows)-len(h),files_paired=2*len(rows),
          current_pairs=len(current),prior_pairs=len(rows)-len(current),current_distinct_png=len(set(x['png_sha'] for x in current)),
          unpaired_png=unpaired,stages=dict(stages),identical_png_groups=[v for v in h.values() if len(v)>1],
          DAG=dict(nodes=len(d['nodes']),edges=len(d['edges']),execution_status_prefix=dict(c),nodes_with_recorded_evidence=sum(bool(x.get('latest_evidence')) for x in d['nodes']),node_status={x['id']:x['execution_status'] for x in d['nodes']}),
          catalog_status=dict(collections.Counter(x['status'] for x in cat)),inventory_sha256=sha(S/'PLOT_INVENTORY56.csv'))
(S/'PROGRESS56.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('stages','identical_png_groups')},ensure_ascii=False,indent=2))
