from pathlib import Path
import json,csv,hashlib,zipfile,collections
S=Path(__file__).resolve().parent;B=S.parent;P=B/'mqft_restore39_20261125/repo';O=B/'mqft_restore57_20261126'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
cat={x['id']:x for x in csv.DictReader((S/'ORIGINAL_TARGET_CATALOG.csv').open())};owners=load(S/'PLOT_GROUPS.json')['original_target_owner'];m=load(S/'plots/manifest.json')
guide='omoknuni_quantum-main/docs/quantum_physics/STATISTICAL_MECHANICS_ANALYSIS_GUIDE.md'
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip'}
with zipfile.ZipFile(paths['omoknuni_quantum-main.zip']) as z:
 g=z.read(guide)
gh=hashlib.sha256(g).hexdigest();assert gh=='bfac951744509e0398f58a801855acb355bf9d95754c0ccbe231baf570228db6'
assert b'Energy as expectation of Q-values' in g and b'energy_evolution.png' in g
qguide='omoknuni_quantum-main/docs/quantum_physics/QUANTUM_PHENOMENA_ANALYSIS_GUIDE.md'
with zipfile.ZipFile(paths['omoknuni_quantum-main.zip']) as z:qg=z.read(qguide)
qh=hashlib.sha256(qg).hexdigest();assert qh=='1789cedff042b0d1da1a2368ca2ce3f5dc60f5ec60e68c1834cef18caaa528aa'
scopes={
'G06':('SCOPED_NATIVE_PANEL','Top-two occupancy gap, entropy, central mass at five L and two B; not spontaneous symmetry breaking.'),
'L07':('SCOPED_NATIVE_PANEL','Actual size-dependent finite-budget concentration; critical point not identified.'),
'G07':('SCOPED_NATIVE_PANEL','Whole-history score covariance and exact conditional final-IID-wave finite contrast; no equilibrium FDT.'),
'Q32':('SCOPED_NATIVE_PANEL','Source response/first detection and fluctuation comparison; no physical heat capacity or critical exponent.'),
'G08':('SCOPED_NATIVE_PANEL','Connected open-boundary legal-pair covariance; fixed total zero mode; no physical xi.'),
'Q29':('SCOPED_NATIVE_PANEL','Spatial correlation at five sizes, no temporal scaling or physical xi.'),
'Q31':('SCOPED_NATIVE_PANEL','Finite-size covariance decomposition at two B, three selector scales; no universal exponent or quantum gap.'),
'G09':('SCOPED_NATIVE_PANEL','Size-grid counting/adaptive drift/cross contributions; no critical point.'),
'Q16':('SCOPED_NATIVE_PANEL','Actual finite-size diagnostics; no scaling function or collapse certified.'),
'G18':('SCOPED_NATIVE_PANEL','Actual k8 negative selector score on legal grid and minimax path; 2D map, not original 3D quantum potential.'),
'G19':('SCOPED_NATIVE_PANEL','Actual first-selection detection and right censoring; not quantum tunneling.'),
'G20':('SCOPED_NATIVE_PANEL','Geometric excess barrier with actual selection propensity; not a physical energy barrier.'),
'L10':('SCOPED_NATIVE_PANEL','Barrier and first detection; no WKB or tunneling identification.'),
'L24':('SCOPED_NATIVE_PANEL','Actual k8 score and visited-only Q cubical filtration, H0 intervals and Betti curves; not topological quantum state.'),
'L23':('SCOPED_CLASSICAL_REFERENCE','Offline copied actual root W arrays with isolated single/double numerical fault tests and stored parity; not native live repair or QEC.')
}
assert set(scopes)=={i for x in m['groups'].values() for i in x['original_items']}
rows=[]
for i,(status,scope) in sorted(scopes.items()):
 x=cat[i];nodes=[k for k,v in m['groups'].items() if i in v['original_items']]
 if x['archive'] in paths:
  with zipfile.ZipFile(paths[x['archive']]) as z:raw=z.read(x['member'])
  assert hashlib.sha256(raw).hexdigest()==x['member_sha256'];src=x['member'];h=x['member_sha256']
 else:
  assert x['archive']=='guide_only'
  src,h,gg=(qguide,qh,qg) if i in ('G18','G19','G20') else (guide,gh,g)
  assert x['member'].encode() in gg
 rows.append(dict(id=i,primary_node=owners[i],evidence_figures=';'.join(nodes),source_member=src,source_sha256=h,legacy_status=x['status'],current_evidence=status,scope=scope))
def writecsv(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
writecsv(S/'ORIGINAL_OUTPUT_EVIDENCE58.csv',rows)
prev={x['id']:x for x in csv.DictReader((O/'ORIGINAL_OUTPUT_PROGRESS57.csv').open())}
assert set(prev)==set(cat);cur={x['id']:x for x in rows};ledger=[]
for i,x in sorted(cat.items()):
 a=prev[i].copy();a['R58_evidence']=cur[i]['scope'] if i in cur else ''
 a['R58_scope_status']=cur[i]['current_evidence'] if i in cur else ''
 assert a['final_completion']=='NOT_CERTIFIED'
 ledger.append(a)
writecsv(S/'ORIGINAL_OUTPUT_PROGRESS58.csv',ledger)
# Existing stage-root/plots inventory with exact byte identities.
figs=[];unpaired=[]
for s in sorted(B.glob('mqft_restore*')):
 if not s.is_dir():continue
 for p in sorted(list(s.glob('*.png'))+list((s/'plots').rglob('*.png'))):
  if {'qa','failed_attempts','first_render','analysis_failed'}&set(p.relative_to(s).parts) or p.is_symlink():continue
  q=p.with_suffix('.pdf')
  if not q.is_file():unpaired.append(str(p.relative_to(B)));continue
  figs.append(dict(stage=s.name,plot=str(p.relative_to(s).with_suffix('')),png_sha=sha(p),pdf_sha=sha(q),png_bytes=p.stat().st_size,pdf_bytes=q.stat().st_size))
writecsv(S/'PLOT_INVENTORY58.csv',figs)
h=collections.defaultdict(list)
for x in figs:h[x['png_sha']].append(x['stage']+'/'+x['plot'])
d=load(P/'docs/DAG.json');counts=collections.Counter(x['execution_status'].split('_')[0] for x in d['nodes'])
ids={x['id'] for x in ledger if any(x.get(f'R{k}_evidence') for k in range(55,59))}
out=dict(scope='retained standalone RESTORE stage-root or plots/ paired PNG/PDF; excludes contact/QA, first_render, failed attempts and archive-only',figure_pairs=len(figs),distinct_png_sha=len(h),duplicate_png_occurrences=len(figs)-len(h),files_paired=2*len(figs),current_pairs=sum(x['stage']==S.name for x in figs),prior_pairs=sum(x['stage']!=S.name for x in figs),unpaired_png=unpaired,
         stages=dict(collections.Counter(x['stage'] for x in figs)),identical_png_groups=[v for v in h.values() if len(v)>1],DAG=dict(nodes=len(d['nodes']),edges=len(d['edges']),execution_status_prefix=dict(counts),nodes_with_recorded_evidence=sum(bool(x.get('latest_evidence')) for x in d['nodes']),node_status={x['id']:x['execution_status'] for x in d['nodes']}),
         catalog_status=dict(collections.Counter(x['status'] for x in cat.values())),
         original_output_evidence=dict(current=len(cur),R55_R58_union=len(ids),baseline_blocked_with_scoped_evidence=sorted(i for i in ids if cat[i]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'),native_definition_replotted=sorted(i for i,x in cur.items() if x['current_evidence']=='NATIVE_DEFINITION_REPLOTTED'),conditional_reference_replotted=sorted(i for i,x in cur.items() if x['current_evidence']=='CONDITIONAL_REFERENCE_REPLOTTED'),final_certified=0,scope='explicit R55-R58 item overlays; not exhaustive historical overlay; no automatic promotion'),
         original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE58.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS58.csv'),inventory_sha256=sha(S/'PLOT_INVENTORY58.csv'))
(S/'PROGRESS58.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('stages','identical_png_groups','DAG')},ensure_ascii=False,indent=2))
print('DAG',out['DAG']['execution_status_prefix'])
