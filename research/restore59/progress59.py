import csv,json,collections
from pathlib import Path
from predict59 import sha,dump
S=Path(__file__).resolve().parent;B=S.parent;O=B/'mqft_restore58_20261126'
def read(p):return list(csv.DictReader(p.open()))
def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
p=json.loads((O/'PROGRESS58.json').read_text())
assert sha(O/'PLOT_INVENTORY58.csv')==p['inventory_sha256']
inv=read(O/'PLOT_INVENTORY58.csv');assert len(inv)==459
for x in inv:
 base=B/x['stage']/x['plot']
 assert sha(Path(str(base)+'.png'))==x['png_sha'] and sha(Path(str(base)+'.pdf'))==x['pdf_sha']
pm=json.loads((S/'PLOT_MANIFEST59.json').read_text())
for x in pm['figures']:
 base=S/'plots'/x['figure'];assert sha(Path(str(base)+'.png'))==x['png_sha'] and sha(Path(str(base)+'.pdf'))==x['pdf_sha']
 inv.append(dict(stage=S.name,plot='plots/'+x['figure'],png_sha=x['png_sha'],pdf_sha=x['pdf_sha'],png_bytes=Path(str(base)+'.png').stat().st_size,pdf_bytes=Path(str(base)+'.pdf').stat().st_size))
assert len({x['png_sha'] for x in inv})==len(inv)==463
write(S/'PLOT_INVENTORY59.csv',inv)
cat={x['id']:x for x in read(S/'ORIGINAL_TARGET_CATALOG.csv')}
rows=read(O/'ORIGINAL_OUTPUT_PROGRESS58.csv');assert len(rows)==121
scope={
'L39':'Measured finite-wave pooled and root/phase-centered pair correlations; no exponential decay claim.',
'L40':'Prospective nested state/history mark risk and exploratory train/test clock-conditioned categorical prediction; not a distributional Markov sufficiency test.',
'L41':'Exact conditional-mean projection identity and fitted finite-risk comparison; no original bath analytic model.',
'L42':'Eight board/core strata empirical lag-1 correlations, not synthetic distribution.',
'L43':'Measured three-bin wave-mean and residual transition counts plus conditional heldout NLL.',
'L44':'Five measured wave lags and mixture-confound diagnostic; no physical timescale separation.',
'L48':'Scoped report with trace identities, numerical audit and residual limitations; not all original tests.',
'L49':'Scoped heldout calibration, core effects and lag diagnostics, not final certification.'}
e=[]
for x in rows:
 k=x['id'];x['R59_evidence']=scope.get(k,'');x['R59_scope_status']='SCOPED_NATIVE_PANEL' if k in scope else ''
 if k in scope:
  y=cat[k];e.append(dict(id=k,primary_node=x['owner'],evidence_figures=';'.join(f['figure'] for f in pm['figures'] if k in f['ids']),source_member=y['member'],source_sha256=y['member_sha256'],legacy_status=y['status'],current_evidence='SCOPED_NATIVE_PANEL',scope=scope[k]))
assert len(e)==8 and all(x['evidence_figures'] for x in e)
write(S/'ORIGINAL_OUTPUT_EVIDENCE59.csv',e);write(S/'ORIGINAL_OUTPUT_PROGRESS59.csv',rows)
union=[x['id'] for x in rows if any(x.get(f'R{i}_evidence') for i in range(55,60))]
p.update(figure_pairs=len(inv),distinct_png_sha=len(inv),files_paired=2*len(inv),current_pairs=4,prior_pairs=459,original_evidence_sha256=sha(S/'ORIGINAL_OUTPUT_EVIDENCE59.csv'),original_progress_sha256=sha(S/'ORIGINAL_OUTPUT_PROGRESS59.csv'),inventory_sha256=sha(S/'PLOT_INVENTORY59.csv'))
p['stages'][S.name]=4
p['original_output_evidence']=dict(current=8,R55_R59_union=len(union),baseline_blocked_with_scoped_evidence=[k for k in union if cat[k]['status']=='BLOCKED_ORIGINAL_DEDICATED_INPUT'],final_certified=0,scope='Explicit R55-R59 overlays, not exhaustive historical evidence; no automatic promotion.')
dump(S/'PROGRESS59.json',p)
print('VERIFIED',len(inv),'pairs;',len(union),'union; new 8 scoped')
