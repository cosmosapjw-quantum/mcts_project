"""One-to-one original artifact closure, preserving original claim status."""
from base61 import *
Q=read(S/'SUMMARY88.json');P=read(S/'PLOT_MANIFEST88.json');R=read(S/'PERSISTENCE_RANK88.json');A=read(S/'ORIGINAL_SOURCE88.json');C=read(S/'CONVERGENCE88.json')
assert R['status']=='PASS_SCOPED' and R['rank_checks']==6048 and len(P['figures'])==8
assert all(sha(S/f)==h for f,h in P['sources'].items())
details={
'L22':('All three original panels recreated (mean normalized Wilson trace, gauge readout error vs stability, graph V/E) plus joint orientation. True unitary links, all fundamental loops and transported effects.','48 actual legal graph contexts, 4005 holonomies; all rank3. Gauge error 3.11e-15; fixed-effect change is a changed probe.','Declared local SU2 model, no native physical gauge identification.',['GAUGE88.json','INDEPENDENT88.json']),
'L24':('Actual prefix tree and quotient, graph/cubical lower-star filtration, GF2 birth-death pairs and explicit discrete critical thresholds and plateau extrema; original stability panel corrected.','144 contexts, 25648 threshold checks, 6048 inclusion ranks and144 dual-hole checks. Chronological quotient H1 births19; final legal graph H1 sum12064 vs filled531.','Local minimum/maximum plateau is not Lyapunov stability; no continuum Morse index or topological phase.',['TOPOLOGY88.json','INDEPENDENT88.json','PERSISTENCE_RANK88.json']),
'L10':('Actual barrier-height vs all-risk hit rate, include censoring and frozen comparison; original postselected tunneling probability replaced by explicitly labelled empirical risk.','3072 paths;3071 entrants,2783 hits,288 censored,1 nonentry; 16 historical games in RECOVERY88 and coherent reference in TRANSPORT88.','No native tunneling probability or physical propagation; descriptive bins not causal.',['BARRIER88.json','RECOVERY88.json','TRANSPORT88.json']),
'G18':('Signed Q and actual U at H32 with all legal actions, fixed oblique projection of sampled (x,y,U), Q heatmap and minimax path.','96 native representative contexts, source/target fixed at H32, score identity exact. Three predetermined core0 examples.','Matplotlib Axes3D unavailable; static oblique 3D projection, no smooth interpolation or interactive surface. Q=0 for unvisited is initialization.',['LANDSCAPE88.json']),
'G19':('Full actual risk-set survival and same-side recovery, threshold sensitivity, all censored/non-entry rows.','3071 eligible native roots; 16 real historical games, threshold .2:13 entrants,3 recoveries,10 censored,3 nonentry.','Native first-selection clock and same-side game decision clock differ; game-end censoring may be informative.',['BARRIER88.json','RECOVERY88.json']),
'G20':('Explicit nearest-neighbor legal graph, V=-U, minimax excess height and shortest width at sublevel threshold; certificate via Kruskal and below-threshold BFS.','3071 native certificates; all96 rep0 threshold mergers exact, prior scalar and risk features exact.','Score units only, local geometric path not native jump sequence.',['BARRIER88.json','TOPOLOGY88.json']),
'G21':('Explicit declared hopping/energy/units and lead boundary exact scattering; lattice transfer, rectangular and continuum WKB controls, actual path action error.','96 graph references x3 energies; dense solve/expm subset12, max1.26e-14; all prior controls identical.','Local WKB exponent lacks general prefactor/interference; no native physical calibration.',['TRANSPORT88.json']),
'G22':('Integrated actual landscape, risk, barrier geometry, exact lead T, coherent/dephased monitored detector and nonneighbor native hits.','96 model contexts, 3072 native risk records, all controls preserved; four-panel final summary.','Native and model clocks/units/protocols distinct; no physical quantum tunneling inference.',['LANDSCAPE88.json','BARRIER88.json','TRANSPORT88.json','RECOVERY88.json'])
}
items=[]
for c in A['items']:
 id=c['id'];method,result,limit,evidence=details[id];p=next(x for x in P['figures'] if x['ids']==[id]);assert all((S/f).is_file() for f in evidence)
 items.append(dict(id=id,original_source=c['original_source'],acceptance_contract=c['acceptance'],decision='ACCEPTED_CORRECTED_SCOPED_ARTIFACT',method=method,result=result,remaining_limit=limit,figure=p['figure'],png_sha=p['png_sha'],pdf_sha=p['pdf_sha'],evidence={f:sha(S/f) for f in evidence},original_science='NOT_CERTIFIED'))
assert len(items)==8 and len({x['figure'] for x in items})==8
dump(S/'ITEM_RECONSTRUCTION88.json',dict(status='EIGHT_CORRECTED_SCOPED_ARTIFACTS_ACCEPTED',items=items,prior_report_artifacts=['L31'],corrected_artifacts_total=9,original_scientific_certifications=0,scope='R79 item contracts evaluated against corrected finite definitions. Corrected output completeness is distinct from the original physical hypothesis; no blanket final science promotion.'))
lines=['# MQFT original structural output report','RESTORE88 · RESTORE69 scientific authority','Eight item-specific static figures, full data and explicit limitations. Original claims and corrected output artifacts are tracked separately.']
for x in items:
 lines += ['',f"## {x['id']}",f"**Original acceptance:** {x['acceptance_contract']}",f"**Method:** {x['method']}",f"**Result:** {x['result']}",f"**Limit:** {x['remaining_limit']}",f"![{x['id']}](plots/{x['figure']}.png)",f"[PDF](plots/{x['figure']}.pdf) · Data: "+', '.join(x['evidence']),"**Disposition:** ACCEPTED_CORRECTED_SCOPED_ARTIFACT; original physical scientific claim NOT_CERTIFIED."]
(S/'ORIGINAL_OUTPUT_REPORT88.md').write_text('\n\n'.join(lines)+'\n')
C['corrected_report_artifacts_accepted']=['L31']+[x['id'] for x in items];C['new_corrected_artifacts']=8;C['artifact_reconstruction_sha']=sha(S/'ITEM_RECONSTRUCTION88.json')
for c in C['claims']:
 if c['id']=='STRUCT-PERSISTENCE-02':c['evidence']+=['PERSISTENCE_RANK88.json']
dump(S/'CONVERGENCE88.json',C)
print('REPORT',len(items),'total corrected9')
