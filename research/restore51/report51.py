from pathlib import Path
import json,statistics as st,collections,math
S=Path(__file__).resolve().parent;r=json.loads((S/'PHYSICS_RESULTS.json').read_text())
counts=[v for a in r['counts'].values() for v in a.values()]
keys=('jobs','searches','backups','parent_queries','child_queries','candidate_samples','rng_root','rng_samples','cuts','exact_prefix_pairs','exact_prefix_draws','runtime_retries','compressed_bytes')
tot={k:sum(x.get(k,0) for x in counts) for k in keys};errs={k:max(x.get(k,0) for x in counts) for k in ('max_root_u_error','max_sample_log_error','max_score_error','max_ratio_balance')}
assert tot['jobs']==1719 and tot['rng_root']==1716*16+168*16 and tot['exact_prefix_pairs']==672 and tot['exact_prefix_draws']==7872
a=[]
def emit(*x):a.append(' '.join(str(v) for v in x))
def f(x):return f'{x:.8g}'
def means(v,ks):return '/'.join(f(st.mean(x[k] for x in v)) for k in ks)
emit('# RESTORE51: measured native and conditional physical results')
emit('Parent 3ac8d282abdb15231e7c7cc8050c4bf48f0ab824; exact RESTORE49')
emit('L13/17/21 common actual MixNet, original 13 nodes/121 output IDs.')
emit('Pilot selection from h0 only, no treatment outcome used. Three')
emit('new central templates x two peripheral variants x three sizes,')
emit('approximate .06 density; not 18 independent central geometries.')
emit('Pilot108, confirm432, B16 pulse1008, B32 pulse168; three games.')
emit('No engine changes; prior 4020 marked histories not recounted.')
emit('\n## Actual totals and independent replay')
emit(tot);emit('max residuals',errs)
emit('Cumulative marked histories',4020+r['fresh_histories'],'fresh',r['fresh_histories'],'unique seed groups',r['unique_seed_groups'])
emit('Analysis bytes',sum(p.stat().st_size for p in (S/'analysis').rglob('*') if p.is_file()))
emit('Games',[(g['L'],g['status'],g['plies'],g['searched'],g['winner']) for g in r['games']])
emit('All receipts verified. Same-evaluator retries retained; root cause unresolved.')
emit('\n## Independent source calibration')
for L in (13,17,21):
    z=[x for x in r['diagnostics'] if x['campaign']=='confirm' and x['L']==L]
    emit(L,'hits/expected',sum(x['hits']['total'] for x in z),f(sum(x['expected_hits']['total'] for x in z)),'pilot/confirm Fisher',f(st.mean(x['pilot']['pilot_F'] for x in z)),f(st.mean(x['Fisher']['mean'] for x in z)))
emit('Pulse baselines, equal law groups, independent confirmation:')
for L in (13,21):
    for c in ('pulse_high','pulse_medium','pulse_rare','pulse32_high'):
        z=[x for x in r['diagnostics'] if x['L']==L and x['campaign']==c]
        emit(L,c,'hits/expected/F per history',sum(x['hits']['total'] for x in z),f(sum(x['expected_hits']['total'] for x in z)),f(st.mean(x['Fisher']['mean'] for x in z)),'no-hit paths',sum(x['no_hit_paths'] for x in z))
        t=[x for x in r['temporal'] if x['L']==L and x['campaign']==c]
        for block in sorted({x['block'] for x in t}):
            v=[x for x in t if x['block']==block]
            emit(' block',block,'target LR/direct/diffSE',f(st.mean(x['values'][1][block][4] for x in v)),f(st.mean(x['values'][2][block][4] for x in v)),f(math.sqrt(sum(x['se'][3][block][4]**2 for x in v))/3))
emit('672 exact middle/late pre-intervention prefixes, 7872 prior selections.')
emit('All future-source direct past responses zero. LR vs direct finite contrasts')
emit('not equal to zero-field derivatives. Sparse target and modest n remain.')
emit('\n## Spatial covariance, rank and heldout prediction')
for L in (13,17,21):
    v=[x for x in r['summary'] if x['L']==L];emit(L,'entropy/trace C/S_low/central mass',means(v,('H','cov_trace','S_low','central_mass')))
v=[x['cov_trace'] for x in r['peripheral_variants']]
emit('27 within-core peripheral contrasts: mean absolute/max',f(st.mean(abs(x) for x in v)),f(max(abs(x) for x in v)))
emit('All S0 exactly zero, Parseval and fixed-total response checked.')
for key in ('actions','actions_n16'):
    v=r[key];emit(key,'train rank',dict(collections.Counter(x['train_rank'] for x in v)))
    for lam in (.01,.001,.0001):
        z=[y for x in v for y in x['fits'] if y['ridge']==lam]
        emit(' ridge',lam,'NLL minus isotropic',f(st.mean(x['delta'] for x in z)),'improved',sum(x['delta']<0 for x in z),'/',len(z))
    z=[y for x in v for y in x['output_tilts']]
    emit(' output tilt zero/quadratic/empirical',means(z,('zero','quadratic','empirical')))
emit('No general predictive-action admission from rank or invertibility.')
emit('54 exact shared-snapshot restricted five-action references, 162 lattice cases.')
for M in (8,16,32):
    v=[y for x in r['loops'] for y in x['values'] if y['M']==M]
    emit('M',M,'median leading/corrected',f(st.median(x['leading_error'] for x in v)),f(st.median(x['one_loop_error'] for x in v)),'improved',sum(x['one_loop_error']<x['leading_error'] for x in v),'/',len(v))
emit('DP residual',f(max(x['dp_residual'] for a in r['loops'] for x in a['values'])))
emit('Exact sample noise decomposition residual',f(max(x['residual'] for x in r['noise'])))
emit('Score covariance bound min eigenvalue',f(min(x['min_schur_eigenvalue'] for x in r['score_bounds'])))
for l in ('iid','pl','reserve'):
    z=[x for x in r['score_bounds'] if x['law']==l];emit(l,'empirical score variance/bracket',means(z,('score_variance','mean_bracket')),'target/mark normalized squared response',f(st.mean(x['response_fraction'][3] for x in z)),f(st.mean(x['response_fraction'][4] for x in z)))
    z=[x for x in r['memory'] if x['law']==l]
    for mode in ('state','clock','memory','clock_memory'):
        emit(l,mode,'heldout sector/target/mark',means([x['models'][mode] for x in z],('sector','target','mark')))
emit('Leave-one-central-template-out, three geometry folds, not 18 independent geometries.')
emit('Preserve negative sector and positive mark scoped evidence, no universal closure.')
emit('\n## Conditional quantum and continuous thermal physics')
for t in (0.,math.pi/2,math.pi):
    z=[x for x in r['quantum'] if x['theta']==t]
    emit('theta',t,'MI/negativity/future gap',means(z,('MI','negativity','future_gap')),'positive MI',sum(x['MI']>1e-12 for x in z),'/',len(z))
emit('1296 exact conditional cases; 432 arrays of two conditional E states.')
emit('Matrix/analytic max probability residual',f(max(x['residual'] for x in r['quantum'])))
emit('Environment dephased/reset destroys memory in declared model; compatible')
emit('classical hidden-variable models not ruled out, no native phase identification.')
for T in (16.,64.,256.):
    z=[x for x in r['thermal'] if x['tau']==T]
    emit('tau',T,'work/sigma/lag/friction/tauW',means(z,('work','sigma','lag','friction','tau_work')),'mean relative gap',f(st.mean(x['relative_gap'] for x in z)))
emit('162 continuous reset ODE cases; 9 independent Radau checks max',f(max(x['max_error'] for x in r['thermal_check'])))
emit('Energy/entropy/mass residual',f(max(x['residual'] for x in r['thermal'])))
emit('Slow-driving metric on declared Gibbs model, not native heat or temperature.')
emit('\n## Scope')
emit('Nine grouped PNG/PDF figures. Same 13 nodes,13 edges and 121 output')
emit('owners. Prior negative evidence and full original obligations preserved.')
emit('Native 1PI, criticality, physical phase/beta/hbar and objectivity open.')
emit('Independent native subagent NOT_RUN; actual independent numerical')
emit('and RNG replay checks. No selector readmission. Next absolute')
emit('information threshold and multi-action sector-field intervention.')
(S/'RESULTS_KO.md').write_text('\n'.join(a)+'\n')
print('\n'.join(a))
