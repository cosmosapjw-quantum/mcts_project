# RESTORE51: measured native and conditional physical results
Parent 3ac8d282abdb15231e7c7cc8050c4bf48f0ab824; exact RESTORE49
L13/17/21 common actual MixNet, original 13 nodes/121 output IDs.
Pilot selection from h0 only, no treatment outcome used. Three
new central templates x two peripheral variants x three sizes,
approximate .06 density; not 18 independent central geometries.
Pilot108, confirm432, B16 pulse1008, B32 pulse168; three games.
No engine changes; prior 4020 marked histories not recounted.

## Actual totals and independent replay
{'jobs': 1719, 'searches': 1731, 'backups': 30384, 'parent_queries': 32050, 'child_queries': 116803, 'candidate_samples': 128200, 'rng_root': 30144, 'rng_samples': 128200, 'cuts': 8823, 'exact_prefix_pairs': 672, 'exact_prefix_draws': 7872, 'runtime_retries': 12, 'compressed_bytes': 151167236}
max residuals {'max_root_u_error': 1.1102230246251565e-16, 'max_sample_log_error': 1.7763568394002505e-15, 'max_score_error': 0.0, 'max_ratio_balance': 2.220446049250313e-16}
Cumulative marked histories 5736 fresh 1716 unique seed groups 708
Analysis bytes 375007335
Games [(13, 'natural_terminal', 17, 7, 1), (17, 'natural_terminal', 23, 5, 1), (21, 'natural_terminal', 29, 3, 1)]
All receipts verified. Same-evaluator retries retained; root cause unresolved.

## Independent source calibration
13 hits/expected 77.0 79.421948 pilot/confirm Fisher 0.54706549 0.48393224
17 hits/expected 267.0 235.43777 pilot/confirm Fisher 1.0083173 0.87964258
21 hits/expected 213.0 217.87471 pilot/confirm Fisher 0.81853225 0.87664245
Pulse baselines, equal law groups, independent confirmation:
13 pulse_high hits/expected/F per history 1.0 2.6048276 0.10375944 no-hit paths 23
 block 0 target LR/direct/diffSE 0.0079879381 0.026041667 0.028310312
 block 2 target LR/direct/diffSE 0 0 0
 block 3 target LR/direct/diffSE 0 0.026041667 0.026041667
13 pulse_medium hits/expected/F per history 11.0 6.3637338 0.24507039 no-hit paths 18
 block 0 target LR/direct/diffSE 0.021942454 0.026041667 0.028827212
 block 2 target LR/direct/diffSE 0.031680866 0 0.014566574
 block 3 target LR/direct/diffSE 0.019230669 0.026041667 0.028148972
13 pulse_rare hits/expected/F per history 1.0 1.4618729 0.060659343 no-hit paths 23
 block 0 target LR/direct/diffSE 0 0 0
 block 2 target LR/direct/diffSE 0 0 0
 block 3 target LR/direct/diffSE 0.010552194 0 0.010552196
13 pulse32_high hits/expected/F per history 2.0 2.1334757 0.17335605 no-hit paths 11
 block 0 target LR/direct/diffSE 0 0.10416667 0.10416667
 block 4 target LR/direct/diffSE 0.017330075 0 0.017330082
 block 7 target LR/direct/diffSE 0 0 0
21 pulse_high hits/expected/F per history 126.0 121.03328 2.1381636 no-hit paths 0
 block 0 target LR/direct/diffSE 0.066846243 0.078125 0.046425524
 block 2 target LR/direct/diffSE 0.12849863 0.13020833 0.07501034
 block 3 target LR/direct/diffSE 0.072394868 0.15625 0.060210832
21 pulse_medium hits/expected/F per history 0.0 0.51196347 0.021296682 no-hit paths 24
 block 0 target LR/direct/diffSE 0 0 0
 block 2 target LR/direct/diffSE 0 0 0
 block 3 target LR/direct/diffSE 0 0 0
21 pulse_rare hits/expected/F per history 0.0 0.50324602 0.020934432 no-hit paths 24
 block 0 target LR/direct/diffSE 0 0 0
 block 2 target LR/direct/diffSE 0 0 0
 block 3 target LR/direct/diffSE 0 0 0
21 pulse32_high hits/expected/F per history 106.0 118.55289 4.8474868 no-hit paths 0
 block 0 target LR/direct/diffSE 0.11741179 0.10416667 0.056457096
 block 4 target LR/direct/diffSE 0.15042945 0.10416667 0.12047656
 block 7 target LR/direct/diffSE 0.22252973 0.052083333 0.10373428
672 exact middle/late pre-intervention prefixes, 7872 prior selections.
All future-source direct past responses zero. LR vs direct finite contrasts
not equal to zero-field derivatives. Sparse target and modest n remain.

## Spatial covariance, rank and heldout prediction
13 entropy/trace C/S_low/central mass 2.4423402/0.088270399/0.088655516/0.23828125
17 entropy/trace C/S_low/central mass 2.4144795/0.073792473/0.064273656/0.11935764
21 entropy/trace C/S_low/central mass 2.6040531/0.060368614/0.062696541/0.081163194
27 within-core peripheral contrasts: mean absolute/max 0.01072152 0.031947545
All S0 exactly zero, Parseval and fixed-total response checked.
actions train rank {3: 49, 2: 5}
 ridge 0.01 NLL minus isotropic 0.21135916 improved 23 / 54
 ridge 0.001 NLL minus isotropic 4.7439469 improved 5 / 54
 ridge 0.0001 NLL minus isotropic 21.752199 improved 4 / 54
 output tilt zero/quadratic/empirical 0.15072711/0.15089088/0.1506642
actions_n16 train rank {3: 6}
 ridge 0.01 NLL minus isotropic 0.12483364 improved 2 / 6
 ridge 0.001 NLL minus isotropic 0.73420742 improved 2 / 6
 ridge 0.0001 NLL minus isotropic 0.94853601 improved 1 / 6
 output tilt zero/quadratic/empirical 0.15380921/0.15235815/0.15276096
No general predictive-action admission from rank or invertibility.
54 exact shared-snapshot restricted five-action references, 162 lattice cases.
M 8 median leading/corrected 2.1850445 0.069585868 improved 54 / 54
M 16 median leading/corrected 2.7985885 0.038235167 improved 54 / 54
M 32 median leading/corrected 3.5219236 0.020451805 improved 54 / 54
DP residual 2.7755576e-14
Exact sample noise decomposition residual 7.4267849e-18
Score covariance bound min eigenvalue 0
iid empirical score variance/bracket 1.12256/0.89928832 target/mark normalized squared response 0.67779058 0.2484662
iid state heldout sector/target/mark 0.22400552/0.0408048/0.10089395
iid clock heldout sector/target/mark 0.22436036/0.040162138/0.09730523
iid memory heldout sector/target/mark 0.22889798/0.039221575/0.082405237
iid clock_memory heldout sector/target/mark 0.22836501/0.038831758/0.08110339
pl empirical score variance/bracket 0.46692183/0.58807997 target/mark normalized squared response 0.62192053 0.21892413
pl state heldout sector/target/mark 0.19875149/0.013756184/0.07331421
pl clock heldout sector/target/mark 0.19930901/0.013510356/0.068116487
pl memory heldout sector/target/mark 0.20165134/0.013631514/0.051175027
pl clock_memory heldout sector/target/mark 0.20180486/0.013398909/0.049841631
reserve empirical score variance/bracket 0.84929494/0.75284898 target/mark normalized squared response 0.82312947 0.24291596
reserve state heldout sector/target/mark 0.23712813/0.026103104/0.09012485
reserve clock heldout sector/target/mark 0.2377041/0.025545202/0.085942292
reserve memory heldout sector/target/mark 0.24559748/0.025268019/0.066388083
reserve clock_memory heldout sector/target/mark 0.24484006/0.024842659/0.065435
Leave-one-central-template-out, three geometry folds, not 18 independent geometries.
Preserve negative sector and positive mark scoped evidence, no universal closure.

## Conditional quantum and continuous thermal physics
theta 0.0 MI/negativity/future gap 0.1500329/0.27335434/0.3612263 positive MI 240 / 432
theta 1.5707963267948966 MI/negativity/future gap 8.9768727e-16/0.27335434/7.324388e-17 positive MI 0 / 432
theta 3.141592653589793 MI/negativity/future gap 0.1500329/0.27335434/0.3612263 positive MI 240 / 432
1296 exact conditional cases; 432 arrays of two conditional E states.
Matrix/analytic max probability residual 3.8857806e-16
Environment dephased/reset destroys memory in declared model; compatible
classical hidden-variable models not ruled out, no native phase identification.
tau 16.0 work/sigma/lag/friction/tauW 0.013859389/0.013099962/0.00075942721/0.28006112/0.22175022 mean relative gap -0.22126541
tau 64.0 work/sigma/lag/friction/tauW 0.004149071/0.0041026876/4.6383304e-05/0.28006112/0.26554054 mean relative gap -0.055358846
tau 256.0 work/sigma/lag/friction/tauW 0.0010798584/0.0010769904/2.8680008e-06/0.28006112/0.27644374 mean relative gap -0.013821687
162 continuous reset ODE cases; 9 independent Radau checks max 1.0302037e-12
Energy/entropy/mass residual 5.2180482e-15
Slow-driving metric on declared Gibbs model, not native heat or temperature.

## Scope
Nine grouped PNG/PDF figures. Same 13 nodes,13 edges and 121 output
owners. Prior negative evidence and full original obligations preserved.
Native 1PI, criticality, physical phase/beta/hbar and objectivity open.
Independent native subagent NOT_RUN; actual independent numerical
and RNG replay checks. No selector readmission. Next absolute
information threshold and multi-action sector-field intervention.
