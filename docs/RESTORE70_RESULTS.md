# RESTORE70 결과

실제 adaptive marked path W 및 통계적 유효작용. 기존 R69의 원본 목표 유지.

288 fresh fixed-root searches: 256 trained g0/g3, 32 pinned Rapfi NNUE. 36,864 backups; same 16 root blocks with independent new RNG. 576 raw histories verified including R69. R65 49,152 waves reused for count closure. No games/training.

| evaluator | frozen NLL | adaptive IID | exact adaptive | reduced history | mark RMSE clock/state/history |
|---|---:|---:|---:|---:|---|
| g0 + | 2.02932 | 1.02768 | 1.02170 | 1.28454 | 0.64948/0.63669/0.46997 |
| g0 - | 2.34228 | 1.14666 | 1.14065 | 1.42454 | 0.65638/0.62638/0.47628 |
| g3 + | 3.21092 | 1.33747 | 1.32633 | 1.61864 | 0.57483/0.45438/0.29149 |
| g3 - | 3.37877 | 1.38289 | 1.36959 | 1.65725 | 0.58224/0.43828/0.29208 |
| Rapfi NNUE | 0.63419 | 0.41324 | 0.41324 | 0.55786 | 0.82900/0.96769/0.38099 |

| M | worst Gaussian relative error | worst corrected relative error | improvements |
|---:|---:|---:|---:|
| 8 | 0.0613157 | 0.0308983 | 12/15 |
| 16 | 0.02479 | 0.00106408 | 13/15 |
| 32 | 0.0120206 | 0.000259228 | 14/15 |
| 64 | 0.00591793 | 6.15195e-05 | 15/15 |
| 128 | 0.00293642 | 1.50022e-05 | 15/15 |
| 256 | 0.00146304 | 3.70535e-06 | 15/15 |

All 20 pointwise core-block martingale intervals contain 1; not simultaneous inference. Separate cumulant and direct-sum check max error 1.51e-11; analytic binomial max log mass error 4.09e-14. At J=(50,-50,25) minimum ESS 1.00016 and min determinant 1.28e-21. Repeat marked W max difference 0.2594 (g0+): empirical source law not population certification. Reduced count closure remains worse than conditional exact law; mark predictor not joint transition.

13 existing DAG nodes, 553 distinct retained PNG/PDF pairs (+10); R55-R70 union 90/121, final-certified 0. R71 plus R70-P and R72 pending. See THEORY.md, NEXT_DYNAMICS.md and DRIFT_WATCHDOG.md.
