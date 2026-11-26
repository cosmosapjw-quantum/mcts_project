# RESTORE71 결과
64 신규 geometry, train 48/test 16, 9216 actual searches, 1179648 backup events, 294912 waves. g0/g3 each 4096, Rapfi 1024. 320 full replay, all raw audit. Autonomous model count/mark/next-state, 15 model-family combinations, 63 source evaluations, directional moments to fourth, 8192 synthetic fitted-model simulations per combination.
| evaluator | clock W RMSE | state | state+lag |
|---|---:|---:|---:|
| g0+ | .709051 | .556038 | .416257 |
| g0- | .725297 | .575750 | .438591 |
| g3+ | .606376 | .456998 | .371360 |
| g3- | .624663 | .486050 | .396572 |
| Rapfi | .663086 | .398494 | .344178 |
All families improve W RMSE, but Rapfi joint Brier worsens (.669689 to .689690); 56/63 pointwise fixed-fit block intervals exclude zero for lag W residual. g0+/- 0/63, g3+ 9/63, g3- 12/63. No equivalence claim. Root-prefix RMSE not universally improved; zero joint support remains. No autonomous true-law closure certification. 10 new figure pairs; cumulative 563. 13 existing nodes, 90/121 scoped original item union, 0 final certifications. No additional completed original obligation.
