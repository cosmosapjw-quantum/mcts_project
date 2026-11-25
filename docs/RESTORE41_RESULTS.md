# RESTORE41 results — common MixNet primary
- Original 15 commissioning + 2 matched-leaf uniform-rollout games and
  192 uniform-root pilot histories: diagnostic only after user's correction.
  Rapfi won all 10 asymmetric games; not evidence of schedule effect.
- Primary: same Quartz engine and actual Rapfi mix9svq NNUE policy/WDL
  for BOTH players, five modes, four candidate samples per evaluator
  call, budget 16, two prescribed four-ply openings. 10/10 natural
  terminal games; 539 engine-selected plies, 8624 backups, 9108 parent
  and 34369 child NNUE query requests. 579 plies including opening moves.
  Lengths opening0/opening1: serial 56/37, root 35/23, tree 58/37,
  leaf 45/31, wave 225/32. Full board draw 225 independently checked.
- 40 paired actual fixed-root searches (two p12 positions from the new
  serial games, four replicate seed labels, same model, 16 backups each,
  four samples per eval). 8/8 serial/leaf event action and root mark
  signatures exact. Mean final root H serial/leaf .6402902, root 0,
  tree/wave .9840865 nats; root replica initialization 640 candidate
  samples across 8 searches vs 544 each for other modes. Mean root Q
  .57554348 serial/leaf, .64873306 root, .55735826 tree, .55682276 wave.
  No exact minimax or independent strength target; Q is a model estimate.
  Equal B is not equal sample work, and one host timing sample is not
  a throughput guarantee. No mode winner claim.
- 48 actual MixNet stochastic root searches (two roots, 3 laws, 4
  proposals, 2 seed repeats, 16 backups each). Native path-mixture
  max weight 1.922494815; local mixture max 3.244598574. Median ESS
  of 2: path 1.569843 vs local 1.716517; mean absolute SNIS discrepancy
  from noisy direct arm: path .0761671 vs local .0544334. The finite
  bound held but efficiency advantage not established. Two EOF failures
  preserved under failed_attempts; first failed job exact replay passed,
  both ultimately successful with same binary/plan/model. No causal
  failure diagnosis or full production reliability claim.
- Common MixNet: 98 jobs (10 games, 88 fixed-root searches), 10032
  backups, 3135 global cuts, 3715 positive local cuts and 4005 replica
  cuts including zero. 10628 parent and 40449 child NNUE query requests.
  Raw compressed receipts 14034334 bytes. All six cohorts total 307 jobs,
  32496 backups, 5811 global cuts, 7347 replica cuts; diagnostic and
  primary separated in SUMMARY.json. Old 9x9 data not counted.
- Actual game final replica cuts: finite beta 285, zero 245, infinite
  158, flat 1. No universal continuous annealing curve. Native score
  projection not physical temperature. Max native raw-moment discrepancy
  in common campaigns 1.78e-15; KKT max 2.562e-12, score balance max
  2.22e-16. Rust 383 passed/14 ignored; Python 23/23 incl actual model
  and parity fixtures; release build passed. Same-context review,
  independent code reduction; independent native subagent NOT_RUN.
- Plots: common MixNet game 8, matched root 9, stochastic law 10 figure
  pairs (PNG/PDF). Diagnostic cohorts separate 26 pairs; all manifests
  hashed. Original function/guide catalog 121 rows unchanged. Source
  response panel blocked because source_h=0 only; physical energy,
  calibrated beta, loop coefficient, SBS/phase, Crooks/JE/FDT and
  actual strength not established. No new selector admitted.
- Actual evaluator is the pinned MixNet family from Rapfi, not a
  separately located MixNet checkpoint or a trained MQFT network.
