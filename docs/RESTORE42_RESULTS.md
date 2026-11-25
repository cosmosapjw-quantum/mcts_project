# RESTORE42: common MixNet color-swapped arena and source response

- Fixed source and runtime: parent RESTORE41 main 1ef5edc49979d0a8ffba34f405b5dc9afdd628f0;
  v2 native binary SHA 6fef50c0d3b935c6c0bec7cfddbadd6a1a830a813ef945c44cedf423123bf40f.
  Same actual pinned Rapfi 250615 mix9svq NNUE policy/WDL, four sampled one-step
  candidates per evaluation on both sides and all modes. No uniform rollout or
  external full-alpha-beta opponent in scientific data.
- 16/16 naturally completed games, 8 color-swapped opening pairs at B=32,
  4 D4-distinct opening boards overall; balanced incomplete-block design
  (root/tree on two openings, leaf/wave on the other two). One seed per
  block. Different modes therefore cannot be directly ranked on common
  four-opening support; two blocks per mode, not a powered population test.
  828 plies including 64 prescribed opening plies, 764 actual searches,
  24448 completed backups. One natural full-board draw at 225 moves.
- Candidate paired scores (each two color-swapped games averaged):
  root [0.5,0.5], tree [0,0.25], leaf [0.5,0.5], wave [0.5,0.5].
  Tree's observed score .125 (0 wins,1 draw,3 losses) is a pilot signal,
  not a rejection or estimate of general strength. Root/leaf/wave each
  2 wins and 2 losses, not evidence of population equivalence.
  Leaf's two pairs exactly reproduce whole game moves and every path/mark
  signature across swapped modes, 33/29 searched plies (62 matched pairs,
  1984 corresponding backups); exact candidate and serial NNUE request
  counts 9384 each. Concurrent campaign timing ratio serial/leaf mean
  per-search wall time 2.082, not an isolated speed benchmark.
  Requests per backup candidate vs serial: root 5.2563/4.7666;
  tree 4.8697/4.8057; leaf 4.7298/4.7298; wave 4.4666/4.3876.
  Candidate root per-backup overhead ~10.3%. No equal-query strength
  claim. Pilot pair SD zero for three modes is NOT evidence of zero
  population variance. Tree sample SD .17678 implies illustrative normal
  25 pairs for absolute .10 score change, alpha .05 two-sided, power .8,
  but only two blocks and no reliable powered sample-size estimate.
- Source-response: 48 completed B=64 searches, 3072 backups, actual four
  D4-distinct prior-game p12 roots (from only two older opening clusters),
  serial and wave4, h=-.2,0,+.2 normalized prior source, 2 replicates.
  16 paired finite contrasts; zero/frozen/adaptive Q diagnostic MAE:
  serial .278320/.230344/.179563; wave .424805/.287289/.246765.
  Two negative and one zero serial responses; no negative wave. Frozen
  predicted response nonnegative all 16; 2 active-face changes serial.
  Maximum absolute Q field shift .653011 serial and .447027 wave.
  Adaptive diagnostic consumes perturbed Q and is not held-out prediction.
  Finite difference at one source magnitude not differential limit.
  Fixed Q prior-tilt target occupancy monotone (derived); actual negative
  response refutes interpreting this reference as exact adaptive law.
  No 1PI or one-loop claim; no new selector admission.
- Combined actual 64 jobs, 812 searches, 27520 backups, 4920 global cuts,
  5947 positive local cuts, 6342 replica cuts including zeros, 27885
  parent and 104938 child NNUE requests; 111540 candidate samples
  including exact terminal evaluations. Raw moments independent max
  discrepancy 2.8422e-14, KKT 7.5495e-15, score balance 2.4980e-16.
  All receipts and analysis/figure manifests checked. Rust 383 passed,
  14 ignored; Python 25 passed incl actual two-sided role/response
  integration fixtures. Release build verified. No independent agent.
- Runtime hardening: subprocess reader + 10 s response deadline,
  stderr ring, PID/exit status on query failure, one same-identity
  transport retry, group cleanup. No other model fallback. Zero nonempty
  diagnostics/transport failures or retries observed across this campaign.
  Earlier EOF root cause and failure branch full stress validation remain
  unresolved; zero events in finite data is not a proof of reliability.
- Existing renderer: arena_A 8, arena_B 8, source 10 PNG/PDF pairs (26).
  Six original observable axes covered as finite native diagnostics.
  No stochastic path-law data in this run, no native quantum phase,
  physical temperature, thermal fluctuation relation, true one-loop
  or universal critical exponent established.
