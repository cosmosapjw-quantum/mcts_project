# RESTORE43: actual crossed arena, prospective response, request-cost control
Run started 2026-11-25 KST and crossed local midnight 2026-11-26;
folder naming follows the start date. Parent main a8f9976ee1e53de0a47d5fe5ffa493ceffeb57dc.
All scientific jobs use the unchanged, previously verified MixNet v2 binary
SHA 6fef50c0d3b935c6c0bec7cfddbadd6a1a830a813ef945c44cedf423123bf40f,
actual pinned Rapfi mix9svq, same 15x15 freestyle Quartz on both sides,
four candidate samples per evaluator; no uniform rollout.

## Crossed game pilot
16/16 new natural-terminal games, 693 plies including openings, 629 searches
and 20128 backups; complete the missing B32 cells of the RESTORE42 design.
Previous 16 games reused via verified prior summary, not recounted as new.
Now four D4-distinct chosen opening boards x four candidates x two colors
=32 natural games, 16 pairs; same per-opening seed labels across modes
verified in registration. Pair scores root [.5,.5,.5,.5], tree [0,.25,.5,.5],
leaf [.5,.5,.5,.5], wave [.5,.5,.5,0] (order may differ by opening).
Candidate W/D/L: root 4/0/4, tree 2/1/5, leaf 4/0/4, wave 3/0/5.
Mean pair scores .5,.3125,.5,.375 respectively. Four chosen blocks,
one seed each; no population strength, ranking or Elo.
New leaf exact same color-swapped complete trajectory and all path/mark
signatures on both blocks (47 and 48 searched plies); prior two blocks
62 searched plies; cumulative 157 matched searches, 5024 backups per arm,
identical logical query counts. This is a powerful implementation control,
not universal equality. Concurrent timing ratio serial/leaf 2.0343,
not isolated benchmarking. Candidate vs serial requests/backup:
root 5.13359/4.61975, tree 4.81865/4.77058, leaf 4.41600/4.41600,
wave 4.61580/4.54688. Root equal-B extra work is real.
Tree sample block SD .23936 -> illustrative 45 pairs for .10 absolute
score shift alpha .05 power .8; wave .25 -> 50. Four selected blocks
insufficient for confirmatory sample-size claim. Zero sample SD for
root/leaf is not zero population uncertainty.

## Prospective source model
FROZEN_MODEL.json SHA 69ab27a408de323f409afceb4e03acb7c2434bae709fafa675bbee15fb7da29f:
trained on prior 16 actual RESTORE42 contrasts only, two coefficients/mode
and separate scalar gain with fixed ridge .05. Model fit predates new
generation; four D4-disjoint roots from two opening clusters absent
from training; p12,p24 of two actual leaf arena games, source chosen
by maximal recorded prior, no new outcome-based selection.
104/104 new actual searches, 7424 backups, B32/128 h=+/- .2,0;
B64 h=+/- .1,.2,.4,0; 40 paired finite contrasts (20 each mode).
MAE serial zero/frozen/gain/prospective/observed-Q:
.341796875/.277732630/.319919369/.275525831/.084668075.
Wave .177734375/.091894625/.092024506/.178495683/.181533570.
Q-field derivative weighted MAE zero/prospective serial .085408838/.085664837,
wave .085551022/.103546116. Two negative serial responses; seven zero
serial and seven zero wave. Prospective closure NOT_ADMITTED: weak serial
improvement, wave deterioration, field error worsens, no retuning.
At B64 alone 24 contrasts: serial zero/frozen/gain/prospective .369466/.331097/.351357/.330639;
wave .159505/.074431/.076066/.189722. Budget comparison at fixed h=.2
must be distinguished from aggregation of 3 h values at B64.
No physical one-loop inference. Actual observed-Q substitution can help
serial while worsening wave; it is a diagnostic, not prospective.
Separate actual field prediction and finite-search residual in SUMMARY.

## Exact realized request calibration
Eight root searches at B61 and B64, four fixed held-out roots.
Registered initial replica compensation C=5(B+R-T)-terminal child samples
before generation; no terminal events/samples in these particular runs.
All four root61 have 325 successful logical requests and 260 samples,
identical to existing serial64 h0 baselines; root64 has 340 and 272.
This is actual query matching, not a query-matched full-game strength test.
Seed depends on B, so B61 does not share the B64 label. Transport
attempts and wall cost are not included in successful logical request count.

## Validation
128/128 new registered jobs, 16 games +112 root searches, 741 native
searches, 28052 backups, 4558 global cuts, 5362 positive local cuts, 5662
replica cuts, 28344 parent +103107 child logical NNUE requests, 113376
candidate samples. Max raw moment error 7.1054e-14, KKT 7.1054e-15,
score-balance 2.8796e-16. Rust 385 passed/14 ignored, Python 26 passed,
two protocol-only Rust fault fixtures (EOF exit19 stderr, semantic error
classification); actual run still unchanged v2 binary. Two real successful
transport retries; logs contain success but no corresponding failure
details, so startup/error provenance remains incomplete, root cause unknown.
No failed or missing jobs. No independent native subagent.
Original figures: cross_A 8, cross_B 8, source_h 9, source_scale 9 pairs;
four new discriminator pairs, total 38 PNG/PDF pairs (76 files).
Original target catalog remains 121 rows, not all original claims solved.
