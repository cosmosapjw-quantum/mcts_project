# RESTORE79 R1–R10 / C1–C10
| Stage | Actual work | Evidence |
|---|---|---|
| R1 | R69 authority, R78 parent, L31 nine-panel original | THEORY.md |
| R2 | Source spans, guide labels, prior published results, proper scoring literature | ORIGINAL_SOURCE79.json |
| R3 | Native likelihood vs endpoint projection vs physical T; source/certification types | THEORY.md |
| R4 | Three constructive routes: source/claim audit, L31 report and independent calibration | THEORY.md |
| R5 | Conditional softmax score likelihood, log-partition curvature, endpoint KL projection | THEORY.md, calibrate79.py |
| R6 | Source-linked corrections, finite null/identifiability, work units, archived claims | ITEM_ACCEPTANCE79.json, HONEST_SUMMARY79.json |
| R7 | Wolfram variance, independent scalar refits and endpoint gradient checks | INDEPENDENT79.json |
| R8 | Native/model/negative/partial report dispositions | QUALITY_MATRIX79.csv |
| R9 | Nine panel quantitative report and exact item-specific acceptance | HONEST_SUMMARY79.json |
| R10 | Convergence research routes and stopping conditions | NEXT_DYNAMICS.md |
| C1 | Fixed R78 training and 768 new registered paths | REGISTRATION.json |
| C2 | Native raw law and source inventory schemas | inventory79.py, calibrate79.py |
| C3 | Stable likelihood, exact slope, occupancy projection | calibrate79.py |
| C4 | 12 native/audit workers and 8 Rapfi workers | generate79.py, RUN_SHARDS.json |
| C5 | Actual new 768 paths and reused 192 authenticated paths | AUDIT79.json |
| C6 | Full native path law/cut replay, historic 640 pairs and 781 manifest files | INVENTORY_AUDIT79.json, AUDIT79.json |
| C7 | 9 heldout scalar fits, exact KL and 4608 occupancy fits; all nine panel data | CALIBRATION79.json, HONEST_SUMMARY79.json |
| C8 | Independent 9 scalar refits, 144 raw reductions, gradient checks | INDEPENDENT79.json |
| C9 | 4 unique plot pairs and original 121 per-item acceptance matrix | PLOT_MANIFEST79.json, QUALITY_MATRIX79.csv |
| C10 | Publish DAG, source, results, verified snapshots and cleanup | Publication/backup/cleanup receipts |
Theory contract before generation; R6–R8 conclusions depend on subsequent numerical results. Logical stage coverage rather than independent native subagents or 20 isolated serial executions. Full legacy regression not rerun.
