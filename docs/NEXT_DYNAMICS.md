# RESTORE42 common MixNet matched pilot
Active source: github cosmosapjw-quantum/mcts_project main, RESTORE41.
Primary backend binary .local/bin/mqft-backend-mixnet-v1, explicit
evaluator rapfi_nnue_lookahead_v1, pinned Rapfi 250615 mix9svq weights.
Never compare a full external alpha-beta engine against uniform rollout
as evidence of parallelization. Do not mix previous diagnostic data.
1. Before larger campaign, record child Rapfi exit status/stderr, query
   timeout and bounded retry/cleanup; two observed EOFs are an open
   runtime issue. No fallback. Keep model/backend identity and raw law.
2. Implement native color-swapped serial-vs-each-mode arena using the
   SAME evaluator and sample count on both sides. Match openings,
   seeds, sample/NNUE-query budgets separately; actual counts and
   terminal/truncated/failed outcomes. No Elo or strength inference
   from same-mode self games.
3. Freeze at least four distinct independent opening blocks for the
   pilot, use B=32,64 (resource-qualified), record actual per-game cost.
   Distinguish sample-matched leaf from root replica initialization.
   Predefine effect size, use independent block paired variance to
   size the subsequent study; current two blocks not a powered test.
4. Fixed actual held-out roots: prior source +/-h vs zero with identical
   MixNet; logit source and causal probability response separate.
   Compare frozen KKT prediction and adaptive Q response, reuse exact
   finite path laws. Increase replicates only after runtime reliability.
5. Reproduce original six-panel outputs with group-level uncertainty.
   Revisit full original action/one-loop, physical temperature,
   quantum dynamics and thermal fluctuation relations only with
   discriminating inputs. No automatic new selector admission.

Current exact reproduction:
    . .venv/bin/activate
    mqft plan configs/mixnet_games.json runs/mixnet-games-v1.json
    mqft generate runs/mixnet-games-v1.json data/mixnet-games-v1 --backend .local/bin/mqft-backend-mixnet-v1 --rapfi .external/rapfi --execute
    mqft analyze runs/mixnet-games-v1.json data/mixnet-games-v1 runs/mixnet-analysis
    mqft plot runs/mixnet-analysis runs/mixnet-plots
This recreates the registered two-opening feasibility design with new
tree scheduling variability. Exact old receipts remain in RESTORE41.
Do not automatically rerun or launch large templates.
