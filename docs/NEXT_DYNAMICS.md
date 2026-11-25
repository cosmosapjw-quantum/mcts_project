# RESTORE41 commissioning handoff

No full game or scientific generation in RESTORE40. The seven previous
checklist items map to these concrete artifacts:

1. Local clean main, release binary, Python editable venv and exact source/runtime receipt.
2. Rapfi pinned release with BEGIN and BOARD parity software probes.
3. 15-job commissioning plan (configs/commissioning_games.json). Planned only.
4. mqft preflight records per-search backups, rollout plies, calls, compressed bytes,
   wall time and process high-water RSS for five modes plus matched serial. These
   fixed-root software probes are not full-game throughput or resource estimates.
5. configs/commissioning_matched_leaf.json: serial and leaf both samples_per_eval=4.
   Equal B or equal sample-count setting does not ensure equal actual rollout work.
   Report actual costs; use matched opening/block and independent repeats.
6. Use commissioning/pilot actual paired block variance for power and storage.
   mqft.preflight.paired_blocks(sd,effect,alpha,power) is a normal approximation,
   not a claim about adequate N. Freeze train/holdout opening/root lists before
   science; example roots are placeholders only.
7. mqft analyze, plot, closure, model-plot tested in temporary software fixtures.
   Missing-input panels remain blocked. Actual figures follow actual receipted data.

Once user explicitly authorizes full commissioning, using the active repository:

    . .venv/bin/activate
    mqft plan configs/commissioning_games.json runs/commissioning-v2.json
    mqft preflight runs/commissioning-v2.json runs/local-preflight.json --backend .local/bin/mqft-backend --rapfi .external/rapfi
    mqft generate runs/commissioning-v2.json data/commissioning-v2 --backend .local/bin/mqft-backend --rapfi .external/rapfi --execute
    mqft analyze runs/commissioning-v2.json data/commissioning-v2 runs/analysis-v2
    mqft plot runs/analysis-v2 runs/plots-v2

Do not start large templates without measured costs, power design and actual
authorization. Root law template: 192 planned fixed-root jobs, separate from
the 15 commissioning games and 2 matched-leaf self games. Use v2 schema only.
