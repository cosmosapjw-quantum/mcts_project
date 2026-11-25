# MQFT: MCTS statistical field-theory measurement package

Research-only migration of the original MQFT / mcts_study project, RESTORE47 (grouped Layer A/Q physics; explicit evaluator identity).
**Rust Quartz backend + Python analysis frontend**, with Rapfi as a separate external opponent.
This repository's main branch is now dedicated to MQFT. Legacy implementation
remains in Git history (previous main eb19e5f4ca9a502d645ccab45a8dcfbcc733f537).
No self-play campaign is launched by installation, import, planning, tests or plotting.

## Scope and evidence

- Original goals Q28/Q30, T/A/R/D/N/C; 13-node research DAG retained.
- Actual 15x15 freestyle Gomoku, native Quartz PUCT, full legal support.
- Serial control and root / shared-tree / leaf / synchronous wave parallelism.
- Fixed deterministic seed mapping; async scheduling is **not** promised bitwise deterministic.
- One backend subprocess per campaign worker, independent game and matched-root blocks.
- Rapfi 250615 source and hash-pinned runtime installer, separate Piskvork process.
- No Python/C++ legacy MCTS used. Primary scientific experiments use the same
  actual pinned Rapfi MixNet NNUE policy/WDL evaluator on both sides and every
  schedule. Uniform rollout remains a diagnostic/legacy option only.
- Real N, W, S2, path/event/evaluation accounting; quiescent cuts, pending VL zero checks.
- Python analysis and original-target PNG/PDF renderers. Missing physical inputs
  cause a blocked status, not invented results.
- Conditional complex-density and finite-bath model calculations are separated
  from native classical observables.

See [theory](docs/THEORY.md), [drift watchdog](docs/DRIFT_WATCHDOG.md),
[original target catalog](docs/ORIGINAL_TARGET_CATALOG.csv),
[stage ledger](docs/STAGE_LEDGER.md), [DAG](docs/DAG.json),
[provenance](docs/PROVENANCE.json), and [next work](docs/NEXT_DYNAMICS.md).

## Install (Linux / CPU)

    cargo build --manifest-path backend/Cargo.toml --locked --release
    python3 -m venv .venv
    . .venv/bin/activate
    pip install -e .
    mqft inspect --backend backend/target/release/mcts_demo

Existing local cargo cache can use --offline. For tests:

    cargo test --manifest-path backend/Cargo.toml --locked --lib mqft::tests
    MQFT_TEST_BACKEND="$PWD/backend/target/release/mcts_demo" python -m unittest discover -s tests -v

Tests use small mathematical examples and fixed-board in-memory/temporary
software fixtures; no study dataset or completed game is produced.

### External opponent

Obtain the pinned Rapfi-engine.7z release archive from the source release:
https://github.com/dotb1ue/rapfi/releases/tag/250615

    mqft install-rapfi /path/to/Rapfi-engine.7z .external/rapfi
    mqft check-rapfi .external/rapfi

The SHA-256 is enforced before unpacking; the executable, config and freestyle
NNUE weights are rehashed afterward. Requires 7z and AVX2 on x86-64 Linux.
Source snapshot and original license are under third_party/rapfi-source.
The prebuilt release and weights are **not** committed to Git; install locally.
The source snapshot and pinned executable have separate identities; no claim
of reproducing the executable byte-for-byte from source. The three upstream
submodule IDs are recorded in PROVENANCE.json; fetch them from upstream if
building from source. Rapfi is a strong external proxy, not exact minimax or
a source of MCTS internal statistics.

## RESTORE47 scale-dependent physics

Four new complete common-MixNet games and 240 marked-law histories
(B32/64, h=.1/.2), empirical joint Legendre curvature, nonstationary
count/mark projection and held-out memory, declared CP process tomography
and cyclic thermal path sum. See [theory](docs/RESTORE47_THEORY.md) and
[results](docs/RESTORE47_RESULTS.md). Low replication, no native physical
phase or 1PI admission.

## RESTORE46 precision replication and memory dynamics

720 additional common MixNet histories balance eight roots at 16 repeats
per law/source arm. An independent Rust RNG replay verifies root uniforms
and evaluator selections. Finite discrete projected memory and conditional
causal-break probe use actual drive records. See [theory](docs/RESTORE46_THEORY.md)
and [results](docs/RESTORE46_RESULTS.md). No production backend change.

## RESTORE45 grouped multivariate action, path response and quantum probes

Same 13 research nodes, 121 original output subtasks. Fresh common MixNet
marked-law pilot, leave-one-out score, exact lattice determinant, nonthermal
reversed-word path entropy, declared operational CP qubit probes and CQ
records. See [theory](docs/RESTORE45_THEORY.md) and [results](docs/RESTORE45_RESULTS.md).
Full precision replication not yet launched.

## RESTORE44 original physics and compressed DAG

The user-supplied RESTORE32 theory distinguishes native marked laws (Layer A)
and explicitly declared complex states/CP dynamics (Layer Q).
[Grouped plot ownership](docs/PLOT_GROUPS.json) maps all 121 original
output IDs to the existing 13 nodes, without adding one node per plot.
See [theory](docs/RESTORE44_THEORY.md), [results](docs/RESTORE44_RESULTS.md),
[legacy crosswalk](docs/PLOT_GROUP_CROSSWALK.csv).
Nine new scoped multi-panel figures, no physical identification from
classical logs. Data and exact complex states in verified local backup.

## RESTORE43 crossed arena and prospective response

configs/restore43_cross_A.json and cross_B fill the missing B32 cells.
The prospective.py model uses only baseline features for predictions, with
training coefficients frozen from RESTORE42. Held-out Q-field closure
was not admitted. cost_root compares Root B61/B64 with Serial B64 at four
actual roots, yielding exact 325 logical requests for Root61/Serial64.
See [results](docs/RESTORE43_RESULTS.md), [theory](docs/RESTORE43_THEORY.md).

## Common MixNet matched-arena and source-response workflow (RESTORE42)

Use configs/restore42_arena_A.json and restore42_arena_B.json for a
color-swapped Serial-vs-candidate arena with the same MixNet on both sides.
Use configs/restore42_source.json for actual game-derived paired prior
source perturbations. The four opening blocks are incomplete-crossed
(two per mode); do not rank modes from that pilot. Runtime v2 adds bounded
Rapfi response timeout, diagnostics and one same-identity transport retry.
See [results](docs/RESTORE42_RESULTS.md) and [theory](docs/RESTORE42_THEORY.md).

## Common MixNet campaign workflow (RESTORE41)

Use configs/mixnet_games.json for same-evaluator self-play. The actual Rapfi
mix9svq policy scores (ScoreScale=32) and rounded black-view WDL are read
through persistent local processes. No fallback or uniform terminal rollout.
Four NNUE-guided one-step samples per eval, same model and rule set on both
sides. This is Quartz search with MixNet evaluation, not Rapfi full alpha-beta.
Two observed process EOF failures were preserved; same-identity remaining jobs
completed. Large-study runtime reliability is still open. See
[RESTORE41 results](docs/RESTORE41_RESULTS.md) and
[next experiment](docs/NEXT_DYNAMICS.md).

## Legacy diagnostic campaign workflow

    mqft plan configs/commissioning_games.json runs/commissioning-plan.json

Planning writes only a preregistered metadata grid and upper search/ply counts.
Review disk cap, CPU limits, search budget, evaluator, openings and opponents.
Default templates are experimental design choices, not population-power claims.
For a later explicitly authorized run only:

    mqft generate runs/commissioning-plan.json data/commissioning \
      --backend backend/target/release/mcts_demo --rapfi .external/rapfi --execute
    mqft analyze runs/commissioning-plan.json data/commissioning runs/commissioning-analysis
    mqft plot runs/commissioning-analysis runs/commissioning-plots

Generation requires explicit --execute; no implicit generation. A game reaching
natural terminal has a distinct status from a truncated game. All root search
cuts, evaluator IDs, actual backup marks and legality are audited. Each gzip
JSONL shard is independently published with SHA-256 receipt; partial shards
are not silently counted. A process lock and immutable run identity prevent
concurrent coordinators. Restart the same plan only after inspecting failures.
Storage ceiling is conservative per-shard; set adequate remaining quota and
disk space. High-volume storage/throughput and real Rapfi games are not yet
benchmarked; commissioning is mandatory before broad campaigns.

- commissioning_games: 15 games planned, 32 backups, one fixed opening.
- pilot_games: 240 planned games, 128/512 backups, 2 openings, 2 c values.
- large_games.template: long games and parameter grid; deliberately not run,
  requires separate disk sizing and independent-game power design.
- roots.template: independent repeats of example fixed positions and paired
  prior source +/-h, 0. Replace example positions with a frozen actual-game
  held-out root list before inference. Root repetitions are not whole games.

Equal backup count is not equal evaluation/sample/wall cost: leaf mode uses
multiple rollouts per backup; root mode has multiple initialization evaluations.
Report physical rollout samples, rollout plies, evaluator calls and wall time
separately. For causal leaf-parallel comparison, add a serial control with
matching samples_per_eval. Root replicas are not collapsed into a fictitious
shared score or global chronology.

## Opt-in causal root laws and pre-study checks

See [three repair passes](docs/RESTORE40_AUDIT.md) and [framework audit](docs/FRAMEWORK_MATRIX.md).
Default is unchanged baseline greedy PUCT with repaired statistics.
For opt-in use configs/root_laws.template.json (planned only); IID, PL and
reservation laws, P/Q/local/path proposals, actual root select/backup lock
chronology. Only serial, leaf and wave are supported for stochastic root law.
Record rollout actions with record_rollouts=true. Older v1 traces/receipts
are not interchangeable with v2. Use mqft closure ANALYSIS SPLIT OUTPUT for
fixed-protocol position-disjoint memory ablation; see closure.py for split
fields. Use mqft preflight PLAN OUTPUT --backend EXE --rapfi DIR for software
fixtures and Rapfi protocol, not research data. Full-game commissioning and
sample size remain to be established.

## Conditional model panels

    mqft model-plot path/to/declared_model.json output-dir

Supply a contract with schema "mqft.declared_density.v1", model, measure, basis,
provenance, npz (relative path), sha256, dims, and states (list of key and
parameter pairs referring to complex NPZ matrices). Optional bath has energies,
beta, theta (list), and energy_unit, for a two-factor system-memory state.
No density is synthesized from classical occupancy. This produces separate
declared-model QMI/Holevo/basis-MI and, with the bath, finite-reset heat and
entropy balance plots; these are not physical native quantum observations.

## Licenses

Quartz extracted code: original MIT terms in backend/LICENSE and
LICENSE-QUARTZ-MIT. Rapfi source: GPL-3.0, copyright and original license in
third_party/rapfi-source/Copying.txt; independent external process, not linked
into the Rust engine. No relicense of original research material is asserted.
