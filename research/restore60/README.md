# RESTORE60
Actual 9x9 CPU iterative policy/value self-play through optional canonical Rust MctsEngine evaluator bridge.
Build from this commit using cargo build --release --offline --features mqft-train --bin mqft_train.
The source binary is linked to OpenBLAS; weights are R23 BoardNet little-endian float32 tensors, shape and lineage in registration.
Run scripts in the full local stage context (R23 source, R59 predecessor, data, checkpoint artifacts and plots in cumulative archive). The Git copy is a source/results handoff, not a standalone dataset. Use exact stage registration and correction receipt; do not overwrite the registered plans. See docs/RESTORE60_RESULTS.md and RUN_MANIFEST60.json.
No claim of physical temperature, phase, hbar, quantum advantage, universal strength, 15x15 or original 19-channel ResNet compatibility.
