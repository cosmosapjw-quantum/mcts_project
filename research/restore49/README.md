Size-family reproduction:
- generate_family.py takes an absent stage family/LXX directory and
  L=13,15,17,19,21; it uses git archive of pinned RESTORE48 parent.
- Build each uniquely named crate with cargo build --offline --locked
  --release --bin mcts_demo_lXX, using CARGO_PROFILE_RELEASE_LTO=false
  and CARGO_PROFILE_RELEASE_CODEGEN_UNITS=8. Copy verified binaries
  to stage/runtime_verified/mqft-LXX. Never use first failed cache builds.
- Use generated family/LXX/python as PYTHONPATH for family_gate.py,
  plan49.py, CLI generation/analysis, and run49.py XX. Scripts expect
  the restored stage directory and previous sibling stages; common
  physics49 is loaded from the active repository. Use pinned Rapfi.
- For state fixture, append state_test49.rs only to a test copy of
  generated backend/src/games/gomoku15.rs; cargo test --lib
  size_family_contract49. Legacy other-size tests are not certified.
- Use active repo Python path for run49.py combined, spatial_ref49.py,
  phase49.py, render49.py. Existing output folders must be absent.
- Source, raw, plans, compiled executables, logs, figures and identities
  are preserved in consolidated backup. Do not regenerate verified data.
