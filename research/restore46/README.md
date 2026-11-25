Use run46.py in the restored RESTORE46 stage directory, with the sibling
RESTORE44/45 stages and their exact receipts and manifests. Compile
cargo build --manifest-path backend/Cargo.toml --offline --locked
--example mqft_rng_replay, copy resulting helper into runtime/.
Use installed mqft or PYTHONPATH to the checked-out source. Run
run46.py then render46.py (fresh grouped output folder).
Actual generation uses configs/restore46_*.json with the pinned v3
binary and common Rapfi MixNet runtime; do not rerun already verified
histories. Large raw data live only in local consolidated archive.
