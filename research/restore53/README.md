Restore exact R52/R53 sibling stages and active repository. Generate
v5 family using generate_vector.py STAGE L (13 or21). Build uniquely
named crate mcts_demo_lXX_vector53 with offline locked Cargo release,
copy to STAGE/runtime/mqft-LXX. Use generated family/LXX/python
PYTHONPATH for plan53.py L, execute53.sh and run53.py L. Plans include
separate h0 pilot, g0/g1 13-arm vector pulses, peripheral baseline,
actual game. Do not regenerate completed data. pilot53.py and
checkpoint53.py audit actual raw shards via R46 RNG helper; run53.py
combines audited cohorts. combine53.py and render53.py use canonical
Python and exact stage layout, fresh output directory. publish53.py
checks parent, raw/derived manifests and original mapping. Exact
work samples and complex states stored in stage archive.
