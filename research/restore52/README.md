Restore exact sibling stages R49/R51/R52 and active repo. Generate
separate group family via generate_sector.py STAGE L (13 or21).
Build uniquely named crate mcts_demo_lXX_sector52 with offline locked
Cargo release build; copy to STAGE/runtime/mqft-LXX and use the
matching STAGE/family/LXX/python PYTHONPATH. plan52.py L reads sealed
R51 confirmation data, gates actual native source semantics, registers
new seed plans. execute52.sh generates/analyses only new registered
plans. Existing verified raw data must not be regenerated. run52.py L
audits with R46 RNG helper. combine52.py combines two sealed sizes and
conditional Q/N; render52.py expects absent grouped figure directory.
publish52.py is scoped to exact R51 parent and validates artifacts.
