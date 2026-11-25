These scripts were run from the RESTORE44 stage directory with the
installed mqft package and its exact data/analysis/registration files.
Copy into that restored stage directory (not run from here). The
group_dag.py script is an audit of the transition from parent
RESTORE43; rerun only against that parent DAG in a temporary location.
Do not overwrite existing data; render_groups.py expects absent output
folder. PHYSICS_RESULTS and state NPZ, raw records and plot manifests
are archived locally with checksums, not published as large raw data.
