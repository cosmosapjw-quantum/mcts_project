import json
from pathlib import Path
from predict59 import sha,dump
S=Path(__file__).resolve().parent; O=S.parent/'mqft_restore58_20261126'
p=json.loads((S/'PROGRESS59.json').read_text());r=json.loads((S/'PHYSICS_RESULTS.json').read_text())
dag=json.loads((O/'DAG.json').read_text())
old_nodes=dag['nodes'];old_edges=dag['edges']
dag.update(revision='RESTORE59_20261126',scope='Native adaptive conditional mark prediction; prospective new-root test; scoped Markovian visualizer and training compatibility triage',previous_entry=dag['entry'],entry='X-MEMORY',supersedes_next=dag['next_stage_id'],next_stage_id='MQFT-RESTORE60-ITERATIVE-SELFPLAY-TRAINING')
ev=dict(parent_dag_sha256=sha(O/'DAG.json'),status='scoped_predictive_memory_diagnostic_negative_history_increment',fresh_games=0,fresh_marked_histories=64,total_marked_histories=9624,fresh_backups=2048,training_histories=180,training_marks=5760,heldout_sizes=[13,21],new_core_geometries=4,original_evidence_items=8,original_evidence_union_R55_R59=p['original_output_evidence']['R55_R59_union'],state_mse=r['mse'][3],history_mse=r['mse'][4],history_improvement=r['mean_improvement'],history_gain='not established',native_subagent='NOT_RUN',training_bridge='incompatible canonical evaluator interface; iterative training NOT_RUN',next_contract='docs/NEXT_DYNAMICS.md',prior_admissions='preserved',additional_plot_nodes=0,original_plot_tasks=121,figures=4,retained_pairs=463)
for f in ['MODEL59.json','PHYSICS_RESULTS.json','DIAGNOSTICS59.json','ORIGINAL_OUTPUT_EVIDENCE59.csv','ORIGINAL_OUTPUT_PROGRESS59.csv','PLOT_INVENTORY59.csv','PROGRESS59.json','PLOT_MANIFEST59.json','REGISTRATION_L13.json','REGISTRATION_L21.json','TRAIN_MANIFEST.json','HELDOUT_MANIFEST.json']:
 ev[f+'_sha256']=sha(S/f)
dag['restore59']=ev
assert dag['nodes']==old_nodes and dag['edges']==old_edges and len(old_nodes)==len(old_edges)==13
dump(S/'DAG.json',dag)
prov=json.loads((O/'PROVENANCE.json').read_text());prov['restore59']=dict(ev,dag_sha256=sha(S/'DAG.json'),parent='15bbe2a124ec7146eec65d93c2bc1f6e77ca4830');dump(S/'PROVENANCE.json',prov)
print('DAG',sha(S/'DAG.json'))
