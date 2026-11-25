"""Compress legacy target catalog by physical question, preserving exact IDs.
No new plot-per-node expansion; existing 13 nodes retain historical fields."""
from pathlib import Path
import json,csv,hashlib
S=Path(__file__).resolve().parent;P=S.parent/"mqft_restore39_20261125/repo";prev=S.parent/"mqft_restore43_20261125"
def ids(prefix,a,b=None):
 return [f"{prefix}{i:02d}" for i in range(a,(b or a)+1)]
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")
L=lambda a,b=None:ids("L",a,b);Q=lambda a,b=None:ids("Q",a,b);G=lambda a,b=None:ids("G",a,b)
# Unique primary ownership; other relevant plots cross-reference owners, never
# duplicate a scientific execution. Catalog includes report/export/animations.
spec={
"REPLAN-01":(L(2)+L(5)+L(12)+L(31)+L(34),"native data/measurement integrity; combined reports","Actual games, full histories, collision rates; do not mark old comprehensive reports complete."),
"REPLAN-02":(L(3,4)+L(25,26)+L(45)+G(1,3)+G(5)+G(15),"score scale, entropy, concentration and temperature","Actual conditional scale and occupancy fit; physical temperature calibration separate."),
"REPLAN-03":(L(16),"marked path action, effective action and controlled one-loop","Exact restricted-wave count saddle correction; true adaptive 1PI and field loop not solved."),
"REPLAN-04.R":(L(13)+L(29)+L(35,38)+Q(9,13),"RG, coarse action, Fisher and graph flow","Exact pushforward and weighted native graph; autonomous beta functions/fixed points remain open."),
"REPLAN-04.C":(L(1)+L(7)+L(14)+L(17)+L(27,28)+Q(2)+Q(14,18)+Q(29,34)+G(6,9),"finite-size, criticality, susceptibility and correlations","Native response pilot and independent-replica scaling; board-size scaling/critical exponents unresolved."),
"REPLAN-05":(L(9)+L(15)+L(19)+L(30)+Q(24,28)+Q(37)+G(13,14)+G(16,17)+G(23),"decoherence, pointer states, classicality and redundant records","Explicit controlled record state, partial trace and Holevo; native objectivity/SBS/envariance not established."),
"REPLAN-06":(L(6)+L(8)+L(18)+L(32,33)+Q(1)+Q(3,8)+Q(19,23)+Q(38)+G(4)+G(10,12),"nonequilibrium entropy, work, heat and fluctuation relations","Exact conditional Gibbs quench/reset; native likelihood response not physical FDT; calibrated bath unresolved."),
"REPLAN-07":([],"physical predictions for selection versus vanilla PUCT","Previous same-evaluator arena and failed selector/closure remain; no new selector in this round."),
"X-BARRIER":(L(10)+G(18,22),"barrier crossing, tunneling and WKB","Existing conditional models preserved; independent physical phase/dynamics needed, no new data this round."),
"X-REP":(L(11)+L(20,21)+Q(35,36)+Q(39,45)+G(24,27),"complex representation, CP dynamics, effective hbar, entanglement","Explicit Gram states, negativity and dissipator obstructions; physical hbar, Bell intervention and full native quantum map unresolved."),
"X-STRUCT":(L(22,24),"gauge, topology and quantum error correction","Preserved opt-in; specify actual encoding/observable before data or plots."),
"X-MEMORY":(L(39,44)+L(48,49),"Markovity, temporal correlations, memory and closure","Actual history prediction, conditional random-unitary memory and graph projected semigroup."),
"X-TRAIN":(L(46,47),"training traces and resource accounting","Existing artifacts preserved; no new training curve claimed.")
}
# Supplemental crosslinks (not new primary owners).
secondary={"REPLAN-03":["Q21","Q43","G10"],"REPLAN-05":["L20","Q35","Q36","Q42","G24"],"X-REP":["L19","Q24","Q25"],"X-MEMORY":["Q29","Q37","L20"],"REPLAN-06":["G01","G02","G03","G06"],"REPLAN-07":["L01","L12"],"REPLAN-04.C":["Q09","Q10"],"REPLAN-02":["G04"],"X-TRAIN":["L05","L34"]}
fref={"REPLAN-01":["F01"],"REPLAN-02":["F02"],"REPLAN-03":["F03"],"REPLAN-04.R":[],"REPLAN-04.C":[],"REPLAN-05":[],"REPLAN-06":[],"REPLAN-07":[],"X-BARRIER":[],"X-REP":["F04","F07"],"X-STRUCT":[],"X-MEMORY":["F05","F06"],"X-TRAIN":[]}
cat=list(csv.DictReader((S/"ORIGINAL_TARGET_CATALOG.csv").open()));by={r["id"]:r for r in cat};assert len(cat)==len(by)==121
owner={}
for node,(ls,_,_) in spec.items():
 for id in ls:
  assert id in by and id not in owner,(id,node)
  owner[id]=node
assert set(owner)==set(by),set(by)-set(owner)
old=read(prev/"DAG.json");assert len(old["nodes"])==len(old["edges"])==13
assert set(spec)=={n["id"] for n in old["nodes"]}
manifest=read(S/"plots/grouped/manifest.json");assert manifest["source_sha256"]==h(S/"PHYSICS_RESULTS.json")
groups=[]
for n in old["nodes"]:
 node=n["id"];ls,topic,limit=spec[node]
 evidence=manifest["groups"].get(node)
 groups.append(dict(node=node,name=n["name"],topic=topic,original_targets=ls,target_count=len(ls),secondary_references=secondary.get(node,[]),
  restore32_figures=fref[node],current_status="scoped_group_figure" if evidence else "retained_not_newly_executed",
  evidence=evidence,remaining=limit))
assert len({v for g in groups for v in g["restore32_figures"]})==7
mapping=dict(schema="mqft.plot_groups.v1",catalog_sha256=h(S/"ORIGINAL_TARGET_CATALOG.csv"),authority="RESTORE32 user-supplied confirmed core + original Q28/Q30",
 original_target_count=121,source_function_count=94,guide_only_count=27,
 existing_node_count=13,additional_plot_nodes=0,group_count=len(groups),
 scope="Group by scientific question, not number of legacy plots; each original ID has one primary owner; per-item catalog remains authoritative and unchanged.",
 groups=groups,original_target_owner=owner)
save(S/"PLOT_GROUPS.json",mapping)
# Full original rows and scope preserved, not promoted based on one summary figure.
with (S/"PLOT_GROUP_CROSSWALK.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,lineterminator="\n",fieldnames=["id","primary_node","function","member","legacy_status","this_round_scope"]);w.writeheader()
 for r in cat:
  g=next(x for x in groups if x["node"]==owner[r["id"]])
  w.writerow(dict(id=r["id"],primary_node=g["node"],function=r["function"],member=r["member"],legacy_status=r["status"],this_round_scope=g["current_status"]+"; no automatic item completion"))
d=read(P/"docs/DAG.json");assert d==old
d.update(revision="RESTORE44_20261126",scope="RESTORE32 Layer A/Q physics and compressed original-output ownership; 13 retained research nodes, 121 legacy targets, 9 scoped group figures",previous_entry=d["entry"],entry="REPLAN-03",supersedes_next=d.get("next_stage_id"),next_stage_id="MQFT-RESTORE45-GROUPED-PHYSICS-REPLICATION")
d["plot_group_contract"]={"schema":mapping["schema"],"path":"docs/PLOT_GROUPS.json","sha256":h(S/"PLOT_GROUPS.json"),"crosswalk":"docs/PLOT_GROUP_CROSSWALK.csv","legacy_catalog_unchanged":True,"primary_owner_unique":True,"additional_nodes":0,"current_group_figures":len(manifest["groups"])}
r=read(S/"PHYSICS_RESULTS.json")
d["restore44"]={"parent_dag_sha256":h(prev/"DAG.json"),"attachment_sha256":"30e15a087b55810f024117963188308b942c5b10e8563f56efcc1f78b03322ef","status":"scoped_grouped_physics_and_native_marked_law_pilot","new_games":4,"new_fixed_roots":144,"backups":sum(v["backups"] for v in r["counts"].values()),"exact_restricted_cases":len(r["exact"]),"conditional_quantum_cases":len(r["quantum"]),"conditional_thermal_cases":len(r["thermal"]),"group_figures":len(manifest["groups"]),"physics_results_sha256":h(S/"PHYSICS_RESULTS.json"),"physics_states_sha256":h(S/"PHYSICS_STATES.npz"),"prior_admissions":"preserved","remaining":"larger independent histories, native phase/readout identification, controlled adaptive one-loop, board-size scaling","evidence":["docs/RESTORE44_THEORY.md","docs/RESTORE44_RESULTS.md","docs/PLOT_GROUPS.json"],"next_contract":"docs/NEXT_DYNAMICS.md"}
assert d["nodes"]==old["nodes"] and d["edges"]==old["edges"]
save(S/"DAG.json",d)
print("group ownership",[(g["node"],g["target_count"],g["current_status"]) for g in groups])
print("unique assignments",len(owner),"new nodes",0,"original nodes",len(d["nodes"]),"edges",len(d["edges"]))
