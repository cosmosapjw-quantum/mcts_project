"""Reproduce the 94-function/27-guide inventory of the frozen linked plan."""
import argparse, ast, csv, hashlib, zipfile
from pathlib import Path
base=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
parser=argparse.ArgumentParser()
parser.add_argument("--omok",type=Path,default=Path.home()/"Dropbox/omoknuni_quantum-main.zip")
parser.add_argument("--quantum",type=Path,default=Path.home()/"quantum.zip")
args=parser.parse_args()
archives={
 "L":(args.omok,"915edcf90ef15a155d1fa1ba520f6d9cefc445e084c4980be8e8679c487124dd"),
 "Q":(args.quantum,"5216ad17dc09757bd0b0f2b71669998ab684a251d57761cc2e2740cae18768c9")}
panels={"T":"T_temperature_entropy","A":"A_action_curvature;A_C_paired_source_response","R":"R_budget_and_coarse_flow;R_coarse_graining","C":"C_finite_budget_response;C_graph_correlation;A_C_paired_source_response","D":"D_decision_records;D_repeated_root_information","N":"N_score_balance","X":""}
limits={"T":"Score scale and occupancy only, not physical temperature","A":"Frozen KKT and measured prior-source response, not calibrated quantum loop","R":"Budget and prefix blocks, not autonomous RG","C":"Finite-budget and graph measures; board-size scaling and critical exponent not established","D":"Classical records/MI; native quantum phase and density unidentifiable","N":"Exact discrete score balance, no physical heat/work/reverse","X":"Requires dedicated original experiment or model; not implemented as native plot"}
blocked={"L10","L11","L19","L20","L21","L22","L23","L24","L28","L37","L39","L40","L41","L42","L43","L44","L46","L47","L48","L49","Q03","Q05","Q06","Q08","Q14","Q15","Q19","Q20","Q22","Q24","Q30","Q35","Q37","Q41","Q42","Q43","Q44"}
results=[]
for line in (base/"original_functions.tsv").read_text().splitlines():
    i,f,stem,fn=line.split("|")
    path,h=archives[i[0]]
    assert sha(path.read_bytes())==h
    with zipfile.ZipFile(path) as z:
        found=[]
        for member in z.namelist():
            if member.rsplit("/",1)[-1]!=stem+".py":continue
            raw=z.read(member)
            try:tree=ast.parse(raw.decode("utf-8-sig"))
            except SyntaxError:continue
            for n in ast.walk(tree):
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==fn:found.append((member,raw,n))
        if len(found)!=1:raise ValueError((i,[(x[0],x[2].lineno) for x in found]))
        member,raw,n=found[0]
    fs=f.split(";");p=";".join(dict.fromkeys(x for k in fs for x in panels[k].split(";") if x))
    if i in blocked or not p:status="BLOCKED_ORIGINAL_DEDICATED_INPUT";p=""
    elif "animation" in fn or "report" in fn or "summary" in fn or i in ("L01","L02"):status="PARTIAL_STATIC_REPLACEMENT"
    else:status="PARTIAL_NATIVE_PANEL_PREPARED"
    results.append(dict(id=i,family=f,archive=path.name,archive_sha256=h,member=member,member_sha256=sha(raw),function=fn,start=n.lineno,end=n.end_lineno,panel=p,status=status,limit="; ".join(limits[k] for k in fs)))
for line in (base/"guide_targets.tsv").read_text().splitlines():
    fn,f=line.split("|");i="G"+str(len(results)-93).zfill(2)
    results.append(dict(id=i,family=f,archive="guide_only",archive_sha256="",member=fn+".png",member_sha256="",function="",start="",end="",panel=panels[f] if fn in ("energy_evolution","temperature_evolution","entropy_evolution","thermodynamics_summary","order_parameter","susceptibility","correlation_function","critical_summary","participation_ratio","pointer_states") else "",status="PARTIAL_NATIVE_PANEL_PREPARED" if fn in ("energy_evolution","temperature_evolution","entropy_evolution","thermodynamics_summary","order_parameter","susceptibility","correlation_function","critical_summary","participation_ratio","pointer_states") else "BLOCKED_ORIGINAL_DEDICATED_INPUT",limit=limits[f]))
assert len(results)==121
with (base/"ORIGINAL_TARGET_CATALOG.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=results[0],lineterminator="\n");w.writeheader();w.writerows(results)
from collections import Counter
print("Verified original functions:",94,"guide entries:",27,"status:",dict(Counter(x["status"] for x in results)))
