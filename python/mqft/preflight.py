"""Local commissioning preflight. Fixed-board software probes only, no games."""
import gzip
import math
import os
import platform
import shutil
import time
from pathlib import Path
from statistics import NormalDist
from .client import Backend,canonical
from .campaign import load_plan,atomic_json
from .analysis import check_trace
from .rapfi import Rapfi,verify

def paired_blocks(sd,effect,alpha=.05,power=.8):
    """Normal approximation for a mean paired contrast; pilot SD required."""
    if not all(math.isfinite(x) for x in (sd,effect,alpha,power)) or sd<=0 or effect<=0 or not 0<alpha<1 or not .5<power<1:
        raise ValueError("pilot SD, nonzero effect and valid alpha/power required")
    z=NormalDist()
    return max(2,math.ceil(((z.inv_cdf(1-alpha/2)+z.inv_cdf(power))*sd/effect)**2))

def run(plan_path,backend,external,output):
    p=load_plan(plan_path);cfg=p["config"]
    cpus=len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else os.cpu_count()
    if cfg["cpu_limit"]>cpus:raise ValueError("CPU quota exceeds affinity")
    dst=Path(output).resolve();dst.parent.mkdir(parents=True,exist_ok=True)
    disk=shutil.disk_usage(dst.parent)
    if p["max_output_bytes"]+256*1024**2>disk.free:raise ValueError("disk quota exceeds available space")
    report=dict(scope="software-only fixed-root fixture; not commissioning or study data",
                plan_hash=p["plan_hash"],planned_jobs=len(p["jobs"]),planned_upper_backups=p["upper_backups"],
                max_output_bytes=p["max_output_bytes"],available_bytes=disk.free,cpu_affinity=cpus,python=platform.python_version(),
                rapfi=verify(external),handshakes=[],fixtures=[])
    # One separate process per schedule: VmHWM is a process high-water mark, not a precise tree allocation.
    moves=[112,113,97,127]
    for mode,workers,samples in (("serial",1,1),("root",4,1),("tree",4,1),("leaf",4,4),("wave",4,1),("serial",1,4)):
        spec=dict(mode=mode,workers=workers,budget=32,cuts=[1,8,32],seed=40001,c_puct=1.4,
                  rollout_depth=225,samples_per_eval=samples,prior="local",source_action=None,source_h=0.,record_rollouts=True)
        with Backend(backend,60) as be:
            start=time.monotonic();tr=be.search(moves,spec);elapsed=time.monotonic()-start
            q=check_trace(tr,moves,spec)["global_rows"][-1]
            peak=next((int(x.split()[1])*1024 for x in Path(f"/proc/{be.proc.pid}/status").read_text().splitlines() if x.startswith("VmHWM:")),None)
            raw=canonical(tr).encode()
            report["backend_sha256"]=be.sha;report["backend_identity"]=be.identity
            report["fixtures"].append(dict(mode=mode,workers=workers,samples_per_eval=samples,
                actual_backups=q["k"],rollout_samples=q["rollout_samples"],rollout_plies=q["rollout_plies"],eval_calls=q["eval_calls"],
                trace_json_bytes=len(raw),trace_gzip_bytes=len(gzip.compress(raw,mtime=0)),search_wall_seconds=elapsed,process_peak_rss_bytes=peak))
    eng=Rapfi(external,100,1,12)
    for moves in ([],[112],[112,113]):
        r=eng.best(moves);report["handshakes"].append(dict(plies=len(moves),action=r["action"],wall_seconds=r["wall_seconds"],verified=True))
    report["limitations"]=["not long-game throughput or storage estimate","no game generated","pilot paired-block variance needed for scientific sample size","backend CPU uniform rollout, not learned evaluator"]
    atomic_json(dst,report);return report
