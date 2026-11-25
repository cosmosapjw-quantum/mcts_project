"""Explicitly planned, resumable, sharded campaigns; no auto-generated measurements."""
import argparse
import concurrent.futures
import gzip
import hashlib
import itertools
import json
import math
import os
import random
import shutil
from pathlib import Path
from .client import Backend, sha256, digest, canonical
from . import rapfi

PLAN_SCHEMA="mqft.campaign.v1"
def read(path):return json.loads(Path(path).read_text())
def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".part")
    with tmp.open("x",encoding="utf-8") as f:
        f.write(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
    tmp.replace(path)
def seed(*parts):
    return int(digest(list(parts))[:16],16)
def win_at(b,a,p):
    y,x=divmod(a,15)
    for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
        n=1
        for sign in (-1,1):
            yy,xx=y+sign*dy,x+sign*dx
            while 0<=yy<15 and 0<=xx<15 and b[yy*15+xx]==p:
                n+=1;yy+=sign*dy;xx+=sign*dx
        if n>=5:return True
    return False
def valid_moves(moves):
    b=[0]*225;terminal=False
    for i,a in enumerate(moves):
        if terminal or type(a)!=int or not 0<=a<225 or b[a]:
            raise ValueError("invalid moves or history past terminal")
        b[a]=1+i%2;terminal=win_at(b,a,b[a])
    return b
def win(b,p):
    return any(v==p and win_at(b,a,p) for a,v in enumerate(b))
def verify_state(state,moves):
    b=valid_moves(moves);w=1 if win(b,1) else -1 if win(b,2) else None
    terminal=bool(w is not None or len(moves)==225)
    if state["board"]!=b or state["side"]!=(1 if len(moves)%2==0 else -1) or state["terminal"]!=terminal or state["winner"]!=w:
        raise ValueError("Rust/Python board, side or terminal mismatch")
    if sorted(state["legal"])!= ([] if terminal else [i for i,v in enumerate(b) if not v]):
        raise ValueError("Rust/Python legal support mismatch")

def make_plan(cfg):
    allowed={"kind","seed","replicates","budgets","c_puct","modes","openings","opponents","positions","sources","rollout_depth","prior","samples_per_eval","move_temperature","max_plies","jobs","cpu_limit","max_output_bytes","search_timeout","rapfi_timeout_ms","rapfi_threads"}
    if set(cfg)-allowed:raise ValueError(f"unknown config keys {sorted(set(cfg)-allowed)}")
    kind=cfg["kind"];reps=int(cfg["replicates"]);base=int(cfg["seed"])
    if kind not in ("games","roots") or not 1<=reps<=100000:raise ValueError("kind/replicates")
    jobs=int(cfg["jobs"]);cpus=int(cfg["cpu_limit"])
    if not 1<=jobs<=cpus<= (os.cpu_count() or 1):raise ValueError("invalid CPU budget")
    max_bytes=int(cfg["max_output_bytes"])
    if max_bytes<=0:raise ValueError("max_output_bytes required")
    max_plies=int(cfg.get("max_plies",225));depth=int(cfg.get("rollout_depth",225));temp=float(cfg.get("move_temperature",1))
    if not 1<=max_plies<=225 or not 1<=depth<=225 or not math.isfinite(temp) or temp<0:raise ValueError("invalid move/rollout limit")
    modes=cfg["modes"];budgets=cfg["budgets"];cs=cfg["c_puct"]
    if not modes or not budgets or not cs:raise ValueError("nonempty parameter grid required")
    for c in cs:
        if not math.isfinite(c) or c<=0:raise ValueError("invalid c_puct")
    rows=[]
    if kind=="games":
        positions=[dict(id=f"opening{i}",moves=m,source_action=None) for i,m in enumerate(cfg.get("openings",[[]]))]
        opponents=cfg.get("opponents",["self"])
        if not opponents or not set(opponents)<={"self","rapfi_black","rapfi_white"}:raise ValueError("invalid opponents")
        sources=[0.]
    else:
        positions=cfg["positions"];opponents=["none"];sources=cfg.get("sources",[0.])
    if not positions or len({p["id"] for p in positions})!=len(positions):raise ValueError("position ids must be distinct")
    for pos in positions:
        b=valid_moves(pos["moves"])
        if win(b,1) or win(b,2) or len(pos["moves"])==225 or (kind=="games" and len(pos["moves"])>=max_plies):
            raise ValueError("opening/root must be nonterminal with future move")
        a=pos.get("source_action")
        if a is not None and (type(a)!=int or not 0<=a<225 or b[a]):raise ValueError("source_action must be legal")
        if any(h!=0 for h in sources) and a is None:raise ValueError("source action must be fixed before perturbation")
    for mode,budget,c,pos,opp,h,rep in itertools.product(modes,budgets,cs,positions,opponents,sources,range(reps)):
        if set(mode)-{"mode","workers","samples_per_eval"}:raise ValueError("unknown mode keys")
        name=mode["mode"];workers=int(mode["workers"]);samples=int(mode.get("samples_per_eval",cfg.get("samples_per_eval",1)))
        if name not in ("serial","root","tree","leaf","wave") or not 1<=workers<=64 or not 1<=samples<=64 or not 1<=budget<=65536 or (name=="root" and workers>budget) or (name=="serial" and workers!=1) or (name=="leaf" and samples<2):
            raise ValueError("invalid schedule")
        if jobs*max(workers,int(cfg.get("rapfi_threads",1)) if opp.startswith("rapfi") else 1)>cpus:raise ValueError("nested CPU oversubscription")
        if not math.isfinite(h) or abs(h)>20:raise ValueError("source_h bound")
        cuts=sorted(set([budget]+[2**i for i in range(budget.bit_length()) if 2**i<budget]))
        # Same opening and replicate are a matched block across schedules, not independent positions.
        block=digest([base,pos["id"],pos["moves"],rep])[:20]
        meta=dict(kind=kind,position_id=pos["id"],opening=pos["moves"],replicate=rep,block=block,opponent=opp,
            max_plies=max_plies,move_temperature=temp,search_timeout=float(cfg.get("search_timeout",180)),
            rapfi_timeout_ms=int(cfg.get("rapfi_timeout_ms",500)),rapfi_threads=int(cfg.get("rapfi_threads",1)))
        spec=dict(mode=name,workers=workers,budget=budget,cuts=cuts,c_puct=c,rollout_depth=depth,samples_per_eval=samples,prior=cfg.get("prior","local"),source_action=pos.get("source_action"),source_h=h)
        job_id=digest([meta,spec,base])[:24]
        # Separate fixed streams; no reliance on Python salted hash. Source +/- share initial seed for paired finite differences.
        spec["seed"]=seed(base,pos["id"],pos["moves"],rep,name,workers,budget,c,opp)
        meta.update(id=job_id,spec=spec,game_seed=seed(base,job_id,"moves"))
        rows.append(meta)
    if not rows or len({x["id"] for x in rows})!=len(rows):raise ValueError("empty/duplicate grid")
    result=dict(schema=PLAN_SCHEMA,kind=kind,config=cfg,jobs=rows,max_parallel_jobs=jobs,max_output_bytes=max_bytes,
        scope="planned_only; no data created by this operation",
        upper_searched_plies=sum((x["max_plies"]-len(x["opening"])) if kind=="games" else 1 for x in rows),
        upper_backups=sum(x["spec"]["budget"]*((x["max_plies"]-len(x["opening"])) if kind=="games" else 1) for x in rows))
    result["plan_hash"]=digest(result)
    return result

def load_plan(path):
    p=read(path);h=p.pop("plan_hash")
    if p["schema"]!=PLAN_SCHEMA or digest(p)!=h:raise ValueError("invalid plan hash")
    p["plan_hash"]=h
    if make_plan(p["config"])!=p:raise ValueError("plan/config mismatch")
    return p

def choose(counts,temperature,rng):
    items=sorted((int(a),int(n)) for a,n in counts if n>0)
    if not items:raise ValueError("empty positive support")
    if temperature==0:
        return max(items,key=lambda x:(x[1],-x[0]))[0]
    logs=[math.log(n)/temperature for _,n in items];mx=max(logs)
    weights=[math.exp(x-mx) for x in logs]
    return rng.choices([a for a,_ in items],weights=weights,k=1)[0]

def run_job(job,outdir,backend,backend_sha,rapfi_dir,plan_hash,byte_cap):
    outdir=Path(outdir);jid=job["id"];path=outdir/f"{jid}.jsonl.gz";rec=outdir/f"{jid}.receipt.json"
    if path.exists() or rec.exists():
        if not path.exists() or not rec.exists():raise FileExistsError(f"incomplete previous publication {jid}")
        r=read(rec)
        if r["plan_hash"]!=plan_hash or r["backend_sha"]!=backend_sha or r["sha256"]!=sha256(path) or r["job_id"]!=jid or r["job_hash"]!=digest(job):raise ValueError("resume identity mismatch")
        return r
    part=path.with_name(path.name+".part")
    if part.exists():raise FileExistsError(f"partial shard {part}; inspect before removing")
    moves=list(job["opening"]);rng=random.Random(job["game_seed"]);count=0;uncompressed=0;backups=0;status="failed"
    with Backend(backend,job["search_timeout"]) as be:
        if be.sha!=backend_sha:raise ValueError("backend bytes changed")
        ext=rapfi.Rapfi(rapfi_dir,job["rapfi_timeout_ms"],job["rapfi_threads"],job["search_timeout"]) if job["opponent"].startswith("rapfi") else None
        with part.open("xb") as raw:
            with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as f:
                def emit(obj):
                    nonlocal uncompressed
                    s=(canonical(obj)+"\n").encode()
                    uncompressed+=len(s)
                    f.write(s);f.flush()
                    if raw.tell()>byte_cap:raise RuntimeError("per-shard compressed byte ceiling reached")
                emit(dict(type="header",schema=PLAN_SCHEMA,plan_hash=plan_hash,job=job,backend_sha=backend_sha,
                          backend=be.identity,external=ext.identity if ext else None))
                st=be.state(moves);verify_state(st,moves)
                if job["kind"]=="roots":
                    tr=be.search(moves,job["spec"])
                    emit(dict(type="root",moves=moves,trace=tr));backups=job["spec"]["budget"];count=1;status="root_complete"
                else:
                    while not st["terminal"] and len(moves)<job["max_plies"]:
                        ply=len(moves)
                        use_ext=ext is not None and ((job["opponent"]=="rapfi_black" and st["side"]==1) or (job["opponent"]=="rapfi_white" and st["side"]==-1))
                        if use_ext:
                            er=ext.best(moves);a=er["action"];row=dict(type="ply",ply=ply,state=st,action=a,engine="rapfi",external=er)
                        else:
                            spec=job["spec"].copy();spec["seed"]=seed(spec["seed"],ply);spec["source_action"]=None;spec["source_h"]=0.
                            tr=be.search(moves,spec);a=choose(tr["move_counts"],job["move_temperature"],rng)
                            row=dict(type="ply",ply=ply,state=st,action=a,engine="quartz",trace=tr)
                            backups+=spec["budget"]
                        if a not in st["legal"]:raise ValueError("chosen action illegal")
                        emit(row);moves.append(a);count+=1;st=be.state(moves);verify_state(st,moves)
                    status="natural_terminal" if st["terminal"] else "truncated"
                emit(dict(type="end",status=status,moves=moves,terminal=st["terminal"],winner=st["winner"] if st["terminal"] else None,
                          searched_records=count,backups=backups))
            raw.flush();os.fsync(raw.fileno())
            if raw.tell()>byte_cap:raise RuntimeError("compressed shard cap at final close")
    part.replace(path)
    receipt=dict(job_id=jid,job_hash=digest(job),plan_hash=plan_hash,backend_sha=backend_sha,status=status,sha256=sha256(path),
                 file=path.name,bytes=path.stat().st_size,uncompressed_bytes=uncompressed,records=count,backups=backups)
    atomic_json(rec,receipt)
    return receipt

def _execute_unlocked(plan,output,backend,rapfi_dir=None,authorized=False):
    if not authorized:raise PermissionError("data generation requires explicit --execute")
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    exe=Path(backend).resolve(strict=True);h=sha256(exe)
    needs_rapfi=any(x["opponent"].startswith("rapfi") for x in plan["jobs"])
    if needs_rapfi:
        if rapfi_dir is None:raise ValueError("external runtime directory required")
        rapfi.verify(rapfi_dir)
    lock={"plan_hash":plan["plan_hash"],"backend_sha":h,"external":rapfi.verify(rapfi_dir) if needs_rapfi else None}
    lp=out/"run_lock.json"
    if lp.exists():
        if read(lp)!=lock:raise ValueError("run lock mismatch")
    else:atomic_json(lp,lock)
    # Conservative hard compressed quota per shard, so concurrent jobs cannot exceed planned total.
    cap=plan["max_output_bytes"]//len(plan["jobs"])
    if cap<1024:raise ValueError("per-shard byte quota too small")
    remaining=sum(not (out/f'{j["id"]}.receipt.json').exists() for j in plan["jobs"])
    if shutil.disk_usage(out).free<remaining*cap+268435456:raise OSError("disk space insufficient for declared remaining quota + 256 MiB reserve; set an appropriate quota/location")
    receipts=[];failures=[]
    # One Python coordinator, one native subprocess per active job. All CPU limits come from plan.
    with concurrent.futures.ThreadPoolExecutor(max_workers=plan["max_parallel_jobs"]) as ex:
        it=iter(plan["jobs"]);pending={}
        def submit(job):
            pending[ex.submit(run_job,job,out,exe,h,rapfi_dir,plan["plan_hash"],cap)]=job["id"]
        for j in itertools.islice(it,plan["max_parallel_jobs"]):submit(j)
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                jid=pending.pop(f)
                try:receipts.append(f.result())
                except Exception as e:failures.append({"job_id":jid,"error":f"{type(e).__name__}: {e}"})
                # Fail closed: finish in-flight bounded jobs but launch no additional jobs.
                if not failures:
                    j=next(it,None)
                    if j is not None:submit(j)
    report=dict(plan_hash=plan["plan_hash"],backend_sha=h,expected=len(plan["jobs"]),completed=sorted(receipts,key=lambda x:x["job_id"]),failures=failures,unattempted=len(plan["jobs"])-len(receipts)-len(failures))
    atomic_json(out/("report-"+digest(report)[:16]+".json"),report)
    return report


def execute(plan,output,backend,rapfi_dir=None,authorized=False):
    if not authorized:raise PermissionError("data generation requires explicit --execute")
    import fcntl
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/".campaign.lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("another coordinator is active") from None
        return _execute_unlocked(plan,out,backend,rapfi_dir,authorized=True)
