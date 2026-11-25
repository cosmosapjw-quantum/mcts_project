"""CLI: only generate --execute may create research data."""
import argparse,json
from .campaign import make_plan,load_plan,read,atomic_json,execute
from .client import Backend
from . import rapfi

def main(argv=None):
    p=argparse.ArgumentParser(prog="mqft")
    s=p.add_subparsers(dest="command",required=True)
    a=s.add_parser("plan");a.add_argument("config");a.add_argument("output")
    a=s.add_parser("generate");a.add_argument("plan");a.add_argument("output");a.add_argument("--backend",required=True);a.add_argument("--rapfi");a.add_argument("--execute",action="store_true")
    a=s.add_parser("analyze");a.add_argument("plan");a.add_argument("data");a.add_argument("output")
    a=s.add_parser("plot");a.add_argument("analysis");a.add_argument("output")
    a=s.add_parser("model-plot");a.add_argument("contract");a.add_argument("output")
    a=s.add_parser("preflight");a.add_argument("plan");a.add_argument("output");a.add_argument("--backend",required=True);a.add_argument("--rapfi",required=True)
    a=s.add_parser("closure");a.add_argument("analysis");a.add_argument("split");a.add_argument("output")
    a=s.add_parser("inspect");a.add_argument("--backend",required=True)
    a=s.add_parser("install-rapfi");a.add_argument("archive");a.add_argument("directory")
    a=s.add_parser("check-rapfi");a.add_argument("directory")
    a=p.parse_args(argv)
    if a.command=="plan":
        r=make_plan(read(a.config));atomic_json(a.output,r)
        r={k:v for k,v in r.items() if k not in ("jobs","config")}|{"jobs":len(r["jobs"])}
    elif a.command=="generate":
        if not a.execute:p.error("generation requires --execute; no data created")
        r=execute(load_plan(a.plan),a.output,a.backend,a.rapfi,authorized=True)
    elif a.command=="analyze":
        from .analysis import analyze
        r=analyze(load_plan(a.plan),a.data,a.output)
    elif a.command=="plot":
        from .plots import render
        r=render(a.analysis,a.output)
    elif a.command=="model-plot":
        from .conditional import render_model
        r=render_model(a.contract,a.output)
    elif a.command=="preflight":
        from .preflight import run
        r=run(a.plan,a.backend,a.rapfi,a.output)
    elif a.command=="closure":
        from .closure import run
        r=run(a.analysis,a.split,a.output)
    elif a.command=="inspect":
        with Backend(a.backend,15) as be:r=be.identity
    elif a.command=="install-rapfi":r=rapfi.install(a.archive,a.directory)
    else:r=rapfi.verify(a.directory)
    print(json.dumps(r,indent=2,sort_keys=True,allow_nan=False))
    return 1 if a.command=="generate" and r["failures"] else 0

if __name__=="__main__":raise SystemExit(main())
