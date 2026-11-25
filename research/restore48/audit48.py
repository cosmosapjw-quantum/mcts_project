"""Bounded independent audit and reusable per-cohort checkpoint."""
import json,sys,subprocess
from pathlib import Path
from mqft.client import sha256
from mqft.campaign import atomic_json
from raw47 import cohort
S=Path(__file__).resolve().parent;prev=S.parent/'mqft_restore46_20261126'
reg=json.loads((S/'REGISTRATION.json').read_text());e=prev/'runtime/mqft-rng-replay'
assert sha256(e)==json.loads((prev/'PHYSICS_RESULTS.json').read_text())['rng_replay_sha256']
for name in sys.argv[1:]:
 p=subprocess.Popen([str(e)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
 try:
  c,a,g=cohort(name,reg,p);assert not g
  p.stdin.close();assert p.wait(timeout=15)==0
 finally:
  if p.poll() is None:p.kill();p.wait()
 q=S/'audit';q.mkdir(exist_ok=True)
 atomic_json(q/(name+'.json'),dict(name=name,counts=c,native=a,helper_sha=sha256(e),plan_sha=sha256(S/'plans'/f'{name}.json'),manifest_sha=sha256(S/'analysis'/name/'manifest.json'),report_sha=sha256(next((S/'data'/name).glob('report-*.json'))),reducer_sha=sha256(S/'raw47.py')))
 print('AUDITED',name,c['jobs'],c['backups'],flush=True)
