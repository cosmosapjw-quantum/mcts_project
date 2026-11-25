from run52 import *
L=int(sys.argv[1]);reg=read(S/f'REGISTRATION_L{L}.json')
old=B/'mqft_restore46_20261126';exe=old/'runtime/mqft-rng-replay';assert sha256(exe)==read(old/'PHYSICS_RESULTS.json')['rng_replay_sha256']
p=subprocess.Popen([str(exe)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
q=S/'audits';q.mkdir(exist_ok=True)
try:
 for n in sys.argv[2:]:
  stem=f'{n}-L{L}';c,a,g=audit(L,n,p,reg)
  atomic_json(q/f'{stem}.json',dict(counts=c,native=a,games=g,registration_sha=sha256(S/f'REGISTRATION_L{L}.json'),reducer_sha=sha256(S/'run52.py'),report_sha=sha256(next((S/'data'/stem).glob('report-*.json'))),analysis_sha=sha256(S/'analysis'/stem/'manifest.json')))
  print('CHECKPOINT',stem,c['jobs'],flush=True)
 p.stdin.close();assert p.wait(timeout=15)==0
finally:
 if p.poll() is None:p.kill();p.wait()
