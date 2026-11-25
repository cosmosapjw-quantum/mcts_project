from run53 import *
L=int(sys.argv[1]);reg=read(S/f'REGISTRATION_L{L}.json');old=B/'mqft_restore46_20261126';exe=old/'runtime/mqft-rng-replay'
assert sha256(exe)==read(old/'PHYSICS_RESULTS.json')['rng_replay_sha256']
p=subprocess.Popen([str(exe)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
try:
 c,a,g=audit(L,'pilot',p,reg);p.stdin.close();assert p.wait(timeout=15)==0
finally:
 if p.poll() is None:p.kill();p.wait()
out=[]
for pos in reg['positions']:
 for law in ('iid','pl','reserve'):
  z=[r for r in a if r['position']==pos['id'] and r['law']==law];assert len(z)==2
  q=vector_responses({(-1,0,0,0):z});out.append(dict(position=pos['id'],law=law,eigen=q['fisher_eigen'],trace=sum(np.diag(q['fisher']))))
atomic_json(S/f'PILOT_L{L}.json',dict(counts=c,groups=out,scope='separate actual h0 pilot; no direction selection or post-treatment tuning'))
print(L,'pilot minimum eigen',min(x['eigen'][0] for x in out),'mean minimum',st.mean(x['eigen'][0] for x in out))
