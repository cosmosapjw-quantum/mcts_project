from base61 import *
import subprocess
p=B/'mqft_restore61_20261126/mqft_trained_rng61'
a=read(S/'AUDIT80.json')
with np.load(S/'MARKED80.npz') as z:u=z['uniforms']
by={}
for i,r in enumerate(a['raw_receipts']):
 if r['cohort']!=1:continue
 sd=r['seed'];v=u[i].tolist()
 if sd in by:assert by[sd]==v
 by[sd]=v
r=[dict(seed=k,u=v) for k,v in sorted(by.items())];dump(S/'RNG_INPUT80.json',r)
q=subprocess.run([str(p),str(S/'RNG_INPUT80.json')],capture_output=True,text=True,timeout=15,check=True);obj=json.loads(q.stdout)
assert obj['streams']==len(r)==768 and obj['independent_uniforms']==128*len(r) and obj['max_error']<1e-12
obj.update(binary_sha=sha(p),input_sha=sha(S/'RNG_INPUT80.json'));dump(S/'RNG_AUDIT80.json',obj);print(obj,flush=True)
