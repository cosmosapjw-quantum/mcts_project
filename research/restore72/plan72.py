from base61 import *
import sys,math
from mqft.campaign import make_plan,atomic_json,valid_moves,win
from mqft.rapfi import verify
L=int(sys.argv[1]);O=B/'mqft_restore49_20261126';P=B/'mqft_restore39_20261125/repo';Ls=[13,15,17,19,21]
r=read(O/f'REGISTRATION_L{L}.json');exe=O/'runtime_verified'/f'mqft-L{L}'
assert sha(exe)==r['backend_sha'] and sha(O/'family'/f'L{L}'/'SOURCE_MANIFEST.json')==r['source_manifest_sha']
assert verify(P/'.external/rapfi')==r['model']
def center(a,L):
 y,x=divmod(a,21);return (y-10+L//2)*L+x-10+L//2
def orbit(m,L):
 a=np.zeros((L,L),np.int8)
 for i,x in enumerate(m):a.ravel()[x]=1+i%2
 return min(x.tobytes().hex() for k in range(4) for x in (np.rot90(a,k),np.fliplr(np.rot90(a,k))))
# Same absolute 8-stone core, exact nesting of peripheral additions across sizes.
# Build full 21-board history with no terminal at ANY prefix; select disjoint D4 orbits.
rng=np.random.default_rng(720001);cores=[];seen=set();src=10*21+11
pool=[y*21+x for y in range(7,14) for x in range(7,14) if y*21+x not in (220,src)]
for i in range(8):
 while True:
  core=[220]+rng.choice(pool,7,replace=False).tolist()
  k=orbit(core,21)
  if k in seen:continue
  extra=[];nprev=8
  for l in Ls:
   n=2*math.floor(.03*l*l+.5);r0=l//2
   prev=0 if l==13 else (l-2)//2
   ring=[y*21+x for y in range(10-r0,11+r0) for x in range(10-r0,11+r0) if max(abs(y-10),abs(x-10))>=(4 if l==13 else prev+1)]
   extra+=rng.choice(ring,n-nprev,replace=False).tolist();nprev=n
  # independent terminal check on 21×21 full history
  b=np.zeros((21,21),int);ok=True
  for t,a in enumerate(core+extra):
   y,x=divmod(a,21);z=1+t%2;b[y,x]=z
   for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
    c=1
    for s in (-1,1):
     yy,xx=y+s*dy,x+s*dx
     while 0<=yy<21 and 0<=xx<21 and b[yy,xx]==z:c+=1;yy+=s*dy;xx+=s*dx
    if c>=5:ok=False
   if not ok:break
  if not ok:continue
  seen.add(k);cores.append(dict(id=i,core=core,extra=extra,orbit21=k));break
positions=[];n=2*math.floor(.03*L*L+.5)
for c in cores:
 for arm in ('fixed','dense'):
  mv=[center(a,L) for a in c['core']+(c['extra'][:n-8] if arm=='dense' else [])]
  b=valid_moves(mv);assert not win(b,1) and not win(b,2) and b[center(src,L)]==0
  positions.append(dict(id=f'c{c["id"]:02d}_{arm}',moves=mv,source_action=center(src,L)))
assert len({orbit(x['moves'],L) for x in positions})==16
base=dict(kind='roots',seed=720100+L,replicates=4,budgets=[32],c_puct=[1.4],sources=[0.],rollout_depth=1,prior='local',evaluator='rapfi_nnue_lookahead_v1',record_rollouts=True,paired_schedule_seed=True,jobs=1,cpu_limit=8,max_output_bytes=2**29,search_timeout=90,max_plies=L*L,positions=positions,modes=[dict(mode='wave',workers=4,samples_per_eval=4,root_law=dict(kind='iid',beta_p=[10/3],beta_q=[10/3],proposal='p',logit_h=0.))])
a=make_plan(base);assert len(a['jobs'])==64
atomic_json(S/'plans'/f'L{L}.json',a)
atomic_json(S/f'REGISTRATION_L{L}.json',dict(L=L,cores=cores,positions=positions,backend_sha=r['backend_sha'],source_manifest_sha=r['source_manifest_sha'],model=r['model'],plan_hash=a['plan_hash'],jobs=len(a['jobs']),density_fixed=8/L**2,density_dense=n/L**2,scope='eight fresh deterministic cores shared across five sizes, nested peripheral stones, 4 seeds per core/arm B32 fixed beta=10/3; prefixes not independent budgets'))
print(L,len(a['jobs']),flush=True)
