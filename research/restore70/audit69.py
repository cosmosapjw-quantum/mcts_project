from base61 import *
from audit65 import audit_one,rho,free,effects,dagger
from audit68 import audit_search
import gzip
def native(j):
 tag='phase'+str(j['generation']);v=audit_one((tag,j));u=v.pop('uniforms');v.pop('rows');v.pop('waves');v.pop('counts');v.pop('marks');v.pop('prior')
 p=S/'data'/tag/(j['id']+'.json.gz')
 with gzip.open(p,'rt') as f:d=json.load(f)
 es=d['trace']['events'];a,b=j['sources'];out=[]
 for start in range(0,128,4):
  q=es[start:start+4];ph=q[0]['root_selection']['law']['phase65'];assert ph['wave']==start//4
  m=ph['pair_mass'];r=ph['population_a'];p0=np.array([m*ph['e0'],m*(1-ph['e0']),1-m]);p1=np.array([m*ph['e1'],m*(1-ph['e1']),1-m])
  y=np.zeros(3,int)
  for e in q:y[0 if e['actions'][0]==a else 1 if e['actions'][0]==b else 2]+=1
  out.append([*ph['prepared'],m,r,ph['rotation'],*y,*p0,*p1])
 return v,u,out
def rapfi(x):
 j=x['job'];p=S/'data/rapfi'/(j['id']+'.json.gz');assert sha(p)==x['sha256']
 with gzip.open(p,'rt') as f:d=json.load(f)
 assert d['input']==j;r=d['trace'];q=audit_search(r,j['moves'],'clean',j['seed'],128)
 a,b=j['sources'];es=r['events'];out=[];lastgap=0.
 for start in range(0,128,4):
  e=es[start]['root_selection']['law'];p=np.asarray(e['p']);acts=e['actions'];ia=acts.index(a);ib=acts.index(b);m=float(p[ia]+p[ib]);r0=float(p[ia]/m);phi=6*lastgap/32 if start else 0.
  lastgap=float(e['scores'][ia])-float(e['scores'][ib]);out.append([m,r0,phi])
 return dict(job=j,sha256=x['sha256'],bytes=x['bytes'],**q),[e['root_selection']['law']['u'] for e in es],out
def main():
 reg=read(S/'REGISTRATION.json');assert sha(S/'mqft_train')==reg['binary_sha']
 for x in reg['weights'].values():assert sha(B/x['path'])==x['sha256']
 jobs=[j for g in (0,3) for j in read(S/f'plans/phase{g}.json')]
 assert len(jobs)==256
 for tag in ('phase0','phase3','rapfi'):
  assert sha(S/f'plans/{tag}.json')==reg['plans'][tag]['sha256'] and len(list((S/'data'/tag).glob('*.gz')))==reg['plans'][tag]['n']
 with cf.ProcessPoolExecutor(max_workers=12) as p:n=list(p.map(native,jobs,chunksize=4));rr=list(p.map(rapfi,read(S/'RAPFI_RECEIPTS69.json'),chunksize=2))
 v=[x[0] for x in n];w=[x[0] for x in rr];streams={}
 for x,u,_ in n+rr:
  sd=x['job']['seed']
  if sd in streams:assert streams[sd]==u
  else:streams[sd]=u
 dump(S/'RNG_INPUT69.json',[dict(seed=int(k),u=v) for k,v in sorted(streams.items())])
 np.savez_compressed(S/'NATIVE_RECORDS69.npz',meta=np.array([[j['generation'],j['core'],j['rep'],int(j['arm']=='persist_minus')] for j in jobs]),native=np.array([x[2] for x in n]),rapfi=np.array([x[2] for x in rr]),rapfi_meta=np.array([[x['job']['core'],x['job']['rep']] for x in w]))
 a=dict(native_paths=len(n),rapfi_paths=len(rr),backups=128*(len(n)+len(rr)),evals=sum(x['evaluations'] for x in v)+sum(x['evals'] for x in w),raw_bytes=sum(x['bytes'] for x in v+w),max_probability=max(x['max_probability'] for x in v+w),max_score=max(x['max_score'] for x in v+w),max_state=max(x['max_state'] for x in v),max_log=max(x['max_log'] for x in v+w),streams=len(streams),array_sha=sha(S/'NATIVE_RECORDS69.npz'),native=v,rapfi=w)
 dump(S/'AUDIT69.json',a);print('AUDIT',{k:v for k,v in a.items() if k not in ('native','rapfi')},flush=True)
if __name__=='__main__':main()
