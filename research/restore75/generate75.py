from base61 import *
from arena68 import Native,verify,P
import gzip,subprocess,os,time,shutil,sys,resource
R=read(S/'REGISTRATION.json')
assert sha(S/'mqft_train')==R['binary_sha'] and verify(P/'.external/rapfi')==R['external']
def rapbatch(js):
 out=[];n=Native(S/'logs'/('rapfi_'+js[0]['id']+'.stderr.log'))
 try:
  for j in js:
   p=S/'data/rapfi'/(j['id']+'.json.gz');assert not p.exists()
   v=n.query(j['moves'],j['seed'],'clean',128,j['sources'][0],j['pulse74']);q=p.with_suffix('.part')
   with gzip.open(q,'wt',compresslevel=1) as f:json.dump(dict(input=j,trace=v),f,separators=(',',':'),allow_nan=False)
   q.replace(p);out.append(dict(job=j,sha256=sha(p),bytes=p.stat().st_size))
 finally:n.close()
 return out
def main():
 t0=time.monotonic();rec=[]
 for g in (0,3):
  tag=f'schedule{g}';dest=S/'data'/tag;dest.mkdir()
  plan=read(S/'plans'/f'{tag}.json');assert len(plan)==1024
  for p in sorted((S/'plans').glob(tag+'_shard*.json')):
   t=time.monotonic();jobs=read(p);sd=S/'data'/(p.stem+'.tmpdir');assert not sd.exists()
   cmd=[str(S/'mqft_pulse74'),'schedule66',str(B/R['weights'][str(g)]['path']),str(p),str(sd),str(8 if p.stem.endswith('shard0') else R['workers'])]
   with (S/'logs'/(p.stem+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
   fs=list(sd.glob('*.json.gz'));assert len(fs)==len(jobs) and {x.name for x in fs}=={j['id']+'.json.gz' for j in jobs}
   size=sum(x.stat().st_size for x in fs)
   for x in fs:x.rename(dest/x.name)
   sd.rmdir()
   rr=dict(tag=tag,shard=p.stem,n=len(jobs),bytes=size,seconds=time.monotonic()-t,plan_sha256=sha(p),argv=cmd)
   rec.append(rr);dump(S/'RUN_SHARDS.json',rec);print(rr,flush=True)
   if p.stem.endswith('shard0'):
    # basic pilot sanity, trace count and input match. No scientific selection on pilot.
    j=jobs[0]
    with gzip.open(dest/(j['id']+'.json.gz'),'rt') as f:q=json.load(f)
    assert q['input']==j and len(q['trace']['events'])==128
    assert shutil.disk_usage(S).free>20*1024**3
  assert len(list(dest.glob('*.json.gz')))==1024
 (S/'data/rapfi').mkdir();jobs=read(S/'plans/rapfi.json');out=[]
 for i in range(0,len(jobs),128):
  t=time.monotonic();chunk=jobs[i:i+128]
  with cf.ProcessPoolExecutor(max_workers=8) as pool:
   for r in pool.map(rapbatch,[chunk[k:k+16] for k in range(0,len(chunk),16)]):out+=r
  dump(S/'RAPFI_RECEIPTS75.json',out);print('RAPFI',len(out),'seconds',time.monotonic()-t,flush=True)
 assert len(out)==1024
 print('COMPLETE',len(out)+2048,'seconds',time.monotonic()-t0,flush=True)
if __name__=='__main__':main()
