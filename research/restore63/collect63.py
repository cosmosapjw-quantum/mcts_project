from audit62 import *
R=B/'mqft_restore62_20261126'
def main():
 reg=read(S/'REGISTRATION.json');old=read(R/'REGISTRATION.json');receipt=read(R/'AUDIT62.json')
 assert sha(S/'mqft_train')==reg['binary_sha']==old['binary_sha'];assert sha(R/'AUDIT62.json')==reg['old_audit_sha']
 assert all(sha(B/v['path'])==v['sha256'] for v in reg['weights'].values())
 args=[];prev={}
 for tag,g in [('response0w4',0),('response3w4',3)]:
  p=R/'plans'/f'{tag}.json';assert sha(p)==old['plans'][tag]['sha256']
  for j in read(p):
   if j['arm']=='z':args.append((tag,j,R/'data'/tag/(j['id']+'.json.gz')))
 for x in receipt['jobs']:
  j=x['job']
  if x['tag'] in ('response0w4','response3w4') and j['arm']=='z':prev[j['id']]=(x['sha256'],x['bytes'])
 nold=len(args);assert nold==1536
 for tag,v in sorted(reg['plans'].items()):
  p=S/'plans'/f'{tag}.json';assert sha(p)==v['sha256'];js=read(p);assert len(js)==v['n'] and len(list((S/'data'/tag).glob('*.gz')))==len(js) and not list((S/'data'/tag).glob('*.part'))
  args.extend((tag,j) for j in js)
 assert len(args)==10752
 tags=sorted({x[0] for x in args});arms=['z','am35','ap35'];rec=[];meta=[];final=[];waves={};streams={}
 # separate matrices arrays from receipts, no JSON encoding numpy vectors
 keys=['C','G','F','D','logw','prior','obs','M','V']
 A={k:[] for k in keys}
 with cf.ProcessPoolExecutor(max_workers=12) as pool:
  for i,x in enumerate(pool.map(audit_one,args,chunksize=8)):
   j=x['job'];tag=x['tag'];u=x.pop('uniforms');sd=j['seed']
   if sd in streams:assert u==streams[sd]
   else:streams[sd]=u
   if i<nold:assert (x['sha256'],x['bytes'])==prev[j['id']]
   r=x.pop('rows')[-1];assert r['k']==128
   for k in keys:A[k].append(x.pop(k) if k in ('M','V') else r[k])
   for k in ('wave_p','wave_y','wave_c','wave_q'):
    v=x.pop(k)
    if tag!='legacy0':waves.setdefault(k,[]).append(v)
   if tag!='legacy0':waves.setdefault('meta',[]).append([i,tags.index(tag),j['generation'],j['core'],j['rep']])
   meta.append([i,tags.index(tag),j['generation'],j['width'],j['core'],j['rep'],arms.index(j['arm'])]);rec.append(x)
   if (i+1)%1024==0:print('AUDIT',i+1,flush=True)
 assert len(rec)==10752
 np.savez_compressed(S/'FINAL63.npz',meta=np.asarray(meta),tags=np.array(tags),arms=np.array(arms),**{k:np.asarray(v) for k,v in A.items()})
 np.savez_compressed(S/'WAVES63.npz',tags=np.array(tags),**{k:np.asarray(v) for k,v in waves.items()})
 dump(S/'RNG_INPUT63.json',[dict(seed=int(k),u=v) for k,v in sorted(streams.items())])
 new=rec[nold:];assert len(new)==9216 and sum(x['job']['budget'] for x in new)==1179648
 dump(S/'AUDIT63.json',dict(n_new=len(new),n_reused=nold,backups_new=sum(x['job']['budget'] for x in new),backups_reused=nold*128,evaluations_new=sum(x['evaluations'] for x in new),raw_gzip_bytes=sum(x['bytes'] for x in new),max_score=max(x['max_score'] for x in rec),max_probability=max(x['max_prob'] for x in rec),max_log=max(x['max_log'] for x in rec),min_margin=min(x['min_margin'] for x in rec),rng_streams=len(streams),jobs=rec,final_sha=sha(S/'FINAL63.npz'),wave_sha=sha(S/'WAVES63.npz')))
 print('AUDIT COMPLETE',len(new),nold,len(streams),flush=True)
if __name__=='__main__':main()
