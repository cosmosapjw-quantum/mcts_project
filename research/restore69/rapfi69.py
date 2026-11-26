from arena68 import *
def one(j):
 p=S/'data'/'rapfi'/(j['id']+'.json.gz');assert not p.exists()
 n=Native(S/'logs'/(j['id']+'.stderr.log'))
 try:r=n.query(j['moves'],j['seed'],'clean',j['budget'])
 finally:n.close()
 tmp=p.with_suffix('.part')
 with gzip.open(tmp,'wt',compresslevel=1) as f:json.dump(dict(input=j,trace=r),f,separators=(',',':'))
 tmp.replace(p)
 return dict(job=j,sha256=sha(p),bytes=p.stat().st_size)
def main():
 reg=read(S/'REGISTRATION.json');assert sha(S/'mqft_train')==reg['binary_sha'] and verify(P/'.external/rapfi')==reg['external'];(S/'data/rapfi').mkdir()
 js=read(S/'plans/rapfi.json');assert sha(S/'plans/rapfi.json')==reg['plans']['rapfi']['sha256']
 with cf.ProcessPoolExecutor(max_workers=4) as pool:out=list(pool.map(one,js))
 dump(S/'RAPFI_RECEIPTS69.json',out);print('RAPFI',len(out),flush=True)
if __name__=='__main__':main()
