"""Independent explicit received-codeword reconstruction of optimized mask-domain decoder."""
from base61 import *
import gzip
def one(x):
 j=x['job'];p=S/'data'/x['tag']/(j['id']+'.json.gz')
 with gzip.open(p,'rt') as f:q=json.load(f)['trace']['fault67']
 for r in q:
  bit=int(math.copysign(1.,r['clean'])<0)
  received=[bit^v for v in r['mask']]
  sy=(received[0]^received[1],received[1]^received[2])
  assert list(sy)==r['syndrome']
  c={(0,0):0,(1,0):1,(1,1):2,(0,1):3}[sy] if j['arm'].startswith('rep') else 0
  if c:received[c-1]^=1
  assert c==r['correction']
  decoded=math.copysign(abs(r['clean']),-1. if received[0] else 1.)
  assert decoded==r['delivered'] and (received[0]^bit)==r['residual']
 return len(q)
def main():
 a=read(S/'AUDIT67.json')
 with cf.ProcessPoolExecutor(max_workers=12) as p:n=sum(p.map(one,a['jobs'],chunksize=8))
 assert n==a['evaluations']
 dump(S/'RECEIVED_CODEWORD67.json',dict(records=n,semantic_equal=True,scope='explicit received sign codewords, parity and decoder reconstructed independently; original optimized Rust mask-domain representation uses exact parity identity, not independent hardware transport'))
 print('RECEIVED CODEWORDS',n)
if __name__=='__main__':main()
