from base61 import *
import gzip,shutil
p=S/'SCALE92.json';q=S/'SCALE92.json.gz'
assert p.is_file() and not q.exists()
with p.open('rb') as f,q.open('xb') as o:
 with gzip.GzipFile(filename='',fileobj=o,mode='wb',mtime=0,compresslevel=4) as z:shutil.copyfileobj(f,z,4<<20)
h=hashlib.sha256()
with gzip.open(q,'rb') as f:
 for c in iter(lambda:f.read(4<<20),b''):h.update(c)
a=sha(p);assert a==h.hexdigest()
dump(S/'SCALE_PACKING92.json',dict(uncompressed_bytes=p.stat().st_size,uncompressed_sha256=a,compressed_bytes=q.stat().st_size,compressed_sha256=sha(q),verified_decompressed_sha256=h.hexdigest()))
p.unlink()
