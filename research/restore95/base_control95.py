from entangle95 import *
# lambda=0 is not a tagged record: coherent history amplitudes factor from system.
a=read(S/'ENTANGLE95.json');tree=read(O/'TREE83.json')['contexts']
with np.load(S/'JOINT95.npz') as z:D={k:z[k] for k in z.files}
idx={(int(g),int(c),p,m):i for i,(g,c,p,m) in enumerate(D['meta'])};errors={};out=[]
def ck(k,x,y):
 e=norm(np.asarray(x)-np.asarray(y));errors[k]=max(e,errors.get(k,0));assert e<1e-9,(k,e)
for c in tree:
 for p in PROTOCOLS:
  i=idx[c['family'],c['core'],p,'base81'];weights={'':1.};sys=RHO0.copy()
  for d in range(4):
   J=D['J'+str(d)][i];h=sorted(weights);v=np.sqrt([weights[x] for x in h]);s=float(v@v);v/=np.sqrt(s)
   pure=np.outer(v,v);ck('base_factorization',J,np.kron(pure,sys))
   ck('base_S_record_negativity',neg(J,[d],d+1) if d else 0,0)
   if d:
    row=next(x for x in a['rows'] if (x['family'],x['core'],x['protocol'],x['mode'],x['depth'],x['xi'])==(c['family'],c['core'],p,'base81',d,1.))
    ck('base_success',row['success']/s,1)
    for k,pair in enumerate(row['pairs']):
     if d in pair:ck('base_S_pair_concurrence',row['metrics'][k][3],0)
    out.append(dict(family=c['family'],core=c['core'],protocol=p,depth=d,success=s,record_purity=float(np.trace(pure@pure)),max_RR_CHSH=max([row['metrics'][k][5] for k,pair in enumerate(row['pairs']) if d not in pair] or [0.])))
   if d==3:break
   nw={}
   for h0,w in weights.items():
    m,r=c['nodes'][h0]
    for b,x in enumerate([r,1-r]):nw[h0+str(b)]=w*((m*x)**4 if p=='native_repeat' else x)
   weights=nw;axis,lam,theta,eta,erase=schedule('base81',32+4*d);assert lam==0
   sys=free(sys,theta,eta,erase)
assert len(out)==432
dump(S/'BASE_CONTROL95.json',dict(status='PASS_SCOPED',checks=errors,rows=out,scope='lambda=0 on branch settings harvested from plus81 forced histories; coherent amplitude encoding of adaptive probabilities can generate record-record entanglement despite no measurement backaction on S. Not an actual native baseline trajectory ensemble or tagged null.'))
print('BASE CONTROL',len(out),errors,'max RR CHSH',max(x['max_RR_CHSH'] for x in out),flush=True)
