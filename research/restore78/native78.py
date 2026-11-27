"""Native D4 raw orientation audit, independent seed difference statistics, actual graph contexts."""
from base61 import *
import gzip,re
from audit_native74 import one as audit9
from audit_rapfi74 import audit_search
from scipy.stats import t as student
FAMILIES=['g0','g3','Rapfi'];CUT=[4,8,16,32,64,128]
def perms(L):
 a=np.arange(L*L).reshape(L,L)
 return np.array([np.argsort(np.rot90(g,k=-k).ravel()) for g in (a,np.fliplr(a)) for k in range(4)])
def tv(a,b):return float(.5*np.sum(abs(a-b)))
def graph(es,m,L,law):
 # actual tree and state quotient: full ordered prefixes vs alternating colored sets.
 nodes={()};edges=set();qn={};qe=set()
 def state(x):
  a=m+list(x);return (tuple(sorted(a[::2])),tuple(sorted(a[1::2])))
 for e in es:
  a=tuple(e['actions'])
  for k in range(1,len(a)+1):
   x,y=a[:k-1],a[:k];nodes.update([x,y]);edges.add((x,y));u,v=state(x),state(y);qn[u]=1;qn[v]=1;qe.add((u,v))
 assert len(edges)==len(nodes)-1 and len(qe)>=len(qn)-1
 v=list(map(int,law['actions']));q=set(v);links=[]
 for i in v:
  y,x=divmod(i,L)
  if x+1<L and i+1 in q:links.append([i,i+1])
  if y+1<L and i+L in q:links.append([i,i+L])
 return dict(L=L,vertices=v,edges=links,score=law['scores'],p=law['p'],tree_vertices=len(nodes),tree_edges=len(edges),tree_cycles=0,quotient_vertices=len(qn),quotient_edges=len(qe),quotient_cycles=len(qe)-len(qn)+1)
def one(arg):
 tag,j=arg;f=S/'data'/tag/(j['id']+'.json.gz')
 with gzip.open(f,'rt') as h:d=json.load(h)
 assert d['input']==j;tr=d['trace'];g=2 if tag=='rapfi' else int(j['generation']==3);L=15 if g==2 else 9
 a=audit_search(tr,j['moves'],'clean',j['seed'],128,j['sources'][0],0.) if g==2 else audit9((tag,j))
 o=int(re.search(r'_o(\d)_',j['id'])[1]);pp=perms(L)[o]
 c={q['k']:q for q in tr['cuts']};es=tr['events'];pr=np.array(c[4]['prior'],float)[pp];pr/=pr.sum()
 v=float(tr['evaluations'][0]['clean' if g==2 else 'value'])
 P=[];W=[];N=[];M=[];U=[];acts=[];paths=[];marks=[]
 for e in es:
  law=e['root_selection']['law'];p=np.zeros(L*L);p[law['actions']]=law['p'];P.append(p[pp])
  acts.append(int(np.argsort(pp)[e['actions'][0]]));paths.append(tuple(map(int,np.argsort(pp)[e['actions']])));marks.append(e['root_value']);U.append(law['u'])
 for k in CUT:
  N.append(np.array(c[k]['n'],float)[pp]/k);W.append(np.array(c[k]['w'],float)[pp]/k)
 # group comparisons use full native board coordinates, not cross-family pooling.
 context=None
 if o==j['rep']==0:
  context=graph(es,j['moves'],L,es[32]['root_selection']['law'])
  context.update(family=g,core=j['core'],source=int(np.argmax(c[4]['prior'])),source_sha=sha(f),source_path=str(f.relative_to(B)),moves=j['moves'])
 rec=dict(path=str(f.relative_to(B)),sha256=sha(f),bytes=f.stat().st_size,seed=j['seed'],family=g,core=j['core'],orientation=o,rep=j['rep'],max_score=a['max_score'],max_probability=a['max_probability'],max_log=a['max_log'])
 return [g,j['core'],o,j['rep']],pr,v,np.array(N),np.array(W),np.array(P)[[0,4,16,32,64,124]],U,acts,paths,marks,context,rec
def ci(x):
 x=np.asarray(x);m=x.mean(0);h=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return [m.tolist(),(m-h).tolist(),(m+h).tolist()]
def main():
 R=read(S/'REGISTRATION.json');args=[]
 for tag in ('schedule0','schedule3','rapfi'):
  p=S/'plans'/f'{tag}.json';assert sha(p)==R['plans'][tag]['sha256'];js=read(p);assert len(js)==512 and {j['id']+'.json.gz' for j in js}=={x.name for x in (S/'data'/tag).glob('*.gz')};args.extend((tag,j) for j in js)
 rows={};ctx=[];receipts=[]
 with cf.ProcessPoolExecutor(max_workers=12) as pool:
  for i,(m,p,v,n,w,q,u,a,paths,z,c,r) in enumerate(pool.map(one,args,chunksize=8)):
   key=tuple(m);assert key not in rows;rows[key]=(p,v,n,w,q,u,a,paths,z);receipts.append(r)
   if c is not None:ctx.append(c)
   if (i+1)%256==0:print('AUDIT',i+1,flush=True)
 assert len(rows)==1536 and len(ctx)==48
 out=[];allroot=[];divs=[]
 for g in range(3):
  roots=[];L=15 if g==2 else 9;P=perms(L);I=np.arange(L*L)
  assert all(any(np.array_equal(x[y],z) for z in P) for x in P for y in P)
  for c in range(16):
   z=[[rows[g,c,o,r] for r in range(4)] for o in range(8)]
   prior=np.array([[a[0] for a in b] for b in z]);value=np.array([[a[1] for a in b] for b in z]);N=np.array([[a[2] for a in b] for b in z]);W=np.array([[a[3] for a in b] for b in z]);q=np.array([[a[4] for a in b] for b in z])
   assert abs(prior-prior[:,0:1]).max()<1e-9 and abs(value-value[:,0:1]).max()<1e-8
   pr=prior[:,0];avg=pr.mean(0)
   # Explicit group average using covariant relabeling of all eight actual orientations.
   err=0.
   for h in range(8):
    # index k for composition P_g ∘ P_h
    ix=[next(k for k in range(8) if np.array_equal(P[i][P[h]],P[k])) for i in range(8)]
    native=np.array([pr[k][np.argsort(P[k])] for k in ix])
    # pull back native f(g h x) by g to transformed frame h x.
    a=np.mean([native[i][P[i]] for i in range(8)],0)
    err=max(err,float(abs(a[P[h]]-avg).max()))
   assert err<1e-12
   d=N[1:]-N[:1];dw=W[1:]-W[:1]
   # Cross-products across independently sampled seeds, paired between orientations.
   unbiased=lambda a:np.mean([np.sum(a[:,i]*a[:,j],axis=-1) for i in range(4) for j in range(4) if i!=j],axis=(0,1))
   ns=np.mean(.5*np.sum(abs(N[1:].mean(1)-N[0].mean(0)),axis=-1),axis=0)
   seedtv=np.mean(.5*np.sum(abs(d),axis=-1),axis=(0,1))
   noise=np.mean(.5*np.sum(abs(N[:,0:2].mean(1)-N[:,2:4].mean(1)),axis=-1),axis=0)
   pair=[]
   for o in range(1,8):
    for r in range(4):
     a,b=z[0][r],z[o][r];t0=next((t for t in range(128) if a[7][t]!=b[7][t] or abs(a[8][t]-b[8][t])>1e-7),128)
     # shared random seed is not permutation-covariant inverse CDF. Check first-wave uniform equality and marginal p equality.
     row=dict(family=g,core=c,orientation=o,rep=r,first_divergence=t0,first_root_action_equal=a[6][0]==b[6][0],uniform0_equal=a[5][0]==b[5][0],first_law_TV=tv(a[4][0],b[4][0]))
     pair.append(row);divs.append(row)
   root=dict(core=c,prior_mean_TV=float(np.mean([tv(pr[0],p) for p in pr[1:]])),prior_max_TV=max(tv(pr[0],p) for p in pr[1:]),value_range=float(np.ptp(value[:,0])),first_law_TV=float(np.mean([tv(q[0,0,0],p[0,0]) for p in q[1:]])),visit_mean_TV=ns.tolist(),paired_visit_TV=seedtv.tolist(),split_noise_TV=noise.tolist(),unbiased_mean_sq_visit=unbiased(d).tolist(),unbiased_mean_sq_mark=unbiased(dw).tolist(),projected_readout_error=err,all_paths_equal=sum(x['first_divergence']==128 for x in pair),first_divergence0=sum(x['first_divergence']==0 for x in pair),uniform0_equal=sum(x['uniform0_equal'] for x in pair),prior_identity=pr[0].tolist(),prior_projected=avg.tolist())
   roots.append(root)
   allroot.append(dict(family=g,**root))
  fields=['prior_mean_TV','value_range','first_law_TV','visit_mean_TV','paired_visit_TV','split_noise_TV','unbiased_mean_sq_visit','unbiased_mean_sq_mark']
  out.append(dict(family=FAMILIES[g],roots=16,comparisons=448,ci={k:ci([a[k] for a in roots]) for k in fields},max_prior_TV=max(a['prior_max_TV'] for a in roots),projected_readout_max_error=max(a['projected_readout_error'] for a in roots),first_divergence0=sum(a['first_divergence0'] for a in roots),all_paths_equal=sum(a['all_paths_equal'] for a in roots),uniform0_equal=sum(a['uniform0_equal'] for a in roots)))
 a=dict(paths=len(rows),backups=128*len(rows),raw_bytes=sum(r['bytes'] for r in receipts),max_score=max(r['max_score'] for r in receipts),max_probability=max(r['max_probability'] for r in receipts),max_log=max(r['max_log'] for r in receipts),raw_receipts=receipts)
 dump(S/'AUDIT78.json',a);dump(S/'NATIVE78.json',dict(scope='raw D4 orientations, seed-paired paths not coordinate-coupled; empirical squared difference U-statistic may be negative; exact symmetrized prior readout not native production change',results=out,roots=allroot,comparisons=divs,contexts=ctx,cuts=CUT))
 print('AUDIT', {k:v for k,v in a.items() if k!='raw_receipts'},flush=True)
 for r in out:print(r['family'],r['ci']['prior_mean_TV'],r['ci']['value_range'],r['ci']['visit_mean_TV'][0][-1],r['ci']['unbiased_mean_sq_visit'][0][-1],r['first_divergence0'],r['all_paths_equal'],flush=True)
if __name__=='__main__':main()
