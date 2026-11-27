"""Independent direct log-normalizer, all model loss/gradient and raw subset verification."""
from base61 import *
import gzip,csv
from scipy.optimize import minimize_scalar
A=read(S/'AUDIT79.json');R=read(S/'CALIBRATION79.json');I=read(S/'INVENTORY_AUDIT79.json');O=read(S/'ORIGINAL_SOURCE79.json');L=read(S/'EVIDENCE_LINKS79.json')
assert I['verified_pairs']==I['plot_pairs']==640 and not I['issues'] and I['guide_literal_matches']==27 and I['function_spans']==94
err=collections.defaultdict(float);raw_count=0;fit_count=0;occ_count=0
def check(k,a,b,tol=1e-9):
 v=float(np.max(abs(np.asarray(a)-np.asarray(b))));err[k]=max(err[k],v);assert v<tol,(k,v)
def logprob(x,b,mask):
 z=b*x;mx=np.max(np.where(mask,z,-np.inf),axis=-1,keepdims=True);e=np.where(mask,np.exp(z-mx),0.);norm=e.sum(-1,keepdims=True)
 return np.where(mask,z-mx-np.log(norm),0.),e/norm
def nll(x,b,mask,y):
 l,p=logprob(x,b,mask);return -np.sum(y*l,axis=-1)
for g,f in enumerate(['g0','g3','Rapfi']):
 p=S/f'CALIB_{f}.npz';assert sha(p)==A['arrays'][p.name]
 with np.load(p) as z:d={k:z[k] for k in z.files}
 m=d['meta'];tr=m[:,0]==0;te=~tr;mask=d['mask'][:,None,:];P=d['p'];Y=d['y'];sc=d['score'];Q=d['Q'];assert tr.sum()==64 and te.sum()==256
 result=R['results'][g];ref=nll(sc,10/3,mask,Y);ent=nll(sc,10/3,mask,P)
 for row in result['models']:
  name=row['model'];x={'score':sc,'Q':Q,'log_prior':np.broadcast_to(d['logprior'][:,None,:],sc.shape),'uniform':np.zeros_like(sc),'known_score':sc}[name];b=row['beta']
  if row['status']=='interior':
   xt=x[tr,1:];mt=mask[tr];yt=Y[tr,1:];_,pp=logprob(xt,b,mt);grad=float(np.sum((pp-yt)*xt))
   assert abs(grad)<1e-6
   alt=minimize_scalar(lambda v:float(nll(xt,v,mt,yt).sum()),bounds=(-32.,32.),method='bounded',options={'xatol':1e-10})
   assert alt.success;check('independent_beta_fit',alt.x,b,tol=1e-5);fit_count+=1
  l=nll(x[te],b,mask[te],Y[te]);kl=nll(x[te],b,mask[te],P[te])-ent[te];dif=l-ref[te]
  for i,k in enumerate(R['cuts']):
   n=k//4;check('heldout_nll',np.mean(l[:,:n]),row['nll'][i][0]);check('heldout_excess',np.mean(dif[:,:n]),row['excess_nll'][i][0]);check('heldout_KL',np.mean(kl[:,:n]),row['conditional_KL'][i][0])
 for row in result['occupancy_fits']:
  i=np.flatnonzero(np.all(m==[1,row['core'],row['rep']],axis=1));assert len(i)==1;i=int(i[0]);k=R['cuts'].index(row['B']);x=d['Qcut'][i,k];y=d['occ'][i,k];ma=d['mask'][i];b=row['beta']
  if b is None:assert row['status']=='flat';continue
  _,p=logprob(x,b,ma);grad=np.dot(p-y,x)
  if row['status']=='interior':assert abs(grad)<1e-9
  elif row['status']=='upper_bound':assert grad<=1e-9 and b==32
  elif row['status']=='lower_bound':assert grad>=-1e-9 and b==-32
  else:raise AssertionError(row['status'])
  check('occupancy_KL',nll(x,b,ma,y)-entropy(y),row['occupancy_KL'])
  occ_count+=1
 for r in [x for x in A['raw_receipts'] if x['family']==g and x['rep']==0]:
  p=B/r['path'];assert sha(p)==r['sha256']
  with gzip.open(p,'rt') as z:raw=json.load(z)
  j=raw['input'];i=np.flatnonzero(np.all(m==[r['cohort'],r['core'],0],axis=1));assert len(i)==1;i=int(i[0]);tr0=raw['trace'];es=tr0['events'];N0=np.zeros(sc.shape[-1]);W0=N0.copy()
  for a in range(32):
   law=es[4*a]['root_selection']['law'];ac=law['actions'];pp=np.array(law['p']);ss=np.array(law['scores']);check('raw_score',sc[i,a,ac],ss);check('raw_p',P[i,a,ac],pp);check('raw_Q',Q[i,a],np.divide(W0,N0,out=np.zeros_like(W0),where=N0>0))
   vals=np.bincount([e['actions'][0] for e in es[4*a:4*a+4]],minlength=len(N0))/4;check('raw_counts',Y[i,a],vals)
   for e in es[4*a:4*a+4]:k=e['actions'][0];N0[k]+=1;W0[k]+=e['root_value']
   if 4*(a+1) in R['cuts']:
    k=R['cuts'].index(4*(a+1));check('raw_occupancy',d['occ'][i,k],N0/(4*(a+1)))
  raw_count+=1
 assert result['score_log_slope_count']==7936 and result['unidentifiable_waves']==256
 print('VERIFIED',f,flush=True)
assert raw_count==144 and fit_count==9 and occ_count==4608
assert len(O['spans'])==121 and len({r['id'] for r in O['spans']})==121
C=list(csv.DictReader((S/'acceptance79.tsv').open(),delimiter='\t'));assert len(C)==121 and {x['id'] for x in C}=={r['id'] for r in O['spans']}
assert sum(bool(x) for x in L['items'].values())==120
# Exact two-point conditional curvature from direct variance and closed form.
for b in [-3.,0.,2.]:
 x=np.array([-.4,1.2]);_,p=logprob(x,b,np.ones(2,bool));v=p@((x-p@x)**2);q=np.exp(b*(x.sum()))*(x[0]-x[1])**2/(np.exp(b*x).sum()**2);check('two_point_curvature',v,q)
dump(S/'INDEPENDENT79.json',dict(status='PASS_SCOPED',raw_paths=raw_count,independent_fits=fit_count,occupancy_gradient_checks=occ_count,errors=dict(err),scope='Independent direct normalizer and numerical optimum, all heldout aggregate losses, all endpoint gradient/constraint checks, raw subset full reductions, source catalog/acceptance identity; same-agent independent implementation.'))
print(dict(err),flush=True)
