"""Actual phase paths and declared coherent record: distinct evidence and provenance."""
from base61 import *
import gzip
from itertools import product
from scipy.stats import t as st
O81=B/'mqft_restore81_20261127';O83=B/'mqft_restore83_20261127';O80=B/'mqft_restore80_20261127';O85=B/'mqft_restore85_20261127'
def one(r):
 p=B/r['path'];assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes']
 with gzip.open(p,'rt') as f:d=json.load(f)
 tr=d['trace'];ev=tr['events'];assert len(ev)==128
 a=np.array([e['actions'][0] for e in ev]);z=np.array([e['root_value'] for e in ev]);q=[];states=[];m=[];u=[]
 for k,e in enumerate(ev):
  l=e['root_selection']['law'];assert l['t']==k and l['actions'][l['picked']]==a[k];u.append(l['u'])
  if 32<=k<44:
   c=l['phase65'];src=c['sources'];assert src==r['sources'];p0=np.array(l['p']);v=np.array(l['actions']);pa=p0[v==src[0]][0];pb=p0[v==src[1]][0]
   assert abs(pa+pb-c['pair_mass'])<2e-12 and abs(pa-c['output_a'])<2e-12
   q.append([pa,pb,1-pa-pb]);states.append(c['post']);m.append([c['pair_mass'],c['population_a']])
 return a,z,np.asarray(q),np.asarray(states),np.asarray(m)[::4],np.asarray(u)
def ci(x):
 m=x.mean(0);h=st.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return np.array([m,m-h,m+h])
def rehash(r):
 p=B/r['path'];assert sha(p)==r['sha256'],r['path'];return r['path']
def td(a,b):return float(.5*sum(abs(np.linalg.eigvalsh(a-b))))
def partial(r,n):
 dim=2**n;d=8//dim;return np.trace(r.reshape(dim,d,dim,d),axis1=1,axis2=3)
def main():
 A=read(O81/'AUDIT81.json');E=read(O81/'EFFECTS81.json');assert sha(O81/'PATHS81.npz')==A['array_sha']==E['input_sha']
 with np.load(O81/'PATHS81.npz') as z:D={k:z[k] for k in z.files}
 meta=D['meta'];idx={tuple(r):i for i,r in enumerate(meta)};rows=A['raw_receipts'];assert len(rows)==len(idx)==3456
 err=collections.defaultdict(float)
 def chk(k,a,b,tol=1e-10):
  e=float(abs(np.asarray(a)-np.asarray(b)).max());err[k]=max(e,err[k]);assert e<tol,(k,e)
 with cf.ProcessPoolExecutor(max_workers=6) as p:
  for r,out in zip(rows,p.map(one,rows,chunksize=8)):
   i=idx[r['family'],r['core'],r['rep'],r['arm']];a,z,q,s,m,u=out;arm=r['arm']
   assert np.array_equal(a,D['actions'][i]) and np.array_equal(z,D['marks'][i]) and np.array_equal(u,D['uniforms'][i])
   chk('raw_group_probability',q,D['group_probs'][i,:,arm]);chk('raw_state_post',s,D['states'][i]);chk('raw_baseinfo',m,D['baseinfo'][i])
 families=[]
 for g in range(3):
  out=[];pred=[];active=[];end=[]
  for c in range(24):
   rr=[];ss=[];aa=[];ee=[]
   for r in range(8):
    ix=[idx[g,c,r,k] for k in range(6)];act=D['actions'][ix];src=D['sources'][ix];z=D['marks'][ix];assert np.array_equal(act[2],act[3]) and np.array_equal(z[2],z[3]) and np.array_equal(act[:,:32],np.tile(act[0,:32],(6,1)))
    signed=(act==src[:,0,None]).astype(float)-(act==src[:,1,None]).astype(float)
    rr.append(signed[0,32]-signed[1,32]);ss.append(2*.65*.8*.9*D['baseinfo'][ix[5],0,0]);aa.append((signed[0,32:44]-signed[1,32:44]).mean());ee.append(z[0].mean()-z[1].mean())
   out.append(np.mean(rr));pred.append(np.mean(ss));active.append(np.mean(aa));end.append(np.mean(ee))
  f=E['families'][g];v=ci(np.array(out));p=ci(np.array(pred));a=ci(np.array(active));e=ci(np.array(end));chk('phase_first_ci',v,np.array(f['comparisons'][0]['direct_ci'])[:,0]);chk('phase_prediction_ci',p,f['first_law_signed_pred_ci']);chk('phase_active_ci',a,np.array(f['comparisons'][0]['direct_ci'])[:,1]);chk('phase_final_mark',e,np.array(f['comparisons'][0]['direct_ci'])[:,7])
  families.append(dict(family=f['family'],first_ci=v.tolist(),predicted_ci=p.tolist(),active_ci=a.tolist(),final_mark_ci=e.tolist()))
 R=read(O83/'RECORD83.json');T=read(O83/'TREE83.json');A83=read(O83/'AUDIT83.json');assert sha(O83/'RECORD_MATRICES83.npz')==R['matrix_sha'] and sha(O83/'TREE83.json')==R['tree_sha']==A83['tree_sha']
 with np.load(O83/'RECORD_MATRICES83.npz') as z:met=z['meta'];mat=z['R'];labels=list(z['labels'])
 H=np.array([[1,1],[1,-1]])/np.sqrt(2);HH=np.kron(np.kron(H,H),H);bits=np.array(list(product(range(2),repeat=3)));dist=(bits[:,None,:]!=bits[None,:,:]).sum(2)
 contexts=[]
 for k,c in enumerate(R['contexts']):
  assert list(met[k])==[c['family'],c['core']];res={}
  for protocol in ('native_repeat','pair_diagnostic'):
   a=mat[k,labels.index(protocol+'_adaptive')];r=c['protocols'][protocol]
   assert abs(np.trace(a)-1)<1e-12 and np.linalg.eigvalsh(a).min()>-1e-12
   chk('record_diag',a.diagonal().real,r['conditional_prob']);chk('record_success_diag',a.diagonal().real*r['success'],r['leaf_prob'])
   if protocol=='native_repeat':
    t=T['contexts'][k];assert t['family']==c['family'] and t['core']==c['core']
    actual=np.array([l['probability'] for l in t['leaves']]);chk('native_leaf_relative',a.diagonal().real*r['success']/actual,1,1e-9)
   read0=np.diag(HH@a@HH.T).real;tagged=np.diag(HH@np.diag(a.diagonal())@HH.T).real;v=float(.5*abs(read0-tagged).sum())
   chk('hadamard_TV',v,r['conditional_Hadamard_TV']);chk('stopped_TV',v*r['success'],r['unconditional_Hadamard_TV'])
   # These are reductions of FINAL SELECTED record, not native causal time series.
   profile=[]
   for n in (1,2,3):
    z=partial(a,n);b=np.diag(z.diagonal());profile.append(td(z,b))
   tags=[]
   for z in r['overlap_controls']:
    b=a*z['overlap']**dist;v0=np.diag(HH@b@HH.T).real;chk('tag_readout',v0,z['readout']);tags.append(float(.5*abs(v0-tagged).sum()))
   res[protocol]=dict(success=r['success'],conditional_tv=v,stopped_tv=v*r['success'],profile=profile,tag_tv=tags)
  contexts.append(dict(family=c['family'],core=c['core'],protocols=res))
 # Actual forced diagnostic source identity, no new native outcomes.
 more=A83['raw_receipts']+read(O85/'BRANCH_ENERGY85.json')['raw_receipts']
 unique={r['path']:r for r in more};assert all(unique[r['path']]['sha256']==r['sha256'] for r in more)
 with cf.ProcessPoolExecutor(max_workers=4) as p:assert len(list(p.map(rehash,unique.values(),chunksize=16)))==len(unique)
 F=read(O80/'FORCED_REPLAY80.json')
 def clean(x):
  if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ('worker','elapsed_ns','wall_ns','u')}
  if isinstance(x,list):return [clean(v) for v in x]
  return x
 for r in F['cases']:
  assert sha(B/r['source'])==r['source_sha'] and sha(B/r['path'])==r['sha256']
  with gzip.open(B/r['source'],'rt') as f:a=json.load(f)['trace']
  with gzip.open(B/r['path'],'rt') as f:b=json.load(f)['trace']
  keys=('events','cuts','evaluations','state_hash','side');assert all(clean(a[k])==clean(b[k]) for k in keys)
  digest=hashlib.sha256(json.dumps({k:clean(a[k]) for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest();assert digest==r['semantic_sha']
 dump(S/'PHASE90.json',dict(status='PASS_SCOPED',actual_phase_paths=3456,matched_blocks=576,declared_record_contexts=72,forced_replay=24,extra_diagnostic_source_paths=len(unique),raw_receipts=rows+list(unique.values())+[dict(path=r['path'],sha256=r['sha256']) for r in F['cases']],families=families,records=contexts,errors=dict(err),scope='Actual programmed phase intervention, separately declared shared-bath record dilation. Reduced final postselected record is not temporal native decoherence; tag controls and stopped probability retained. No physical hbar or coherent hardware.'))
 print('PHASE COMPLETE',len(rows),len(contexts),len(unique),dict(err),families,flush=True)
if __name__=='__main__':main()
