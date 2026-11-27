"""Full native trace audit for all registered arms, exact causal likelihood cross-evaluation."""
from instrument81 import *
from arena68 import board as board15,won
import gzip
from scipy.special import logsumexp
TAGS=['schedule0','schedule3','rapfi'];CUT=[4,8,16,32,64,128]
def semhash(tr,n):
 # Exclude only worker/time and auxiliary phase record, retain native path/evaluation and law fields.
 es=[]
 for e in tr['events'][:n]:
  z=dict(e);z.pop('worker',None);rs=dict(z['root_selection']);l=dict(rs['law']);l.pop('phase65',None);rs['law']=l;z['root_selection']=rs;es.append(z)
 eids={e['eval_id'] for e in es if e['eval_id'] is not None}|{0}
 rec=[r for r in tr['evaluations'] if r['id'] in eids];cuts=[c for c in tr['cuts'] if c['k']<=n]
 return hashlib.sha256(json.dumps([es,rec,cuts],sort_keys=True,separators=(',',':')).encode()).hexdigest()
def audit_one(arg):
 tag,j=arg;g=TAGS.index(tag);L=15 if g==2 else 9;N=L*L;f=S/'data'/tag/(j['id']+'.json.gz')
 with gzip.open(f,'rt') as h:d=json.load(h)
 assert d['input']==j;tr=d['trace'];mode=j['arm'];arm=ARMS.index(mode)
 if g<2:assert d['schema']=='mqft.trained.phase65.v1' and tr['job']['cohort']=='phase65'
 else:assert tr['phase81']==mode and tr['pair81']==j['sources'] and tr['arm']=='clean'
 b=board15(j['moves']) if g==2 else board(j['moves']);legal=np.flatnonzero(b==0);aa,bb=j['sources'];assert aa in legal and bb in legal
 es=tr['events'];rec=tr['evaluations'];cuts={c['k']:c for c in tr['cuts']};assert len(es)==128 and set(cuts)==set(CUT);side=1 if len(j['moves'])%2==0 else -1
 assert tr['side']==side and rec[0]['side']==side and rec[0]['state_hash']==tr['state_hash']
 assert [r['id'] for r in rec]==list(range(len(rec)))
 prior=np.array(cuts[4]['prior']);assert abs(prior.sum()-1)<2e-6 and np.all(prior[legal]>0) and np.all(prior[b!=0]==0)
 if g==2:
  for e in rec:
   w0=np.array(e['wdl']);assert np.all(w0>=0) and abs(w0.sum()-1)<.001;assert abs(float(np.float32(e['side']*(w0[0]-w0[1])/w0.sum()))-e['clean'])<1e-7 and e['delivered']==e['clean'] and e['mask']==[0,0,0]
  a0=np.array(rec[0]['scores']);assert np.array_equal(a0[:,0],legal);z=np.exp((a0[:,1]-a0[:,1].max())/32);pred=(1-1e-6)*z/z.sum()+1e-6/len(legal);assert abs(prior[legal]-pred.astype(np.float32)).max()<2e-7
 n=np.zeros(N,int);w=np.zeros(N);s2=np.zeros(N);state=RHO0.copy();vs=np.tile([.8,0.,0.],(6,1));ids=[];seen=set();lp=0.;errs=collections.defaultdict(float);min_eig=1.;u=[];actions=[];marks=[];rat=np.zeros((12,6));ql=np.zeros((12,6,3));st=np.zeros((12,3));baseinfo=[]
 def chk(key,a,b,tol=2e-10):
  e=float(np.max(np.asarray(a)!=np.asarray(b))) if np.asarray(a).dtype==bool else float(np.max(abs(np.asarray(a)-np.asarray(b))));errs[key]=max(errs[key],e);assert e<tol,(key,e,tag,j['id'])
 for start in range(0,128,4):
  Q=np.divide(w.astype(np.float32),n.astype(np.float32),out=np.zeros(N,np.float32),where=n>0)
  U=Q+np.float32(1.4)*prior.astype(np.float32)*np.sqrt(np.float32(start))/(np.float32(1)+n.astype(np.float32))
  for l,e in enumerate(es[start:start+4]):
   t=start+l;rs=e['root_selection'];law=rs['law'];a=np.array(law['actions']);sc=np.array(law['scores']);p=np.array(law['p']);ph=law['phase65']
   assert law['kind']=='iid' and law['proposal']=='p' and law['source'] is None and law['logit_h']==0 and law['beta_p']==law['beta_q']==10/3 and all(law['active'])
   assert e['phase']==start//4+1 and e['complete_seq']==law['t']==t and rs['clock']==2*start+l and e['root_backup_clock']==2*start+4+l
   assert np.array_equal(np.sort(a),legal);chk('score',sc,U[a],2e-6)
   base=np.exp((10/3)*(sc-sc.max()));base/=base.sum();ia=int(np.flatnonzero(a==aa)[0]);ib=int(np.flatnonzero(a==bb)[0]);pick=law['picked']
   # Active instrument audited via complex matrix, all 6 counterfactual state updates follow same observed history.
   if 32<=t<44:
    want,post,fields,ce,norm=matrix_step(mode,t,state,base,ia,ib,pick);chk('Kraus_complete',ce,0);chk('conditional_trace',norm,want[pick])
    for k,v in fields.items():chk('phase_record_'+k,v,ph[k])
    for r0 in (state,post,rho(fields['prepared'])):
     chk('trace',np.trace(r0),1.);chk('Hermiticity',r0,adj(r0));ev=np.linalg.eigvalsh(r0);assert ev.min()>-1e-10;min_eig=min(min_eig,float(ev.min()))
    state=post;st[t-32]=bloch(state)
    for i,arm0 in enumerate(ARMS):
     q,vs[i]=scalar_step(arm0,t,vs[i],base,ia,ib,pick);assert np.all(q>0) and abs(q.sum()-1)<1e-12
     rat[t-32,i]=np.log(q[pick]/base[pick]);ql[t-32,i]=[q[ia],q[ib],1-q[ia]-q[ib]]
     if i==arm:chk('independent_scalar_p',q,want);chk('independent_scalar_state',vs[i],bloch(state))
    if t%4==0:baseinfo.append([float(base[ia]+base[ib]),float(base[ia]/(base[ia]+base[ib]))])
   else:
    want=base;chk('inactive_pre',ph['pre'],bloch(state));chk('inactive_prepared',ph['prepared'],bloch(state));chk('inactive_post',ph['post'],bloch(state));assert ph['lambda81']==0 and ph['rotation']==0 and ph['eta81']==1 and not ph['erase81']
    chk('inactive_output',ph['output_a'],base[ia]);chk('inactive_mass',ph['pair_mass'],base[ia]+base[ib])
   assert ph['sources']==j['sources'] and ph['wave']==t//4 and ph['outcome']==e['actions'][0] and ph['tau']==start/128 and ph['exposure']==0
   chk('probability',p,want);assert np.all(p>0) and abs(p.sum()-1)<1e-12
   uu=law['u'];assert int(np.searchsorted(np.cumsum(p),uu,side='right'))==pick and int(a[pick])==e['actions'][0] and rs['edge_index']==pick;u.append(uu)
   ll=math.log(p[pick]);lp+=ll;chk('log_probability',ll,law['log_p']);chk('log_path',lp,law['log_p_path'],1e-9)
   path=e['actions'];assert path and e['parent_hashes'][0]==tr['state_hash'];bx=b.copy();sd=side;terminal=False
   for a0 in path:
    assert bx[a0]==0 and not terminal;bx[a0]=1 if sd==1 else 2;terminal=(won(bx,a0) if g==2 else win(bx,a0)) or np.all(bx!=0);sd=-sd
   lv=float(e['leaf_value']);mark=float(e['root_value']);assert mark==(-1)**len(path)*lv and bool(e['terminal'])==bool(terminal) and abs(mark)<=1
   if terminal:assert e['eval_id'] is None and lv==(-1 if (won(bx,path[-1]) if g==2 else win(bx,path[-1])) else 0)
   else:
    i=e['eval_id'];assert i is not None and rec[i]['state_hash']==e['leaf_hash'] and rec[i]['side']==sd and rec[i]['delivered' if g==2 else 'value']==lv;ids.append(i)
   ac=path[0];n[ac]+=1;w[ac]+=mark;s2[ac]+=mark*mark;actions.append(ac);marks.append(mark);seen.update(tuple(path[:k]) for k in range(1,len(path)+1))
  k=start+4
  if k in cuts:
   c=cuts[k];assert np.array_equal(n,c['n']) and abs(w-c['w']).max()<1e-9 and abs(s2-c['s2']).max()<1e-9 and np.array_equal(prior,c['prior'])
   if g<2:assert len(seen)==c['visited_prefix_edges']
 assert sorted(ids)==list(range(1,len(rec))) and len(baseinfo)==3
 actions=np.array(actions);marks=np.array(marks);recommend=int(n.argmax())
 if g==2:assert tr['action']==recommend
 r=dict(family=g,core=j['core'],rep=j['rep'],arm=arm,seed=j['seed'],sources=j['sources'],path=str(f.relative_to(B)),sha256=sha(f),bytes=f.stat().st_size,errors=dict(errs),min_eig=min_eig,semantic32=semhash(tr,32),semantic36=semhash(tr,36),semantic128=semhash(tr,128),elapsed_ns=tr['elapsed_ns'],evaluations=len(rec),recommend=recommend,first_action=int(actions[32]))
 return r,actions,marks,n,w,rat,ql,st,np.array(baseinfo),u
def main():
 R=read(S/'REGISTRATION.json');assert sha(S/'mqft_phase81')==R['phase_binary_sha'] and R['arms']==ARMS
 args=[]
 for tag in TAGS:
  p=S/'plans'/f'{tag}.json';assert sha(p)==R['plans'][tag]['sha256'];js=read(p);assert len(js)==1152 and {j['id']+'.json.gz' for j in js}=={x.name for x in (S/'data'/tag).glob('*.gz')};args.extend((tag,j) for j in js)
 rows=[]
 with cf.ProcessPoolExecutor(max_workers=12) as pool:
  for i,r in enumerate(pool.map(audit_one,args,chunksize=8)):
   rows.append(r)
   if (i+1)%288==0:print('AUDIT',i+1,flush=True)
 rec=[r[0] for r in rows];M=np.array([[r[k] for k in ('family','core','rep','arm')] for r in rec]);assert len(set(map(tuple,M)))==3456
 # Different board sizes, pad native final counts to 225 for storage only.
 actions,marks,rat,ql,st,baseinfo,u=map(np.stack,zip(*[(r[1],r[2],r[5],r[6],r[7],r[8],r[9]) for r in rows]))
 counts=np.stack([np.pad(r[3],(0,225-len(r[3]))) for r in rows]);weights=np.stack([np.pad(r[4],(0,225-len(r[4]))) for r in rows])
 np.savez_compressed(S/'PATHS81.npz',meta=M,actions=actions,marks=marks,counts=counts,weights=weights,ratios=rat,group_probs=ql,states=st,baseinfo=baseinfo,uniforms=u,sources=np.array([r['sources'] for r in rec]),elapsed=np.array([r['elapsed_ns'] for r in rec])/1e9,evaluations=np.array([r['evaluations'] for r in rec]))
 streams={}
 for i,r in enumerate(rec):
  sd=r['seed'];v=u[i].tolist()
  if sd in streams:assert streams[sd]==v
  else:streams[sd]=v
 dump(S/'RNG_INPUT81.json',[dict(seed=k,u=v) for k,v in sorted(streams.items())])
 errors={k:max(r['errors'].get(k,0.) for r in rec) for k in set().union(*(r['errors'] for r in rec))}
 out=dict(paths=len(rec),backups=128*len(rec),bytes=sum(r['bytes'] for r in rec),errors=errors,min_eig=min(r['min_eig'] for r in rec),raw_receipts=rec,array_sha=sha(S/'PATHS81.npz'),rng_streams=len(streams),scope='Independent complex matrix and alternate scalar audit all fresh actual histories; recorded actual fine probabilities and full native backup replay; counterfactual likelihood on same observed history.')
 dump(S/'AUDIT81.json',out);print('AUDIT COMPLETE',{k:v for k,v in out.items() if k!='raw_receipts'},flush=True)
if __name__=='__main__':main()
