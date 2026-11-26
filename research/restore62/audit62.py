"""Replay every event in native canonical coordinates, reconstruct three-group field law."""
from base61 import *
from plan62 import frame
import gzip
CUTS=[4,8,16,32,64,128]
# For each cut: counts, score, predictable covariance (F), last-wave exact moments,
# historical likelihood ratios, native observables and prefix features.
TARGETS=np.array([[-.1,0.],[.1,0.],[-.2,0.],[.2,0.],[-.35,0.],[.35,0.],[0.,-.2],[0.,.2]])
def audit_one(arg):
 tag,j=arg;path=S/'data'/tag/(j['id']+'.json.gz')
 with gzip.open(path,'rt') as f:q=json.load(f)
 assert q['schema']=='mqft.trained.control62.v1' and q['input']==j
 m,src,fs=frame(j['moves'],j['sources'],j['cohort']=='legacy')
 g=fs[j['frame_rank']];inv=np.argsort(PERM[g]);assert q['frames']==fs and q['frame']==g and q['inverse']==inv.tolist() and q['canonical_sources']==src
 r=q['trace'];assert r['schema']=='mqft.trained.control61.v1' and r['job']['moves']==m and r['job']['source']==src[0] and r['job']['h']==0.
 for k in ('id','seed','budget','width','beta','d4','generation','core','orientation','rep','cohort'):assert r['job'][k]==j[k]
 b=board(m);legal=np.flatnonzero(b==0);side=1 if len(m)%2==0 else -1
 assert r['board']==b.tolist() and r['side']==side and len(set(src))==len(src) and all(a in legal for a in src)
 BUD=j['budget'];width=j['width'];es=r['events'];cuts={c['k']:c for c in r['cuts']};assert len(es)==BUD and set(cuts)=={k for k in CUTS if k<=BUD}
 rec=r['evaluations'];assert rec[0]['state_hash']==r['state_hash'] and rec[0]['side']==side and [e['id'] for e in rec]==list(range(len(rec)))
 p0=np.asarray(cuts[4]['prior'],float);assert abs(p0.sum()-1)<2e-6 and np.all(p0[legal]>0) and np.all(p0[b!=0]==0)
 group=np.full(N,2,int);group[src[0]]=0
 if len(src)==2:group[src[1]]=1
 h=np.array(j['field']+[0.]*(3-len(src)));hf=h[group]
 n=np.zeros(N,int);w=np.zeros(N);s2=np.zeros(N);depth=term=0;visited=set();ids=[];logpath=0.
 C=np.zeros(3);D=np.zeros(3);F=np.zeros((3,3));rw=np.zeros(len(TARGETS));u=[];rows=[];last_features=[];last_true=[]
 errscore=errprob=errlog=0.;margin=1.
 for start in range(0,BUD,width):
  Q=np.divide(w.astype(np.float32),n.astype(np.float32),out=np.zeros(N,np.float32),where=n>0)
  expected=Q+np.float32(1.4)*p0.astype(np.float32)*np.sqrt(np.float32(start))/(np.float32(1)+n.astype(np.float32))
  cb=C.copy();gb=C-D;db=D.copy();fb=F.copy();pc=None
  # At start of final wave, count/Q features contain no outcomes from that wave.
  prior_group=np.bincount(group,weights=p0,minlength=3)
  qsrc=[Q[src[0]],Q[src[1]] if len(src)==2 else 0.]
  for l,e in enumerate(es[start:start+width]):
   t=start+l;rs=e['root_selection'];law=rs['law'];a=np.asarray(law['actions'],int);sc=np.asarray(law['scores']);pr=np.asarray(law['p'])
   assert e['phase']==start//width+1 and e['complete_seq']==t and law['t']==t and rs['clock']==2*start+l and e['root_backup_clock']==2*start+width+l
   assert law['kind']=='iid' and law['proposal']=='p' and law['source'] is None and law['logit_h']==0. and law['beta_p']==law['beta_q']==j['beta'] and all(law['active'])
   assert np.array_equal(np.sort(a),legal)
   zz=j['beta']*sc+hf[a];zz-=zz.max();pp=np.exp(zz);pp/=pp.sum()
   se=float(abs(sc-expected[a]).max());pe=float(abs(pp-pr).max());errscore=max(errscore,se);errprob=max(errprob,pe);assert se<2e-6 and pe<2e-12
   uu=float(law['u']);pick=int(law['picked']);cdf=np.cumsum(pr);assert 0<=uu<1 and int(np.searchsorted(cdf,uu,side='right'))==pick
   margin=min(margin,uu-(cdf[pick-1] if pick else 0.),cdf[pick]-uu)
   assert int(a[pick])==e['actions'][0] and rs['edge_index']==pick and rs['score']==law['scores'][pick]
   lp=math.log(pr[pick]);logpath+=lp;le=max(abs(lp-law['log_p']),abs(logpath-law['log_p_path']));errlog=max(errlog,le);assert le<1e-9
   pg=np.bincount(group[a],weights=pr,minlength=3);assert abs(pg.sum()-1)<1e-12
   if pc is None:pc=pg.copy()
   else:assert np.array_equal(pc,pg)
   I=np.zeros(3);I[group[a[pick]]]=1
   C+=I;D+=pg;F+=np.diag(pg)-np.outer(pg,pg);u.append(uu)
   # likelihood from zero field (only used for zero-field histories)
   if all(x==0 for x in j['field']):
    rw+=TARGETS@I[:2]-np.log1p(np.expm1(TARGETS)@pg[:2])
   path0=e['actions'];assert path0 and e['parent_hashes'][0]==r['state_hash'];bb=b.copy();s=side;terminal=False
   for act in path0:
    assert bb[act]==0 and not terminal
    bb[act]=1 if s==1 else 2;terminal=win(bb,act) or np.all(bb!=0);s=-s
   z=float(e['root_value']);lv=float(e['leaf_value']);assert z==(-1)**len(path0)*lv and abs(lv)<=1 and bool(e['terminal'])==bool(terminal)
   if terminal:assert e['eval_id'] is None and lv==(-1 if win(bb,path0[-1]) else 0)
   else:
    ei=e['eval_id'];assert ei is not None and rec[ei]['state_hash']==e['leaf_hash'] and rec[ei]['side']==s and rec[ei]['value']==lv;ids.append(ei)
   n[path0[0]]+=1;w[path0[0]]+=z;s2[path0[0]]+=z*z;depth+=len(path0);term+=int(terminal);visited.update(tuple(path0[:i]) for i in range(1,len(path0)+1))
  k=start+width
  if k in cuts:
   c=cuts[k];assert np.array_equal(n,c['n']) and abs(w-np.asarray(c['w'])).max()<1e-9 and abs(s2-np.asarray(c['s2'])).max()<1e-9 and len(visited)==c['visited_prefix_edges']
   assert np.array_equal(c['prior'],p0)
   G=C-D;assert abs(C.sum()-k)<1e-12 and abs(G.sum())<1e-10 and abs(F.sum(0)).max()<1e-10
   lastF=F-fb;assert np.max(abs(lastF-width*(np.diag(pc)-np.outer(pc,pc))))<1e-10
   # dimensions stable: closure uses cut observables, never final data
   ov=observ(n,w,s2,p0,depth,k,len(visited),rec[0]['value'],term)
   Qnow=np.divide(w,n,out=np.zeros(N),where=n>0)
   marks=[Qnow[src[0]],Qnow[src[1]] if len(src)==2 else 0.]
   rows.append(dict(k=k,C=C.copy(),G=G.copy(),F=F.copy(),D=D.copy(),preC=cb,preG=gb,lastP=pc.copy(),lastF=lastF,logw=rw.copy(),obs=ov,prior=prior_group.copy(),marks=marks,action=-logpath))
 assert sorted(ids)==list(range(1,len(rec)))
 # mapped path signature for symmetry comparison, including mapped conditional
 # probabilities (not unordered arrays), marks, and final moments.
 sig=None
 if j['cohort']=='symmetry':
  z=[]
  for e in es:
   law=e['root_selection']['law'];p=np.zeros(N);p[inv[law['actions']]]=law['p']
   z.append([inv[e['actions']].tolist(),e['root_value'],p.tolist(),law['u']])
  # return digest in submitted input coordinates and per-orientation pullback.
  baseinv=np.argsort(PERM[j['orientation']]);out=[]
  for path1,mark,p,uu in z:out.append([baseinv[path1].tolist(),mark,np.asarray(p)[PERM[j['orientation']]].tolist(),uu])
  sig=hashlib.sha256(json.dumps(out,separators=(',',':')).encode()).hexdigest()
 return dict(job=j,tag=tag,sha256=sha(path),bytes=path.stat().st_size,evaluations=len(rec),max_score=errscore,max_prob=errprob,max_log=errlog,min_margin=margin,rows=rows,uniforms=u,signature=sig)
def collect():
 r=read(S/'REGISTRATION.json');assert sha(S/'mqft_train')==r['binary_sha'] and sha(S/'plan62.py')==r['plan_code_sha']
 for v in r['weights'].values():assert sha(B/v['path'])==v['sha256']
 args=[]
 for tag,q in sorted(r['plans'].items()):
  p=S/'plans'/f'{tag}.json';assert sha(p)==q['sha256'];js=read(p);assert len(js)==q['n'] and len(list((S/'data'/tag).glob('*.gz')))==len(js) and not list((S/'data'/tag).glob('*.part'));args.extend((tag,j) for j in js)
 for tag in ('response0w1','symmetry0'):
  js=read(S/f'plans/{tag}.json');sub=read(S/f'plans/{tag}_smoke.json')+read(S/f'plans/{tag}_rest.json');assert sub==js
 keys=['C','G','F','D','preC','preG','lastP','lastF','logw','obs','prior','marks','action'];arr={k:[] for k in keys};meta=[];receipts=[];streams={};nback=0;sy=[]
 with cf.ProcessPoolExecutor(max_workers=12) as pool:
  for i,x in enumerate(pool.map(audit_one,args,chunksize=8)):
   j=x['job'];uu=x.pop('uniforms');sd=j['seed']
   if sd in streams:assert uu[:min(len(uu),len(streams[sd]))]==streams[sd][:min(len(uu),len(streams[sd]))]
   if sd not in streams or len(streams[sd])<len(uu):streams[sd]=uu
   for row in x.pop('rows'):
    meta.append([i,j['generation'],j['width'],j['core'],j['rep'],row['k'],int(j['cohort']=='response'),int(j['cohort']=='legacy'),list(r['arms']).index(j['arm'])])
    for k in keys:arr[k].append(row[k])
   nback+=j['budget']
   if j['cohort']=='symmetry':sy.append(dict(job=j,signature=x['signature']))
   receipts.append(x)
   if (i+1)%2048==0:print('AUDIT',i+1,nback,flush=True)
 assert len(receipts)==20448 and nback==2586624
 # For every explicit frame rank, compare multiset of canonical-law pullbacks.
 groups=collections.defaultdict(list)
 for x in sy:
  j=x['job'];groups[(j['generation'],j['core'],j['rep'],j['orientation'])].append(x['signature'])
 comps=0;uniq=0;nontriv=0
 for g,c,rep in sorted({k[:3] for k in groups}):
  ref=sorted(groups[g,c,rep,0])
  for o in range(1,8):
   assert sorted(groups[g,c,rep,o])==ref,(g,c,rep,o)
   comps+=1
   if len(ref)==1:uniq+=1
   else:nontriv+=1
 assert comps==uniq+nontriv
 np.savez_compressed(S/'CONTROL62.npz',meta=np.array(meta),**{k:np.asarray(v) for k,v in arr.items()},obs_fields=np.array(FIELDS),targets=TARGETS,arm_names=np.array(list(r['arms'])),meta_fields=np.array(['job','generation','width','core','rep','cut','response','legacy','arm']))
 dump(S/'RNG_INPUT62.json',[dict(seed=int(k),u=v) for k,v in sorted(streams.items())])
 dump(S/'SYMMETRY62.json',dict(n_jobs=len(sy),orientation_multiset_comparisons=comps,unique_frame_comparisons=uniq,stabilizer_frame_comparisons=nontriv,all_exact=True,old_divergence_witnesses_retested=28,scope='mapped complete paths, marks, conditional p arrays and uniforms; nonunique frames compare multiset, not a deterministic unique selector',signatures=sy))
 dump(S/'AUDIT62.json',dict(n=len(receipts),backups=nback,evaluations=sum(x['evaluations'] for x in receipts),raw_gzip_bytes=sum(x['bytes'] for x in receipts),max_score=max(x['max_score'] for x in receipts),max_probability=max(x['max_prob'] for x in receipts),max_log=max(x['max_log'] for x in receipts),min_margin=min(x['min_margin'] for x in receipts),rng_streams=len(streams),rows=len(meta),jobs=receipts,array_sha256=sha(S/'CONTROL62.npz')))
 print('AUDIT COMPLETE',len(receipts),nback,len(meta),len(streams),'SYMMETRY',comps,uniq,nontriv,flush=True)
if __name__=='__main__':collect()
