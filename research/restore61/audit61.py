"""Independent finite-trace replay, not an inference of a thermal/quantum path."""
from base61 import *
import gzip
CUTS=[4,8,16,32,64,128]
EXTRA=['C','mean_p','score','fisher_pred','action','last_C','last_mean','last_var','lr_minus','lr_plus','logratio_minus','logratio_plus','count_noise','count_drift']
# C source count, p running expectation, score=sum(I-p). Bounded O independent of h.
def audit_one(arg):
 tag,j=arg;path=S/'data'/tag/(j['id']+'.json');q=read(path);assert q['schema']=='mqft.trained.control61.v1' and q['job']==j
 b=board(j['moves']);assert q['board']==b.tolist() and q['side']==(1 if len(j['moves'])%2==0 else -1)
 legal=np.flatnonzero(b==0);assert j['source'] in legal
 BUD=j['budget'];width=j['width'];events=q['events'];cuts={x['k']:x for x in q['cuts']};assert len(events)==BUD and set(cuts)=={k for k in CUTS if k<=BUD}
 p0=np.array(cuts[4]['prior'],float);assert abs(p0.sum()-1)<2e-6 and np.all(p0[legal]>0) and np.all(p0[b!=0]==0)
 assert all(np.array_equal(c['prior'],p0) for c in cuts.values())
 rec=q['evaluations'];assert rec[0]['state_hash']==q['state_hash'] and rec[0]['side']==q['side'] and [e['id'] for e in rec]==list(range(len(rec)))
 n=np.zeros(N,int);w=np.zeros(N);s2=np.zeros(N);depth=0;visited=set();term=0
 uniforms=[];src_p=[];src_i=[];logp=[];psrc0=[];ids=[];max_score=max_prob=max_log=0.;rows=[];prev=0.;margin=1.
 for start in range(0,BUD,width):
  nn=n.copy();ww=w.copy();Q=np.divide(ww.astype(np.float32),nn.astype(np.float32),out=np.zeros(N,np.float32),where=nn>0)
  ex=np.float32(1.4)*p0.astype(np.float32)*np.sqrt(np.float32(start))/(np.float32(1)+nn.astype(np.float32))
  expected=Q+ex
  wave=events[start:start+width]
  for l,e in enumerate(wave):
   t=start+l;rs=e['root_selection'];d=rs['law'];a=np.asarray(d['actions'],int);sc=np.asarray(d['scores'],float);pr=np.asarray(d['p'],float)
   assert e['phase']==start//width+1 and e['complete_seq']==t and d['t']==t
   assert rs['clock']==2*start+l and e['root_backup_clock']==2*start+width+l
   assert d['kind']=='iid' and d['proposal']=='p' and d['source']==j['source'] and d['logit_h']==j['h'] and d['beta_p']==d['beta_q']==j['beta'] and all(d['active'])
   assert np.array_equal(np.sort(a),legal)
   se=float(np.max(abs(sc-expected[a])));max_score=max(max_score,se);assert se<2e-6,(j['id'],t,se)
   z=j['beta']*sc;ix=np.flatnonzero(a==j['source'])[0];z[ix]+=j['h'];z-=z.max();p=np.exp(z);p/=p.sum()
   pe=float(abs(p-pr).max());max_prob=max(max_prob,pe);assert pe<1e-12
   u=float(d['u']);pick=int(d['picked']);cdf=np.cumsum(pr);assert 0<=u<1 and int(np.searchsorted(cdf,u,side='right'))==pick
   lo=cdf[pick-1] if pick else 0.;margin=min(margin,u-lo,cdf[pick]-u)
   assert int(a[pick])==e['actions'][0] and rs['edge_index']==pick and rs['score']==d['scores'][pick]
   lp=math.log(pr[pick]);prev+=lp;le=max(abs(lp-d['log_p']),abs(prev-d['log_p_path']));max_log=max(max_log,le);assert le<1e-9
   # At fixed realized history, h->0 changes only source logit.
   ph=float(pr[ix]);pzero=ph*math.exp(-j['h'])/(1+ph*math.expm1(-j['h']))
   uniforms.append(u);src_p.append(ph);psrc0.append(pzero);src_i.append(int(pick==ix));logp.append(lp)
   path0=e['actions'];assert path0 and e['parent_hashes'][0]==q['state_hash']
   bb=b.copy();side=q['side'];terminal=False
   for depth0,act in enumerate(path0):
    assert bb[act]==0 and not terminal
    bb[act]=1 if side==1 else 2;terminal=win(bb,act) or np.all(bb!=0);side=-side
   r=float(e['root_value']);lv=float(e['leaf_value']);assert r==(-1)**len(path0)*lv and abs(lv)<=1 and bool(e['terminal'])==bool(terminal)
   if terminal:assert e['eval_id'] is None and lv==(-1 if win(bb,path0[-1]) else 0)
   else:
    ei=e['eval_id'];assert ei is not None and rec[ei]['state_hash']==e['leaf_hash'] and rec[ei]['side']==side and rec[ei]['value']==lv;ids.append(ei)
   n[path0[0]]+=1;w[path0[0]]+=r;s2[path0[0]]+=r*r;depth+=len(path0);term+=int(terminal);visited.update(tuple(path0[:i]) for i in range(1,len(path0)+1))
  k=start+width
  if k in cuts:
   c=cuts[k];assert np.array_equal(n,c['n']) and abs(w-np.asarray(c['w'])).max()<1e-9 and abs(s2-np.asarray(c['s2'])).max()<1e-9 and len(visited)==c['visited_prefix_edges']
   p=np.asarray(src_p);I=np.asarray(src_i);pz=np.asarray(psrc0);G=float(np.sum(I-p));F=float(np.sum(p*(1-p)))
   assert abs(float(I.sum())-float(p.sum())-G)<1e-10
   if width>1:assert np.max(abs(p[-width:]-p[-1]))<1e-14
   ratios=[float(np.sum(h*I-np.log1p(pz*np.expm1(h)))) for h in (-.35,.35)]
   rows.append(observ(n,w,s2,p0,depth,k,len(visited),rec[0]['value'],term)+[int(I.sum()),float(p.sum()),G,F,-prev,float(I[-width:].sum()),float(p[-width:].sum()),float(np.sum(p[-width:]*(1-p[-width:]))),math.exp(ratios[0]),math.exp(ratios[1]),ratios[0],ratios[1],G,float(p.sum())])
 assert sorted(ids)==list(range(1,len(rec)))
 # Original raw JSON preserved, no synthetic trajectory. Return compact arrays and raw hash.
 return dict(job=j,sha256=sha(path),bytes=path.stat().st_size,elapsed_ns=q['elapsed_ns'],evaluations=len(rec),max_score=max_score,max_prob=max_prob,max_log=max_log,min_cdf_margin=margin,rows=rows,uniforms=uniforms,root_prior=p0.tolist(),final_n=n.tolist(),final_w=w.tolist())
def main():
 reg=read(S/'REGISTRATION.json');assert reg['binary_sha']==sha(S/'mqft_train')
 for g,v in reg['weights'].items():assert sha(B/v['path'])==v['sha256']
 args=[]
 for f,h in sorted(reg['plans'].items()):
  p=S/'plans'/f;assert sha(p)==h;jobs=read(p);tag=p.stem
  assert len(list((S/'data'/tag).glob('*.json')))==len(jobs) and not list((S/'data'/tag).glob('*.part'))
  args.extend((tag,j) for j in jobs)
 assert len(args)==3584
 # verify commissioning split is a disjoint exhaustive partition of frozen plans
 for g in (0,3):
  js=read(S/f'plans/response{g}.json');z=read(S/f'plans/smoke{g}.json')+read(S/f'plans/rest{g}.json');assert sorted(js,key=lambda x:x['id'])==sorted(z,key=lambda x:x['id'])
 out=[];xs=[];meta=[];prior=[];nn=[];ww=[];u=[];nback=0
 with cf.ProcessPoolExecutor(max_workers=12) as pool:
  for i,r in enumerate(pool.map(audit_one,args,chunksize=8)):
   j=r['job'];prior.append(r.pop('root_prior'));nn.append(r.pop('final_n'));ww.append(r.pop('final_w'));uu=r.pop('uniforms');u.append(uu+[float('nan')]*(128-len(uu)))
   for k,row in zip([k for k in CUTS if k<=j['budget']],r.pop('rows')):
    xs.append(row);meta.append([i,j['generation'],j['core'],j['width'],round(j['h']*100),int(j['d4']),j['orientation'],j['rep'],int(j['cohort']=='symmetry'),k,j['budget']])
   nback+=j['budget'];out.append(r)
   if (i+1)%512==0:print('AUDIT',i+1,nback,flush=True)
 assert nback==425984 and len(out)==3584
 # exact CRN: same seed means identical uniform stream prefix across interventions, widths, orientations, models
 z={}
 for i,r in enumerate(out):
  j=r['job'];a=np.asarray(u[i][:j['budget']]);seed=j['seed']
  if seed in z:assert np.array_equal(a[:min(len(a),len(z[seed]))],z[seed][:min(len(a),len(z[seed]))])
  if seed not in z or len(z[seed])<len(a):z[seed]=a
 np.savez_compressed(S/'CONTROL61.npz',values=np.array(xs),meta=np.array(meta),fields=np.array(FIELDS+EXTRA),meta_fields=np.array(['job','generation','core','width','h100','d4','orientation','rep','symmetry','cut','budget']),prior=np.array(prior),final_n=np.array(nn),final_w=np.array(ww),uniforms=np.array(u))
 # Independent Rust RNG replay reads seeds/uniforms from this file.
 dump(S/'RNG_INPUT61.json',[dict(seed=int(k),u=v.tolist()) for k,v in sorted(z.items())])
 dump(S/'CONTROL_AUDIT61.json',dict(n=len(out),backups=nback,response=3072,symmetry=512,n_eval=sum(x['evaluations'] for x in out),raw_bytes=sum(x['bytes'] for x in out),max_score=max(x['max_score'] for x in out),max_probability=max(x['max_prob'] for x in out),max_log=max(x['max_log'] for x in out),min_cdf_margin=min(x['min_cdf_margin'] for x in out),unique_crn_streams=len(z),jobs=out,array_sha256=sha(S/'CONTROL61.npz'),scope='all native trace replay, root and deep native conservation checked at cuts, independent Python legal path/score/law/count audit; evaluator calls not all independently recomputed'))
 print('AUDIT_COMPLETE',len(out),nback,len(xs),len(z),flush=True)
if __name__=='__main__':main()
