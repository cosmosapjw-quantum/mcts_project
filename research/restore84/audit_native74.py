"""Independent native schedule law and complete deterministic downstream trace audit."""
from base61 import *
import gzip
def one(arg):
 tag,j=arg;path=S/'data'/tag/(j['id']+'.json.gz')
 with gzip.open(path,'rt') as f:d=json.load(f)
 assert d['schema']=='mqft.trained.schedule66.v1' and d['input']==j
 q=d['trace'];assert q['schema']=='mqft.trained.control61.v1'
 for k in ('id','moves','seed','budget','width','beta','generation','core','rep'):assert q['job'][k]==j[k]
 assert q['job']['cohort']=='schedule66' and q['job']['source']==j['sources'][0]
 b=board(j['moves']);side=1 if len(j['moves'])%2==0 else -1;legal=np.flatnonzero(b==0);a0=j['sources'][0];b0=j['sources'][1]
 assert q['board']==b.tolist() and q['side']==side and a0 in legal and b0 in legal
 rec=q['evaluations'];assert rec[0]['state_hash']==q['state_hash'] and rec[0]['side']==side and [r['id'] for r in rec]==list(range(len(rec)))
 cuts={c['k']:c for c in q['cuts']};assert set(cuts)=={4,8,16,32,64,128};prior=np.asarray(cuts[4]['prior']);assert abs(prior.sum()-1)<2e-6 and np.all(prior[legal]>0) and np.all(prior[b!=0]==0)
 es=q['events'];assert len(es)==128
 n=np.zeros(N,int);w=np.zeros(N);s2=np.zeros(N);seen=set();ids=[];uniforms=[];lp=lf=lc=0.;depth=term=0;maxscore=maxp=maxlog=0.;minp=1.;rows=[];wave=[]
 for start in range(0,128,4):
  Q=np.divide(w.astype(np.float32),n.astype(np.float32),out=np.zeros(N,np.float32),where=n>0)
  U=Q+np.float32(1.4)*prior.astype(np.float32)*np.sqrt(np.float32(start))/(np.float32(1)+n.astype(np.float32))
  h=.35 if start==32 else 0.;phase=start//4+1
  ps=[];y=[]
  for l,e in enumerate(es[start:start+4]):
   t=start+l;rs=e['root_selection'];law=rs['law'];a=np.asarray(law['actions'],int);sc=np.asarray(law['scores']);p=np.asarray(law['p'])
   assert law['kind']=='iid' and law['proposal']=='p' and law['source']==a0 and law['beta_p']==law['beta_q']==j['beta'] and all(law['active'])
   assert abs(law['logit_h']-({'plus74':h,'minus74':-h,'zero74':0.}[j['arm']]))<1e-15
   assert e['phase']==phase and e['complete_seq']==t and law['t']==t and rs['clock']==2*start+l and e['root_backup_clock']==2*start+4+l
   assert 'phase64' not in law and 'phase65' not in law and np.array_equal(np.sort(a),legal)
   er=float(abs(sc-U[a]).max());assert er<2e-6;maxscore=max(maxscore,er)
   def soft(f):
    z=j['beta']*sc+f*(a==a0);z-=z.max();r=np.exp(z);return r/r.sum()
   pf=soft(h);pc=soft(-h);want=pf if j['arm']=='plus74' else pc if j['arm']=='minus74' else soft(0.)
   er=float(abs(want-p).max());assert er<2e-12 and np.all(p>0) and abs(p.sum()-1)<1e-12;maxp=max(maxp,er);minp=min(minp,float(min(p.min(),pf.min(),pc.min())))
   pick=int(law['picked']);uu=float(law['u']);assert 0<=uu<1 and int(np.searchsorted(np.cumsum(p),uu,side='right'))==pick
   assert int(a[pick])==e['actions'][0] and rs['edge_index']==pick and rs['score']==law['scores'][pick]
   ll=math.log(p[pick]);lp+=ll;lf+=math.log(pf[pick]);lc+=math.log(pc[pick]);er=max(abs(ll-law['log_p']),abs(lp-law['log_p_path']));assert er<1e-9;maxlog=max(maxlog,er);uniforms.append(uu)
   ia=int(np.flatnonzero(a==a0)[0]);ib=int(np.flatnonzero(a==b0)[0]);pr=p[ia];r=p[ia]/(p[ia]+p[ib]);hb=lambda r: -r*math.log(r)-(1-r)*math.log1p(-r)
   js=entropy((pf+pc)/2)-.5*(entropy(pf)+entropy(pc));assert -1e-13<=js<=math.log(2)+1e-13
   ps.append([entropy(p),-ll,hb(pr),hb(r),js,pr,float(p[pick]),float(pf@np.log(pf/pc)),float(pc@np.log(pc/pf))])
   path0=e['actions'];assert path0 and e['parent_hashes'][0]==q['state_hash'];bx=b.copy();sd=side;terminal=False
   for ac in path0:
    assert bx[ac]==0 and not terminal
    bx[ac]=1 if sd==1 else 2;terminal=win(bx,ac) or np.all(bx!=0);sd=-sd
   mark=float(e['root_value']);lv=float(e['leaf_value']);assert mark==(-1)**len(path0)*lv and abs(lv)<=1 and bool(e['terminal'])==bool(terminal)
   if terminal:assert e['eval_id'] is None and lv==(-1 if win(bx,path0[-1]) else 0)
   else:
    ei=e['eval_id'];assert ei is not None and rec[ei]['state_hash']==e['leaf_hash'] and rec[ei]['side']==sd and rec[ei]['value']==lv;ids.append(ei)
   ac=path0[0];n[ac]+=1;w[ac]+=mark;s2[ac]+=mark*mark;depth+=len(path0);term+=int(terminal);seen.update(tuple(path0[:i]) for i in range(1,len(path0)+1))
  wave.append(np.mean(ps,axis=0).tolist()+[lf-lc])
  k=start+4
  if k in cuts:
   c=cuts[k];assert np.array_equal(n,c['n']) and abs(w-np.asarray(c['w'])).max()<1e-9 and abs(s2-np.asarray(c['s2'])).max()<1e-9 and len(seen)==c['visited_prefix_edges'] and np.array_equal(prior,c['prior'])
   rows.append(observ(n,w,s2,prior,depth,k,len(seen),rec[0]['value'],term)+[lf-lc,int(n[a0]),lp])
 assert sorted(ids)==list(range(1,len(rec)))
 assert j['arm']=='zero74' or abs(lp-(lf if j['arm']=='plus74' else lc))<1e-9
 return dict(job=j,tag=tag,sha256=sha(path),bytes=path.stat().st_size,evaluations=len(rec),max_score=maxscore,max_probability=maxp,max_log=maxlog,min_probability=minp,rows=rows,waves=wave,uniforms=uniforms)
def main():
 r=read(S/'REGISTRATION.json');assert sha(S/'mqft_train')==r['binary_sha'] and sha(S/'plan66.py')==r['plan_code_sha']
 assert all(sha(B/v['path'])==v['sha256'] for v in r['weights'].values())
 args=[]
 for tag,d in sorted(r['plans'].items()):
  p=S/'plans'/f'{tag}.json';assert sha(p)==d['sha256'];js=read(p);assert len(js)==d['n'] and len(list((S/'data'/tag).glob('*.gz')))==len(js) and not list((S/'data'/tag).glob('*.part'));args.extend((tag,j) for j in js)
 assert len(args)==3072
 out=[];rows=[];waves=[];meta=[];streams={}
 with cf.ProcessPoolExecutor(max_workers=12) as pool:
  for i,x in enumerate(pool.map(one,args,chunksize=8)):
   j=x['job'];sd=j['seed'];u=x.pop('uniforms');assert sd not in streams;streams[sd]=u
   meta.append([j['generation'],j['core'],j['rep'],int(j['arm']=='counter')]);rows.append(x.pop('rows'));waves.append(x.pop('waves'));out.append(x)
   if (i+1)%768==0:print('AUDIT',i+1,flush=True)
 np.savez_compressed(S/'NATIVE66.npz',meta=np.array(meta),rows=np.array(rows),waves=np.array(waves),fields=np.array(FIELDS+['L','Na','logP']),wave_fields=np.array(['H','surprisal','Hindicator','Hpair','JS','p_a','p_selected','KL_FC','KL_CF','L']),cuts=np.array([4,8,16,32,64,128]))
 dump(S/'RNG_INPUT66.json',[dict(seed=int(k),u=v) for k,v in sorted(streams.items())])
 report=dict(n=len(out),backups=len(out)*128,evaluations=sum(x['evaluations'] for x in out),raw_gzip_bytes=sum(x['bytes'] for x in out),max_score=max(x['max_score'] for x in out),max_probability=max(x['max_probability'] for x in out),max_log=max(x['max_log'] for x in out),min_probability=min(x['min_probability'] for x in out),streams=len(streams),array_sha=sha(S/'NATIVE66.npz'),jobs=out)
 dump(S/'AUDIT66.json',report);print('AUDIT COMPLETE',{k:v for k,v in report.items() if k!='jobs'},flush=True)
if __name__=='__main__':main()
