from base61 import *
import gzip
MASK=(1<<64)-1
def u(seed,id,k):
 x=((seed^0x67f00dca12345678)+0x9e3779b97f4a7c15*(id*5+k+1))&MASK
 x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK;x=((x^(x>>27))*0x94d049bb133111eb)&MASK
 return ((x^(x>>31))>>11)/2**53
def fault(j,r):
 i=r['id'];mode=j['arm'];common=mode=='rep_common' or (mode.endswith('_mix') and u(j['seed'],i,0)<.5)
 m=[int(i>0 and mode not in ('clean','reference') and u(j['seed'],i,1 if common else k+1)<.12) for k in range(3)]
 s=[m[0]^m[1],m[1]^m[2]];c={(1,0):1,(1,1):2,(0,1):3,(0,0):0}[tuple(s)] if mode.startswith('rep') else 0
 residual=m[0]^(c==1);assert r['mask']==m and r['syndrome']==s and r['correction']==c and r['residual']==residual
 assert r['delivered']==(-r['clean'] if residual else r['clean'])
 if mode.startswith('rep'):assert residual==(sum(m)>=2)
 return [sum(m),m[0],int(any(s)),int(residual),abs(r['clean']-r['delivered']),int(any(s) and not residual),int(sum(m)==3)]
def one(arg):
 tag,j=arg;p=S/'data'/tag/(j['id']+'.json.gz')
 with gzip.open(p,'rt') as f:d=json.load(f)
 assert d['schema']=='mqft.trained.fault67.v1' and d['input']==j
 q=d['trace'];assert q['schema']=='mqft.trained.control61.v1'
 b=board(j['moves']);side=1 if len(j['moves'])%2==0 else -1;legal=np.flatnonzero(b==0);BUD=j['budget']
 assert q['board']==b.tolist() and q['side']==side and q['job']['cohort']=='fault67'
 for k in ('id','moves','seed','budget','width','beta','generation','core','rep'):assert q['job'][k]==j[k]
 rec=q['evaluations'];fr=q['fault67'];assert len(rec)==len(fr) and rec[0]['state_hash']==q['state_hash'] and rec[0]['side']==side
 F=[]
 for i,(e,r) in enumerate(zip(rec,fr)):
  assert e['id']==r['id']==i and e['value']==r['delivered'];v=fault(j,r)
  if i:F.append(v)
 assert not any(fault(j,fr[0]))
 cuts={c['k']:c for c in q['cuts']};assert set(cuts)=={k for k in (4,8,16,32,64,128,256,512,1024) if k<=BUD}
 prior=np.asarray(cuts[4]['prior']);assert abs(prior.sum()-1)<2e-6 and np.all(prior[legal]>0) and np.all(prior[b!=0]==0)
 n=np.zeros(N,int);w=np.zeros(N);s2=np.zeros(N);depth=term=0;seen=set();ids=[];uniforms=[];lp=0.;maxp=maxscore=maxlog=0.;rows=[];es=q['events'];assert len(es)==BUD
 for start in range(0,BUD,4):
  Q=np.divide(w.astype(np.float32),n.astype(np.float32),out=np.zeros(N,np.float32),where=n>0)
  U=Q+np.float32(1.4)*prior.astype(np.float32)*np.sqrt(np.float32(start))/(np.float32(1)+n.astype(np.float32))
  for l,e in enumerate(es[start:start+4]):
   k=start+l;rs=e['root_selection'];law=rs['law'];a=np.asarray(law['actions'],int);sc=np.asarray(law['scores']);pr=np.asarray(law['p'])
   assert law['kind']=='iid' and law['proposal']=='p' and law['source'] is None and law['logit_h']==0. and law['beta_p']==law['beta_q']==j['beta'] and all(law['active'])
   assert e['phase']==start//4+1 and e['complete_seq']==k and law['t']==k and rs['clock']==2*start+l and e['root_backup_clock']==2*start+4+l
   assert np.array_equal(np.sort(a),legal);se=float(abs(sc-U[a]).max());assert se<2e-6;maxscore=max(maxscore,se)
   z=j['beta']*sc;z-=z.max();want=np.exp(z);want/=want.sum();pe=float(abs(want-pr).max());assert pe<2e-12 and np.all(pr>0);maxp=max(maxp,pe)
   pick=law['picked'];uu=law['u'];assert int(np.searchsorted(np.cumsum(pr),uu,side='right'))==pick and int(a[pick])==e['actions'][0] and rs['edge_index']==pick
   ll=math.log(pr[pick]);lp+=ll;le=max(abs(ll-law['log_p']),abs(lp-law['log_p_path']));assert le<1e-9;maxlog=max(maxlog,le);uniforms.append(uu)
   pa=e['actions'];assert pa and e['parent_hashes'][0]==q['state_hash'];bx=b.copy();sd=side;terminal=False
   for a0 in pa:
    assert bx[a0]==0 and not terminal;bx[a0]=1 if sd==1 else 2;terminal=win(bx,a0) or np.all(bx!=0);sd=-sd
   mark=float(e['root_value']);lv=float(e['leaf_value']);assert mark==(-1)**len(pa)*lv and bool(e['terminal'])==bool(terminal)
   if terminal:assert e['eval_id'] is None and lv==(-1 if win(bx,pa[-1]) else 0)
   else:
    i=e['eval_id'];assert i is not None and rec[i]['state_hash']==e['leaf_hash'] and rec[i]['side']==sd and rec[i]['value']==lv;ids.append(i)
   a0=pa[0];n[a0]+=1;w[a0]+=mark;s2[a0]+=mark*mark;depth+=len(pa);term+=int(terminal);seen.update(tuple(pa[:i]) for i in range(1,len(pa)+1))
  k=start+4
  if k in cuts:
   c=cuts[k];assert np.array_equal(n,c['n']) and abs(w-np.asarray(c['w'])).max()<1e-9 and abs(s2-np.asarray(c['s2'])).max()<1e-9 and len(seen)==c['visited_prefix_edges'] and np.array_equal(prior,c['prior'])
   rows.append(observ(n,w,s2,prior,depth,k,len(seen),rec[0]['value'],term)+[int(n.argmax())])
 assert sorted(ids)==list(range(1,len(rec)))
 return dict(job=j,tag=tag,sha256=sha(p),bytes=p.stat().st_size,evaluations=len(rec),fault_counts=np.asarray(F).sum(0).tolist(),max_probability=maxp,max_score=maxscore,max_log=maxlog,rows=rows,counts=n,marks=w,uniforms=uniforms)
def main():
 reg=read(S/'REGISTRATION.json');assert sha(S/'mqft_train')==reg['binary_sha'] and sha(S/'plan67.py')==reg['plan_code_sha']
 assert all(sha(B/v['path'])==v['sha256'] for v in reg['weights'].values())
 args=[]
 for tag,d in sorted(reg['plans'].items()):
  p=S/'plans'/f'{tag}.json';assert sha(p)==d['sha256'];js=read(p);assert len(js)==d['n'] and len(list((S/'data'/tag).glob('*.gz')))==len(js);args.extend((tag,j) for j in js)
 assert len(args)==3200
 out=[];rows=[];meta=[];counts=[];marks=[];faults=[];refs=[];streams={}
 with cf.ProcessPoolExecutor(max_workers=12) as pool:
  for i,x in enumerate(pool.map(one,args,chunksize=8)):
   j=x['job'];u=x.pop('uniforms');sd=j['seed']
   if sd in streams:assert streams[sd]==u
   else:streams[sd]=u
   n=x.pop('counts');w=x.pop('marks');r=x.pop('rows')
   if j['arm']=='reference':refs.append((j['core'],j['rep'],n,w,r))
   else:
    meta.append([j['generation'],j['core'],j['rep'],reg['arms'].index(j['arm'])]);rows.append(r);counts.append(n);marks.append(w);faults.append([x['evaluations']-1,*x['fault_counts']])
   out.append(x)
   if (i+1)%800==0:print('AUDIT',i+1,flush=True)
 refs.sort(key=lambda x:x[:2]);assert len(refs)==128 and len(meta)==3072
 np.savez_compressed(S/'NATIVE67.npz',meta=np.array(meta),rows=np.array(rows),counts=np.array(counts),marks=np.array(marks),faults=np.array(faults),fields=np.array(FIELDS+['recommended']),arms=np.array(reg['arms']),cuts=np.array([4,8,16,32,64,128]),fault_fields=np.array(['n','carrier_flips','first_flip','detected','residual','absolute_error','detected_success','triple']))
 np.savez_compressed(S/'REFERENCE67.npz',meta=np.array([x[:2] for x in refs]),counts=np.array([x[2] for x in refs]),marks=np.array([x[3] for x in refs]))
 dump(S/'RNG_INPUT67.json',[dict(seed=int(k),u=v) for k,v in sorted(streams.items())])
 report=dict(n=len(out),low_budget=3072,reference=128,backups=sum(x['job']['budget'] for x in out),evaluations=sum(x['evaluations'] for x in out),raw_gzip_bytes=sum(x['bytes'] for x in out),max_probability=max(x['max_probability'] for x in out),max_score=max(x['max_score'] for x in out),max_log=max(x['max_log'] for x in out),streams=len(streams),array_sha=sha(S/'NATIVE67.npz'),reference_sha=sha(S/'REFERENCE67.npz'),jobs=out)
 dump(S/'AUDIT67.json',report);print('AUDIT COMPLETE',{k:v for k,v in report.items() if k!='jobs'})
if __name__=='__main__':main()
