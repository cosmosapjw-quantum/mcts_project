from arena68 import *
MASK=(1<<64)-1
def fault(seed,id,k):
 x=((seed^0x67f00dca12345678)+0x9e3779b97f4a7c15*(id*5+k+1))&MASK
 x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK;x=((x^(x>>27))*0x94d049bb133111eb)&MASK
 return ((x^(x>>31))>>11)/2**53
def stat(n,w,s2,p,budget,legal):
 pi=n/budget;q=np.divide(w,n,out=np.zeros(225),where=n>0);u=n>0;m=w.sum()/budget
 var=s2.sum()/budget-m*m;within=(s2[u]-w[u]*q[u]).sum()/budget;between=(pi*(q-m)**2).sum()
 assert abs(var-within-between)<2e-12 and min(var,within,between)>-1e-9
 return dict(H=entropy(pi),Hnorm=entropy(pi)/math.log(legal) if legal>1 else 0.,Hprior=entropy(p),KL=float(pi[u]@np.log(pi[u]/p[u])),mean_value=float(m),variance=max(0.,float(var)),within=max(0.,float(within)),between=max(0.,float(between)),top=float(pi.max()),neff=float(1/(pi@pi)))
def audit_search(r,m,arm,sd,budget):
 b=board(m);side=1 if len(m)%2==0 else -1;legal=np.flatnonzero(b==0);assert r['schema']=='mqft.arena68.search.v1' and r['moves']==m and r['side']==side and r['arm']==arm and r['seed']==sd and r['budget']==budget
 rec=r['evaluations'];es=r['events'];cuts={x['k']:x for x in r['cuts']};assert len(es)==budget and set(cuts)=={k for k in (4,8,16,32,64,128) if k<=budget}
 assert rec[0]['state_hash']==r['state_hash'] and rec[0]['side']==side
 maxp=maxscore=maxlog=maxprior=maxvalue=0.;flips=bad=detected=corrected=0;abs_err=0.
 for i,e in enumerate(rec):
  assert e['id']==i;w0=np.asarray(e['wdl']);assert np.all(w0>=0) and abs(w0.sum()-1)<=.001
  clean=float(np.float32(e['side']*(w0[0]-w0[1])/w0.sum()));maxvalue=max(maxvalue,abs(clean-e['clean']));assert abs(clean-e['clean'])<1e-7
  bits=[int(i>0 and arm!='clean' and fault(sd,i,k+1)<.12) for k in range(3)];sy=[bits[0]^bits[1],bits[1]^bits[2]]
  co={(0,0):0,(1,0):1,(1,1):2,(0,1):3}[tuple(sy)] if arm=='recovery' else 0
  res=bits[0]^(co==1);assert e['mask']==bits and e['syndrome']==sy and e['corrected']==co and e['delivered']==(-e['clean'] if res else e['clean'])
  if i:flips+=sum(bits);bad+=res;detected+=any(sy);corrected+=int(any(sy) and not res);abs_err+=abs(e['delivered']-e['clean'])
 def prior(e,legal):
  a=np.array(e['scores']);assert np.array_equal(a[:,0],legal)
  z=np.exp((a[:,1]-a[:,1].max())/32.);v=(1-1e-6)*z/z.sum()+1e-6/len(legal)
  x=np.zeros(225);x[legal]=v.astype(np.float32);return x
 p=prior(rec[0],legal);p0=np.asarray(cuts[4]['prior']);maxprior=float(abs(p-p0).max());assert maxprior<2e-7
 n=np.zeros(225,int);w=np.zeros(225);s2=np.zeros(225);lp=0.;depth=0;term=0;ids=[];H=[];sur=[];wave=[];summ=[]
 for start in range(0,budget,4):
  Q=np.divide(w.astype(np.float32),n.astype(np.float32),out=np.zeros(225,np.float32),where=n>0)
  U=Q+np.float32(1.4)*p0.astype(np.float32)*np.sqrt(np.float32(start))/(np.float32(1)+n.astype(np.float32))
  for l,e in enumerate(es[start:start+4]):
   k=start+l;rs=e['root_selection'];law=rs['law'];a=np.asarray(law['actions']);s=np.asarray(law['scores']);pr=np.asarray(law['p'])
   assert np.array_equal(np.sort(a),legal) and law['kind']=='iid' and law['proposal']=='p' and law['source'] is None and law['beta_p']==law['beta_q']==10/3 and law['logit_h']==0 and all(law['active'])
   assert e['complete_seq']==law['t']==k and e['phase']==start//4+1 and rs['clock']==2*start+l and e['root_backup_clock']==2*start+4+l
   er=float(abs(s-U[a]).max());maxscore=max(maxscore,er);assert er<3e-6
   z=s*(10/3);z-=z.max();v=np.exp(z);v/=v.sum();er=float(abs(v-pr).max());maxp=max(maxp,er);assert er<2e-12 and np.all(pr>0)
   pick=law['picked'];u=law['u'];assert int(np.searchsorted(np.cumsum(pr),u,side='right'))==pick and int(a[pick])==e['actions'][0] and rs['edge_index']==pick
   ll=math.log(pr[pick]);lp+=ll;er=max(abs(ll-law['log_p']),abs(lp-law['log_p_path']));maxlog=max(maxlog,er);assert er<1e-9
   H.append(entropy(pr));sur.append(-ll)
   pa=e['actions'];assert pa and e['parent_hashes'][0]==r['state_hash'];bx=b.copy();ss=side;terminal=False
   for aa in pa:
    assert bx[aa]==0 and not terminal;bx[aa]=1 if ss==1 else 2;terminal=won(bx,aa) or np.all(bx!=0);ss=-ss
   lv=float(e['leaf_value']);mark=float(e['root_value']);assert mark==(-1)**len(pa)*lv and bool(e['terminal'])==bool(terminal)
   if terminal:assert e['eval_id'] is None and lv==(-1 if won(bx,pa[-1]) else 0)
   else:
    i=e['eval_id'];assert i is not None and rec[i]['state_hash']==e['leaf_hash'] and rec[i]['side']==ss and rec[i]['delivered']==lv;ids.append(i)
    prior(rec[i],np.flatnonzero(bx==0))
   ac=pa[0];n[ac]+=1;w[ac]+=mark;s2[ac]+=mark*mark;depth+=len(pa);term+=int(terminal)
  k=start+4
  if k in cuts:
   c=cuts[k];assert np.array_equal(n,c['n']) and abs(w-c['w']).max()<1e-9 and abs(s2-c['s2']).max()<1e-9 and np.array_equal(p0,c['prior'])
   q=stat(n,w,s2,p0,k,len(legal));q.update(k=k,depth=depth/k,terminal_fraction=term/k,conditional_H=float(np.mean(H)),surprisal=float(np.mean(sur)));summ.append(q)
 assert sorted(ids)==list(range(1,len(rec))) and n.argmax()==r['action']
 return dict(cuts=summ,evals=len(rec),fault_records=len(rec)-1,carrier_flips=flips,sign_errors=bad,syndromes=detected,corrected=corrected,value_abs_error=abs_err,max_probability=maxp,max_score=maxscore,max_log=maxlog,max_prior=maxprior,max_value=maxvalue,chosen_surprisal=-math.log(n[r['action']]/budget),elapsed=r['elapsed_ns']/1e9,root_value=rec[0]['clean'])
def audit_game(x):
 j=x['job'];p=S/'data'/(j['id']+'.json.gz');assert sha(p)==x['sha256'] and p.stat().st_size==x['bytes']
 with gzip.open(p,'rt') as f:d=json.load(f)
 assert d['job']==j and d['status']==x['status'] and d['winner']==x['winner'] and d['moves'][:len(j['opening'])]==j['opening']
 m=j['opening'].copy();b=board(m);assert not any(won(board(m[:i+1]),a) for i,a in enumerate(m));native=[];terminal=False;winner=None
 for row in d['rows']:
  assert not terminal;ply=len(m);side=1 if ply%2==0 else -1;arm=j['focal'] if side==j['color'] else j['opponent']
  assert row['ply']==ply and row['side']==side and row['arm']==arm
  a=row['action'];assert b[a]==0
  if arm=='rapfi':
   assert row['engine']=='rapfi';q=row['external'];assert q['identity']==read(S/'REGISTRATION.json')['external'] and parse(q['stdout'],m)==a and q['timeout_ms']==j['external_ms']
  else:
   assert row['engine']=='native';q=audit_search(row['search'],m,arm,seed(j['seed'],ply),j['budget']);q.update(ply=ply,side=side,arm=arm);native.append(q)
  m.append(a);b[a]=1 if side==1 else 2
  terminal=won(b,a) or len(m)==225
  if terminal:winner=side if won(b,a) else 0
 assert m==d['moves'] and len(m)==x['plies']
 assert (d['status']=='complete')==terminal and (winner==d['winner'] if terminal else d['winner'] is None)
 assert len(native)==x['native'] and len(d['rows'])-len(native)==x['external']
 return dict(id=j['id'],job=j,status=x['status'],winner=winner,error=x['error'],plies=len(m),native=native,external=x['external'],elapsed=d['elapsed'],sha256=x['sha256'],bytes=x['bytes'])
def main():
 reg=read(S/'REGISTRATION.json');camp=read(S/'CAMPAIGN68.json');assert len(camp)==len(reg['jobs']) and {x['id'] for x in camp}=={x['id'] for x in reg['jobs']}
 with cf.ProcessPoolExecutor(max_workers=8) as p:out=list(p.map(audit_game,camp,chunksize=1))
 rows=[y for x in out for y in x['native']]
 report=dict(games=len(out),complete=sum(x['status']=='complete' for x in out),native_searches=len(rows),backups=len(rows)*reg['budget'],evals=sum(x['evals'] for x in rows),external_moves=sum(x['external'] for x in out),raw_bytes=sum(x['bytes'] for x in out),max_probability=max(x['max_probability'] for x in rows),max_score=max(x['max_score'] for x in rows),max_log=max(x['max_log'] for x in rows),max_prior=max(x['max_prior'] for x in rows),max_value=max(x['max_value'] for x in rows),rows=out)
 dump(S/'AUDIT68.json',report);print('AUDIT',{k:v for k,v in report.items() if k!='rows'},flush=True)
if __name__=='__main__':
 if '--smoke' in sys.argv:print(audit_search(read(S/'SMOKE68.json'),[112,113,97,98],'recovery',123,64))
 else:main()
