"""Independent full-path replay and complex128 reference of every completed wave."""
from channel64 import *
import gzip
CUTS=[4,8,16,32,64,128]
EXTRA=['Na','Nb','Npair','mean_mark_a','mean_mark_b','path_nll','base_path_nll','calibrated_nll','dephased_nll','score_action','root_best_visit_action']
WF=['tau','exposure','theta','eta','mass','r','X','Y','q_a','q_b','base_a','base_b','dephased_a','coherent_shift','local_TV','entropy_vN','purity','min_eig','mean_wave_mark','pair_hits','a_hits','b_hits']
def audit_one(arg):
 tag,j,fit=arg;path=S/'data'/tag/(j['id']+'.json.gz')
 with gzip.open(path,'rt') as f:d=json.load(f)
 assert d['schema']=='mqft.trained.phase64.v1' and d['input']==j
 q=d['trace'];assert q['schema']=='mqft.trained.control61.v1'
 for name in ('id','moves','seed','budget','width','beta','generation','core','rep'):assert q['job'][name]==j[name]
 assert q['job']['cohort']=='phase64' and q['job']['source']==j['sources'][0]
 b=board(j['moves']);side=1 if len(j['moves'])%2==0 else -1;legal=np.flatnonzero(b==0)
 assert q['board']==b.tolist() and q['side']==side
 aa,bb=j['sources'];assert aa in legal and bb in legal and aa!=bb
 width=j['width'];budget=j['budget'];mode=j['arm']
 es=q['events'];assert len(es)==budget
 rec=q['evaluations'];assert rec[0]['state_hash']==q['state_hash'] and rec[0]['side']==side and [r['id'] for r in rec]==list(range(len(rec)))
 cuts={c['k']:c for c in q['cuts']};assert set(cuts)==set(CUTS);prior=np.asarray(cuts[4]['prior'])
 assert abs(prior.sum()-1)<2e-6 and np.all(prior[legal]>0) and np.all(prior[b!=0]==0)
 n=np.zeros(N,int);w=np.zeros(N);s2=np.zeros(N);depth=term=0;seen=set();ids=[];uniforms=[];exposure=0.;lp=lbase=lfit=ldeph=score_action=0.
 maxscore=maxp=maxphase=maxlog=0.;minmargin=1.;min_eig=1.;pathwords=[];rows=[];waves=[];rho_saved=[];hashes=[]
 for start in range(0,budget,width):
  t=start//width;wave=es[start:start+width];tau=start/budget
  Q=np.divide(w.astype(np.float32),n.astype(np.float32),out=np.zeros(N,np.float32),where=n>0)
  U=Q+np.float32(1.4)*prior.astype(np.float32)*np.sqrt(np.float32(start))/(np.float32(1)+n.astype(np.float32))
  z=j['beta']*U[legal].astype(float);z-=z.max();base=np.exp(z);base/=base.sum()
  pa=float(base[np.searchsorted(legal,aa)]);pb=float(base[np.searchsorted(legal,bb)]);m=pa+pb;r=pa/m
  theta0={'plus':np.pi/2,'minus':-np.pi/2,'rotate0':0.}.get(mode,0.)
  active=mode in ('plus','minus','rotate0')
  theta=theta0+6*exposure if active else 0.
  eta=0. if mode=='dephased' else .8*np.exp(-1.2*tau)
  alpha=0. if mode=='baseline' else .35
  rho=rho_channel(r,theta,eta);eig=np.linalg.eigvalsh(rho);min_eig=min(min_eig,float(eig.min()))
  assert abs(rho-rho.conj().T).max()<1e-13 and abs(np.trace(rho)-1)<1e-13 and eig.min()>-1e-13
  out,xy=readout(rho,alpha);out_a=float(m*out);out_b=m-out_a
  if mode=='baseline':out_a,out_b=pa,pb
  deph=float(m*readout(rho_channel(r,0.,0.),alpha)[0])
  thfit=theta0+fit[2]*exposure if active else 0.
  etafit=0. if mode=='dephased' else fit[0]*np.exp(-fit[1]*tau)
  outf=float(m*readout(rho_channel(r,thfit,etafit),alpha)[0]) if mode!='baseline' else pa
  assert min(out_a,out_b)>0 and max(out_a,out_b)<m+1e-13
  # exact phase sign contrast at same conditional root state, independent from realized trajectory
  opposite=float(m*readout(rho_channel(r,theta+np.pi,eta),alpha)[0])
  assert abs((out_a+opposite)/2-deph)<1e-13
  mu=[];a_hits=b_hits=0
  for l,e in enumerate(wave):
   k=start+l;rs=e['root_selection'];d0=rs['law'];a=np.asarray(d0['actions'],int);sc=np.asarray(d0['scores']);p=np.asarray(d0['p'])
   assert d0['kind']=='iid' and d0['proposal']=='p' and d0['source'] is None and d0['logit_h']==0. and d0['beta_p']==d0['beta_q']==j['beta'] and all(d0['active'])
   assert e['phase']==t+1 and e['complete_seq']==k and d0['t']==k and rs['clock']==2*start+l and e['root_backup_clock']==2*start+width+l
   assert np.array_equal(np.sort(a),legal);se=float(abs(sc-U[a]).max());maxscore=max(maxscore,se);assert se<2e-6
   # Use recorded score array for independent softmax recomputation (same float32 convention).
   z=j['beta']*sc;z-=z.max();base0=np.exp(z);base0/=base0.sum();ia=int(np.flatnonzero(a==aa)[0]);ib=int(np.flatnonzero(a==bb)[0])
   assert abs(base0[ia]-pa)<1e-12 and abs(base0[ib]-pb)<1e-12
   want=base0.copy();want[ia]=out_a;want[ib]=out_b
   pe=float(abs(want-p).max());maxp=max(maxp,pe);assert pe<2e-12 and np.all(p>0) and abs(p.sum()-1)<1e-12
   assert abs(p[ia]+p[ib]-m)<1e-12
   ph=d0['phase64'];assert ph['sources']==j['sources'] and ph['wave']==t
   expected=[tau,exposure,theta,eta,m,r,*xy,out_a];measured=[ph[x] for x in ('tau','exposure','theta','eta','pair_mass','population_a','x','y','output_a')]
   er=float(abs(np.asarray(expected)-measured).max());maxphase=max(maxphase,er);assert er<2e-12
   pick=int(d0['picked']);uu=float(d0['u']);cdf=np.cumsum(p);assert 0<=uu<1 and int(np.searchsorted(cdf,uu,side='right'))==pick
   minmargin=min(minmargin,uu-(cdf[pick-1] if pick else 0.),cdf[pick]-uu)
   assert int(a[pick])==e['actions'][0] and rs['edge_index']==pick and rs['score']==d0['scores'][pick]
   ll=math.log(p[pick]);lp+=ll;lbase+=math.log(base0[pick])
   pfit=outf if pick==ia else m-outf if pick==ib else base0[pick]
   pdeph=deph if pick==ia else m-deph if pick==ib else base0[pick]
   lfit+=math.log(pfit);ldeph+=math.log(pdeph);score_action+=float(sc[pick])/budget
   er=max(abs(ll-d0['log_p']),abs(lp-d0['log_p_path']));maxlog=max(maxlog,er);assert er<1e-9
   uniforms.append(uu);pa0=e['actions'];pathwords.append(pa0);assert pa0 and e['parent_hashes'][0]==q['state_hash']
   bx=b.copy();sd=side;terminal=False
   for act in pa0:
    assert bx[act]==0 and not terminal
    bx[act]=1 if sd==1 else 2;terminal=win(bx,act) or np.all(bx!=0);sd=-sd
   mark=float(e['root_value']);lv=float(e['leaf_value']);assert mark==(-1)**len(pa0)*lv and abs(lv)<=1 and bool(e['terminal'])==bool(terminal)
   if terminal:assert e['eval_id'] is None and lv==(-1 if win(bx,pa0[-1]) else 0)
   else:
    ei=e['eval_id'];assert ei is not None and rec[ei]['state_hash']==e['leaf_hash'] and rec[ei]['side']==sd and rec[ei]['value']==lv;ids.append(ei)
   ac=pa0[0];n[ac]+=1;w[ac]+=mark;s2[ac]+=mark*mark;depth+=len(pa0);term+=int(terminal);seen.update(tuple(pa0[:i]) for i in range(1,len(pa0)+1))
   mu.append(mark);a_hits+=int(ac==aa);b_hits+=int(ac==bb)
  ev=eig[eig>0];Sv=float(-ev@np.log(ev));pur=float(np.trace(rho@rho).real)
  waves.append([tau,exposure,theta,eta,m,r,*xy,out_a,out_b,pa,pb,deph,out_a-deph,abs(out_a-pa),Sv,pur,float(eig.min()),float(np.mean(mu)),a_hits+b_hits,a_hits,b_hits])
  if start in (0,28,60,124):rho_saved.append(rho)
  k=start+width;exposure+=(float(U[aa])-float(U[bb]))*width/budget
  if k in cuts:
   c=cuts[k];assert np.array_equal(n,c['n']) and abs(w-np.asarray(c['w'])).max()<1e-9 and abs(s2-np.asarray(c['s2'])).max()<1e-9 and len(seen)==c['visited_prefix_edges'] and np.array_equal(prior,c['prior'])
   vals=observ(n,w,s2,prior,depth,k,len(seen),rec[0]['value'],term)
   vals+=[int(n[aa]),int(n[bb]),int(n[aa]+n[bb]),float(w[aa]/max(1,n[aa])),float(w[bb]/max(1,n[bb])),-lp,-lbase,-lfit,-ldeph,score_action,int(n.argmax())]
   rows.append(vals);hashes.append(hashlib.sha256(json.dumps(pathwords,separators=(',',':')).encode()).hexdigest())
 assert sorted(ids)==list(range(1,len(rec)))
 # semantic digest of actual engine law/marks, independent of phase metadata
 norm=[dict(actions=e['actions'],root=e['root_value'],leaf=e['leaf_value'],phase=e['phase'],law={k:v for k,v in e['root_selection']['law'].items() if k!='phase64'}) for e in es]
 native_hash=hashlib.sha256(json.dumps(norm,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return dict(native_hash=native_hash,job=j,tag=tag,sha256=sha(path),bytes=path.stat().st_size,evaluations=len(rec),max_score=maxscore,max_probability=maxp,max_phase=maxphase,max_log=maxlog,min_eigenvalue=min_eig,min_margin=minmargin,rows=rows,waves=waves,rho=np.array(rho_saved),hashes=hashes,uniforms=uniforms)
def main():
 reg=read(S/'REGISTRATION.json');fit=read(S/'CALIBRATION64.json')['fitted']
 assert sha(S/'mqft_train')==reg['binary_sha'] and sha(S/'plan64.py')==reg['plan_code_sha'] and sha(S/'ROOT_PREPARATION.json')==reg['root_preparation_sha']
 assert all(sha(B/v['path'])==v['sha256'] for v in reg['weights'].values())
 args=[]
 for tag,d in sorted(reg['plans'].items()):
  p=S/'plans'/f'{tag}.json';assert sha(p)==d['sha256'];js=read(p);assert len(js)==d['n'] and len(list((S/'data'/tag).glob('*.gz')))==len(js) and not list((S/'data'/tag).glob('*.part'))
  args.extend((tag,j,fit) for j in js)
 assert len(args)==6144
 out=[];rows=[];waves=[];rho=[];meta=[];hashes=[];streams={}
 with cf.ProcessPoolExecutor(max_workers=12) as pool:
  for i,r in enumerate(pool.map(audit_one,args,chunksize=8)):
   j=r['job'];sd=j['seed'];u=r.pop('uniforms')
   if sd in streams:assert u==streams[sd]
   else:streams[sd]=u
   meta.append([j['generation'],j['core'],j['rep'],reg['arms'].index(j['arm'])])
   rows.append(r.pop('rows'));waves.append(r.pop('waves'));rho.append(r.pop('rho'));hashes.append(r.pop('hashes'));out.append(r)
   if (i+1)%1024==0:print('AUDIT',i+1,flush=True)
 A=np.asarray(rows);M=np.asarray(meta);W=np.asarray(waves);H=np.asarray(hashes);D=np.asarray(rho)
 assert A.shape==(6144,6,len(FIELDS+EXTRA)) and W.shape==(6144,32,len(WF)) and D.shape==(6144,4,2,2)
 # At identical seeds real vs dephased have identical probabilities by design; compare actual path and marks.
 real=reg['arms'].index('real');deph=reg['arms'].index('dephased');ir=np.flatnonzero(M[:,3]==real);idp=np.flatnonzero(M[:,3]==deph)
 assert np.array_equal(M[ir,:3],M[idp,:3]) and np.array_equal(H[ir],H[idp])
 # Ignore calibrated NLL because independently fitted eta has no effect on theta=0 either.
 # Native law/marks must match exactly; complex128 reference arithmetic
 # can differ at rounding level in independently computed likelihoods.
 assert all(out[i]['native_hash']==out[j]['native_hash'] for i,j in zip(ir,idp))
 delta=abs(A[ir]-A[idp]);ref_delta=delta.max((0,1))
 print('REAL_DEPHASED_DIFF',dict(zip(FIELDS+EXTRA,ref_delta.tolist())),flush=True)
 exact=[i for i,k in enumerate(FIELDS+EXTRA) if k not in ('calibrated_nll','dephased_nll','base_path_nll','path_nll')]
 assert np.array_equal(A[ir][:,:,exact],A[idp][:,:,exact]) and delta.max()<1e-10
 np.savez_compressed(S/'PHASE64.npz',meta=M,rows=A,waves=W,rho=D,hashes=H,fields=np.array(FIELDS+EXTRA),wave_fields=np.array(WF),cuts=np.array(CUTS),arms=np.array(reg['arms']),rho_starts=np.array([0,28,60,124]))
 dump(S/'RNG_INPUT64.json',[dict(seed=int(k),u=v) for k,v in sorted(streams.items())])
 summary=dict(n=len(out),backups=len(out)*128,evaluations=sum(x['evaluations'] for x in out),raw_gzip_bytes=sum(x['bytes'] for x in out),max_score=max(x['max_score'] for x in out),max_probability=max(x['max_probability'] for x in out),max_phase=max(x['max_phase'] for x in out),max_log=max(x['max_log'] for x in out),min_eigenvalue=min(x['min_eigenvalue'] for x in out),min_margin=min(x['min_margin'] for x in out),real_dephased_path_matches=len(ir),real_dephased_native_law_exact=True,real_dephased_reference_max_difference=float(delta.max()),rng_streams=len(streams),wave_records=len(out)*32,jobs=out,array_sha=sha(S/'PHASE64.npz'))
 dump(S/'AUDIT64.json',summary);print('AUDIT COMPLETE',{k:v for k,v in summary.items() if k!='jobs'},flush=True)
if __name__=='__main__':main()
