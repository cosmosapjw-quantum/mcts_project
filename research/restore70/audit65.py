"""Independent complex matrix instrument replay, native paths and high-budget references."""
from channel64 import *
import gzip
RX=np.cos(.35)*I-1j*np.sin(.35)*X
def bloch(r):return np.array([np.trace(r@a).real for a in (X,Y,Z)])
def rho(v):return (I+v[0]*X+v[1]*Y+v[2]*Z)/2
def free(r,phi,eta):
 U=np.diag([np.exp(-.5j*phi),np.exp(.5j*phi)]);v=U@r@dagger(U)
 return (1+eta)/2*v+(1-eta)/2*Z@v@Z
def effects(r):
 e=np.array([.35*r+.65,.35*r]);A=dagger(RX)@np.diag(np.sqrt(e))@RX;B=dagger(RX)@np.diag(np.sqrt(1-e))@RX
 assert abs(dagger(A)@A+dagger(B)@B-I).max()<2e-15
 return e,A,B
def checkrho(r):
 assert abs(r-dagger(r)).max()<1e-12 and abs(np.trace(r)-1)<1e-12
 e=np.linalg.eigvalsh(r);assert e.min()>-1e-11
 return float(e.min()),float(-e[e>0]@np.log(e[e>0])),float(np.trace(r@r).real)
FIELDS65=FIELDS+['Na','Nb','Npair','path_nll','recommended']
WAVE65=['tau','X','Y','Z','entropy','purity','post_X','post_Y','post_Z','pair_mass','q_a','base_a','reset_a','abs_memory_delta','measurement_TV','min_eig','pair_hits','a_hits','b_hits','probability_change_within_wave']
def audit_one(arg):
 tag,j=arg;path=S/'data'/tag/(j['id']+'.json.gz')
 with gzip.open(path,'rt') as f:d=json.load(f)
 assert d['schema']=='mqft.trained.phase65.v1' and d['input']==j
 q=d['trace'];assert q['schema']=='mqft.trained.control61.v1'
 for name in ('id','moves','seed','budget','width','beta','generation','core','rep'):assert q['job'][name]==j[name]
 assert q['job']['cohort']=='phase65' and q['job']['source']==j['sources'][0]
 b=board(j['moves']);side=1 if len(j['moves'])%2==0 else -1;legal=np.flatnonzero(b==0)
 assert q['board']==b.tolist() and q['side']==side
 aa,bb=j['sources'];assert aa in legal and bb in legal and aa!=bb
 BUD=j['budget'];width=j['width'];mode=j['arm'];es=q['events'];assert len(es)==BUD
 rec=q['evaluations'];assert rec[0]['state_hash']==q['state_hash'] and rec[0]['side']==side and [r['id'] for r in rec]==list(range(len(rec)))
 cuts={c['k']:c for c in q['cuts']};assert set(cuts)=={k for k in (4,8,16,32,64,128,256,512,1024) if k<=BUD}
 prior=np.asarray(cuts[4]['prior']);assert abs(prior.sum()-1)<2e-6 and np.all(prior[legal]>0) and np.all(prior[b!=0]==0)
 n=np.zeros(N,int);w=np.zeros(N);s2=np.zeros(N);depth=term=0;seen=set();ids=[];uniforms=[];lp=0.;exposure=0.;lastgap=0.
 persist=mode.startswith('persist') or mode=='reset_plus'
 state=rho([0.,-.8 if mode=='persist_minus' else 0. if mode=='persist_dephased' else .8,0.])
 maxscore=maxp=maxstate=maxlog=0.;mineig=1.;minmargin=1.;rows=[];waves=[]
 for start in range(0,BUD,width):
  t=start//width;tau=start/BUD;dt=width/BUD
  Q=np.divide(w.astype(np.float32),n.astype(np.float32),out=np.zeros(N,np.float32),where=n>0)
  U=Q+np.float32(1.4)*prior.astype(np.float32)*np.sqrt(np.float32(start))/(np.float32(1)+n.astype(np.float32))
  gap=float(U[aa])-float(U[bb]);phase=6*lastgap*dt if start else 0.
  if start:exposure+=lastgap*dt
  lastgap=gap
  zz=j['beta']*U[legal].astype(float);zz-=zz.max();base=np.exp(zz);base/=base.sum()
  pa,pb=[float(base[np.searchsorted(legal,a)]) for a in (aa,bb)];m=pa+pb;r=pa/m
  e0,A,Bm=effects(r) if persist else (None,None,None)
  xx=[];yhit=0;ahit=bhit=0;qp=[]
  reset=rho_channel(r,np.pi/2+6*exposure,.8*np.exp(-1.2*tau))
  reset_a=float(m*np.trace(A@reset@dagger(A)).real) if persist else pa
  for l,e in enumerate(es[start:start+width]):
   k=start+l;rs=e['root_selection'];law=rs['law'];a=np.asarray(law['actions'],int);sc=np.asarray(law['scores']);p=np.asarray(law['p'])
   assert law['kind']=='iid' and law['proposal']=='p' and law['source'] is None and law['logit_h']==0. and law['beta_p']==law['beta_q']==j['beta'] and all(law['active'])
   assert e['phase']==t+1 and e['complete_seq']==k and law['t']==k and rs['clock']==2*start+l and e['root_backup_clock']==2*start+width+l
   assert np.array_equal(np.sort(a),legal);se=float(abs(sc-U[a]).max());maxscore=max(maxscore,se);assert se<2e-6
   z=j['beta']*sc;z-=z.max();base0=np.exp(z);base0/=base0.sum();ia=int(np.flatnonzero(a==aa)[0]);ib=int(np.flatnonzero(a==bb)[0]);assert abs(base0[ia]-pa)<1e-12 and abs(base0[ib]-pb)<1e-12
   pre=state.copy()
   if persist:
    if l==0 and start:state=free(state,phase,np.exp(-1.2*dt))
    if mode=='reset_plus':state=reset.copy()
    if mode=='persist_dephased':state=np.diag(np.diag(state))
    prepared=state.copy()
    vmin,ent,pur=checkrho(prepared);mineig=min(mineig,vmin)
    qa=float(m*np.trace(A@prepared@dagger(A)).real);qb=float(m*np.trace(Bm@prepared@dagger(Bm)).real)
    assert abs(qa+qb-m)<1e-12
    ph=law['phase65'];assert 'phase64' not in law and ph['sources']==j['sources'] and ph['wave']==t
    ex=[tau,exposure,phase if l==0 else 0.,m,r,*e0,qa];ac=[ph[x] for x in ('tau','exposure','rotation','pair_mass','population_a','e0','e1','output_a')]
    err=max(float(abs(np.array(ex)-ac).max()),float(abs(bloch(pre)-ph['pre']).max()),float(abs(bloch(prepared)-ph['prepared']).max()))
    assert err<2e-11;maxstate=max(maxstate,err)
   elif mode=='fresh64':
    prepared=reset;vmin,ent,pur=checkrho(prepared);mineig=min(mineig,vmin)
    qa=float(m*readout(prepared)[0]);qb=m-qa
    ph=law['phase64'];assert 'phase65' not in law and ph['sources']==j['sources'] and ph['wave']==t
    ex=[tau,exposure,np.pi/2+6*exposure,.8*np.exp(-1.2*tau),m,r,*bloch(prepared)[:2],qa]
    err=float(abs(np.array(ex)-[ph[x] for x in ('tau','exposure','theta','eta','pair_mass','population_a','x','y','output_a')]).max());assert err<2e-11;maxstate=max(maxstate,err)
   else:
    assert 'phase64' not in law and 'phase65' not in law
    qa,qb=pa,pb;prepared=rho([0.,0.,2*r-1]);vmin,ent,pur=checkrho(prepared)
   want=base0.copy();want[ia]=qa;want[ib]=qb
   err=float(abs(want-p).max());maxp=max(maxp,err);assert err<2e-11 and np.all(p>0) and abs(p.sum()-1)<1e-12
   pick=int(law['picked']);uu=float(law['u']);cdf=np.cumsum(p);assert 0<=uu<1 and int(np.searchsorted(cdf,uu,side='right'))==pick
   minmargin=min(minmargin,uu-(cdf[pick-1] if pick else 0.),cdf[pick]-uu)
   action=int(a[pick]);assert action==e['actions'][0] and rs['edge_index']==pick and rs['score']==law['scores'][pick]
   if persist:
    assert ph['outcome']==action
    K=A if action==aa else Bm if action==bb else I
    s=K@prepared@dagger(K);d0=float(np.trace(s).real);assert d0>0
    state=s/d0
    err=float(abs(bloch(state)-ph['post']).max());maxstate=max(maxstate,err);assert err<2e-11
    checkrho(state)
    if action not in (aa,bb):assert abs(state-prepared).max()<1e-13
   else:state=prepared
   v=bloch(prepared);vp=bloch(state);mtv=float(np.linalg.norm(vp-v)/2)
   xx.append([tau,*v,ent,pur,*vp,m,qa,pa,reset_a,abs(qa-reset_a),mtv,vmin])
   qp.append(qa);ahit+=int(action==aa);bhit+=int(action==bb)
   ll=math.log(p[pick]);lp+=ll;le=max(abs(ll-law['log_p']),abs(lp-law['log_p_path']));maxlog=max(maxlog,le);assert le<1e-9
   uniforms.append(uu);path0=e['actions'];assert path0 and e['parent_hashes'][0]==q['state_hash'];bx=b.copy();sd=side;terminal=False
   for ac0 in path0:
    assert bx[ac0]==0 and not terminal
    bx[ac0]=1 if sd==1 else 2;terminal=win(bx,ac0) or np.all(bx!=0);sd=-sd
   mark=float(e['root_value']);lv=float(e['leaf_value']);assert mark==(-1)**len(path0)*lv and abs(lv)<=1 and bool(e['terminal'])==bool(terminal)
   if terminal:assert e['eval_id'] is None and lv==(-1 if win(bx,path0[-1]) else 0)
   else:
    ei=e['eval_id'];assert ei is not None and rec[ei]['state_hash']==e['leaf_hash'] and rec[ei]['side']==sd and rec[ei]['value']==lv;ids.append(ei)
   n[action]+=1;w[action]+=mark;s2[action]+=mark*mark;depth+=len(path0);term+=int(terminal);seen.update(tuple(path0[:i]) for i in range(1,len(path0)+1))
  waves.append(np.mean(xx,axis=0).tolist()+[ahit+bhit,ahit,bhit,max(qp)-min(qp)])
  k=start+width
  if k in cuts:
   c=cuts[k];assert np.array_equal(n,c['n']) and abs(w-np.asarray(c['w'])).max()<1e-9 and abs(s2-np.asarray(c['s2'])).max()<1e-9 and len(seen)==c['visited_prefix_edges'] and np.array_equal(prior,c['prior'])
   rows.append(observ(n,w,s2,prior,depth,k,len(seen),rec[0]['value'],term)+[int(n[aa]),int(n[bb]),int(n[aa]+n[bb]),-lp,int(n.argmax())])
 assert sorted(ids)==list(range(1,len(rec)))
 return dict(job=j,tag=tag,sha256=sha(path),bytes=path.stat().st_size,evaluations=len(rec),max_score=maxscore,max_probability=maxp,max_state=maxstate,max_log=maxlog,min_eig=mineig,min_margin=minmargin,rows=rows,waves=waves,uniforms=uniforms,counts=n,marks=w,prior=prior)
def main():
 r=read(S/'REGISTRATION.json');assert sha(S/'mqft_train')==r['binary_sha'] and sha(S/'plan65.py')==r['plan_code_sha']
 assert sha(B/'mqft_restore64_20261126/CALIBRATION64.json')==r['prior_calibration_sha']
 assert all(sha(B/v['path'])==v['sha256'] for v in r['weights'].values())
 args=[]
 for tag,d in sorted(r['plans'].items()):
  p=S/'plans'/f'{tag}.json';assert sha(p)==d['sha256'];js=read(p);assert len(js)==d['n'] and len(list((S/'data'/tag).glob('*.gz')))==len(js) and not list((S/'data'/tag).glob('*.part'));args.extend((tag,j) for j in js)
 assert len(args)==4800
 out=[];rows=[];waves=[];meta=[];nr=[];wr=[];pr=[];refs=[];streams={}
 with cf.ProcessPoolExecutor(max_workers=12) as pool:
  for i,x in enumerate(pool.map(audit_one,args,chunksize=8)):
   j=x['job'];sd=j['seed'];u=x.pop('uniforms')
   if sd in streams:assert u==streams[sd]
   else:streams[sd]=u
   n=x.pop('counts');w=x.pop('marks');p=x.pop('prior');v=x.pop('rows');wave=x.pop('waves')
   if j['arm']=='reference':refs.append((j['core'],j['rep'],n,w,p,v))
   else:
    meta.append([j['generation'],j['core'],j['rep'],r['arms'].index(j['arm'])]);rows.append(v);waves.append(wave);nr.append(n);wr.append(w);pr.append(p)
   out.append(x)
   if (i+1)%800==0:print('AUDIT',i+1,flush=True)
 assert len(meta)==4608 and len(refs)==192
 np.savez_compressed(S/'PHASE65.npz',meta=np.array(meta),rows=np.array(rows),waves=np.array(waves),counts=np.array(nr),marks=np.array(wr),prior=np.array(pr),fields=np.array(FIELDS65),wave_fields=np.array(WAVE65),cuts=np.array([4,8,16,32,64,128]),arms=np.array(r['arms']))
 refs.sort(key=lambda x:x[:2]);np.savez_compressed(S/'REFERENCE65.npz',meta=np.array([x[:2] for x in refs]),counts=np.array([x[2] for x in refs]),marks=np.array([x[3] for x in refs]),prior=np.array([x[4] for x in refs]),rows=np.array([x[5] for x in refs]))
 dump(S/'RNG_INPUT65.json',[dict(seed=int(k),u=v) for k,v in sorted(streams.items())])
 report=dict(n=len(out),phase_paths=4608,reference_paths=192,backups=sum(x['job']['budget'] for x in out),evaluations=sum(x['evaluations'] for x in out),raw_gzip_bytes=sum(x['bytes'] for x in out),max_score=max(x['max_score'] for x in out),max_probability=max(x['max_probability'] for x in out),max_state=max(x['max_state'] for x in out),max_log=max(x['max_log'] for x in out),min_eig=min(x['min_eig'] for x in out),min_margin=min(x['min_margin'] for x in out),streams=len(streams),array_sha=sha(S/'PHASE65.npz'),reference_sha=sha(S/'REFERENCE65.npz'),jobs=out)
 dump(S/'AUDIT65.json',report);print('AUDIT COMPLETE',{k:v for k,v in report.items() if k!='jobs'},flush=True)
if __name__=='__main__':main()
