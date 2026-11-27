"""Reauthenticate R79/R80 train, R81 lambda=0 validation, R82 independent test. No intervened histories."""
from base61 import *
from audit_native74 import one as audit9
from audit_rapfi74 import audit_search
import gzip
TAGS=['schedule0','schedule3','rapfi'];FAMS=['g0','g3','Rapfi']
ROOT_FIELDS=['ply','root_value','log_pair_mass','log_prior_max','prior_entropy_norm','log_pair_ratio','pair_distance','mean_radial_stone_distance','same_color_adjacency']
HIST_FIELDS=['prefix_pair_fraction','prefix_signed_mean','prefix_signed_variance','prefix_visit_entropy','prefix_visited_fraction','Q_a','Q_b','log_next_pair_mass','next_pair_log_odds','prefix_pair_martingale_residual']
def one(arg):
 st,group,tag,j,old=arg;g=TAGS.index(tag);L=15 if g==2 else 9;N=L*L
 d0=B/f'mqft_restore{st}_20261127';f=d0/'data'/tag/(j['id']+'.json.gz');h=sha(f)
 if old is not None:assert h==old['sha256']
 with gzip.open(f,'rt') as q:d=json.load(q)
 assert d['input']==j;tr=d['trace'];es=tr['events'];cs={c['k']:c for c in tr['cuts']};assert len(es)==128 and set(cs)=={4,8,16,32,64,128}
 if st==82:
  au=audit_search(tr,j['moves'],'clean',j['seed'],128,j['sources'][0],0.) if g==2 else audit9((tag,j))
  err0={k:au[k] for k in ['max_score','max_probability','max_log']}
 else:
  err0={k:old.get(k,old.get('errors',{}).get({'max_score':'score','max_probability':'probability','max_log':'log_path'}[k],0.)) for k in ['max_score','max_probability','max_log']}
 prior=np.array(cs[4]['prior'],float);legal=np.flatnonzero(prior>0);assert len(legal)==N-len(j['moves']) and abs(prior.sum()-1)<2e-6
 assert set(legal)==set(range(N))-set(j['moves']);aa,bb=j['sources'];assert aa in legal and bb in legal and aa!=bb
 ev=tr['evaluations'][0];v=float(ev['clean' if g==2 else 'value']);assert abs(v)<=1
 z=np.array([e['root_value'] for e in es]);acts=np.array([e['actions'][0] for e in es]);hit=np.isin(acts,[aa,bb])
 assert np.all(abs(z)<=1) and np.all(np.isin(acts,legal))
 n=np.zeros(N,int);w=np.zeros(N);s2=np.zeros(N);err=collections.defaultdict(float);lp=0.;mass=[];uniform=[];hist=None
 for start in range(0,128,4):
  Q=np.divide(w.astype(np.float32),n.astype(np.float32),out=np.zeros(N,np.float32),where=n>0)
  U=Q+np.float32(1.4)*prior.astype(np.float32)*np.sqrt(np.float32(start))/(np.float32(1)+n.astype(np.float32))
  for k in range(start,start+4):
   e=es[k];l=e['root_selection']['law'];a=np.asarray(l['actions']);sc=np.asarray(l['scores']);p=np.asarray(l['p']);assert np.array_equal(np.sort(a),legal) and all(l['active']) and l['beta_p']==10/3 and l['logit_h']==0 and l['t']==k and e['complete_seq']==k
   if st==81:
    assert j['arm']=='base81' and l['phase65']['lambda81']==0 and l['source'] is None
   else:assert j['arm']=='zero74' and l['source']==aa
   err['score']=max(err['score'],float(abs(sc-U[a]).max()));x=sc*(10/3);q=np.exp(x-x.max());q/=q.sum()
   err['probability']=max(err['probability'],float(abs(q-p).max()));assert np.all(p>0) and abs(p.sum()-1)<1e-12
   pick=int(l['picked']);u=l['u'];assert int(np.searchsorted(np.cumsum(p),u,side='right'))==pick and a[pick]==acts[k]
   lp+=math.log(p[pick]);err['log_path']=max(err['log_path'],abs(lp-l['log_p_path']))
   assert z[k]==(-1)**len(e['actions'])*e['leaf_value'];mass.append(float(p[np.isin(a,[aa,bb])].sum()));uniform.append(u)
   if k==32:
    pp=p[np.flatnonzero(a==aa)[0]];qq=p[np.flatnonzero(a==bb)[0]];hist=[float(hit[:32].mean()),float(z[:32].mean()),float(z[:32].var()),entropy(n[n>0]/32)/math.log(len(legal)),float(np.count_nonzero(n)/len(legal)),float(Q[aa]),float(Q[bb]),math.log(pp+qq),math.log(pp/qq),float(hit[:32].sum()-sum(mass[:32]))/32]
  for k in range(start,start+4):
   a=acts[k];n[a]+=1;w[a]+=z[k];s2[a]+=z[k]**2
  if start+4 in cs:
   c=cs[start+4];assert np.array_equal(n,c['n']) and abs(w-c['w']).max()<1e-9 and abs(s2-c['s2']).max()<1e-9 and np.array_equal(prior,c['prior'])
 assert err['score']<2e-6 and err['probability']<2e-12 and err['log_path']<1e-9 and hist is not None
 # All static root features are available from the immutable pre-search evaluator and board.
 pos=np.array([divmod(a,L) for a in j['moves']],float);cen=(L-1)/2
 same=0;den=0
 for i in range(len(pos)):
  for l in range(i):
   if (i-l)%2==0:
    den+=1;same+=int(np.max(abs(pos[i]-pos[l]))<=1)
 p0=prior[legal]/prior[legal].sum()
 root=[len(pos),v,math.log(prior[aa]+prior[bb]),math.log(prior.max()),entropy(p0)/math.log(len(legal)),math.log(prior[aa]/prior[bb]),float(np.linalg.norm(np.array(divmod(aa,L))-np.array(divmod(bb,L)))/(L-1)),float(np.mean(np.linalg.norm(pos-cen,axis=1))/(L-1)),same/max(1,den)]
 A=np.array([hit[:32].sum()/128,z[:32].sum()/128]);V=np.array([hit[32:].mean(),z[32:].mean()]);X=np.array([hit.mean(),z.mean()]);assert abs(X-A-.75*V).max()<1e-14
 rec=dict(stage=st,group=group,family=g,core=j['core'],rep=j['rep'],seed=j['seed'],path=str(f.relative_to(B)),sha256=h,bytes=f.stat().st_size,errors=dict(err),prior_audit=err0)
 return rec,root,hist,A,V,X,uniform
def main():
 R={i:read(B/f'mqft_restore{i}_20261127/REGISTRATION.json') for i in (79,80,81,82)}
 assert all(R[i]['weights']==R[80]['weights'] and R[i]['external']==R[80]['external'] and R[i]['binary_sha']==R[80]['binary_sha'] and R[i]['pulse_binary_sha']==R[80]['pulse_binary_sha'] for i in R)
 assert sha(S/'mqft_pulse74')==R[80]['pulse_binary_sha']
 for i in (79,80,81,82):
  for c in R[i]['cores']:assert orbit(board(c['moves'])).hex()==c['orbit']
 for a,b in zip(R[79]['cores'],R[80]['cores'][:32]):assert all(a[k]==b[k] for k in ('moves','orbit','sources'))
 keys=[set(c['orbit'] for c in R[i]['cores']) for i in (80,81,82)]
 assert list(map(len,keys))==[48,24,32] and all(not a&b for i,a in enumerate(keys) for b in keys[i+1:])
 old80=read(B/'mqft_restore80_20261127/AUDIT80.json');old81=read(B/'mqft_restore81_20261127/AUDIT81.json')
 old={q['path']:q for q in old80['raw_receipts']+old81['raw_receipts'] if q.get('arm',5)==5}
 args=[]
 for st in (79,80,81,82):
  group=0 if st<81 else st-80;root=B/f'mqft_restore{st}_20261127'
  for tag in TAGS:
   p=root/'plans'/f'{tag}.json';assert sha(p)==R[st]['plans'][tag]['sha256']
   for j in read(p):
    if st==81 and j['arm']!='base81':continue
    f=root/'data'/tag/(j['id']+'.json.gz');args.append((st,group,tag,j,old[str(f.relative_to(B))] if st!=82 else None))
 assert len(args)==4032
 with cf.ProcessPoolExecutor(max_workers=10) as pool:
  out=[]
  for i,v in enumerate(pool.map(one,args,chunksize=8)):
   out.append(v)
   if (i+1)%384==0:print('AUDIT',i+1,flush=True)
 rec=[r[0] for r in out];M=np.array([[r[k] for k in ('group','family','core','stage','rep')] for r in rec]);assert len(set(map(tuple,M)))==4032
 root,hist,A,V,X,u=map(np.stack,zip(*(r[1:] for r in out)));assert np.isfinite(root).all() and np.isfinite(hist).all()
 for g in range(3):
  for gr,n0,n1 in [(0,48,640),(1,24,192),(2,32,512)]:
   ix=(M[:,0]==gr)&(M[:,1]==g);assert len(np.unique(M[ix,2]))==n0 and ix.sum()==n1
   for c in np.unique(M[ix,2]):assert np.array_equal(root[ix&(M[:,2]==c)],np.tile(root[ix&(M[:,2]==c)][0],(np.sum(ix&(M[:,2]==c)),1)))
 # Check no reused random seed across geometry/cohort (g0/g3 common seed allowed).
 seen={};streams={}
 for i,r in enumerate(rec):
  key=(r['group'],r['core'],r['stage'],r['rep'],r['family']==2)
  sd=r['seed']
  if sd in seen:assert seen[sd]==key and streams[sd]==u[i].tolist()
  seen[sd]=key;streams[sd]=u[i].tolist()
 fresh=[r for r in rec if r['group']==2];fresh_seeds={r['seed'] for r in fresh}
 assert len(fresh_seeds)==1024
 dump(S/'RNG_INPUT82.json',[dict(seed=k,u=streams[k]) for k in sorted(fresh_seeds)])
 np.savez_compressed(S/'POPULATION82.npz',meta=M,root=root,history=hist,A=A,V=V,X=X,root_fields=ROOT_FIELDS,history_fields=HIST_FIELDS)
 a=dict(paths=1536,backups=196608,prior_paths=2496,prior_backups=319488,raw_bytes=sum(r['bytes'] for r in fresh),all_bytes=sum(r['bytes'] for r in rec),errors={k:max(r['errors'][k] for r in rec) for k in ('score','probability','log_path')},root_counts=[48,24,32],fresh_streams=1024,array_sha=sha(S/'POPULATION82.npz'),raw_receipts=rec,scope='Reauthenticated 1920 train, 576 R81 clean base, 1536 fresh paths; all raw score/CDF/cut replay, fresh full downstream audit; root/prefix features only.')
 dump(S/'AUDIT82.json',a);print('AUDIT',{k:v for k,v in a.items() if k!='raw_receipts'},flush=True)
if __name__=='__main__':main()
