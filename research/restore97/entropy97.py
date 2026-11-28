from base61 import *
from itertools import product,combinations
from scipy.special import xlogy
import time,gzip,zipfile,csv,platform
O=B/'mqft_restore96_20261128';P95=B/'mqft_restore95_20261128';P94=B/'mqft_restore94_20261128';P92=B/'mqft_restore92_20261128';P91=B/'mqft_restore91_20261128';P68=B/'mqft_restore68_20261126';P85=B/'mqft_restore85_20261127'
FAMS=['g0','g3','Rapfi'];I=np.eye(2);X=np.array([[0,1],[1,0]]);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1]);PAULI=np.array([X,Y,Z]);ERR=collections.defaultdict(float)
def ck(k,a,b,tol=1e-9):
 e=float(np.max(abs(np.asarray(a)-np.asarray(b))));ERR[k]=max(e,ERR[k]);assert e<tol,(k,e)
def ent(p,axis=-1):
 p=np.maximum(np.asarray(p).real,0.);return -np.sum(xlogy(p,p),axis=axis)
def spec(a):
 e=np.linalg.eigvalsh(a);assert e.min()>-1e-10;return np.maximum(e,0)
def vn(a):return float(ent(spec(a)))
def red(a,K,n):
 K=list(K);R=[j for j in range(n) if j not in K];p=K+R;d=2**len(K);e=2**len(R)
 return np.trace(a.reshape([2]*(2*n)).transpose(p+[j+n for j in p]).reshape(d,e,d,e),axis1=1,axis2=3)
def direct(a,K,n):
 K=list(K);R=[j for j in range(n) if j not in K];idx=np.zeros((2**len(K),2**len(R)),int)
 for i,j in product(range(idx.shape[0]),range(idx.shape[1])):
  for t,k in enumerate(K):idx[i,j]|=((i>>(len(K)-1-t))&1)<<(n-1-k)
  for t,k in enumerate(R):idx[i,j]|=((j>>(len(R)-1-t))&1)<<(n-1-k)
 return sum(a[v[:,None],v[None,:]] for v in idx.T)
def pt(a,K,n):
 p=list(range(2*n))
 for i in K:p[i],p[i+n]=p[i+n],p[i]
 return a.reshape([2]*(2*n)).transpose(p).reshape(a.shape)
def neg(a,K,n):
 e=np.linalg.eigvalsh(pt(a,K,n));return float(-e[e<0].sum())
def tag(a,n,xi):
 k=np.arange(2**n);return a*np.array([[xi**(((int(i)^int(j))>>1).bit_count()) for j in k] for i in k])
def maskset(mask,n):return [i for i in range(n) if mask&(1<<i)]
def spectral_log_cross(a,b):
 e,v=np.linalg.eigh(b);w=np.diag(v.conj().T@a@v).real
 assert w.min()>-1e-10 and e.min()>-1e-10
 m=e>1e-14;lost=float(abs(w[~m]).sum());assert lost<1e-10
 ERR['discarded_spectral_weight']=max(ERR['discarded_spectral_weight'],lost)
 return float(-w[m]@np.log(e[m]))
q=(np.arange(128)+.5)/128;az=np.arange(128)*np.pi*(3-np.sqrt(5));zz=1-2*q
DIR=np.vstack([np.c_[np.sqrt(1-zz*zz)*np.cos(az),np.sqrt(1-zz*zz)*np.sin(az),zz],np.eye(3)])
def measured(r,B0,n):
 # B0 is one qubit; A is all other qubits. Outcome-conditioned entropy S(A|Y).
 A=[j for j in range(n) if j!=B0];a=red(r,[B0]+A,n);d=2**len(A);u=a.reshape(2,d,2,d)
 a0=red(r,A,n);b0=red(r,[B0],n);sa=vn(a0);sb=vn(b0);sab=vn(r);mi=sa+sb-sab
 bvec=np.array([np.trace(b0@p).real for p in PAULI])
 dirs=np.vstack([DIR,bvec/np.linalg.norm(bvec)]) if np.linalg.norm(bvec)>1e-14 else DIR
 m=np.einsum('ki,ijl->kjl',dirs,PAULI);e=(I+m)/2
 w=np.einsum('kij,jaib->kab',e,u);ww=np.stack([w,a0[None]-w],axis=1)
 # entropy of unnormalized blocks minus entropy of their traces
 ev=np.linalg.eigvalsh(ww);assert ev.min()>-1e-10
 pp=np.trace(ww,axis1=-2,axis2=-1).real;cond=ent(ev,axis=-1).sum(-1)-ent(pp)
 assert cond.min()>-1e-9
 chi=sa-cond;k=int(np.argmax(chi));zidx=130
 assert np.array_equal(dirs[zidx],[0,0,1])
 zchi=float(chi[zidx]);best=float(chi[k]);lower=max(0.,sb-sab);upper=max(0.,mi-best)
 assert lower<=upper+1e-9 and upper<=mi-zchi+1e-9 and best<=min(sa,mi)+1e-9
 # Independent rank-one measurement via explicit eigenvectors, each candidate winner and Z
 for kk in [k,zidx]:
  v=np.linalg.eigh(m[kk])[1];c=0.
  for j in range(2):
   vec=v[:,j];b=np.kron(vec[:,None],np.eye(d));w0=b.conj().T@a@b;p=float(np.trace(w0).real)
   if p>1e-15:c+=p*vn(w0/p)
  ck('projective_conditional',c,cond[kk])
 diag=red(r,[B0]+A,n).diagonal().real.reshape(2,d)
 cl=float(ent(diag.sum(0))+ent(diag.sum(1))-ent(diag.ravel()))
 assert cl<=zchi+1e-9 and zchi<=best+1e-9
 return dict(qubit=B0,MI=mi,classic_Z=cl,chi_Z=zchi,chi_best=best,fixed_gap=mi-zchi,discord_lower=lower,discord_upper=upper,direction=dirs[k].tolist())
def calc(arg):
 g,c,p,m,d,xi,s,r,old=arg;n=d+1;dim=2**n;full=dim-1
 a=tag(r,n,xi);ck('trace',np.trace(a),1);ck('hermitian',a,a.conj().T);e,v=np.linalg.eigh(a);assert e.min()>-1e-10
 root=(v*np.sqrt(np.maximum(e,0)))@v.conj().T;ck('purifier',root@root.conj().T,a)
 E={0:0.};R={0:np.ones((1,1))};rows=[]
 for k in range(1,dim):
  K=maskset(k,n);b=red(a,K,n);b2=direct(a,K,n);ck('partial_trace',b,b2)
  ev=spec(b);S0=float(ent(ev));E[k]=S0;R[k]=b
  # purification |sqrt(rho)> has left-system marginal rho. A|rest+purifier singular values.
  rest=[j for j in range(n) if j not in K];mat=root.reshape([2]*n+[dim]).transpose(K+rest+[n]).reshape(2**len(K),-1);sv=np.linalg.svd(mat,compute_uv=False)**2
  ck('schmidt_spectrum',np.sort(sv),ev);ck('schmidt_entropy',float(ent(sv)),S0)
  H=float(ent(b.diagonal().real));assert H>=S0-1e-9
  rows.append(dict(mask=k,size=len(K),S=S0,Renyi2=float(-np.log(ev@ev)),Hdiag=H,purification_cut_entropy=S0,spectrum=ev[::-1].tolist()))
 cuts=[]
 for k in range(1,full):
  if not k&1:continue
  t=full^k;K=maskset(k,n);T=maskset(t,n);sa,sb,sab=E[k],E[t],E[full];mi=sa+sb-sab;na=neg(a,K,n)
  assert mi>=-1e-9 and abs(sa-sb)<=sab+1e-9
  assert sa<=len(K)*np.log(2)+1e-9 and sb<=len(T)*np.log(2)+1e-9
  if min(sab-sa,sab-sb)<-1e-9:assert na>1e-10
  rsort=red(a,K+T,n)
  # Use factor eigenvalues: discarding tiny products can remove genuine joint weight.
  ea,va=np.linalg.eigh(R[k]);eb,vb=np.linalg.eigh(R[t]);V=np.kron(va,vb)
  w=np.diag(V.conj().T@rsort@V).real.reshape(len(ea),len(eb))
  ck('product_basis_marginal_A',w.sum(1),ea);ck('product_basis_marginal_B',w.sum(0),eb)
  ma=ea>1e-14;mb=eb>1e-14
  assert abs(w[~ma,:]).sum()+abs(w[:,~mb]).sum()<1e-10
  ds=float(-w.sum(1)[ma]@np.log(ea[ma])-w.sum(0)[mb]@np.log(eb[mb]))-sab
  ck('product_relative_entropy',ds,mi,1e-7)
  # independent explicit PT integer index; swap A row/col bits only
  bmask=sum(1<<(n-1-i) for i in K);ii,jj=np.indices(a.shape);flip=(ii^jj)&bmask
  ev=np.linalg.eigvalsh(a[ii^flip,jj^flip]);ck('PT_direct',na,float(-ev[ev<0].sum()))
  if old is not None:
   q=old['cuts'].index(K);ck('R95_negativity',na,old['cut_negativity'][q])
  cuts.append(dict(mask=k,other=t,size=min(len(K),len(T)),temporal_boundary=sum((i in K)!=(i+1 in K) for i in range(n-1)),SA=sa,SB=sb,MI=mi,conditional_A=sab-sb,conditional_B=sab-sa,N=na,success_weighted_N=s*na))
 cmis=[]
 for a0,b0,c0 in product(range(1,dim),repeat=3):
  if a0>=c0 or (a0&b0 or a0&c0 or b0&c0) or a0|b0|c0!=full:continue
  cmi=E[a0|b0]+E[b0|c0]-E[b0]-E[full];assert cmi>=-1e-9
  cmis.append(cmi)
 diag=np.diag(a.diagonal().real);coh=float(ent(a.diagonal().real))-E[full]
 ck('dephase_relative_entropy',spectral_log_cross(a,diag)-E[full],coh,1e-7)
 ms=[measured(a,b,n) for b in range(n)] if n>1 else []
 return dict(family=g,core=c,protocol=p,mode=m,depth=d,xi=xi,success=s,S=E[full],Hdiag=float(ent(a.diagonal().real)),coherence=coh,purity=float(np.trace(a@a).real),subsets=rows,cuts=cuts,conditional_MI=cmis,measurements=ms),dict(ERR)
def rehash(t):
 k,h=t;assert sha(B/k)==h,k;return k,h
def raw68(q):
 f=P68/'data'/(q['id']+'.json.gz');assert f.stat().st_size==q['bytes'] and sha(f)==q['sha256']
 with gzip.open(f,'rt') as z:a=json.load(z)
 assert a['status']==q['status'] and a['job']==q['job'] and len(a['moves'])==q['plies']
 rows=[];old={x['ply']:x for x in q['native']}
 for r in a['rows']:
  if r['engine']!='native':continue
  x=r['search'];events=x['events'];assert len(events)==x['budget'];acts=[e['actions'][0] for e in events];assert [e['complete_seq'] for e in events]==list(range(len(events)))
  for cut in x['cuts']:
   k=cut['k'];n=np.bincount(acts[:k],minlength=225);assert np.array_equal(n,cut['n'])
   H=float(ent(n/k));o=next(v for v in old[r['ply']]['cuts'] if v['k']==k);ck('R68_visit_entropy',H,o['H'])
   ph=float(np.mean([ent(e['root_selection']['law']['p']) for e in events[:k]]));ck('R68_conditional_entropy',ph,o['conditional_H'])
   rows.append(dict(game=q['id'],cohort='external' if q['job']['match'].startswith('ext_') else 'internal',arm=r['arm'],core=q['job']['core'],ply=r['ply'],budget=k,H=H,conditional_H=ph))
 return str(f.relative_to(B)),q['sha256'],rows,dict(ERR)
def refs():
 def dm(v):v=np.asarray(v,complex);return np.outer(v,v.conj())
 b=dm([1/np.sqrt(2),0,0,1/np.sqrt(2)]);cl=np.diag([.5,0,0,.5]);mix=np.eye(4)/4;ghz=dm([1/np.sqrt(2)]+[0]*6+[1/np.sqrt(2)])
 r=[]
 for name,a,n,expected in [('Bell',b,2,[0,np.log(2),.5]),('classical',cl,2,[np.log(2),np.log(2),0]),('mixed_product',mix,2,[2*np.log(2),np.log(2),0]),('GHZ',ghz,3,[0,np.log(2),.5])]:
  vals=[vn(a),vn(red(a,[0],n)),neg(a,[0],n)];ck('analytic_controls',vals,expected);m=measured(a,0,n);r.append(dict(name=name,values=vals,measurement=m))
  if name=='classical':ck('classical_discord',m['discord_upper'],0)
  if name=='Bell':ck('Bell_discord',m['discord_upper'],np.log(2))
 return r
def main():
 start=time.monotonic();sources={}
 def pinned(P,f):
  n=P.name.split('restore')[1].split('_')[0]
  m=read(P/f'RUN_MANIFEST{n}.json');h=sha(P/f);assert h==m['files'][f],(P,f);sources[str((P/f).relative_to(B))]=h;return P/f
 for P,f in [(P95,'JOINT95.npz'),(P95,'ENTANGLE95.json'),(O,'DARWIN96.json'),(O,'SUMMARY96.json'),(O,'INDEPENDENT96.json'),(P94,'NATIVE94.json'),(P94,'NATIVE94.npz'),(P94,'RECORD94.json'),(P92,'SUMMARY92.json'),(P91,'THERMAL91.json')]:pinned(P,f)
 raw=read(O/'INDEPENDENT96.json')['raw_receipts'];assert len(raw)==4104
 with cf.ProcessPoolExecutor(max_workers=10) as pool:raw=dict(pool.map(rehash,raw.items(),chunksize=16))
 print('RAW RECORDS',len(raw),flush=True)
 a=read(P95/'ENTANGLE95.json');d96=read(O/'DARWIN96.json')
 with np.load(P95/'JOINT95.npz') as z:D={k:z[k] for k in z.files}
 ix={(int(g),int(c),p,m):i for i,(g,c,p,m) in enumerate(D['meta'])};assert len(ix)==576
 args=[]
 for (g,c,p,m),i in ix.items():args.append((g,c,p,m,0,1.,1.,D['J0'][i],None))
 for r in a['rows']:
  g,c,p,m,d,xi,s=[r[k] for k in ['family','core','protocol','mode','depth','xi','success']];args.append((g,c,p,m,d,xi,s,D[f'J{d}'][ix[g,c,p,m]],r))
 assert len(args)==5760;rows=[]
 with cf.ProcessPoolExecutor(max_workers=10) as pool:
  for i,(r,e) in enumerate(pool.map(calc,args,chunksize=12)):
   rows.append(r)
   for k,v in e.items():ERR[k]=max(ERR[k],v)
   if (i+1)%1440==0:print('STATE',i+1,flush=True)
 keys=lambda r:tuple(r[k] for k in ['family','core','protocol','mode','depth','xi'])
 rr={keys(r):r for r in rows};assert len(rr)==5760
 # Match existing R96 pointer-Z information. First-record scope excludes S and is checked on reduced R-only state.
 for r in d96['rows']:
  z=rr[keys(r)];n=r['depth']+1
  for ptr in r['pointers']:
   f=ptr['fragments'][-1]
   if ptr['pointer']=='system_Z':
    m=next(x for x in z['measurements'] if x['qubit']==n-1);ck('R96_system_MI',m['MI'],f['QMI']);ck('R96_system_chiZ',m['chi_Z'],f['chi'])
   else:
    g,c,p,m,d,xi=[r[k] for k in ['family','core','protocol','mode','depth','xi']]
    J=tag(D[f'J{d}'][ix[g,c,p,m]],n,xi);redR=red(J,list(range(d)),n);m0=measured(redR,0,d);ck('R96_record_MI',m0['MI'],f['QMI']);ck('R96_record_chiZ',m0['chi_Z'],f['chi'])
 # raw game risk sets; separate clock and historical population.
 audit=read(P68/'AUDIT68.json');assert audit['games']==80
 sources[str((P68/'AUDIT68.json').relative_to(B))]=sha(P68/'AUDIT68.json')
 games=[];native=[]
 with cf.ProcessPoolExecutor(max_workers=10) as pool:
  for k,h,v,e in pool.map(raw68,audit['rows'],chunksize=2):
   assert k not in raw;raw[k]=h;native.extend(v);games.append(k)
   for k,v in e.items():ERR[k]=max(ERR[k],v)
 assert len(native)==853*5 and len(raw)==4184
 # Reference thermal curves, rederive from explicit fixed spectra rather than infer native T
 t=read(P91/'THERMAL91.json');m91=read(P91/'RUN_MANIFEST91.json');f=P85/'BRANCH_ENERGY85.json';assert sha(f)==m91['inputs'][str(f.relative_to(B))];sources[str(f.relative_to(B))]=sha(f)
 ctx={(c['protocol'],c['family'],c['core']):c for c in read(f)['contexts']};theta=np.array(t['theta']);therm=[]
 for r in t['capacity']:
  e=np.array(ctx[r['protocol'],r['family'],r['core']]['e0']);v=np.array(r['curve']);vals=[];der=[]
  def gib(th):
   z=-e/th;z-=z.max();p=np.exp(z);p/=p.sum();U=float(p@e);S0=float(ent(p));C=float(p@((e-U)**2)/th**2);return U,S0,C
  for th,prior in zip(theta,v):
   U,S0,C=gib(th);ck('R91_capacity',[C,U,S0],prior[[0,2,3]])
   h=th*1e-4;u1,s1,_=gib(th+h);u0,s0,_=gib(th-h);dU=(u1-u0)/(2*h);dS=(s1-s0)/(2*h)
   ERR['thermal_C_fd']=max(ERR['thermal_C_fd'],abs(dU-C));ERR['thermal_entropy_fd']=max(ERR['thermal_entropy_fd'],abs(dS-C/th))
   if C>1e-5:der.append([float(th),float(th*dS/dU-1)])
   vals.append([U,S0,C,U-th*S0])
  therm.append(dict(protocol=r['protocol'],family=r['family'],core=r['core'],curve=vals,derivative_relative=der))
 assert len(therm)==104
 for r in t['cases']:
  # beta recover from protocol context? R91 cases do not store beta; cases use beta 10/3 throughout.
  ck('R91_first_law',r['DE'],r['W']+r['Q'])
  ck('R91_conditional_entropy_balance',r['dS_cond']-(10/3)*r['Q'],r['sigma_rel'])
  ck('R91_feedback_balance',(10/3)*r['W']+r['I0'],r['sigma_rel']+r['D_final']+r['I1'])
 # Original source, fixed authority; R91/R92 references are anchored, not re-run native.
 paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
 orig=read(O/'ORIGINAL_SOURCE96.json')['archives'];assert all(sha(p)==orig[k] for k,p in paths.items())
 qm={x['id']:x for x in csv.DictReader(open(O/'QUALITY_MATRIX96.csv'))};items=[];code='quantum/research/visualization/plot_entropy_analysis.py'
 with zipfile.ZipFile(paths['quantum.zip']) as z:
  b=z.read(code);h=hashlib.sha256(b).hexdigest();ls=b.decode().splitlines()
  for k in read(S/'REGISTRATION.json')['targets']:
   x=qm[k];assert h==x['source_sha'];l,r=map(int,x['source_span'].split('-'));assert x['source_function'] in ls[l-1]
   items.append(dict(id=k,archive='quantum.zip',member=code,member_sha=h,span=[l,r],source_function=x['source_function'],source_excerpt=ls[l-1:r],acceptance=x['final_acceptance']))
 study=[]
 with zipfile.ZipFile(paths['mcts_study.zip']) as z:
  assert z.testzip() is None
  for prior in read(O/'ORIGINAL_SOURCE96.json')['study_selected']:
   k=prior['member'];b=z.read(k);assert hashlib.sha256(b).hexdigest()==prior['sha256'];ls=b.decode().splitlines();match=[(i+1,s) for i,s in enumerate(ls) if any(w in s.lower() for w in ['entropy','entanglement','discord','thermodynamic','area law','partition'])]
   study.append(dict(member=k,sha256=prior['sha256'],lines=len(ls),relevant_line_count=len(match),excerpts=match[:40]))
 dump(S/'ORIGINAL_SOURCE97.json',dict(archives=orig,items=items,study_selected=study,authority='R69 byte-identical originals, exact function spans; archived mcts_study; original source not edited'))
 ref=refs();print('CONTROLS',ref,flush=True)
 import sys
 # compact raw rows gzip; JSON byte stream reproducible except elapsed metadata external.
 with gzip.GzipFile(filename=str(S/'ENTROPY97.json.gz'),mode='wb',mtime=0) as z:z.write(json.dumps(rows,separators=(',',':'),allow_nan=False).encode())
 dump(S/'NATIVE_THERMAL97.json',dict(status='PASS_SCOPED',native=native,games=len(games),thermal=therm,theta=theta.tolist(),feedback_cases=len(t['cases']),scope='R68 native actual count replay, separate R91 fixed-spectrum reference; no physical calibration'))
 dump(S/'INDEPENDENT97.json',dict(status='PASS_SCOPED',checks=dict(ERR),raw_receipts=raw,unique_raw=len(raw),source_hashes=sources,software=dict(python=sys.version,numpy=np.__version__),independence='same-agent distinct indexed partial trace, SVD purification, PT bit swap, explicit rank-one measurements, raw count replay, reference thermodynamics',controls=ref))
 dump(S/'ENTROPY97.json',dict(status='PASS_SCOPED',raw_receipts=raw,states=len(rows),subsets=sum(len(x['subsets']) for x in rows),cuts=sum(len(x['cuts']) for x in rows),conditional_MI=sum(len(x['conditional_MI']) for x in rows),measurement_cases=sum(len(x['measurements']) for x in rows),raw_gzip_sha=sha(S/'ENTROPY97.json.gz'),elapsed_s=time.monotonic()-start,scope='Declared finite current-prefix quantum model and independent historical native visit/thermal reference; not spatial area law or optimized POVM discord'))
 print('RESULT',read(S/'ENTROPY97.json')|{'raw_receipts':len(raw)},dict(ERR),flush=True)
if __name__=='__main__':main()
