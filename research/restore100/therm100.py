from native100 import *
from scipy.special import xlogy
R85=B/'mqft_restore85_20261127';R91=B/'mqft_restore91_20261128';BETA=10/3;RS=[.1,.5,1.];NMC=4096;NS=[64,256,1024,4096]
def gib(e,b=BETA):
 z=-b*e;return np.exp(z-logsumexp(z))
def K(p,r):return (1-r)*np.eye(len(p))+r*p[None,:]
def H(p):return float(-np.sum(xlogy(p,p)))
def KL(p,q):return float(np.sum(xlogy(p,p/q)))
def F(e,b=BETA):return float(-logsumexp(-b*e)/b)
def refresh(rng,x,p,r):
 j=rng.choice(len(p),len(x),p=p);return np.where(rng.random(len(x))<r,j,x)
def stats(w,p):
 m=float(np.sum(p*w));v=float(np.sum(p*(w-m)**2));return [m,v,float(np.sum(p*(w-m)**3)),float(np.sum(p*(w-m)**4)-3*v*v)]
def sample_metrics(w,df):
 out=[]
 for n in NS:
  z=-BETA*(w[:n]-df);s=logsumexp(z);lw=s-np.log(n);wt=np.exp(z-z.max());ess=wt.sum()**2/(wt@wt);rel_se=float(wt.std(ddof=1)/(np.sqrt(n)*wt.mean()))
  out.append(dict(n=n,estimate=float(df-lw/BETA),log_IFT=float(lw),ess=float(ess),se=float(rel_se/BETA),within_2se=bool(abs(lw)<=2*rel_se)))
 return out
def one(c,ci0,r,ri,err):
 e=np.array(c['e0']);e1=np.array(c['em'][0]);em=(e+e1)/2;n=len(e)
 p,pm,p1=[gib(x) for x in (e,em,e1)];A=K(pm,r);B1=K(p1,r)
 for a,g in [(A,pm),(B1,p1)]:
  ck(err,'row_stochastic',a.sum(1),1);ck(err,'detailed_balance',g[:,None]*a,g[None,:]*a.T);ck(err,'stationary',g@a,g)
 # terminal relaxation summed out; paired reverse probability over (i,j).
 pf=p[:,None]*A;pr=p1[None,:]*A.T;w1=em-e;w2=e1-em;w=w1[:,None]+w2[None,:];df=F(e1)-F(e);dm=F(em)-F(e)
 sig=BETA*(w-df);ck(err,'Crooks_log_ratio',np.log(pf/pr),sig);ck(err,'forward_normalization',pf.sum(),1);ck(err,'reverse_normalization',pr.sum(),1)
 ck(err,'Jarzynski_forward',np.sum(pf*np.exp(-sig)),1);ck(err,'Jarzynski_reverse',np.sum(pr*np.exp(sig)),1)
 ck(err,'prefix_Jarzynski',p@np.exp(-BETA*(w1-dm)),1)
 mf=stats(w,pf);mr=stats(-w,pr);cov=float(np.sum(pf*(w1[:,None]-p@w1)*(w2[None,:]-(p@A)@w2)))
 st1=stats(w1,p);st2=stats(w2,p@A);ck(err,'work_variance_components',mf[1],st1[1]+st2[1]+2*cov)
 # ensemble entropy, signed heat at each relaxation; nonequilibrium free energy residual.
 a=p@A;b=a@B1;W1=float(p@w1);Q1=float((a-p)@em);W2=float(a@w2);Q2=float((b-a)@e1)
 s1=H(a)-H(p)-BETA*Q1;s2=H(b)-H(a)-BETA*Q2;Dend=KL(b,p1);dE=float(b@e1-p@e)
 ck(err,'first_law',dE,W1+W2+Q1+Q2);ck(err,'mean_work',mf[0],W1+W2);ck(err,'dissipation_split',BETA*(mf[0]-df),s1+s2+Dend);assert min(s1,s2,Dend,BETA*(mf[0]-df),BETA*(mr[0]+df))>-1e-9
 ck(err,'relaxation_KL_1',s1,KL(p,pm)-KL(a,pm));ck(err,'relaxation_KL_2',s2,KL(a,p1)-KL(b,p1))
 # actual full F/R path enumeration on fixed small subset, reverse step ordering verified
 if ci0<3:
  f3=pf[:,:,None]*B1[None,:,:];r3=p1[None,None,:]*B1.T[None,:,:]*A.T[:,:,None]
  ck(err,'full_path_Crooks',np.log(f3/r3),sig[:,:,None]);ck(err,'full_path_reverse_marginal',r3.sum(2),pr)
  q=em[None,:,None]-em[:,None,None]+e1[None,None,:]-e1[None,:,None]
  ck(err,'full_path_first_law',w[:,:,None]+q,e1[None,None,:]-e[:,None,None])
 # fresh independent auxiliary MC draws; actual F and R use disjoint seed sequences.
 rng=np.random.default_rng(np.random.SeedSequence([100,ci0,ri,0]));rv=np.random.default_rng(np.random.SeedSequence([100,ci0,ri,1]))
 i=rng.choice(n,NMC,p=p);j=refresh(rng,i,pm,r);k=refresh(rng,j,p1,r);kk=rv.choice(n,NMC,p=p1);jj=refresh(rv,kk,p1,r);ii=refresh(rv,jj,pm,r)
 fw=w[i,j];rw=-w[ii,jj];fq=em[j]-em[i]+e1[k]-e1[j];rq=em[ii]-em[jj]+e1[jj]-e1[kk]
 ck(err,'MC_F_first_law',fw+fq,e1[k]-e[i]);ck(err,'MC_R_first_law',rw+rq,e[ii]-e1[kk])
 # finite exact CGF and parametric convex dual; no empirical asymptotic large deviation inference
 lam=np.linspace(-2*BETA,2*BETA,33);z=np.log(pf.ravel())[None]+lam[:,None]*w.ravel();psi=logsumexp(z,axis=1);mu=np.exp(z-psi[:,None])@w.ravel();rate=lam*mu-psi
 ck(err,'CGF_zero',psi[16],0);assert min(rate)>-1e-9
 # Crooks on work law binned using probability-ratio integral, not center substitution.
 edges=np.linspace(float(w.min())-1e-10,float(w.max())+1e-10,41)
 ph=np.histogram(w.ravel(),edges,weights=pf.ravel())[0];rh=np.histogram(w.ravel(),edges,weights=pr.ravel())[0];rr=np.histogram(w.ravel(),edges,weights=(pr*np.exp(sig)).ravel())[0];ck(err,'binned_Crooks_integral',ph,rr)
 # stable exact plus finite MC, keep raw indices for regeneration.
 z=dict(protocol=c['protocol'],family=c['family'],core=c['core'],context=ci0,n=n,refresh=r,df=df,mid_df=dm,forward_cumulants=mf,reverse_cumulants=mr,work_component_cumulants=[st1,st2],work_component_cov=cov,mean_heat=Q1+Q2,mean_reverse_heat=float(np.mean(rq)),dE=dE,sigma=[s1,s2,Dend],mean_diss=BETA*(mf[0]-df),reverse_mean_diss=BETA*(mr[0]+df),stages=[[float(p@e),H(p),0,0],[float(p@em),H(p),W1,0],[float(a@em),H(a),W1,Q1],[float(a@e1),H(a),W1+W2,Q1],[float(b@e1),H(b),W1+W2,Q1+Q2]],F=sample_metrics(fw,df),R=sample_metrics(rw,-df),hist_edges=edges.tolist(),exact_F=ph.tolist(),exact_R_minus=rh.tolist(),MC_F=(np.histogram(fw,edges)[0]/NMC).tolist(),MC_R_minus=(np.histogram(-rw,edges)[0]/NMC).tolist(),tilt=dict(lam=lam.tolist(),psi=psi.tolist(),mean=mu.tolist(),dual=rate.tolist()),sample_work_means=[float(fw.mean()),float(rw.mean())],sample_work_se=[float(fw.std(ddof=1)/np.sqrt(NMC)),float(rw.std(ddof=1)/np.sqrt(NMC))])
 return z,np.stack([np.c_[i,j,k],np.c_[ii,jj,kk]]).astype(np.int16)
def cycle(c,ci0,a,r,err):
 e=np.array(c['e0']);pc=gib(e,1/.3);ph=gib(a*e,1/.9)
 den=2*r-r*r;x=(r*pc+(1-r)*r*ph)/den;y=(1-r)*x+r*ph
 ck(err,'cycle_closure',(1-r)*y+r*pc,x);z=pc.copy()
 for _ in range(200):z=(1-r)*((1-r)*z+r*ph)+r*pc
 ck(err,'cycle_iteration',z,x)
 Uc=float(x@e);Uh=float(y@e);qh=a*(Uh-Uc);qc=Uc-Uh;w=(a-1)*(Uc-Uh);sigma=-qh/.9-qc/.3
 sc=[H(y)-H(x)-qh/.9,H(x)-H(y)-qc/.3];ck(err,'cycle_first_law',w+qh+qc,0);ck(err,'cycle_entropy',sum(sc),sigma);assert qh>0 and w<0 and min(sc)>-1e-10
 eta=-w/qh;ck(err,'Otto_efficiency',eta,1-1/a);assert eta<1-.3/.9
 wi=(a-1)*float((pc-ph)@e);ck(err,'cycle_finite_relaxation_output',w/wi,r/(2-r))
 return dict(context=ci0,protocol=c['protocol'],family=c['family'],core=c['core'],n=len(e),scale=a,refresh=r,Qh=qh,Qc=qc,W=w,eta=eta,sigma=sigma,entropy_strokes=sc,ideal_work=wi,stages=[[Uc,H(x),0,0],[a*Uc,H(x),(a-1)*Uc,0],[a*Uh,H(y),(a-1)*Uc,qh],[Uh,H(y),w,qh],[Uc,H(x),w,qh+qc]])
def main():
 t=time.monotonic();sources={};raw={};err={};out=[];samples=[];cycles=[];therm=[]
 def pin(stage,f):
  p=next(B.glob(f'mqft_restore{stage}_20*'));m=read(p/f'RUN_MANIFEST{stage}.json');k=str((p/f).relative_to(B));assert sha(p/f)==m['files'][f];sources[k]=m['files'][f];return p/f
 src=read(pin(85,'BRANCH_ENERGY85.json'));old=read(pin(91,'THERMAL91.json'));pin(91,'NATIVE91.json');pin(74,'PETZ74.json');pin(93,'RESULTS_KO.md')
 for x in src['raw_receipts']:
  assert x['path'] not in raw or raw[x['path']]==x['sha256'];raw[x['path']]=x['sha256']
 assert len(raw)==2448
 with cf.ThreadPoolExecutor(max_workers=6) as pool:
  assert all(pool.map(lambda x:sha(B/x[0])==x[1],raw.items()))
 print('THERMAL RAW',len(raw),flush=True)
 cap={(c['protocol'],c['family'],c['core']):c for c in old['capacity']}
 for ci0,c in enumerate(src['contexts']):
  for ri,r in enumerate(RS):
   z,rawidx=one(c,ci0,r,ri,err);out.append(z);samples.append(rawidx)
  for a in [1.5,2.]:
   for r in [.25,.75,1.]:cycles.append(cycle(c,ci0,a,r,err))
  e=np.array(c['e0']);curve=[]
  for th,prev in zip(old['theta'],cap[c['protocol'],c['family'],c['core']]['curve']):
   p=gib(e,1/th);U=float(p@e);s=H(p);C=float(p@((e-U)**2)/th**2);ck(err,'R91_thermal_reference',[C,U,s],np.asarray(prev)[[0,2,3]]);ff=F(e,1/th);ck(err,'free_energy',ff,U-th*s);curve.append([U,s,C,ff])
  lo=math.floor(e.min()*10);hi=math.floor(e.max()*10);edges=np.arange(lo,hi+2)/10;ct=np.histogram(e,edges)[0];assert ct.sum()==len(e)
  therm.append(dict(context=ci0,n=len(e),protocol=c['protocol'],family=c['family'],core=c['core'],curve=curve,dos=[dict(low=float(edges[i]),high=float(edges[i+1]),count=int(n),S=float(np.log(n))) for i,n in enumerate(ct) if n>0]))
 assert len(out)==312 and len(cycles)==624 and len(therm)==104
 np.savez_compressed(S/'THERMAL_SAMPLES100.npz',indices=np.array(samples),meta=np.array([[x['context'],RS.index(x['refresh'])] for x in out]))
 dump(S/'THERMAL100.json',dict(status='PASS_SCOPED',contexts=104,cases=out,cycles=cycles,capacity=therm,theta=old['theta'],samples_per_direction=NMC,raw_receipts=raw,source_hashes=sources,checks=err,array_sha=sha(S/'THERMAL_SAMPLES100.npz'),elapsed_s=time.monotonic()-t,scope='New explicit auxiliary thermal model on authenticated score spectra; independently drawn F/R paths; not native calibrated bath.'))
 print('THERMAL DONE',len(out),len(cycles),err,flush=True)
if __name__=='__main__':main()
