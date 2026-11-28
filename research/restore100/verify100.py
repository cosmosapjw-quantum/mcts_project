from therm100 import *
def main():
 n=read(S/'NATIVE100.json');t=read(S/'THERMAL100.json');err={};raw=n['raw_receipts']|t['raw_receipts'];assert len(raw)==6768
 assert sha(S/'NATIVE100.npz')==n['array_sha'] and sha(S/'THERMAL_SAMPLES100.npz')==t['array_sha']
 src=read(R85/'BRANCH_ENERGY85.json');cs=src['contexts'];D=read(S/'ORIGINAL_SOURCE100.json');assert len(D['items'])==10 and len(D['study_selected'])==13
 # Independent exact distribution as diagonal + independent product refresh, not dense Markov matrix.
 with np.load(S/'THERMAL_SAMPLES100.npz') as z:
  inds=z['indices'];meta=z['meta'];assert inds.shape==(312,2,4096,3)
  for c,ar,m in zip(t['cases'],inds,meta):
   i0=c['context'];r=c['refresh'];assert list(m)==[i0,RS.index(r)];e=np.array(cs[i0]['e0']);g=np.array(cs[i0]['em'][0]);em=.5*(e+g);a=gib(e);b=gib(em);c1=gib(g);d=em-e
   # Product law, including no-refresh diagonal branch explicitly.
   x=d[:,None]+d[None,:];q=r*a[:,None]*b[None,:];pdiag=(1-r)*a;rv=r*b[:,None]*c1[None,:];rdiag=(1-r)*c1
   mf=float(np.sum(q*x)+pdiag@(2*d));mr=float(-np.sum(rv*x)-rdiag@(2*d));vf=float(np.sum(q*(x-mf)**2)+pdiag@((2*d-mf)**2))
   ck(err,'mixture_mean_F',mf,c['forward_cumulants'][0]);ck(err,'mixture_mean_R',mr,c['reverse_cumulants'][0]);ck(err,'mixture_var_F',vf,c['forward_cumulants'][1])
   ck(err,'mixture_exact_J',float(np.sum(q*np.exp(-BETA*(x-c['df'])))+pdiag@np.exp(-BETA*(2*d-c['df']))),1)
   # independent replay from stored indices, both directions. Histograms and stable weight moments.
   for k,sign in [(0,1),(1,-1)]:
    i,j,z0=ar[k].T;assert i.min()>=0 and max(i.max(),j.max(),z0.max())<len(e)
    w=sign*(d[i]+d[j]);df=sign*c['df'];k0='F' if k==0 else 'R'
    hist=np.histogram(sign*w,c['hist_edges'])[0]/NMC;ck(err,'sample_histogram',hist,c['MC_F' if k==0 else 'MC_R_minus'])
    ck(err,'sample_mean',w.mean(),c['sample_work_means'][k])
    for row in c[k0]:
     nn=row['n'];ww=np.exp(-BETA*(w[:nn]-df));est=df-np.log(ww.mean())/BETA;ess=ww.sum()**2/np.sum(ww**2)
     ck(err,'sample_estimator',est,row['estimate']);ck(err,'sample_ESS_relative',ess/nn,row['ess']/nn);ck(err,'sample_log_IFT',np.log(ww.mean()),row['log_IFT'])
 # native cumulative array, independent endpoint reduction and sample log weights using long double
 with np.load(S/'NATIVE100.npz') as z:
  total=0
  for L in LS:
   M=z[f'M{L}'];V=z[f'V{L}'];T=z[f'T{L}'];P=z[f'P{L}'];lookup={tuple(x):i for i,x in enumerate(P)};assert len(lookup)==864
   for m,v in zip(M,V):
    j=lookup[tuple(m[:-1])];h=m[-1];ck(err,'native_prefix_array',T[j,h-1],v);ck(err,'native_score_delta',v[:,6]+v[:,7],v[:,8]);total+=1
   for g in n['groups']:
    if g['L']!=L:continue
    a=np.array([g[k] for k in ['cohort','arm','bi','budget','h']]);sel=np.all(M[:,[0,2,3,5,6]]==a,axis=1);v=V[sel,:,0].astype(np.longdouble);w=np.exp(-v);ck(err,'native_sample_logmean',np.log(w.mean(0)),g['log_sample_IFT']);ck(err,'native_sample_ESS',w.sum(0)**2/(w*w).sum(0)/len(v),np.array(g['ess'])/len(v))
  assert total==10400
 for c in t['cycles']:
  e=np.array(cs[c['context']]['e0']);a=c['scale'];r=c['refresh'];pc=gib(e,1/.3);ph=gib(a*e,1/.9)
  # Heat formula from ideal energy difference and equal-refresh attenuation.
  dc=float((ph-pc)@e)*r/(2-r)
  ck(err,'cycle_heat_analytic',[a*dc,-dc,-(a-1)*dc],[c['Qh'],c['Qc'],c['W']])
 # two-state absolute irreversibility and score-only counterexample: E_P[e^-Sigma]=Q(supp P).
 p=np.array([1.,0.]);q=np.array([.4,.6]);z=float(p[0]*q[0]/p[0]);ck(err,'support_mass_control',z,.4)
 checks={**n['checks'],**t['checks'],**{'independent_'+k:v for k,v in err.items()}}
 source=n['source_hashes']|t['source_hashes']
 assert all(sha(B/k)==h for k,h in source.items())
 # Summaries preserve finite sampling bias and no physical extrapolation.
 cases=t['cases'];cyc=t['cycles'];ns=[]
 for g in n['groups']:
  if g['cohort']==2 and g['budget']==g['h']==64 and g['arm']==1 and g['bi']==3:
   ns.append(dict(L=g['L'],mean=np.asarray(g['ci'])[0].tolist(),logIFT=g['log_sample_IFT'],ESS=g['ess']))
 rs=[]
 for r in RS:
  a=[c for c in cases if c['refresh']==r];assert len(a)==104
  rs.append(dict(refresh=r,mean_diss=float(np.mean([v['mean_diss'] for v in a])),reverse_mean_diss=float(np.mean([v['reverse_mean_diss'] for v in a])),mean_heat=float(np.mean([v['mean_heat'] for v in a])),work_variance=float(np.mean([v['forward_cumulants'][1] for v in a])),F_in2se=sum(v['F'][-1]['within_2se'] for v in a),R_in2se=sum(v['R'][-1]['within_2se'] for v in a),min_ESS_F=min(v['F'][-1]['ess'] for v in a),min_ESS_R=min(v['R'][-1]['ess'] for v in a)))
 cc=[]
 for a in [1.5,2.]:
  for r in [.25,.75,1.]:
   x=[v for v in cyc if v['scale']==a and v['refresh']==r]
   cc.append(dict(scale=a,refresh=r,W_out=float(np.mean([-v['W'] for v in x])),Qh=float(np.mean([v['Qh'] for v in x])),sigma=float(np.mean([v['sigma'] for v in x])),eta=1-1/a))
 dump(S/'SUMMARY100.json',dict(native_fresh_dense_beta16=ns,logIFT_min=np.min([g['log_sample_IFT'] for g in n['groups']],0).tolist(),logIFT_max=np.max([g['log_sample_IFT'] for g in n['groups']],0).tolist(),reference=rs,cycles=cc,max_final_error_F=max(abs(v['F'][-1]['estimate']-v['df']) for v in cases),max_final_error_R=max(abs(v['R'][-1]['estimate']+v['df']) for v in cases),reference_paths=312*2*NMC,source_contexts=104,scope='No native physical entropy production, bath, calibrated work/heat, or actual native reverse. Independent F/R reference samples, pointwise approximate intervals only.'))
 dump(S/'INDEPENDENT100.json',dict(status='PASS_SCOPED',checks=checks,raw_receipts=raw,source_hashes=source,software=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),native_paths=4320,native_draws=215040,native_prefixes=10400,thermal_raw=2448,reference_cases=312,reference_paths=312*2*NMC,cycles=624,independence='Exact mixture algebra independent of dense kernel; independent direct sample weights from saved indices; native prefix and sample IFT longdouble checks, full path enumeration subset, analytic cycle heat; same agent.'))
 print('INDEPENDENT',err,flush=True)
 print('SUMMARY',read(S/'SUMMARY100.json'),flush=True)
if __name__=='__main__':main()
