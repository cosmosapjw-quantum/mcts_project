"""First-divergence diagnosis, conditional probability vs common-uniform coupling.
Finite 2-step analytical counterexample and empirical coarse CGF sensitivity."""
from base61 import *
from fractions import Fraction as F
reg=read(S/'REGISTRATION.json');perm=np.asarray(reg['permutations'])
out=[];n=0;exact=0;rootdiff=0;desc=0;order=0;probmax=0.;tie=0
for gen in (0,3):
 for c in (0,1,8,9,16,17,24,25):
  for rep in (0,1):
   base=read(S/f'data/symmetry{gen}/s_g{gen}_c{c:02d}_d1_o0_r{rep}.json')
   for o in range(1,8):
    q=read(S/f'data/symmetry{gen}/s_g{gen}_c{c:02d}_d1_o{o}_r{rep}.json');inv=np.argsort(perm[o]);n+=1
    assert q['job']['moves']==perm[o][base['job']['moves']].tolist()
    assert np.array_equal(np.array(q['cuts'][0]['prior'])[perm[o]],base['cuts'][0]['prior'])
    found=False
    for t,(e,f) in enumerate(zip(base['events'],q['events'])):
     a=e['actions'];b=inv[f['actions']].tolist()
     if a!=b:
      l=e['root_selection']['law'];m=f['root_selection']['law'];aa=np.array(l['actions']);bb=inv[m['actions']]
      p=np.zeros(N);p[aa]=l['p'];p2=np.zeros(N);p2[bb]=m['p'];pe=float(abs(p-p2).max());probmax=max(probmax,pe)
      diff=np.flatnonzero(aa!=bb);isorder=bool(len(diff));order+=isorder
      same_prior=bool(isorder and np.all(np.asarray(base['cuts'][0]['prior'])[aa[diff]]==np.asarray(base['cuts'][0]['prior'])[bb[diff]]));tie+=same_prior
      rootdiff+=a[0]!=b[0];desc+=a[0]==b[0]
      out.append(dict(generation=gen,core=c,rep=rep,orientation=o,first_divergence=t,root_action_diff=a[0]!=b[0],same_uniform=l['u']==m['u'],mapped_probability_max_error=pe,edge_order_diff=isorder,tie_permutation=same_prior,path0=a,path_other=b,order_mismatches=len(diff)))
      assert l['u']==m['u'] and pe<1e-12
      found=True;break
     assert e['root_value']==f['root_value']
    if not found:exact+=1
assert n==224 and exact+len(out)==n
# Exact finite 2-step example: P(I1=1)=1/2, P(I2=1|I1=0)=1/4, P(...|1)=3/4.
rows=[]
for a in (0,1):
 p1=F(1,2);p2=F(3,4) if a else F(1,4)
 for b in (0,1):
  prob=(p1 if a else 1-p1)*(p2 if b else 1-p2);C=a+b;G=F(a)-p1+F(b)-p2
  rows.append((prob,C,G,p1*(1-p1)+p2*(1-p2)))
E=lambda f:sum(p*f(c,g,v) for p,c,g,v in rows)
mu=E(lambda c,g,v:c);der=E(lambda c,g,v:c*g);var=E(lambda c,g,v:(c-mu)**2);fish=E(lambda c,g,v:v)
assert E(lambda c,g,v:g)==0 and E(lambda c,g,v:g*g)==fish and (der,var,fish)==(F(9,16),F(3,4),F(7,16))
# Additional empirical quenched-vs-mixture diagnostic, no extrapolation to native rate function.
a=np.load(S/'CONTROL61.npz');X=a['values'];M=a['meta'];k=list(a['fields']).index('C');cg=[]
for g in (0,3):
 for w in (1,4):
  mask=(M[:,1]==g)&(M[:,3]==w)&(M[:,4]==0)&(M[:,8]==0)&(M[:,9]==128)
  m=M[mask];x=X[mask,k];z=x[np.lexsort((m[:,7],m[:,2]))].reshape(32,8)
  for l in np.linspace(-.8,.8,33):
   y=l*z;mx=y.max();wt=np.exp(y-mx);W=float(mx+np.log(wt.mean()))
   mx2=y.max(1);qw=mx2+np.log(np.exp(y-mx2[:,None]).mean(1));Wq=float(qw.mean())
   assert W>=Wq-1e-12
   cg.append(dict(generation=g,width=w,lambda_=float(l),mixture_W=W,mean_conditional_empirical_W=Wq,jensen_gap=W-Wq,mixture_ESS=float(wt.sum()**2/np.sum(wt*wt))))
# Paired count dispersion and LR/FD contrast, block resampling.
r=read(S/'PHYSICS_RESULTS.json');ce=read(S/'CORE_EFFECTS61.json');rng=np.random.default_rng(610121);boot=rng.integers(0,32,(10000,32))
def stat(v):return dict(mean=float(v.mean()),ci95=np.quantile(v[boot].mean(1),[.025,.975]).tolist())
tests=[]
for gen in (0,3):
 for w in (1,4):
  mask=(M[:,1]==gen)&(M[:,3]==w)&(M[:,4]==0)&(M[:,8]==0)&(M[:,9]==128)
  m=M[mask];x=X[mask];x=x[np.lexsort((m[:,7],m[:,2]))].reshape(32,8,-1);f=list(a['fields'])
  n0=x[:,:,f.index('C')];f0=x[:,:,f.index('fisher_pred')].mean(1)
  b=next(q for q in ce if (q['generation'],q['width'],q['k'],q['observable'])==(gen,w,128,'C'))
  lr=np.asarray(b['score']);fd=np.asarray(b['fd']);tests.append(dict(generation=gen,width=w,var_minus_fisher=stat(n0.var(1,ddof=1)-f0),var_minus_score=stat(n0.var(1,ddof=1)-lr),var_minus_fd=stat(n0.var(1,ddof=1)-fd)))
out0=dict(scope='post-primary explanatory diagnostic; CRN coupling is not group-covariant when equal-prior ordering differs; not proof of marginal search-law non-equivariance',comparisons=n,exact_paths_and_marks=exact,first_divergences=len(out),root_action_divergences=rootdiff,descendant_only=desc,root_order_difference=order,tie_permutations=tie,max_mapped_probability_error_at_first_divergence=probmax,witnesses=out,exact_two_step_counterexample=dict(derivative=str(der),variance=str(var),fisher=str(fish)),coarse_cgf_mixture_vs_conditional=cg,dispersion_contrasts=tests)
dump(S/'DIAGNOSTICS61.json',out0);print({k:v for k,v in out0.items() if k not in ['witnesses','coarse_cgf_mixture_vs_conditional']})
