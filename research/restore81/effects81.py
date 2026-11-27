"""Paired actual-path contrasts, causal likelihood transport, phase and order witnesses."""
from instrument81 import *
from scipy.stats import t as student
from scipy.special import logsumexp
import subprocess
def ci(x):
 x=np.asarray(x);mu=x.mean(0);h=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x));return [mu.tolist(),(mu-h).tolist(),(mu+h).tolist()]
def main():
 a=read(S/'AUDIT81.json');assert sha(S/'PATHS81.npz')==a['array_sha']
 with np.load(S/'PATHS81.npz') as z:d={k:z[k] for k in z.files}
 meta=d['meta'];R={tuple(meta[i]):i for i in range(len(meta))};rec=a['raw_receipts'];res=[];pairrows=[]
 fields=['first_signed','active_signed','wave_Y_signed','wave_middle_signed','wave_last_signed','final_source_a_frac','final_pair_frac','final_mean_mark','recommend_a','recommend_pair','elapsed_s','eval_calls']
 for g in range(3):
  group=[]
  for c in range(24):
   q=[]
   for rep in range(8):
    ix=[R[g,c,rep,m] for m in range(6)];rs=[rec[i] for i in ix];a0=d['actions'][ix];z=d['marks'][ix];src=d['sources'][ix];u=d['uniforms'][ix];pr=d['group_probs'][ix];ratio=d['ratios'][ix].sum(1)
    assert np.array_equal(src,np.tile(src[0],(6,1))) and np.array_equal(u,np.tile(u[0],(6,1)))
    assert len({r['semantic32'] for r in rs})==1 and rs[0]['semantic36']==rs[4]['semantic36'] and rs[2]['semantic128']==rs[3]['semantic128']
    assert np.array_equal(a0[2],a0[3]) and np.array_equal(z[2],z[3]) and np.array_equal(d['states'][ix[2]],d['states'][ix[3]])
    assert np.max(abs(ratio[:,2]-ratio[:,3]))<1e-12 and np.max(abs(ratio[:,5]))<1e-12
    pa=pr[:,0,:,0];m=float(d['baseinfo'][ix[5],0,0]);pred=m*.65*.8*.9
    assert abs(pa[0,0]-pa[1,1]-pred)<1e-12 and abs(pa[0,0]+pa[1,1]-2*pa[2,2])<1e-12
    # Same observed history before first intervention; counterfactual first law independent of arm.
    assert np.max(abs(pr[:,0]-pr[0,0]))<1e-12
    s=(a0==src[:,0,None]).astype(float)-(a0==src[:,1,None]).astype(float);hit=np.isin(a0,src[0]);rc=np.array([r['recommend'] for r in rs]);first=s[:,32];active=s[:,32:44].mean(1);waves=s[:,32:44].reshape(6,3,4).mean(2)
    F=np.column_stack([first,active,waves,(s==1).mean(1),hit.mean(1),z.mean(1),rc==src[:,0],np.isin(rc,src[0]),d['elapsed'][ix],d['evaluations'][ix]])
    assert F.shape==(6,len(fields))
    # P_arm/P_base on each actual history, conditional likelihood: expected L=1 under base.
    w=np.exp(ratio[5]);assert w[5]==1 and abs(w[2]-w[3])<1e-12
    # Cost is not an observable of probability transport; excludes elapsed and eval count? eval calls deterministic mark history yes.
    measured=F[:,:10]
    estim=w[:,None]*measured[5,:]
    # Per-realized-history conditional phase sensitivity: use other modes' state on same fine path.
    diff=pr[5,:,0,0]-pr[5,:,1,0];order=.5*abs(pr[5,:,0]-pr[5,:,4]).sum(-1);era=.5*abs(pr[5,:,0]-pr[5,:,2]).sum(-1)
    pairrows.append(dict(family=g,core=c,rep=rep,first_mass=m,first_qa_plus_minus=pred,first_signed_pred=2*pred,first_signed_observed=float(first[0]-first[1]),first_law_qa=[float(pa[i,i]) for i in range(6)],same_prefix32=True,same_prefix36_swap=True,erase_exact=True,final_recommendations=rc.tolist(),final_mark=z.mean(1).tolist(),importance_weights=w.tolist(),first_phase_divergence=next((int(t) for t in range(32,128) if a0[0,t]!=a0[1,t] or z[0,t]!=z[1,t]),128)))
    q.append(dict(F=F,IS=estim,w=w,first_pred=2*pred,phase_diff=diff,order=order,erase=era,rec=rc,raw=ix))
   group.append(q)
  F=np.array([[q['F'] for q in z] for z in group]);IS=np.array([[q['IS'] for q in z] for z in group]);W=np.array([[q['w'] for q in z] for z in group]);rc=np.array([[q['rec'] for q in z] for z in group]);pred=np.array([[q['first_pred'] for q in z] for z in group])
  comparisons=[]
  for name,i,j in [('phase_reversal',0,1),('erase_plus',0,2),('erase_minus',1,3),('erase_sign_null',2,3),('order_swap',0,4),('plus_vs_base',0,5)]:
   direct=F[:,:,i]-F[:,:,j];dr=ci(direct.mean(1));ir=ci((IS[:,:,i]-IS[:,:,j]).mean(1))
   comparisons.append(dict(name=name,arms=[ARMS[i],ARMS[j]],direct_ci=dr,importance_ci=ir,direct_minus_importance_ci=ci((direct[:,:,:10]-(IS[:,:,i]-IS[:,:,j])).mean(1)),recommendation_disagreement_ci=ci((rc[:,:,i]!=rc[:,:,j]).mean(1))))
  weight=W.reshape(-1,6);ess=weight.sum(0)**2/(weight**2).sum(0)
  out=dict(family=FAMS[g],fields=fields,arm_mean_ci={arm:ci(F[:,:,i].mean(1)) for i,arm in enumerate(ARMS)},comparisons=comparisons,first_law_signed_pred_ci=ci(pred.mean(1)),first_observed_minus_pred_ci=ci((F[:,:,0,0]-F[:,:,1,0]-pred).mean(1)),importance_weight_ci=ci(W.mean(1)),importance_ESS=ess.tolist(),importance_max=weight.max(0).tolist(),same_history_phase_qa_diff=np.mean([[q['phase_diff'] for q in z] for z in group],axis=(0,1)).tolist(),same_history_swap_TV=np.mean([[q['order'] for q in z] for z in group],axis=(0,1)).tolist(),same_history_erase_TV=np.mean([[q['erase'] for q in z] for z in group],axis=(0,1)).tolist())
  res.append(out)
  print(FAMS[g],'pred',out['first_law_signed_pred_ci'],'observed',comparisons[0]['direct_ci'][0][0],'active',comparisons[0]['direct_ci'][0][1],'ESS',ess,flush=True)
 dump(S/'EFFECTS81.json',dict(scope='All six arms matched at same geometry/seed, 24 geometry blocks x8 seeds per family; paired pointwise t intervals, actual elapsed cost not equal-time, same-history likelihood reference and importance sampling separate; no game utility/physical phase.',families=res,matched_pairs=pairrows,blocks_per_family=192,arms=ARMS,fields=fields,input_sha=a['array_sha']))
 p=B/'mqft_restore61_20261126/mqft_trained_rng61';q=subprocess.run([str(p),str(S/'RNG_INPUT81.json')],capture_output=True,text=True,timeout=15,check=True);z=json.loads(q.stdout);assert z['streams']==384 and z['independent_uniforms']==49152 and z['max_error']<1e-12
 z.update(binary_sha=sha(p),input_sha=sha(S/'RNG_INPUT81.json'));dump(S/'RNG_AUDIT81.json',z);print('RNG',z,flush=True)
if __name__=='__main__':main()
