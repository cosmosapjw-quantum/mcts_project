from rg98 import *
def main():
 a=read(S/'SUMMARY98.json');x=read(S/'RG98.json');v=read(S/'INDEPENDENT98.json');assert v['independent_prefix_checks']==10400
 groups=a['groups'];f=a['fields'];ix=lambda k:f.index(k);out=[]
 for arm in [0,1]:
  for bi in range(5):
   s=[g for g in groups if (g['cohort'],g['arm'],g['bi'],g['budget'],g['h'])==(2,arm,bi,64,64)]
   assert len(s)==5
   mean=np.mean([g['ci'][0] for g in s],axis=0)
   out.append(dict(arm=arm,bi=bi,means={k:mean[:,ix(k)].tolist() for k in ['KL_A','KL_X','Fisher','beta_field_rms','theta_x','theta_y','projection_TV','push_projection_TV','theta_next_error','iterated_TV','KL_A_vs_frozen_mean_gap']}))
 with np.load(S/'RG98.npz') as q:
  V=np.concatenate([q[f'V{L}'] for L in LS]);M=np.concatenate([q[f'M{L}'] for L in LS])
  mask=(M[:,3]!=0);n=int(mask.sum())
  # Steps with a full-rank next basis; nontrivial push only.
  rank=V[:,:,ix('rank')];tv=V[:,:-1,ix('push_projection_TV')];e=V[:,:-1,ix('theta_next_error')];valid=rank[:,1:]==2
  summary=dict(nonuniform_control_prefixes=n,rank2_next_cases=int(valid.sum()),rank2_projection_mismatch_cases=int(((e>1e-9)&valid).sum()),nonzero_push_projection_cases=int((tv>1e-9).sum()),max_push_projection_TV=float(tv.max()),max_theta_error=float(e[valid].max()),max_iterated_TV=float(V[:,:,ix('iterated_TV')].max()),max_projection_TV=float(V[:,:,ix('projection_TV')].max()),zero_beta_max_push=float(tv[~mask].max()),zero_beta_max_KL=float(abs(V[~mask,:,ix('KL_A')]).max()),min_KL_loss=float(V[:,:-1,ix('KL_step_loss')].min()),min_Fisher_loss=float(V[:,:-1,ix('Fisher_step_loss')].min()))
 crit=read(R87/'CRITICAL87.json');orig=read(R77/'ORIGINAL_RG_AUDIT77.json')
 a['fresh_equal_size_means']=out;a['counts']=summary;a['prior_critical_gates']=crit['gates'];a['prior_WF']=orig
 dump(S/'SUMMARY98.json',a)
 print('COUNTS',summary,flush=True);print('CONTROL',a['controls'],flush=True)
 for q in out:
  if q['arm']==1 and q['bi'] in [0,1,3,4]:
   print('FRESH DENSE',q['bi'],{k:[round(vv,6) for vv in q['means'][k]] for k in ['KL_A','Fisher','push_projection_TV','iterated_TV','beta_field_rms']},flush=True)
 print('PRIOR GATES',str(crit['gates'])[:1600],flush=True)
if __name__=='__main__':main()
