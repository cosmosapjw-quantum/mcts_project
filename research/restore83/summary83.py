from base61 import *
A=read(S/'AUDIT83.json');T=read(S/'TREE83.json');R=read(S/'RECORD83.json');V=read(S/'INDEPENDENT83.json');F=read(S/'FRAGMENTS83.json');build=read(S/'BUILD83.json')
assert A['paths']==648 and A['forced_paths']==576 and A['reference_replays']==72 and A['diagnostic_backups']==82944 and A['fresh_paths']==0
assert len(T['contexts'])==len(R['contexts'])==72 and V['status']=='PASS_SCOPED' and V['raw_forced_paths']==576 and V['independent_record_matrices']==720
assert len(F['contexts'])==144 and build['status']=='PASS_BUILD_AND_LIB_SUITE' and sha(S/'logs/test83.log')==build['test_log_sha']
assert '394 passed; 0 failed; 14 ignored' in (S/'logs/test83.log').read_text()
# Independently recompute fragment information using explicit index loops and eigenvalues.
with np.load(S/'RECORD_MATRICES83.npz') as z:D={k:z[k] for k in z.files}
def S_vn(a):
 q=np.linalg.eigvalsh(a);q=q[q>1e-16];return float(-q@np.log(q))
def partial(a,keep):
 if keep==[0,1]:return a
 b=np.zeros((2,2),complex)
 for i in range(2):
  for j in range(2):
   for k in range(2):
    aa=2*i+k if keep==[0] else 2*k+i;bb=2*j+k if keep==[0] else 2*k+j;b[i,j]+=a[aa,bb]
 return b
ferr=0.
for i,row in enumerate(F['contexts']):
 n=i//2;j=i%2;R0=D['R'][n,j*5];p=[np.trace(R0[k*4:k*4+4,k*4:k*4+4]).real for k in range(2)]
 assert [row['family'],row['core']]==D['meta'][n].tolist()
 for k,keep in enumerate(([0],[1],[0,1])):
  a=[partial(R0[h*4:h*4+4,h*4:h*4+4]/p[h],keep) for h in range(2)]
  v=S_vn(p[0]*a[0]+p[1]*a[1])-sum(p[h]*S_vn(a[h]) for h in range(2));ferr=max(ferr,abs(v-row['fragments'][k]['holevo']))
assert ferr<1e-12
dump(S/'FRAGMENT_AUDIT83.json',dict(status='PASS_SCOPED',fragment_cases=432,max_holevo_error=ferr,scope='Independent explicit partial trace loops; secondary exploratory pointer diagnostic.'))
out=[]
keys=['success','conditional_Hadamard_TV','unconditional_Hadamard_TV','conditional_trace_distance','unconditional_trace_distance','frozen_probability_TV','frozen_trace_distance','frozen_stopped_TV','phase_reversal_TV','phase_reversal_stopped_TV','phase_erasure_TV','offdiag_l1','conditional_entropy','diagonal_entropy']
for g,name in enumerate(['g0','g3','Rapfi']):
 a=[x for x in R['contexts'] if x['family']==g];row=dict(family=name,contexts=24,settings_spread_mean=np.mean([x['settings_spread'] for x in a],0).tolist(),protocol_difference_TV_mean=float(np.mean([x['protocol_difference_TV'] for x in a])),protocols={})
 assert len(a)==24 and all(all(max(s)>1e-10 for s in x['settings_spread']) for x in a)
 for protocol in ('native_repeat','pair_diagnostic'):
  v=[x['protocols'][protocol] for x in a];q=[x for x in F['contexts'] if x['family']==g and x['protocol']==protocol];assert len(q)==24
  z={k:dict(mean=float(np.mean([x[k] for x in v])),median=float(np.median([x[k] for x in v])),min=min(x[k] for x in v),max=max(x[k] for x in v)) for k in keys}
  z.update(NPT_any=sum(max(x['negativity'] for x in y['splits'])>1e-10 for y in v),NPT_split_counts=[sum(y['splits'][i]['negativity']>1e-10 for y in v) for i in range(3)],max_negativity=max(max(x['negativity'] for x in y['splits']) for y in v),mean_QMI=np.mean([[s['QMI'] for s in x['splits']] for x in v],0).tolist(),fragment_nontrivial=sum(x['nontrivial'] for x in q),disjoint_90_Holevo=sum(x['disjoint_90_Holevo'] for x in q),disjoint_90_classical=sum(x['disjoint_90_classical'] for x in q),mean_fragment_Holevo_fraction=np.mean([[s['holevo_fraction'] for s in x['fragments']] for x in q],0).tolist())
  row['protocols'][protocol]=z
 out.append(row)
q=dict(fresh_paths=0,diagnostic_paths=648,forced_paths=576,reference_replays=72,diagnostic_backups=82944,raw_bytes=A['raw_bytes'],contexts=72,actual_leaf_count=576,actual_setting_count=504,contexts_branch_dependent=72,max_leaf_relative_error=max(x['protocols']['native_repeat']['native_leaf_relative_error'] for x in R['contexts']),max_completeness_error=max(z['completeness_error'] for x in R['contexts'] for z in x['protocols'].values()),max_causal_error=max(z['causality_error'] for x in R['contexts'] for z in x['protocols'].values()),max_unconditional_Hadamard_TV=max(x['protocols']['native_repeat']['unconditional_Hadamard_TV'] for x in R['contexts']),min_native_success=min(x['protocols']['native_repeat']['success'] for x in R['contexts']),max_native_success=max(x['protocols']['native_repeat']['success'] for x in R['contexts']),native_NPT_any=sum(x['protocols']['native_repeat']['NPT_any'] for x in out),diagnostic_NPT_any=sum(x['protocols']['pair_diagnostic']['NPT_any'] for x in out),fragment_90_cases=sum(x['disjoint_90_Holevo'] for x in F['contexts']),fragment_audit=read(S/'FRAGMENT_AUDIT83.json'),errors=A['errors'],independent=V,families=out,scope='Declared adaptive coherent extension of exact retained native histories conditional H32; not physical/native coherent record identification. Normalized pair restriction is a different controlled protocol. All actual replay cases diagnostic only.')
dump(S/'SUMMARY83.json',q)
old=read(B/'mqft_restore82_20261127/CONVERGENCE82.json');prev=[]
for c in old['claims']:
 c=dict(c);c['evidence']=[('mqft_restore82_20261127/'+x if '/' not in x else x) for x in c['evidence']]
 if c['id']=='Q-RECORD-03':
  c['restore83_update']='Actual branch-dependent coherent extension constructed for explicit 8-leaf selective tree and separate normalized pair-restricted protocol; observed diagonal law cannot identify unique dilation (history-tag counterexample). Native coherent recombination/objectivity not demonstrated; success mass tiny, fragment 90% packing absent. Original physical identification remains open.'
  c['evidence']+=['SUMMARY83.json','RECORD83.json','FRAGMENTS83.json']
 prev.append(c)
new=[
 dict(id='Q-ADAPT-05',stage=83,status='ACCEPTED_DERIVED_AND_CHECKED',items=['L15','L16','L20','Q24','Q35','Q37','Q42','Q43'],claim='A branch-dependent coherent extension of the specified three-wave selective native history tree exists: product of actual K_i(H)^4 and bath Kraus, positive abort effect and complete stopped instrument, with all 576 native retained fine-history probabilities reproduced. A distinct source-restricted diagnostic protocol gives a normalized eight-outcome instrument using the same actual branch settings.',evidence=['THEORY.md','TREE83.json','RECORD83.json','INDEPENDENT83.json','SUMMARY83.json'],limit='Conditional on 72 actual H32 contexts; restricted all-four-identical source event. Actual replays are forced classical diagnostics. Coherent branch control and erased workspace are declared, not performed. Abort post-state not actual omitted-history reconstruction. Pair restriction/forced fillers not native event conditioning.'),
 dict(id='Q-DILATION-06',stage=83,status='ACCEPTED_DERIVED_AND_CHECKED',items=['L15','Q25','Q35','Q37','Q45'],claim='Identical measured retained path probabilities and stopped classical law admit different explicit coherent record completions by retaining or tracing orthogonal history tags; the observed diagonal does not identify record coherence or entanglement. All 72 native-repeat and 72 diagnostic model contexts have nonzero readout contrast in a declared coherent completion; exact success-weighted bound retained.',evidence=['THEORY.md','RECORD83.json','FRAGMENTS83.json','FRAGMENT_AUDIT83.json','SUMMARY83.json'],limit='Constructive nonidentifiability from the measured record alone in these declared models, not a physical impossibility of coherent controlled implementation. 31/72 native selective conditional states and 67/72 diagnostic states NPT in at least one split; native selective event very rare, no 90% disjoint fragment redundancy. Physical objectivity and actual operational record coherence remain open.')]
dump(S/'CONVERGENCE83.json',dict(status='NINE_BOUNDED_CORRECTED_SUBCLAIMS_ACCEPTED_THREE_OPEN',claims=prev+new,accepted_subclaims=9,new_accepted_subclaims=2,final_original_item_certifications=0,prior_corrected_report_artifacts=['L31'],scope='Seven prior accepted subclaims preserved. Explicit adaptive extension and measured-diagonal nonidentifiability accepted; Q-RECORD-03 physical identification/objectivity still open.'))
print('SUMMARY',q['native_NPT_any'],q['diagnostic_NPT_any'],q['fragment_90_cases'],q['max_unconditional_Hadamard_TV'],'fragment error',ferr,flush=True)
