from base61 import *
P=read(S/'PLOT_MANIFEST89.json');I=read(S/'INDEPENDENT89.json');A=read(S/'ORIGINAL_SOURCE89.json');O=B/'mqft_restore88_20261128';old=read(O/'CONVERGENCE88.json');assert I['status']=='PASS_SCOPED'
details={
'L23':('Exact repetition code and stabilizer recovery, syndrome-conditioned MAP posterior, actual native sign fault rates and redundancy, coherent conditional recovery and matched native game evidence.','3200 R67 roots and80 R68 games,579026 eval records, zero cut error; 24 posterior cases and465 repeated-channel checks. Original value error, redundancy, confidence panels plus channel/utility diagnostics.','Classical sign transport is not hardware QEC; quantum repetition code protects X errors only, known-angle inversion requires calibration; no general strength admission.',['QEC89.json','NATIVE89.json','SUMMARY89.json','INDEPENDENT89.json','SOURCE_CORRECTIONS89.json']),
'L21':('Recreate original 3x3 gauge, QEC and topology report, R88 actual graphs/declared SU2 connection and discrete persistence, R89 explicit actual sign and declared channel.','R88 L22/L24 corrected component identities verified, L23 dedicated output reconstructed; nine panels preserve separate native and model quantities.','No shared physical joint state or native gauge/coherence identification; discrete extrema not dynamical stability.',['QEC89.json','SUMMARY89.json','INDEPENDENT89.json'])
}
items=[]
for c in A['items']:
 k=c['id'];m,r,l,e=details[k];f=next(x for x in P['figures'] if x['ids']==[k])
 items.append(dict(id=k,original_source=c['original_source'],acceptance_contract=c['acceptance'],decision='ACCEPTED_CORRECTED_SCOPED_ARTIFACT',method=m,result=r,remaining_limit=l,figure=f['figure'],png_sha=f['png_sha'],pdf_sha=f['pdf_sha'],evidence={z:sha(S/z) for z in e},prior_evidence=P['prior_sources'] if k=='L21' else {},original_science='NOT_CERTIFIED'))
assert len(items)==2
accepted=old['corrected_report_artifacts_accepted']+[x['id'] for x in items];assert len(set(accepted))==11
dump(S/'ITEM_RECONSTRUCTION89.json',dict(status='TWO_CORRECTED_SCOPED_ARTIFACTS_ACCEPTED',items=items,prior_report_artifacts=old['corrected_report_artifacts_accepted'],corrected_artifacts_total=11,original_scientific_certifications=0))
lines=['# MQFT advanced quantum original output reconstruction','RESTORE89; R69 research authority. Corrected finite model/native artifact scope, not original physical certification.']
for x in items:lines+=['',f"## {x['id']}",f"**Acceptance:** {x['acceptance_contract']}",f"**Method:** {x['method']}",f"**Result:** {x['result']}",f"**Limit:** {x['remaining_limit']}",f"![{x['id']}](plots/{x['figure']}.png)",f"[PDF](plots/{x['figure']}.pdf) · Evidence: "+', '.join(x['evidence']),'**Decision:** ACCEPTED_CORRECTED_SCOPED_ARTIFACT. Original science NOT_CERTIFIED.']
(S/'ORIGINAL_OUTPUT_REPORT89.md').write_text('\n\n'.join(lines)+'\n')
claims=[]
for c in old['claims']:
 c=dict(c);c['evidence']=[('mqft_restore88_20261128/'+z if '/' not in z else z) for z in c['evidence']];claims.append(c)
claims.extend([
dict(id='QEC-POSTERIOR-01',stage=89,status='ACCEPTED_DERIVED_AND_CHECKED',items=['L23','L21'],claim='For the declared bit-flip repetition sign channel, the syndrome identifies complementary masks; equal-prior MAP residual is sum of smaller paired probabilities, majority optimal for p<=1/2 in iid/common mixture. Zero syndrome need not mean no error. Actual R67/R68 sign transport reauthenticated, conditioned success distinct from detection.',evidence=['THEORY.md','QEC89.json','NATIVE89.json','SUMMARY89.json','INDEPENDENT89.json'],limit='Classical sign channel with stated mask distribution and decoder. No phase/hardware correction or general game strength.'),
dict(id='QEC-COHERENT-02',stage=89,status='ACCEPTED_DERIVED_AND_CHECKED',items=['L23','L21'],claim='For identical calibrated coherent X rotations on the explicit three-qubit repetition code, syndrome-conditioned logical Kraus operators are scalar times unitary, hence a known-angle conditional inverse restores any logical state. Ordinary recovered channel differs from Pauli-twirled repeated channel even at equal one-cycle entanglement fidelity.',evidence=['THEORY.md','QEC89.json','INDEPENDENT89.json'],limit='Exact declared auxiliary channel, ideal recovery, known noise angle; mismatch tested, arbitrary unknown noise or physical device not corrected. No superiority over direct calibrated physical inverse.')
])
dump(S/'CONVERGENCE89.json',dict(status='TWENTY_THREE_BOUNDED_CORRECTED_SUBCLAIMS_ACCEPTED_SIX_OPEN',claims=claims,accepted_subclaims=23,new_accepted_subclaims=2,final_original_item_certifications=0,corrected_report_artifacts_accepted=accepted,new_corrected_artifacts=2,artifact_reconstruction_sha=sha(S/'ITEM_RECONSTRUCTION89.json'),scope='R88 accepted artifacts and prior six open claims retained. Corrected artifact completion separate from original physical claims.'))
print('REPORT 2; corrected total11; accepted claims23')
