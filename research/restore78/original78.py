from base61 import *
import zipfile,csv
O=B/'mqft_restore77_20261127';A=B/'mqft_restore69_20261126'
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A/f),f
prev=read(O/'ORIGINAL_SOURCE77.json')
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert all(sha(paths[k])==h for k,h in prev['archives'].items())
cat={x['id']:x for x in csv.DictReader((S/'ORIGINAL_TARGET_CATALOG.csv').open())};members={}
for k in ['L16','L21','L22','L24']:
 r=cat[k]
 with zipfile.ZipFile(paths[r['archive']]) as z:b=z.read(r['member'])
 h=hashlib.sha256(b).hexdigest();assert h==r['member_sha256'];members[r['member']]=dict(archive=r['archive'],member=r['member'],sha256=h)
m='omoknuni_quantum-main/python/mcts/quantum/phenomena/gauge_policy.py'
with zipfile.ZipFile(paths['omoknuni_quantum-main.zip']) as z:b=z.read(m);s=b.decode()
for x in ['identity + perturbation','return torch.eye(self.group_dim, dtype=torch.float32)','virtual_loop = [node, neighbor1, neighbor2, node]','curvature = 0.5 * (curvature + curvature.conj().T)','connection_term += commutator.real','field_traceless = field - torch.trace(field)','link_params[param_idx:param_idx + n_params].reshape','action += 1.0 - wilson_loop.value.real']:
 assert x in s,x
members[m]=dict(archive='omoknuni_quantum-main.zip',member=m,sha256=hashlib.sha256(b).hexdigest())
with zipfile.ZipFile(paths['mcts_study.zip']) as z:
 for m in ['mcts_study/effective-action-mcts.md','mcts_study/qft-answers.md','mcts_study/v5.0/new_quantum_mcts.md']:
  b=z.read(m);members[m]=dict(archive='mcts_study.zip',member=m,sha256=hashlib.sha256(b).hexdigest())
  if 'v5.0' in m:assert 'the action Hessian is diagonal' in b.decode() and 'there are no gauge constraints' in b.decode()
  if 'effective-action' in m:assert 'Gauge Structure' in b.decode()
dump(S/'ORIGINAL_SOURCE78.json',dict(archives=prev['archives'],verified_members=list(members.values()),original_attachment_receipt=prev['original_attachment_receipt'],scope='Three original archive identities, actual original gauge producer, renderer/topological/path plotter, relevant effective action and v5 study member bytes; uploaded original attachment receipt inherited.'))
# Exact algebraic examples and original no-plaquette diagonal Hessian issue.
I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Z=np.diag([1.,-1.]).astype(complex)
a=np.pi/4;u=np.cos(a)*I+1j*np.sin(a)*Z;v=np.cos(a)*I+1j*np.sin(a)*X
trace=lambda m:float(np.trace(m).real/2);w=u@v@u.conj().T@v.conj().T
assert abs(trace(u)-trace(v))<1e-15 and abs(trace(w)-.5)<1e-15
p=np.array([.2,.8]);H=np.diag(p)-p[:,None]*p[None,:];assert abs(H[0,1]+.16)<1e-15 and np.max(abs(H.sum(0)))<1e-15
dump(S/'ORIGINAL_GAUGE_AUDIT78.json',dict(status='DERIVED_AND_NUMERICALLY_CHECKED',source_member=members['omoknuni_quantum-main/python/mcts/quantum/phenomena/gauge_policy.py'],source_defects=['I+perturbation not generally unitary','independently initialized opposite directions not inverse','absent virtual links I not covariantly transformed','open-link commutator not adjoint-covariant','Hermitian part of commutator of Hermitian matrices zero','Hermitian traceless projection Lie algebra not group','complex U1 Wilson legitimate','unconstrained link entries and unnormalized Wilson action'],native_tree_scope='No independent cycles in actual prefix tree; quotient and legal grid distinct. No plaquettes does not imply diagonal softmax Hessian or absence of local basis redundancy.',categorical_hessian=H.tolist(),categorical_p=p.tolist(),SU2_equal_individual_traces=[trace(u),trace(v)],SU2_commuting_pair_commutator_trace=1.,SU2_noncommuting_pair_commutator_trace=trace(w),Wolfram_symbolic='Tr[Uz Ux Uz† Ux†]/2=cos(b)^2+cos(2a) sin(b)^2, Uz†Uz=I; successful exact tool output',toy=read(S/'GAUGE78.json')['toy'],scope='Original producer not executed or modified; explicit algebraic counterexamples and corrected separate graph model. No physical native gauge or ghost identification.'))
print('ORIGINAL SOURCE VERIFIED',len(members),trace(w),H[0,1],flush=True)
