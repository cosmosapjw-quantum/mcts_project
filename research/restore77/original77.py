from base61 import *
import zipfile,csv
import sympy as s
O=B/'mqft_restore76_20261127';A=B/'mqft_restore69_20261126'
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O/f)==sha(A/f),f
prev=read(O/'ORIGINAL_SOURCE76.json')
paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert all(sha(paths[k])==h for k,h in prev['archives'].items())
cat={x['id']:x for x in csv.DictReader((S/'ORIGINAL_TARGET_CATALOG.csv').open())}
ids=['L13','L29','L35','L36','L37','L38','Q09','Q11','Q13'];members={}
for k in ids:
 r=cat[k]
 with zipfile.ZipFile(paths[r['archive']]) as z:b=z.read(r['member'])
 h=hashlib.sha256(b).hexdigest();assert h==r['member_sha256'];members[r['member']]=dict(archive=r['archive'],member=r['member'],sha256=h)
with zipfile.ZipFile(paths['mcts_study.zip']) as z:
 m='mcts_study/v3.0/qft-mcts-statistical.md';b=z.read(m);x=b.decode()
 assert 'dg/dl = β_g = -g/2 + g³/(8πT) - gλ²/(4π²)' in x and 'g* = 2√(2πεT_0)' in x
 members[m]=dict(archive='mcts_study.zip',member=m,sha256=hashlib.sha256(b).hexdigest())
g,t,c,l,ta,ep,eta,t0=s.symbols('g T c lambda tau epsilon eta T0',positive=True)
beta=s.Matrix([-g/2+g**3/(8*s.pi*t)-g*l*l/(4*s.pi**2),-t/ta+g*g*t/(4*s.pi)-l*l*t/(8*s.pi),eta*g*g/(16*s.pi)-c/ta**2+l*c/(4*s.pi*ta),ep*l/ta+g*g*l/(2*s.pi)-l**3/(12*s.pi)])
wf=s.simplify((beta[0]/g).subs({g:2*s.sqrt(2*s.pi*ep*t0),t:t0,l:2*s.pi*ep}));assert s.simplify(wf+(ep-s.Rational(1,2))**2+s.Rational(1,4))==0
no=s.simplify((beta[3]/l).subs(g**2,4*s.pi/ta+l*l/2));assert s.simplify(no-((ep+2)/ta+l*l/(6*s.pi)))==0
fp={g:2*s.sqrt(s.pi/ta),t:1/ta,c:eta*ta/4,l:0}
res=s.simplify(beta.subs(fp));assert res==s.zeros(4,1)
jac=s.simplify(beta.jacobian([g,t,c,l]).subs(fp));x=s.Symbol('x');pol=s.factor(jac.charpoly(x).as_expr());want=(x-(ep+2)/ta)*(x+1/ta**2)*(x*x-x+1/ta);assert s.simplify(pol-want)==0
src=dict(archives=prev['archives'],verified_members=list(members.values()),original_attachment_receipt=prev['original_attachment_receipt'],scope='Three original local archives and exact relevant plotter/claim source members')
audit=dict(status='DERIVED_CHECKED_SYMPY_WOLFRAM',source_member=members[m],scope='Original formulas, finite positive T and frozen tau; original tau(N) is not constant; no alternative physical RG identified',wilson_fisher_beta_g_over_g=str(wf),positive_lambda_residual=str(no),frozen_tau_candidate={str(k):str(v) for k,v in fp.items()},beta_residual=list(map(str,res)),jacobian=[[str(x) for x in row] for row in jac.tolist()],characteristic_polynomial=str(pol),gaussian_singular='g^3/T path-dependent near (0,0): T=g^3 gives 1, T=g^2 gives 0 limit',original_prior_infinite='lambda=infinity not finite-coordinate fixed point')
assert src==read(S/'ORIGINAL_SOURCE77.json') and audit==read(S/'ORIGINAL_RG_AUDIT77.json')
print('ORIGINAL_RG_REPRODUCED',wf,no,res,pol,flush=True)
