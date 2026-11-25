# RESTORE55: longitudinal self-play physics and sequential wave control
R1/R2. Parent RESTORE54 8c0536e71dfd1d27b3ff87bf0be1022529cc5c8e.
Original archive identities and Q28 six physics objectives verified.
Inspect original L03/L25 Q-score thermometry, L35 depth-dependent
Q mean/variance/beta function, L33 thermodynamic evolution placeholder,
L19/L20 and Q24/Q25/Q26 pointer/record/decoherence plots. Distinguish
source objectives from invalid proxies (sequence robustness is not
quantum envariance). Existing Layer A/Q, 13/13 DAG and 121 owner IDs.
Munos, Geometric Variance Reduction in Markov Chains, JMLR 7 (2006)
https://jmlr.org/papers/v7/munos06a.html : sequential control variates
based on conditional expectations; our exact conditional identities
below are independently derived without stationarity. SciSpace
record context https://scispace.com/papers/broadcasting-classicality-environmental-records-redundancy-qk8oaiofm6c0 ;
the distinction between redundancy and objectivity is kept.
Parallel equilibrium Markov response context
https://arxiv.org/html/0710.4394v2 ; actual FDT below derived for
a specified reset generator. Wolfram verifies derivative of
exp(-gamma t) and local tilted probability. No native subagent.

R3/R4. Two new D4-distinct central 8-stone templates, mapped to
L13/21 with approximately .06 density and peripheral extensions.
Two matched opening games per L and B16/32; same actual IID
wave4 root law, beta_s=10/3, four actual NNUE candidate samples,
move temperature0; 8 games, natural terminal unless observed
otherwise. Opening and budget not interchangeable independent
population samples. Separate fixed roots: four root variants,
three laws IID/PL/reserve, 8 paired seeds per law/root/arm,
B16, baseline, early or late +/- .25 field (h,-h,0,0).
480 root jobs, 8 games. Full root probabilities and histories,
R53 v5 binaries and R54 stage-local analyzer unchanged.
No selector tuning, no physical beta/hbar inferred.

R5. Let I_t be sector one-hot, p_t the actual conditional law
before selection. M_t=I_t-p_t is a martingale increment in the
root-selection filtration. Within wave j, E X_j=E (mean p_t) U
for tangent occupancy X_j=(mean I_t) U. If PL/reserve, later p_t
depends on previous selected actions (support/pending). Replacing
I_t by p_t on observed histories is an unbiased sequential
control, not exact joint wave conditional expectation. Under IID
frozen final wave p_t=q before all draws, it recovers R54.
If arbitrary action-predictor m_t(s) is predictable,
C_t=m_t(s_t)-sum_s p_t(s)m_t(s) has conditional mean zero.
Train block/sector mark means on other core at same L/law,
one zero pseudo-count per cell, no treatment results and no
test fitting; condition on separate training sample. Thus
Y_j-mean C_t is unbiased for same mark mean but not conditional
mark integration. For direct paired finite contrasts, subtract
controls in each arm; shared seeds do not change marginal
unbiasedness. For baseline likelihood ratio W=(L_+-L_-)/(.5),
determined after pulse source block, E[W C_t]=0 for later
draws. Do not subtract W C_t inside source block: W not
predictable there. Raw LR is uncentered W*Y, not the older
centered covariance estimator. No variance reduction theorem
for arbitrary martingale control, test variance directly.
Use whole independent seed histories, compare finite response
sample means and SE (not asymptotic equivalence).

Exact restricted wave: from same actual h0 IID k8 snapshot for
all three laws, choose one top prior action in each of four
quadrants plus next highest prior legal action. Keep full
parent n,w,prior and pending score arithmetic, restrict only
draw support; no claim of full native PL/reserve closure.
Enumerate ordered 5^4 words (PL no replacement 5P4=120).
At each draw p_eta from actual score convention and tangent
field; sum word probabilities, score and predictable Fisher.
R=Cov(Y,G), E G=0, E GG^T=E F; analytic first derivative
vs +/-1e-5 finite diff. Compare full restricted finite +/- .25
to IID frozen reference on same parent/support (only IID exact
frozen). Reuse scalar restricted lattice DP for independent
frozen waves M8/16/32, not native adaptive continuum 1PI.

Longitudinal native output: empirical pi=N/B, visited action
Q=W/N and Q=0 convention for unvisited actions (not a measured
unvisited reward). Fit beta>=0 minimizing KL(pi||softmax(beta Q))
on all legal support, separately visited-only and PUCT-score
all legal fits. Flat/zero/infinite/finite statuses explicit.
Reciprocal finite scale only in histogram; dimensionless,
not physical temperature. Fit objective derivative = model
mean Q - empirical mean Q, Hessian nonnegative Var_model Q.
Visited-only fit is support-selected, not unbiased replacement.
Track entropy, top-two gap, return variance, black-frame
mean (side*root mean), path and root entropy. No independent
ply inference; equal-game phase-bin summaries. Same-root
score proxy E=-pi dot score has exact midpoint decomposition
Delta E = -mean(pi) dot Delta score -mean(score) dot Delta pi;
not work/heat, no comparison across changed legal support.

Actual trie: each visited path prefix contributes root-frame
mark N,W,S2. At depth d, weights N_a from surviving paths,
mean Q=sum W/sum N; total mark variance = weighted within
node variance + between-node Q variance. Nested prefix
partition on SAME surviving paths (length >=d+1) gives
Var(E[Y|prefix d+1])-Var(E[Y|prefix d])>=0 by total variance.
Use matched survival for depth increment; raw depth profile
has composition change. This is a running observed coupling
proxy, not autonomous beta function or fixed point. Graph
pair covariance and coarse block variance retained from
actual tree; do not assign continuum criticality.

Q. Declared isometry |s>|0> -> |s>(sqrt(1-g)|0>+
sqrt(g)|s+1>) on each of eight environment fragments.
Actual sector propensity p at first/middle/last searched ply
sets amplitudes sqrt(p_s) exp(i*.37*s); external g=.2,.5,.8.
Offdiagonal overlap (1-g)^m for m traced fragments,
reduced rho_m=diag(p)+overlap*(rho_0-diag(p)).
For global pure state with eight fragments, Holevo pointer
chi_m=S(rho_m), I(S:F_m)=S(rho_8)+S(rho_m)-S(rho_{8-m}).
Local flag measurement accesses (1-(1-g)^m) H(p), yields
a lower bound on accessible information. Count disjoint
fragments that exceed .9 H(p), not proof of spectrum
broadcast/objectivity. CP exact isometry, endpoints g0/1.
For continuous projector-dephasing generator
L(rho)=gamma sum_s(P_s rho P_s-{P_s,rho}/2),
rho_t=diag(p)+exp(-gamma t)(rho0-diag(p)); population fixed,
pointer projectors stationary, trace PSD and complex phases
preserved before decay. No population thermalization.
Orthogonal Schmidt environment reference: paired system and
environment swap gives state residual sqrt(2)|sqrt(p_i)-
sqrt(p_j)|, zero exactly equal coefficients (phases absorbed
in Schmidt basis). Arbitrary sequence robustness does not
establish envariance. Both reference models declared; native
MCTS phase/bath and physical gamma not identified.

N. Declared stationary reset L=gamma(pi 1^T-I),
pi>0, E/E*=-log pi, beta E*=1, gamma=1 declared.
f=(1,-1,0,0), field energy -h f, pi_h=softmax(log pi+h f).
A=U^T I, C0=Cov_pi(A,f); stationary semigroup gives
C(t)=exp(-gamma t) C0. After step field, mu_h(t)=
exp(-gamma t)pi+(1-exp(-gamma t))pi_h, hence derivative
R_step=C0-C(t), kernel gamma exp(-gamma t) C0.
Check direct matrix exp and +/- .1,.025,.0001.
If initial mu0 differs from pi, response derivative remains
(1-exp(-gamma t)) Cov_pi(A,f), while initial two-time
connected correlation equals exp(-gamma t) Cov_mu0(A,f).
Naive replacement pi -> mu0 fails generically; calculate
mu0=softmax(log pi+.7 f). Explicit countercheck, not a
generic nonequilibrium FDT. Earlier Crooks/JE/continuous
work retained. No native physical energy or thermal bath.
R6 exact algebra and independent numerical tests, R7 actual
campaign and references, R8 measured scope and negative results,
R9 original-output crosswalk, R10 next contract.
