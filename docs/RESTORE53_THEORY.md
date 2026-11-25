# RESTORE53: vector spatial source, mixed retarded response and conditional dynamics
R1/R2. Exact RESTORE52 parent bc0d76ff06c0f24ecde418d5efb900012236a9de.
Original archive hashes/Q28 six axes; RESTORE32 Layer A/Q, 13 nodes,
13 edges and 121 output owners. Reuse exact R52 size-specialized
source, distinct v5 schema and crate identity; old runtimes intact.
Exa https://arxiv.org/pdf/2012.15480v2.pdf Sec1 exponential-family
log partition and Legendre duality, local conditional version only.
SciSpace sequential effects context
https://scispace.com/papers/sequentially-independent-effects-4xy6e042s4
(abstract-level; our channel computation direct).
Parallel characteristic-function context
https://arxiv.org/html/1608.07925v2 (quantum jump trajectory, not a
theorem transfer to native MCTS). Wolfram evaluates mixed derivative
of log(1-p-q+p exp(x)+q exp(y)) at zero = -pq and normalized ratio
mixed derivative ij-jp-iq+2pq. All direct finite identities below
do not require those literature hypotheses. No native subagent.

R3/R4. Four spatial quadrants with tie rule y>=L//2,x>=L//2. Let
U be the orthonormal 4x3 Helmert matrix (U^T 1=0). Root field
h_t=U eta_t dimensionless, p_h(a|H_t)=p_0(a|H_t) exp(h_s)/Z_t.
Root conditional law retains full actions, PL active mask and
downstream kernel. Group and scalar fields mutually exclusive with
vector field. v5 schema exact schedule budget x 4, finite |h_s|<=10,
proposal p only. Canonical logits h_s-h_3 enforce common-shift
gauge; full support numerical bound includes max sector span.
Tests group equivalence and common shift, reject wrong length,
dimension and excessive field. First Python replay gate shadowed
pending reservation variable v by vector metadata, corrected before
registered generation; failure retained. No Rust production algorithm
or model change. Two new seeded common central 8-stone geometries
checked D4 against prior selected roots and templates, two nonterminal
peripheral variants at density approximately .06, L13/21.
Independent n2 pilot per root/law (48 histories) before confirm;
fixed axes and arms predeclared, no pilot-driven selection.
B16, beta_s=10/3, wave4 actual NNUE samples4. g0v0 n8,
g1v0 n4 per arm/law; 13 arms: h0, +/- .3 along each of 3 tangent
axes on last block, 4 +/- combinations of axes 0/1 on last block,
+/- axis0 on first block. Additional v1 n4 h0 only. New seeds
separate cohort/root/law, paired within pulse. 1032 fixed-root
jobs and two self games (one/size); no population strength claim.
Budget range not expanded this round; no new critical exponent.

R5/R6. Let I_t=e_{s(a_t)} and q_t conditional sector probability.
f_t=U^T I_t; m_t=U^T q_t; g_t=f_t-m_t;
F_t=U^T(diag q_t-q_t q_t^T)U. A fixed-history normalized
tilt preserves within-sector conditional distributions. Hence
categorical KL = log(sum q_s exp(h_s))-q dot h, exact sector KL.
At each time h=0, gradient log Z=m_t, Hessian F_t, PSD. For
full path likelihood L(eta)=product exp(eta_t dot f_t)/Z_t,
derivatives at zero on block j: G_j=sum g_t, F_j=sum F_t,
partial L / partial eta_i=G_ji, partial_ik L=G_ji G_jk-F_jik.
Different time blocks j,k contribute G_j G_k (no same-time
partition term). Conditional ideal draws imply E G_j=0,
E G_j G_j^T=E F_j, cross-block score covariance zero.
For completed-block observable X_i=(sector tangent occupancy,
mean mark), retarded K_ij=Cov(X_i,G_j) vanishes for j>i.
Prefix likelihood sufficient for response of X_i; directly
verify earlier complete events, linked evals and cuts match
baseline at same seed for all late arms. Finite central first
and second contrasts, mixed four-corner derivative at delta=.3,
compare LR from fixed baseline paths vs actual perturbed paths.
These are finite contrasts, not exact derivative at finite delta;
second derivative uses score outer minus predictable F.
Centered covariance with LR contrast and whole paired seed
jackknife. Causally mask future-source estimated covariances;
retain raw upper triangle norm as finite noise diagnostic.
Joint sample covariance of (mean X,full G) PSD, not a population
Fisher identity in each sample. Use no asymptotic confidence
claim with n4/8. Independent same-root seeds and geometry
variants; fixed total occupancy, zero Fourier mode and Parseval.
Retarded kernel learned from baseline independent seed half,
compare against direct contrasts on other half (fixed zero,
frozen conditional law, score and finite LR predictions).
Frozen conditional probabilities keep observed h0 history,
do not capture adaptive changes; use for occupancy only.
No retuning on test, no native full 1PI from covariance.

Q. Explicit conditional two-qubit S:E reference. After first
interaction exp(-ia Z_S X_E), a=.35 n1, Y measurement and
system reset, E has Bloch vector (y sin2a,0,cos2a).
Nonselective qubit measurement along unit axis n maps
r -> n(n dot r). Choose n=X, m=cos(theta) X+sin(theta) Z.
Then D_m D_n r=m(m dot n)(n dot r);
D_n D_m r=n(n dot m)(m dot r). Read out E Z by second
exp(-ib Z_S Z_E) and system Y, b=.35 n2. The past/future
signed correlation for NM is sin(2a) sin(2b) cos(theta) sin(theta)
and MN zero. At theta=0,pi/2 commute; at nonorthogonal angle
order may differ. Explicit complex matrix calculation checks
analytic expression. Native data supply only counts from
fixed spatial sector (sector 0); no native quantum state
identified, no exclusion of all classical contextual processes.

N. Declared Gibbs reset process pi(s) from actual sector
propensity knots, log-interpolation continuous cyclic.
Units beta E*=1, E=-log pi, normalized clock s, rate tau.
Work w=W/E*=integral E'_{X_s} ds. Column generator
L=tau(pi 1^T-I). Moment vectors a_m=E[w^m 1(X_s=i)]
satisfy a'_m=L a_m+m diag(E') a_{m-1}, a_0(0)=pi0,
a_m(0)=0 for m>0. Characteristic vector g'_k=L g_k+
ik diag(E') g_k, g_k(0)=pi0. E[w^m]=sum a_m,
phi(k)=sum g_k. Continuous exact Poisson refresh at rate tau:
draw state from pi(s) at each event (self redraw allowed).
Between events state fixed and work is exact endpoint energy
difference, no integration time grid for work. No actual
state transition probability sum_i pi_i(0) exp[-tau integral
(1-pi_i) ds]; for cyclic protocol those paths have w=0.
This is a lower bound on zero-work atom; distinct from
probability exp(-tau) of no refresh. Compare sample mean,
characteristic function, no-transition fraction with ODE
and exact integral; finite sample errors recorded, not a
density reconstructed from few CF points. Retain earlier
Crooks/JE and independent mean-work ODE. No native heat/bath.

R7-R10 and C1-C10. Actual registered paths, exact restricted
lattice reference on new actual snapshot, vector Fisher/mixed
response, quantum order and work distribution checks. Results
and remaining limitations in RESULTS_KO and NEXT_DYNAMICS.
Native independent subagent NOT_RUN; actual independent RNG,
CAS, analytic/ODE/MC verification. Preserve all previous
negative action/selector evidence, no new plot nodes.
