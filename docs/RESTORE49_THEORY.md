# RESTORE49: actual size-family and time-local field response
R1/R2: exact RESTORE48 parent b448c0257934edf832a8b4b2f1978a27b2c68180.
Original Q28 six axes, original 3 ZIP identities, RESTORE32 Layer A/Q,
13 existing topic nodes, 121 original output owners unchanged.
Same common actual Rapfi MixNet v3 weight, verified header supports
13..22. Preserve old 15x15 binary/source/trace. New size-specialized
source generation from exact parent, same Quartz algorithms, explicit
size in trace/plan schema and state metadata; distinct crate names
to prevent Cargo target cache collision. For L!=15 size-specific
Zobrist seed; empty bare hash zero, identity is (L,hash,rule).
Freestyle >=5, action a=yL+x, board open edges. Full policy validation
and actual child WDL, child reconstructed board/ply/side checked.
9/11 unsupported, no header override or heuristic fallback.

R3/R4: two actual old p8 positions selected by prior registration
(sorted first root each 44/45 cohort), common absolute center-relative
geometry, source action mapped. Different density 8/L^2, not a
fixed-density design. Five sizes13/15/17/19/21; B16, four seeds each
law/arm, constant h=0,+/-.2; sizes15/21 additionally early and late
four-selection block pulses +/- .2. Seed groups matched within
size/law/root only, not across L. One actual complete serial self
game per L, common first four prescribed moves. No strength ranking.
All fixed root histories use 17 parent and 68 nonterminal child
queries; match logical NNUE calls, not FLOPs or wall time.
Finite-size background: Dohm et al. arxiv:1209.0105 (critical Binder
quantity depends on geometry/anisotropy under equilibrium conditions);
Nakamura 2010 dynamic-correlation-length scaling
https://scispace.com/papers/nonequilibrium-dynamic-correlation-length-scaling-method-37auk7lebm ;
Lee et al. finite-time and finite-size scaling
https://scispace.com/papers/finite-time-and-finite-size-scaling-of-the-kuramoto-9i3tg0kcsl .
No transport of their universality hypotheses to this adaptive process.

## R5 spatial statistics and exact projected count reference
m_a=N_a/B, sum m=1. Fixed root complete history samples only. Sample
C=1/(n-1) sum (m_i-mbar)(m_i-mbar)^T, C 1=0, rank<=n-1.
For lattice coordinates r_a=(y,x) in lattice spacing units, discrete
Fourier S_k=E|sum_a delta m_a exp(-2pi i k·r_a/L)|². Fourier used
as diagnostic basis on open board, not periodic dynamics. S_0=0;
Parseval sum S/L²=tr C. Do not use equilibrium correlation-length
formula S0/Smin or claim inverse full Hessian (rank3 at n4).
Two-sector map Q maps central 5x5 vs complement, C_coarse=Q^T C Q.
Local spatial response to source at a0 is Cov(m,G_a0); sum zero.
Full-score vector G_a=sum_t(1[a_t=a]-p_t(a)); common shift of all
logits cancels, sum G=0. This fixed-count covariance/response is
algorithmic path statistics, not thermodynamic spin susceptibility.
Native entropy H and finite inverse score-scale statuses by cut.
For one-loop reuse exact restricted 5-action width4 reference:
support source + top four L15 initial prior legal actions in common
central 13x13 region, translated to all L. For each L and geometry
use actual IID h0 first seed cut k8, same snapshot for 3 laws;
full parent visits/priors/reservation arithmetic kept. Restrict
and renormalize selection only. 30 exact word measures, three-sector
coarse count DP M8/16/32, tangent determinant and lattice covolume
as in RESTORE45. Independent frozen waves, not native adaptive 1PI.
Fisher pushforward ordered->unordered and target count, exact
information loss; no claim of native continuum RG beta function.

## R5/R6 time-local source and causal check
For field h_t at actual selection clock, fixed downstream law
and same support, P_h/P_0=prod_t exp(h_t I_t)/(1+p_t(expm1 h_t)).
For pulse h on block j of 4 draws, log L_i^j(h)=sum_{t in j,t<=i}
[h I_t-log(1+p_t expm1(h))] up to observable block i.
Prefix LR=1 before j, same pulse ratio thereafter. R_ij =
Cov(X_i,G_j), vanishes before source under ideal law. Score and
finite contrast not interchangeable; centered finite prefix LR
uses sample covariance with [L_i(+h)-L_i(-h)]/(2h). Direct paired
contrast across seed groups, delete-one jackknife complete histories.
Exact causal negative control: late pulse begins selection index12.
Compare full pre-intervention events/evals/cuts at matching seed,
excluding worker/time only. All marks completed before future block
selection; verify timestamps from trace. Direct effects before
intervention must be zero, independent of sampling error.
Actual path-source ratio for arbitrary vector h->h':
delta_t=h'_t-h_t, Sigma=sum(log(1+p expm1(delta))-delta I).
D_pred=sum(log(1+p expm1(delta))-delta p)>=0,
Sigma-D_pred=-sum(delta_t(I_t-p_t)); exp(-Sigma) has mean1 under
ideal law, finite sample normalization not exact. No time reversal
or physical heat identified. Wolfram Bernoulli score/second derivative
checked. Related continuous-time martingale response: Faggionato
and Silvestri, ALEA21 (2024), https://arxiv.org/html/2201.02982v4 ;
finite discrete proof above independent of CT hypotheses.

## Q and N original physics extensions
Actual h0 spatial action records partition into four quadrants,
axis tie rule y>=c/x>=c. Empirical normalized sector occupancy
(not independent draw population) defines p. Declared isometry
controlled on spatial label |a>, fragment |e_a>=sqrt(1-g)|0>+
sqrt(g)|a+1>, g=.35 and eight fragments. Gram overlap (1-g)^m.
Compute QMI, Holevo, specified erasure readout MI and distinguish
them. Explicit conditional complex rho at two retained/dephasing
fragments, F_ab=.65² offdiag, diagonal1, theta rotation with
G=diag(0,1,2,3). rho(theta)=exp(-i theta G)rho exp(i theta G).
d rho/d theta=-i[G,rho], phase QFI, fixed uniform coherent probe,
identical diagonal dephased model QFI=0. CP-positive state and
phase sensitivity conditional, not observed native quantum phase,
SBS or objectivity.
Thermal reference generalizes R48 two-state cycle to four spatial
sectors. Mean actual conditional root probability vectors over
three selected completed blocks (first, third, fourth) and four
h0 histories give fixed external knot triplets. Gibbs weights
pi_j, energy E/E*=-log pi_j, beta E*=1; log-weight interpolation
and renormalization closes cycle. Partial reset T=(1-eta)I+eta 1 pi^T,
eta=1-exp(-tau/(3m)), m32 per segment. Exact tilted transfer DP
Z_F(lambda)=Z_R(1-lambda) for reverse reset/quench sequence;
JE, first law and W=Sigma+KL(final||initial) checked. This is
declared conditional heat/energy model, not native measured
energy or bath. Fixed phase/energy units stated, no inferred hbar.

R7/R8: actual campaigns, independent replay and finite calculations.
Rank constraint and n=4 per root/law/arm preclude exponent, response
equivalence or thermodynamic-limit claims; no model tuning or
selector admission. R9/R10 exact source generation, data and plots,
original objective watchdog, next fixed-density and temporal precision
campaign. Native independent subagent unavailable, separate Rust
helper, independent Python reducer, CAS and exact reference checks.
