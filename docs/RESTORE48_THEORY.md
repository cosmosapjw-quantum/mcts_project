# RESTORE48: scale precision, causal response and conditional physical measurements
R1/R2. Parent 6796fd5e2c9c24464d9cc01c5152dcbc5951c5c7, original
Q28 six axes, exact three archives, RESTORE32 Layer A/Q and fixed 13
topic nodes, 121 original outputs. Do not substitute optimization for
physics. Reuse existing 4 scale anchors and v3 real MixNet. B32 five
arms (0,+/-.1,+/-.2), B64 three (0,+/-.2); add 14 independent seed
labels per arm to reach 16 including RESTORE47 pilot. Three bounded
campaign streams, max 24 declared CPU slots, cap720s/stream. No B16
doubling or new game cohort. Prior B16 16/arm at anchors for reference.
Separate h and B: compare B32 vs B64 at .2; B16 uses .3, not same h.
No population sample of openings, no spatial scaling claim.

## R3–R6: causal native response, mixed sources and error estimation
For fixed full adaptive history gamma, ideal conditional draws,
p_t=P(I_t=1|past), g_t=I_t-p_t, F_t=p_t(1-p_t).
Same law/downstream kernel, log L_h=sum_t [h I_t-log(1+p_t expm1(h))].
G=sum g, F=sum F_t. L'_0=G and L''_0=G²-F. For a bounded observable
X independent of explicit h, derivative E[XG], second E[X(G²-F)].
Wolfram derivatives verify Bernoulli p, p(1-p), (I-p)²-p(1-p).
Mohamed et al., Monte Carlo Gradient Estimation in Machine Learning,
JMLR21 (2020), sec4.3.4, https://jmlr.org/papers/volume21/19-346/19-346.pdf,
general higher density derivatives (search excerpt; above specific formula derived).

For 4-backup blocks i (first may aggregate short phases), define
X_i=(target fraction, mean root-frame return), G_j=sum g in block j,
F_j=sum F_t in block j. A vector field h_j on block j is a mathematical
extension of fixed-history root logit source. Backup boundary is a
stopping/filtration cut; all X_i marks available before selection of
future block. Under ideal law, conditional E[G_j|F_i]=0 if j>i.
Thus R_ij=Cov(X_i,G_j)=0 for future j>i and
d E[mean_i X_i]/dh=(1/d)sum_(i,j<=i) R_ij, d=B/4.
For second order on each block, prefix scores S_i=sum_(j<=i)G_j,
V_i=sum_(j<=i)F_j; E[X_i(S_i²-V_i)] is exact derivative because
suffix probability integrates to one. For mixed h_j,h_l:
E[X_i(G_jG_l-delta_jl F_j)] with j,l<=i; any future index vanishes.
Proof: marginal path probability up to cut is product of preceding
conditionals; differentiation at fixed bounded horizon commutes
with sum. Equivalently conditional martingale orthogonality.
Native block chronology directly audited in new paths and 240 reused
h0 traces. Do not infer actual time-dependent native response from a
direct time-dependent intervention, which is not executed; score
functional predicts it under stated assumptions. Future-score
entries measure finite statistical noise, not retrocausality.

Prefix likelihood Rao-Blackwell: L_i=E_0[L_B|F_i], hence
E[X_i L_B]=E[X_i L_i]. For raw estimator X_i L_i variance cannot
exceed X_i L_B (law of total variance). Apply to signed finite
contrasts D_i=(L_i(h)-L_i(-h))/(2h) and D2_i=(L_i(h)+L_i(-h)-2)/h².
For centered covariance U-statistics and sums across blocks variance
ordering is not automatic, so compare jackknife errors empirically.
Full and prefix LR target SAME finite contrast as direct, whereas
score and G²-F target derivatives at zero. LOO center independent
seed groups (sample covariance) unbiased under ideal law.
Paired delete-one jackknife entire +/-/0 seed group. Fixed-root
conditional SE sqrt(sum_r se_r²)/4, across-root descriptive SE
separate, neither gives population CI. Richardson (4D(.1)-D(.2))/3
and second counterpart formally O(h^4) for smooth expected response;
no such rate for individual discontinuous greedy histories. Covariance
and Binder fourth-moment diagnostic vs backup blocks (not spatial size).
No thermodynamic FDT identified.

## R4–R7: empirical action and memory discrimination
Repeat R47 fixed two-observable generating function and mixed Legendre
derivative on B32/64 h0 with 8+8 seed split, 4 fixed J vectors. Compare
zero, quadratic and full empirical mean errors; no tuning. No new
test opening population. Repeat prior frozen ridge and categorical
memory protocols on enlarged old-root training, unchanged R47 heldout
roots; this is repeated evaluation, not independent fresh holdout.
Preserve exact nonstationary MZ result. Do not infer 1PI from empirical
finite output tilt. Show entropy and inverse score-scale status by
actual backup cut; do not force finite beta.

## Q: finite-shot conditional process reconstruction
Reuse R47 exact four channels on same 12 actual drive-count pairs.
For six Pauli preparations +/- and 3 Pauli measurements sample N
independent Bernoulli shots per setting (N128,512,2048,32 trials).
Estimate PTM T by differences of means and unital offset mean, fix
trace row exactly. Normalized Choi J from T. Finite-shot linear
inversion can violate PSD. Repair along depolarizing direction
J_r=(1-r)J+r I_4/4, r=max(0,-lambda_min/(1/4-lambda_min)).
Trace preserving because both J and I/4 have partial trace I/2.
This is a simple CP repair, not nearest-channel or ML estimate.
Compare raw/repaired trace-norm error; persistent vs matched-block
reset distinguishability against independently sampled same-channel
null. Dephased control retained. Exact conditional computational
simulation, not native quantum phase or hardware shots.
Related literature: https://scispace.com/papers/boosting-projective-methods-for-quantum-process-and-detector-4ugx64hy5n
(projective physical reconstruction; this repair is different).

## N: finite-duration conditional thermal protocol
Reuse R47 declared actual propensity knot triplet p0,p1,p2; close
cycle p3=p0, piecewise-linear log odds. E/E*=-ln pi, beta E*=1.
Duration tau=gamma t, reset generator at fixed knot yields
eta=1-exp(-tau/(3m)). Three segments with m=8,32,128 steps.
Quench then exact partial Gibbs relaxation. Reverse order:
relax at same Hamiltonian then inverse quench. For work tilted
row transfer forward M_lambda=diag(exp(-lambda Delta E)) T,
reverse uses T diag(exp(+lambda Delta E)) in reversed order.
Detailed balance and Gibbs initial states yield Z_F(lambda)=
Z_R(1-lambda) for cyclic Delta F=0; JE at lambda=1. Compute exact
DP for all steps and lambda grid {0,.25,.5,.75,1}; mean W,Q from
probability transport, Delta U=W+Q, beta W=Sigma+KL(final||pi0).
Conditional dimensionless thermal model, not measured native heat.
Convergence in time discretization separately from stochastic error;
slow driving and sudden limit may both have low work for cyclic
protocol, no monotone duration claim. Markov-chain FT context:
https://pmc.ncbi.nlm.nih.gov/articles/PMC9777588/ and
https://iopscience.iop.org/article/10.1088/1367-2630/16/9/095001 .

## Spatial size gate
Rapfi pinned mix9svqfreestyle_bsmix weight header uses mask 0x3ff000
and rule bit 1 (actual read via lz4). Supported L=13..22; verify
same file loaded, side and full legal scores on one fixed geometric
four-move board each L. 9/11 disabled NNUE and fall back, excluded.
Current Rust Gomoku15 fixed N=15,N2=225; mqft.rs and Rapfi parser
and Python legal/trace contracts fixed. Need separate compatible
multi-size state backend and schema with actual board size, not
header overrides or padding heuristics. Spatial criticality blocked
on that implementation; direct evaluator support is enabling result.

R8–R10. Evaluate precision and negative predictions against actual
results; preserve original Q28 objectives and previous negative
admissions. Native independent agent absent; separate code/CAS
checks. No physical phase, hbar, native 1PI or thermal identification
by simulated conditional model. No per-plot node expansion.
