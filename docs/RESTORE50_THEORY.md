# RESTORE50: density-controlled finite spatial statistics and time-local source
R1: parent 97c1e58119a3404b0534fb090ba4ed4c2b4d1309; exact original
three archives, Q28 six axes, RESTORE32 Layer A/Q, 13 topic nodes
and 121 original output owners preserved. Same five verified L-family
executables and pinned MixNet from 49; no change of search algorithm.
R2: Hu, Blote, Deng, Percolation in canonical ensemble, arxiv:1210.3463,
derives constraint-specific finite-size corrections for percolation.
Deng/Blote canonical scaling (SciSpace paper 4ayljkykw3) likewise
motivates separating fixed density and fixed total; their critical
equilibrium hypotheses are NOT assumed here. Regularized covariance
context https://pmc.ncbi.nlm.nih.gov/articles/PMC3667751/ (small n,
large p inverse instability); we project before inversion, no claim
of optimal shrinkage. Wolfram confirms Bernoulli log partition
derivatives p,p(1-p), and 3D tangent Gram determinant 4.
R3/R4: two new common central 8-stone legal templates, independently
drawn from fixed RNG before engine results. Add alternating peripheral
stones outside central 7x7 to target rho=.06, nearest even total
n_L=2 floor(.03 L^2+.5), error <=1/L²; no terminal roots, no source
occupation, D4 collision rejection only. Compare fixed-count and
approximately fixed-density templates, same core and source. Peripheral
patterns differ across L, hence geometry/density not uniquely separated.
Selected templates, not sample of opening population. One actual
self-game from one dense template per L (no strength ranking).
B16 beta=10/3, all laws, n=4 per arm; L15/L21 first dense template
n=8, early/middle/late four-draw source pulses +/- .2 plus baseline.
Selected dense template at L13/17/21, B16 low beta=1, B32/64 beta=1
and 10/3, n=4, h0 only. No constant source contrasts at new anchors.
Seed namespaces separate B, beta, L; pulse arms share seed only within
root/law/B/beta. Fixed density is approximate and conditional.

R5 spatial projection: let X be normalized mass in four spatial
quadrants (open-board diagnostic partition), 1^T X=1. Pseudoinverse
of full sample C has no information outside its rank<=n-1. Choose
explicit orthonormal tangent U (Helmert basis) and y=U^T X. C_y=U^T C U.
Fit on one independent half of histories, test on other; eigenvalues
and rank recorded. Use ridge C_lambda=C_y+lambda I, lambda fixed
at .01, .001, .0001 (dimensionless mass²), not tuned on test. Local
quadratic Gamma curvature H_lambda=C_lambda^-1; it is a regularized
surrogate, not true full 1PI. Validate positive eigenvalues, determinant
identity and heldout Gaussian log score against isotropic covariance
(trace(C_y)/3+lambda)I. Score only a predictive density on R^3,
not a calibrated probability on count lattice. A 3D covariance
from two or four training samples cannot certify full field rank.
For fixed local geometry source a0 and retarded blocks use
L_i(h)=prod_{t<=i} exp(h_t I_t)/(1+p_t expm1 h_t).
Causal prefix and exact future-pulse null follow from martingale
conditional expectation and identical paired RNG before intervention.
Time-local first/second score at h0 are G_j=sum(I-p) in block j
and G_j²-F_j; finite contrast uses prefix LR not derivative.
Compare direct and LR with whole-pair jackknife, no equivalence
claim from nonrejection, use 4-completed-backup block clocks.
For spatial structure factor S(k), S(0)=0 from total normalization,
sum S/L²=tr C; no S(0)/S_min correlation length. Finite time B
and beta distinct from L and rho, no automatic critical collapse.
R6/R7: verify actual clocks, state and runtime receipts, independent
RNG replay; compare matched density contrasts, B/beta pilot, temporal
source predictions, projected quadratic action, conditional
quantum/thermal sector references. R8-R10 after measured outcomes.

## R5/R6: predictable drift, fluctuation, and coarse projection
For spatial sectors a=0..3 let X_t=e_(sector of chosen action),
P_t=conditional sector probability vector given the entire past,
D=B^-1 sum P_t and M=B^-1 sum (X_t-P_t). Then averaged occupancy
A=D+M. The exact sample identity is
C_A=C_D+C_M+Cov(D,M)+Cov(M,D).
Under ideal conditional random draws, the increments X_t-P_t have
zero conditional expectation and are orthogonal across distinct times,
so Cov(M)=E[B^-2 sum_t (diag(P_t)-P_t P_t^T)]. The sample analogue
need not equal the bracket; no independence of D and M is assumed.
All these matrices annihilate 1. Compute with actual full conditional
probabilities from raw native records, not a post-hoc fitted law.
The covariance decomposition is exact; the martingale identity is
an ideal-law ensemble statement. Do not substitute equilibrium FDT.
For coarse memory, use y=(U^T A,target fraction,mean return mark);
fit ridge .05 to next block using current block, optional clock and
optional prior block. Include L and rho in all alternatives.
Train on template0, evaluate template1, equal weight root variants.
Use Euclidean simplex projection for predicted four-sector mass,
clip target fraction to [0,1] and mark to [-1,1]. Ten size/density
variants per template are NOT ten independent geometries. No
hyperparameter tuning on test. Model-specific predictive errors
distinguished from exact Mori-Zwanzig identity retained from R47.

## Q: coherent record intervention, not classical data identified as quantum
For empirical sector weights p_a, define normalized |e_a> =
sqrt(1-g)|0>+sqrt(g)|a+1>, g=.35 and two declared separated fragments.
The explicit pure state is sum_a sqrt(p_a)|a>|e_a>|e_a>.
Gram matrix G_ab=<e_b|e_a> has off-diagonal 1-g. After a rank-one
heralding measurement on the first fragment with
|f_theta>=(|0>+sum_a exp(i theta a)|a+1>)/sqrt(5),
unnormalized reduced system is R_ab=b_a b_b* G_ab,
b_a=sqrt(p_a)<f_theta|e_a>. Herald probability q=Tr R,
conditional state rho=R/q. Independently construct system+remaining
fragment amplitude and multiply by its adjoint. Completeness of a
full orthonormal measurement implies sum_outcomes R=rho_unconditioned,
so no remote change if outcomes are ignored. Compare coherent
projector onto uniform sector superposition with the dephased control.
Phase QFI is for a SEPARATE system rotation generated by diag(0,1,2,3)
on the heralded state; it is not derivative with respect to the
heralding angle theta. Native data specify p only; no physical
phase identification or observed environmental quantum record.

## N and one-loop: mapped support and thermal cycle
Use unperturbed L15 fixed-core prior to select source plus four
highest-prior legal actions within common central 7x7, mapped to both
density arms and all sizes. Exact restricted width4 law on shared
actual h0 IID k8 snapshot, with parent full counts/priors retained.
DP exact coarse three-sector counts for M8/16/32 independent waves;
lattice covolume correction as derived in R45/49, Fisher pushforward
retained. Restricted independent frozen law is not adaptive native 1PI.
Thermal reference as in R49: aggregate actual conditional sector
probabilities in first/third/fourth blocks to fixed external knots,
Gibbs E/E*=-log pi, beta E*=1, close cycle, interpolate log weights
and exact partial reset over 32 intervals per segment. Duration
tau=1,4,16,64. Compute tilted forward/reverse transfer, JE/Crooks
symmetry and first law/entropy balance. Actual likelihood ratios
remain distinct from physical time reversal and heat.

## R8-R10: interpretation and closure
Actual 732 histories and five natural-terminal games; n4 for ordinary
root/law/B/beta, n8 on two pulse roots. Exact pre-intervention
prefixes 192/192 (1536 earlier draws). Sparse source at new L21
dense0 leads to no baseline hits in 24 histories, total expected
hits ~1.06, hence no well-resolved target response, not proof of
zero response. Next source selection must be predeclared by
unperturbed policy/propensity information, not post-treatment.
Full projected covariance rank3 in 102/105 groups but training
half rank1 in 99/105; ridge predictability tested, not true Hessian.
All tested ridge settings have worse mean heldout Gaussian NLL
than isotropic; empirical and quadratic output tilt fail to beat
zero-shift mean error. Keep negative results, do not tune test.
Mark lag improves heldout template but sector prediction worsens;
no universal Markov closure. Finite B covariance roughly 1/B
over tested range, no critical exponent, no spatial universality
or native 1PI claim. Both density and peripheral arrangement change.
All 13 nodes/13 edges and original output ownership preserved.
Independent native subagent unavailable; separate exact references,
CAS and independent RNG/trace reduction used. Next stage must
improve actual discriminating data instead of adding plot nodes.
