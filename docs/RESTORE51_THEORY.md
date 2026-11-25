# RESTORE51: informative local sources, geometric variation and controlled dynamics
R1: exact RESTORE50 parent 3ac8d282abdb15231e7c7cc8050c4bf48f0ab824.
Original three archive identities, Q28 six physics axes, RESTORE32
Layer A/Q, 13 existing topic nodes and 121 original output owners.
Reuse exact RESTORE49 L13/17/21 size-specialized binaries and same
actual Rapfi MixNet. Prior 15/19 and all other data preserved; no
production algorithm, model or source changes. Three new seeded
central 8-stone templates, each two nonterminal D4-distinct
peripheral variants, approximate .06 density (10/18/26 stones).
Same core but peripheral realizations differ across L; 18 root
variants, only 3 distinct core geometries. No population sampling.
R2: Jie and Abbeel, On a Connection between Importance Sampling and
the Likelihood Ratio Policy Gradient, NeurIPS 2010,
https://papers.neurips.cc/paper_files/paper/2010/file/35cf8659cfcb13224cbd47863a34fc58-Paper.pdf,
Sec3 connects trajectory likelihood ratios with gradient. Liu,
Bacon and Brunskill, ICML2020 https://proceedings.mlr.press/v119/liu20a.html
shows finite-horizon per-decision IS need not reduce overall variance;
do not transfer their MDP conditions to native estimates without
checking. Binary local design uses p(1-p), not a claim of optimal
multivariate D-design; Parallel logistic design context
https://www.math.tau.ac.il/~dms/GLM_Design/Technical%20Report%20RP-SOR-0501-%20Approximate%20Local%20D-optimal%20Design%20for%20Binary%20Response.pdf.
SciSpace spatiotemporal quantum noise context
https://scispace.com/papers/quantifying-and-bounding-spatiotemporal-correlations-in-kkhxold5xbzp
(abstract-level, no theorem transfer). Wolfram verifies Bernoulli
second derivative p(1-p), first and second derivatives 1-2p,-2,
and L''(0)=(I-p)^2-p(1-p). Direct derivations below independent.

## R3/R4 prospective source selection
For each actual fixed root, 2 h0 seeds per law (IID/PL/reserve),
B16, beta_s=10/3, actual full conditional action probabilities.
F_a = average over six pilot paths of sum_t p_t(a)(1-p_t(a));
H_a similarly sum_t p_t(a). Pick legal high=max F, medium nearest
.5 max, rare nearest .02 max, deterministic action tie break,
all distinct. The named strata are operational labels, not fixed
absolute Fisher ranges, and may miss target ratios due to finite
support. No treatment results consulted; record all pilot receipts,
selected action/F/H and exact plan before confirmation. Independent
8 h0 seeds per root/law for high source. On L13/L21 g0v0 run all
three strata B16 eight seed groups per law, pulses +/- .2 in
blocks 1,3,4 (four completed draws/block) plus h0; B32 high
four seeds, blocks1,5,8. Distinct seed namespaces across
size/root/campaign/law/B, +/-/0 common within a pulse design.
1716 fixed-root histories planned: pilot108, confirmation432,
pulse B16 1008, pulse B32 168. Three actual self games from
g0v0 dense template, one/size. No strength ranking. Budget and
beta_s fixed within each spatial cohort; no additional B/beta grid.

## R5/R6 finite causal and information bounds
For field h_t, P_h/P_0=product exp(h_t I_t)/(1+p_t expm1(h_t))
on fixed path, unchanged downstream conditional kernel and
support. Let g_t=I_t-p_t; under ideal random law E[g_t|past]=0,
predictable variance p_t(1-p_t). On block j, G_j=sum g_t, F_j
=sum p_t(1-p_t). Conditional mean and orthogonality yield
E[G_j]=0, E[G_j²]=E[F_j]. For bounded Y_i measurable at completed
block i, R_ij=Cov(Y_i,G_j)=0 if j>i. At h=0, first derivative
E[Y_i G_j], second E[Y_i(G_j²-F_j)], and both vanish before j.
Finite +/- .2 prefix likelihood contrast is a distinct quantity.
For j pulse, prefix likelihood equals one before j and full
pulse likelihood thereafter. Whole-seed paired jackknife,
verify exact early paths and all linked evaluations/cuts for
middle/late versus h0. Full sector tangent, target and return
mark responses and first/second contrasts retained.
Let Y=(U^T sector mass,target fraction,mark), covariance C and
R=Cov(Y,G). For any real v, Cauchy-Schwarz:
(v^T R)^2 <= Var(G) v^T C v.
Hence Var(G) C-RR^T PSD; ideal population Var(G)=E F.
For empirical centered histories use actual sample variance V,
not the sample mean bracket as exact denominator. If V=0, R=0;
component normalized response ratio undefined for zero denominator,
stored zero as convention and not evidence of absence. Check
matrix PSD in all 54 groups. This is a path-information bound,
not a thermodynamic FDT. Source hit counts and Fisher pilot
calibration tested only on independent seeds. E[N] bounds
P(N>=1) <= E[N], so a rare source can have no visits at small n;
do not infer zero susceptibility from observed zeros.

## R5 spatial action and RG
Use full connected native occupancy covariance, exact fixed-total
zero mode and Parseval as R49. Three matched core geometries and
two peripheral variants per size, compare absolute difference
within L/core/law to across-L mean changes (no causal isolation of
density or proof of scaling). At sector level use actual
A=D+M, D=mean P_t, M=mean(X_t-P_t), exact sample
C_A=C_D+C_M+Cov(D,M)+Cov(M,D). Ideal-law bracket identity
in expectation only. Project action covariance and conditional
Fisher to sector logits; fixed sum removes one direction.
Reuse R50 explicit Helmert basis and fixed ridge {.01,.001,.0001},
4+4 independent seed split on confirm, plus 8+8 at two selected
high roots using distinct high-pulse baseline seeds. Compare
heldout Gaussian log score with isotropic regularized covariance,
and empirical output tilt vs quadratic/zero shift. Gaussian
on R^3 is conditional surrogate, not exact lattice law or
native 1PI. Fit on train only, no ridge tuning on test.
Mori-Zwanzig nonstationary exact identity retained. Prospective
ridge .05 memory model on sector tangent, target and mark,
leave-one-central-template-out (12 training root variants,
6 test variants per fold; not independent across same template).
Simplex projection and scalar range clipping; compare state/clock/
memory/clock+memory. No selector admission.
Exact restricted-wave action: pick 5 central legal actions by
unperturbed L17 prior for each geometry/variant, map offsets
across L. This is reference target distinct from chosen native
pulse source. Same actual IID h0 k8 snapshot for each law,
preserve full parent visits/priors, restrict only draw support.
54 exact width4 word measures; coarse three-sector lattice DP
M8/16/32, determinant and cell-volume correction as R45.
Native 1PI and continuum scaling not inferred.

## Q: explicit retained quantum environment
For each of 432 confirmed actual histories, sector of selected
source, counts n1,n2 in first and final block define a=.35*n1,
b=.35*n2 dimensionless angles. System S initially |+X>, qubit E
initially |0>. First unitary exp(-i a Z_S tensor X_E) yields
cos(a)|+X>|0>-i sin(a)|-X>|1>. Bipartite negativity
|sin(2a)|/2 (Schmidt coefficients |cos a|,|sin a|).
Measure Y on S, outcome y=+/- with probability1/2. Conditional
environment pure state cos(a)|0>+y sin(a)|1>. Reprepare S=+X
independent of outcome, retain E. Apply phase rotation
Rz(theta)=exp(-i theta Z_E/2), theta=0,pi/2,pi. Second interaction
exp(-i b Z_S tensor X_E), then Y measurement y'. Direct
conditional E X expectation y sin(2a) cos theta, so
P(y,y')=[1+y y' sin(2a)sin(2b)cos(theta)]/4.
Dephase E in Z or reset E=|0> yields uniform independent outcomes,
same S preparation. Full 4x4 unitary density matrices and
unnormalized conditional traces verify analytical result.
Calculate joint MI, negativity and S_vN. Quantum memory is
explicit in DECLARED model; classical hidden-variable models
not ruled out by these measured observables, and native classical
traces provide only drive counts. No physical native phase/bath.

## N: continuous-time thermal reference and friction
For each confirmed root/law, actual conditional sector propensities
averaged at first/third/fourth complete blocks give positive Gibbs
knots pi(s), log-weight linear on 3 normalized-time segments,
pi(1)=pi(0), E_i/E*=-log pi_i, beta E*=1. Declared continuous
reset dynamics in normalized time s: d mu/ds=tau(pi-mu),
tau=gamma t_total, mu(0)=pi(0). Work W/E*=-integral
mu dot(log pi) ds; equilibrium term -pi dot(log pi)=0. For
piecewise smooth pi, away from initial/segment layers, mu=pi-
tau^-1 pi' + O(tau^-2). Boundary layers of width O(tau^-1)
contribute O(tau^-2) to cyclic work if pi continuous, derivative
piecewise bounded, pi strictly positive. Therefore
tau W/E* -> integral sum_i (pi_i')²/pi_i ds = friction metric
integral. In local segment u, derivative s=3 derivative u;
friction=3 sum_segments integral Var_pi(log pi_end-log pi_start) du.
Derive by substitution, not extrapolation from discrete quenches.
Integrate probability and W with DOP853, independent tighter Radau
on 3 roots x3 durations. Compute Q=delta U-W, entropy production
Sigma=delta S/k_B-beta Q, lag KL; verify beta W=Sigma+lag.
Use tau16/64/256, compare scaled work with computed friction.
Native source-path likelihood and thermal dynamics distinct;
continuous-time work MGF not computed in this iteration (earlier
exact discrete Crooks and JE preserved). No physical native heat
or temperature or equilibrium assumption on MCTS.

R7-R10: actual bounded experiments and exact/conditional calculations;
final measured outcomes and next contract in RESULTS_KO and
NEXT_DYNAMICS. Preserve negative evidence; native independent
subagent unavailable, serial/independent RNG/CAS/ODE paths.
