# RESTORE47: scale pilot, empirical joint effective action and time-dependent physical reference
Authority RESTORE32 A/Q, original Q28/Q30, original ZIP identities, parent
023d76ae80eb9ca6d299a61907d69945b77422eb. Existing 13 nodes/13 edges,
121 original output owners unchanged. Same actual Quartz/MixNet v3,
15x15 freestyle, four NNUE candidates, no legacy rollout. This is
bounded science rather than the 32-repetition B16 expansion. Four
preselected old p8 scale anchors (first two sorted roots in each 44/45
cohort), B32 h=0,+/-.1,+/-.2 and B64 h=0,+/-.2, two seeds per arm.
Four fresh natural-terminal self games B16, four new D4-disjoint
p8 heldout roots B64 h0 with four seeds per law. New game openings
geometric sampling with NumPy seed47001 and D4 rejection, not outcomes.
First hand-proposed list failed D4 gate before generation; no data
discarded. Budgets carry different seed labels, so compare at common
root, not paired seed across B. Within a B, h arms share seed.
Original negative selector/closure results preserved.

## R3–R6 actual two-observable source functional
Define X=(N_target/B, mean root-frame mark), dimensionless, and distinguish
J coupled to final output X from actual root logit field h.
K(J,h)=log E_h exp(J·X). Full support within same law, fixed downstream
conditional law. Derivatives at J=h=0:
K_J=mu, K_JJ=C, K_Jh=Cov(X,G), score G=sum(I-p).
For Bernoulli tilt score derivative dG/dh=-F, F=sum p(1-p) on
fixed history, hence d² E_h X / dh² = E[X(G²-F)]. Center both G and
G²-F in independent seed samples; finite LR contrast and derivatives
are different. G²-F has zero mean under ideal law, not exact in a
finite sample. Direct second difference is very noisy at small h;
do not interpret sign/magnitude as critical curvature.
K finite empirical representation: normalized source weights on fixed
training support, not actual adaptive process resampling. Legendre
Gamma(X,h)=sup_J (J·X-K(J,h)), at interior mean and SPD covariance.
Implicit derivative K_JJ dJ/dX=I and K_JJ dJ/dh+K_Jh=0 yields
Gamma_XX=C^-1 and partial_h grad_X Gamma=-C^-1 Cov(X,G).
Finite rank gate eig>1e-10, no inverse outside full rank. Validate
via normalized exact source likelihood reweight of fitted training
measure and independent finite-difference of J stationarity. 24
fixed-root/law B16 training/test splits (8+8 by sorted seed), four
predeclared final-observable tilt vectors (2,1),(-2,-1),(0,2),(2,0).
Compare held-out empirical tilted mean with train exact exponential
tilt, quadratic mean mu+C J and no shift mu. This is a finite sample
out-of-sample check of empirical CGF, not a native source intervention
or loop correction. Exponential-family moment/Hessian background:
https://www.stat.cmu.edu/~siva/teaching/705/lec13.pdf .
Wolfram Bernoulli log partition second derivative p(1-p) checked.
Earlier exact lattice determinant work retained, native 1PI unresolved.

## R4–R7 time-dependent projection and prediction
Only complete four-draw waves after initial short barriers; real
completed-backup clock k. X_t=(wave target count/4, mean root mark).
Fit ridge .05, fixed four alternatives state-only, clock (k/B,log2
B/16,interaction with X_t), lag X_(t-1), clock+lag+interaction.
Training all eight old h0 B16 roots (128 histories/law) plus 16
scale h0 histories/law B32/64; equal root weights. Four genuinely
new D4 roots B64 h0 (16 histories/law) held out. Predict next X,
report count and mark MSE separately; clipping physically admissible
ranges. No future features, no hyperparameter retuning on test.
Separate finite categorical model q=(indicator count>0,indicator
mark>0), four states, empirical pair transition with two clock bins
k/B<.5 and >=.5. Dirichlet .5 and backoff 4, counts pooled over
training transitions (not equal-root fit), evaluate equal-root NLL.
Use declared early4/late4 continuation; not a native homogeneous
generator. Column L_t=K_t^T, E lifts distribution on latest
category with uniform previous-category, R marginalizes, RE=I.
Q=I-ER, A_t=R L_t E, B_t=R L_t Q, C_t=Q L_t E, D_t=Q L_t Q.
T_n=R L_(n-1)...L_0 E, Y_n=Q L_(n-1)...L_0 E; Y_0=0.
T_(n+1)=A_n T_n+B_n Y_n, Y_(n+1)=C_n T_n+D_n Y_n.
Eliminate Y to get exact chronological kernel
T_(n+1)=A_n T_n+sum_(j<n) B_n D_(n-1)...D_(j+1) C_j T_j.
Empty product identity; validate against direct products. Fitted
operators are conditional references, not an identified RG flow.
Literature context nonstationary history-enriched projection:
https://arxiv.org/html/2601.07101 (search excerpt; exact finite
derivation above independent of its continuous hypotheses).

## Q: conditional process tomography and phase identifiability
48 fresh native h0 drives: first/last complete wave counts n1,n2.
Declared auxiliary qubit, Z phase coupling alpha=.5 (dimensionless),
Rx(pi/2) between blocks. Four channels:
independent sign each hit, persistent sign across both blocks,
independent persistent sign in each block (reset_static), and
dephased input/output around persistent channel. Third is essential
matched single-block marginal control; isolates inter-block hidden
memory from different within-block noise. Same fixed actual drive.
Normalized Choi J=sum_(ij) T_ij sigma_j^T tensor sigma_i/4, input first;
T_ij=Tr[sigma_i Phi(sigma_j)]/2. Reconstruct T from six +/- Pauli
preparations and three exact expectation readouts, independently
compare direct matrix evolution and explicit independent sign sum.
Choi PSD and Tr_output J=I/2; compare trace distances and phase
SLD QFI for initial (|0>+exp(i theta)|1>)/sqrt2 at theta=0.
dephased model QFI=0. Exact conditional computational tomography,
not finite-shot lab tomography or observation of native quantum phase.
One block with no hit gives zero matched-block memory contrast.
Earlier causal-break and Gram/QD data preserved. Choi-state
non-Markovianity context https://scispace.com/papers/assessing-non-markovian-dynamics-through-moments-of-the-choi-oszmuc9pb3 ;
we do not infer CP divisibility from a single channel distance.

## N: conditional non-equilibrium thermal cycle
For first/middle/last complete wave choose p as mean recorded
target propensity, fixed external schedule. Declare Gibbs weights
pi_j=(1-p_j,p_j), E_j/E*=-ln pi_j, beta E*=1. Close cycle pi_3=pi_0.
Three forward quenches followed by reversible partial reset
T_j=(1-eta)I+eta 1 pi_(j+1)^T, eta=.25,.5,1. Reverse order:
reset at same Hamiltonian then inverse quench, initial reverse
Gibbs pi_0. All support positive. Path W=quench energy changes
and Q=reset energy changes into system; Delta U=W+Q.
Detailed balance gives log P_F/P_R(reversed)=beta W (Delta F=0).
Thus <exp(-beta W)>=1, and beta<W>=D(P_F||P_R).
With final nonequilibrium distribution p_f, exact entropy balance
beta<W>=Delta S/k_B-beta<Q>+D(p_f||pi_0).
Finite 2^4 path sums crosschecked all identities; final lag vanishes
for eta=1. Recorded propensities supply declared schedule only,
not native measured energy/heat or physical temperature.
Native same-path likelihood-source law stays separate.

R7–R10 actual finite calculations and negative findings below;
native 1PI, physical beta/hbar, continuum/board criticality and
SBS still open. No selector admission, no new plot nodes. Native
independent subagent unavailable; separate RNG helper, matrix and
finite difference verification. No large repetition doubling.
