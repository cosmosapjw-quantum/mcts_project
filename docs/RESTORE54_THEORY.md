# RESTORE54: last-wave conditional response and independent physical probes
Parent RESTORE53 5b449021025417f70c8f7cc59c0ed890c45a1359.
Original three ZIP SHA256 and mcts_study Q28 six physical objectives
rechecked; Layer A/Q and 13 nodes/13 edges/121 output owners retained.
No change of v5 engine, evaluator, weights, game rules or source basis.
R1/R2. Scope: improve actual multicomponent response precision, complete
mixed02 and mixed12, test independent geometry and separate B and h.
Ciosek and Whiteson, Expected Policy Gradients, JMLR21(52) 2020
https://jmlr.org/papers/v21/18-012.html : conditional action integration
can lower gradient variance under its hypotheses. Our last-wave
identity below is direct, not a transfer of their policy-gradient
theorem. SciSpace abstract Huang/Wu/Zhou classical fidelity threshold
https://scispace.com/papers/certifying-quantumness-with-the-classical-fidelity-threshold-ae4ff0tefg
motivates explicit Choi and classical benchmark. Crooks context
https://arxiv.org/pdf/1611.00949 and Parallel work FT review;
use earlier independently derived declared reset-process law.
Wolfram verifies Favg=(3+2k+k^2)/6 and partial transpose eigenvalues
{1/2,1/2,-k^2/2,k^2/2}. No native subagent available.

R3/R4. B16 old L13 g0v0 IID only, 32 paired seeds and 13 arms:
zero, axis0 +/- .15 and .3, four corners for each mixed02 and12.
One fresh common central 8-stone template (D4 distinct from prior
selected cores), two nonterminal peripheral variants at approximate
density .06, L13/21. New v0 B16 full 19-arm stencil, n4, IID at
L13 and all three laws at L21. Separate same v0 scale cohort B16/32/64,
n4 IID and five arms per B (zero, +/- .15, +/- .3 axis0). Variant
v1 h0 n4 IID per size. Two games from v1. All new source acts on
final four-draw block; no early/late mixed-clock claim. Fixed arms
and seed namespaces before outcomes. 848 root jobs + 2 games;
bounded precision, not 32 per every root/law. Matched local core
across L; periphery and density still not pure independent factors.

R5/R6. At pre-wave history H (all previous backups completed),
let q be conditional sector probabilities under h0. For IID frozen
completed-root law the final four sector selections are iid q_eta
conditional on H, independent of evaluator marks for root selection
until the wave completes. q_eta=softmax(log q + U eta). Test this
identity on every actual arm's last four recorded distributions and
verify exact preceding events/evals/cuts against h0 at paired seed.
X=U^T four-draw occupancy; E_eta[X|H]=U^T q_eta.
For a finite stencil with weights c_v, D=sum c_v X_v, conditional
D_RB=sum c_v U^T q_v. Marginals give E[D|H]=D_RB even when arms share
random numbers. Thus Var(D)=Var(D_RB)+E Var(D|H), componentwise or
PSD matrix. This is a theorem for iid final-wave occupancy only;
not for marks, later adaptive blocks, or PL/reservation. In PL/reserve
the first-draw frozen q stencil is a comparison approximation.
Check all same-seed prefix identities; conditional expected response
is exact given H, finite ensemble mean still uncertain.
For tangent f_s=U^T e_s, m=E f, C=Cov(f), derivative
partial_i q_s=q_s(f_si-m_i);
partial_ij q_s=q_s[(f_si-m_i)(f_sj-m_j)-C_ij].
Analytic conditional derivative vs finite stencil separates h bias
from Monte Carlo variability. Path score G_j and F_j as R53.
First/second LR finite contrast covariance estimates and direct
paired contrasts retained. 32-seed nested 8/16/32 precision by
replicate label; no resampling selection. Estimate empirical direct
and conditional variance, residual and delete-whole-seed jackknife.
Conditional exact response is a local analytic state-dependent
predictor, no fitted full path closure. Independent geometry use
of formula, not heldout trained kernel. For PL/reserve not exact.
Full adaptive 1PI open. Fixed-sum covariance and Fourier S0,
Parseval; same B separate from spatial L. Repeat exact restricted
single-source frozen lattice one-loop at actual k8 snapshots,
not adaptive vector 1PI. Retain prior negative prediction evidence.

Q. Explicit finite quantum memory model. S input state, E=|0>,
partial iSWAP U_a on span{|01>,|10>} = cos a I-i sin a X;
a=.35 n1 and b=.35 n2 from fixed sector0 actual first and final
block counts. Discard S, prepare S=|0> independent of previous
input, retain E. Apply U_b then trace E and phase correction Z on S.
Amplitude transfer k=sin a sin b (0<=k<=1 in tested range).
Effective channel amplitude damping with survival eta=k^2:
rho00'=rho00+(1-eta)rho11, rho11'=eta rho11,
rho01'=k rho01. Z correction removes (-i)^2. Retaining E coherently
vs completely dephasing E in Z yields same output diagonal, latter
zero coherence. Reset E yields |0>. Reconstruct normalized Choi from
six specified input +/- eigenstates X/Y/Z and independently matrix
units. Choi PSD by actual CPTP composition, TP Tr_out J=I/2.
For coherent J partial transpose eigenvalues {1/2,1/2,-k^2/2,k^2/2},
negativity k^2/2; dephased separable. Average six-state fidelity
F=(2 Fe+1)/3=(3+2k+k^2)/6, dephased (3+k^2)/6, reset1/2.
Uniform six-state ensemble is a qubit 2-design (direct Pauli
second moments delta_ij/3), hence any entanglement-breaking channel
has Favg<=2/3: separable normalized Choi has overlap Fe<=1/2 with
maximally entangled state. Coherent F>2/3 iff k>sqrt(2)-1.
This is exact declared reference tomography, no finite-shot hardware,
no native physical phase. Operationally distinguishes this coherent
channel from all EB channels under stated input ensemble; classical
simulation of matrices not excluded. Actual data specify drive only.

N. Declared positive 4-state Gibbs protocol pi(s) from actual
propensity knots, log-weight cyclic interpolation and reset generator
L=tau(pi 1^T-I), beta E*=1. R52 Crooks relation for probability
measures incl atoms P_F(dw)=exp(w) P_R(d(-w)). For bounded f,
E_F[exp(-w) f(w)]=E_R[f(-w)]. Use complex smooth f=e^{ik w},
k=0,1,4,16,32; k0 JE. Independent exact-clock Poisson refresh
paths per direction, same energy endpoint work, no grid approximation.
Compare moment and characteristic function ODE from R53, exact
continuous tilted-generator Crooks symmetry from R52 and independent
MC weighted test errors. Do not infer continuous density or exact
tail ratio from noisy histogram. Zero-work atom vs no-transition
lower bound distinguished. Not native measured heat or physical beta.
R7-R10 measured outcomes, grouped figures and next contract after C.

## R54 exact conditional coarse action and CRN covariance
Conditional on the completed pre-wave history H, the IID four-draw
sector count N has full multinomial law
P(N=n|H)=4! product_s q_s^{n_s}/n_s!, sum_s n_s=4.
There are binom(7,3)=35 support points. The negative log mass is
a normalized conditional action; its source score is U^T(n-4q).
Summing exact probabilities gives Fisher 4 U^T(diag(q)-q q^T)U.
Merge sectors (0,1) and (2,3) with mass b=q0+q1. Conditional
score projection on merged count has Fisher
4 b(1-b)(fbar0-fbar1)(fbar0-fbar1)^T, with fbar conditional
means of sector tangent features. Information decreases in PSD
order by the law of total covariance. This is a precise one-wave
coarse map, not a Wilsonian fixed point of the adaptive process.
For N independent frozen draws and interior x=n/N, Stirling gives
-log P(n)=N KL(x||q)+(3/2)log(2 pi N)+(1/2)sum_s log x_s
+sum_s r(n_s)-r(N). Robbins 1/(12n+1)<r(n)<1/(12n)
bounds the remainder; the next correction is (sum 1/n_s-1/N)/12.
The exact conditional width-four law is native; larger N draws
are a declared independent frozen repetition reference, not
actual multi-wave adaptive histories. No zero-count Stirling use.

For paired arms, conditional on H and ideal fresh uniform u,
actual native action-level CDFs partition [0,1]. The finite stencil
Y(u)=sum_v c_v U_{sector(a_v(u)),i} is constant on the union
of their CDF intervals. Integrate its first/second moments exactly;
average of four independent uniforms has conditional covariance
Cov(Y|H)/4. For n independent pre-wave histories, conditional
variance of the mean equals sum Cov(Y|H)/(4 n^2) (using single-
draw covariance). Validate same recorded uniform across all arms
and reconstruct actual observations. Six specified scalar outputs,
32 prefixes, 16384 conditional resamples (2097152 fresh uniform
numbers); max standardized residual calibration, not native
searches and not proof of PRNG independence or exact universal
response. Individual deviations retained, no post-hoc tuning.

Phase extension: retain the environment and insert diag(1,i).
With fixed final phase correction, the coherent transfer has
rho01'=-i k rho01. For input +X, fixed Y measurement gives k,
dephased control 0. Construct complex Choi directly, TP/PSD
and phase readout; preserve the off-diagonal imaginary entries.
For thermal Crooks, complex tilt solves g'=(L+(-1+ik)diag E')g.
Cyclic exact identity sum g_F(1)=phi_R(-k), including atomic mass.
Compare independent exact-clock forward/reverse paths with weighted
test functions, and ODE real work/moments with the prior reference.
No reconstruction of noisy pointwise density ratios.
