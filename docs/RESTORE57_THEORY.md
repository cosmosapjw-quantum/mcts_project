# RESTORE57: source-specific native energy and noncyclic thermodynamics
Parent RESTORE56 4e1ec234265013507a1e5cd5dfca0f32e174114f.
Original three archive hashes and Q28 six axes checked.
Exact G01/G02/G03 guide source:
omoknuni_quantum-main/docs/quantum_physics/STATISTICAL_MECHANICS_ANALYSIS_GUIDE.md
SHA256 bfac951744509e0398f58a801855acb355bf9d95754c0ccbe231baf570228db6,
Theoretical Framework and Implementation Components (energy -Q,
temperature 1/sqrt(N), Shannon entropy) and Output Structure line272.
Q19/Q20 original plot_jarzynski_equality.py and Q03/Q05/Q08
plot_thermodynamics.py re-read. Original mock Gaussian reverse,
arbitrary deltaF and random work components replaced, not accepted
as native data. Original Q08 absolute entropy-rate ansatz is not
general; use signed balance with heat convention below.
Exa Crooks 1999 https://arxiv.org/abs/cond-mat/9901352 :
equilibrium endpoints, stochastic microscopically reversible
dynamics, noncyclic work relation. SciSpace abstract Esposito and
Van den Broeck, Three detailed fluctuation theorems,
https://scispace.com/papers/three-detailed-fluctuation-theorems-nsaj6r6hqaxt
motivates separate boundary measures; no theorem transfer beyond
our explicit reset derivation. Parallel review
https://link.aps.org/doi/10.1103/PhysRevX.7.021051 and trajectory
class relation https://link.springer.com/article/10.1007/s10955-025-03422-z
context. Wolfram verifies log-sum-exp/free energy derivative.
Native subagent unavailable; serial fallback.

R3/R4. Same R53 v5 runtime and R54 scoped analyzer, same actual
Quartz/MixNet. Two new D4-distinct common central templates and
two nonterminal size-root variants per size L13/21. B16 wave4
three laws, baseline and first-wave +/- .25 sector contrast,
n8 paired seeds per arm/law/root, 288 actual jobs. R56 mark
models and coefficients exactly frozen (R55 training only);
R57 has no refit. Reuse 44 R55/R56 verified natural games and
their receipt-bound raw shards, no new game generation or
duplicate native counts. Select first two R56 opening IDs per
size and B16/64, eight total protocols; use first/middle/last
actual searched ply sector probabilities. Freeze selection
independently of work outcomes. Three durations4/16/64, 2048
forward and 2048 reverse paths each; exact independent seeds.
No large uniform stencil or board-size campaign.

R5. Native energy original E_Q=-pi.Q=-mean root return.
E_PUCT=-pi.S differs by exploration term -pi.(S-Q).
Same root and support, E=-pi.S has exact midpoint product rule
Delta E=mean(pi).Delta(-S)+mean(-S).Delta pi.
Across game moves support and root player change: no conservation
or cross-ply work interpretation. Root frame and black frame
kept distinct. All positive backup cuts, actual ply and per-game
phase means. Original T_budget=1/sqrt(k) is a chosen scale, not
thermometry. Conditional law at actual IID selection:
log L(beta)=sum[beta S_t(a_t)-logsumexp(beta S_t)].
Derivative sum chosen minus mean, second derivative negative
sum conditional score variance. Fit each game over changing
legal support without padding; verify actual conditional p at
beta=10/3. Occupancy-fit beta remains a distinct quantity.
Predictable mark control as R56, independent R57 cores; compare
fixed action-unit LR and old scaled/sector controls, no admission
from mark MSE alone.

N model: actual sector propensity p sets declared dimensionless
levels e_i=-log(p_i/p_3), e_3=0. beta E*=1 and heat q positive
into system. This energy zero is a convention; time-dependent
common offset changes W and DeltaF equally. Piecewise linear
e(t) through three knots; pi=softmax(-e), F=-logsumexp(-e).
Here endpoint DeltaF=log(p_3(final)/p_3(initial)) generally
nonzero in the chosen gauge. Generator L=tau(pi 1^T-I) with
rate tau on normalized t in [0,1]. At each refresh, draw from
pi(t) (self draws permitted); offdiagonal rates k_{i<-j}=tau pi_i.
Local detailed balance log(k_{i<-j}/k_{j<-i})=-(e_i-e_j).
Work w=integral e'_X dt between jumps, heat q=sum jumps
(e_new-e_old). State-preserving intervals give exact endpoint
work, not a time-discretized chain. Every path satisfies
Delta e_X=w+q. At observation times split exponential waiting
by memorylessness; no path evolution grid approximation.

Forward starts pi(0), reverse starts pi(1), reversed energy
schedule. Holding factors cancel in path density ratio under
time reversal. Ratio of initial factors contributes e_final-
e_initial - DeltaF; product of jump ratios contributes -q;
hence log(P_F/P_R^rev)=w-DeltaF. This proves measure Crooks
including atoms. Set lambda=1-ik, g_lambda'=L g_lambda-
lambda diag(e')g_lambda. lambda=1 solution
g_1(t)=exp[-(F(t)-F(0))] pi(t), initial pi(0).
Check exact weighted CF vs reflected reverse and independent
MC with shifted w-DeltaF. First four raw moments satisfy
a_m'=L a_m +m diag(e')a_{m-1}, a0=mu, a_m>0(0)=0.
Mean heat independently integrate qdot=mu'.e.
S=-sum mu log mu, J_ij=tau(pi_i mu_j-pi_j mu_i).
sigma_dot=1/2 sum J_ij log(pi_i mu_j/(pi_j mu_i))>=0
by (a-b)log(a/b)>=0. Derive sigma_dot=Sdot-Qdot, not
absolute Sdot. Thus DeltaU=W+Q, Sigma=DeltaS-Q.
For initial equilibrium and final nonequilibrium mu,
W-DeltaF=Sigma+D(mu||pi_final). Do not identify dissipated
work with actual system-plus-bath entropy production absent
final equilibration. Trajectory sigma=log(pi0(X0)/mu1(X1))-q;
its path-ratio reverse starts mu1, not pi1. Gauge shift check
same RNG state path, q and sigma invariant, w shifts by
endpoint offset. ODE moment/tilt/heat/Sigma integrations,
independent exact-clock MC. At fixed equilibrium q and
conjugate f=(1,-1,0,0), derivative of <f> under -h f is
Var_pi(f) (beta E*=1); do not equate generic work variance
with energy susceptibility. Original Q20 arbitrary exploration/
exploitation/quantum decomposition has no identified energy.
For independent identical repetitions of declared work,
psi(l)=log E exp(l w); tilted ODE gives psi and psi' and
parametric rate I(psi')=l psi'-psi on finite l grid.
This is the Cramer reference for independent repeats,
not native temporal large-deviation scaling from a histogram.

Q03: declared diagonal four-level cold spectrum e_c>=0
from middle-game p (common additive shift immaterial),
hot spectrum r e_c with r=1.5. beta_c E*=1, beta_h E*=.5.
Fixed relaxation duration d, u=exp(-d), pi_c and pi_h
Gibbs at respective levels/bath. Exact periodic endpoints
mu_c=(pi_c+u pi_h)/(1+u), mu_h=(pi_h+u pi_c)/(1+u).
Isochoric hot heat Qh=r e_c.(mu_h-mu_c), cold heat
Qc=-e_c.(mu_h-mu_c), work on system Win=(r-1)e_c.(mu_c-mu_h).
Win+Qh+Qc=0, Sigma=-beta_h Qh-beta_c Qc>=0.
When Qh>0 eta=-Win/Qh=1-1/r=1/3 <=1-beta_h/beta_c=.5.
Plot Hamiltonian scale vs generalized force -<e_c>, not
invented physical pressure/volume. Ideal reversible Carnot
reference: hot isotherm scale1.8->1.2, adiabat rescale by
beta_h/beta_c, cold isotherm reversed, final adiabat;
Qh=DeltaS/beta_h,Qc=-DeltaS/beta_c, eta=.5.
No claim native MCTS realizes a two-bath cycle or quantum
efficiency. Missing quantum cycle remains explicit.

R6/R7 independent Wolfram algebra, exact path reconstruction,
ODE/MC, Cramer moment, periodic endpoints and unit tests.
R8 measured scope and prospective negative control.
R9 dedicated native/conditional output mapping with actual
source identity. R10 remaining original-output obligations.
