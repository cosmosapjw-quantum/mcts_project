# RESTORE46: precision of actual path response and memory dynamics
Authority: RESTORE32 two-layer core, original ZIP/Q28/Q30, parent main
bce81c431fef9db06329d4afa214b13780e8d37e; current 13 topic nodes and
121 original outputs, no plot-driven node inflation. Same pinned actual
Quartz/MixNet v3, B16, wave4, four NNUE candidates, beta=1/.3, logit
h=-.3,0,.3, no prior tilt or uniform rollout. Existing eight D4-distinct
actual p8 roots, add 12 seeds to each of four 44 roots and 8 to each of
four 45 roots; unique seed namespaces, paired arms. No new games required.
R1/R2 prior theory and hashes, source Q28 read, exact previous archive
and registrations. R3–R7 derive and test the following three routes.

## A. Ideal conditional stochastic law and exact PRNG replay
For predictable p_t=P(I_t=1|F_{t-1}), conditional ideal uniform draw,
g_t=I_t-p_t, E[g_t|F_{t-1}]=0, E[g_t²|F_{t-1}]=p_t(1-p_t).
For G=sum g, F=sum p(1-p), cross terms vanish and E(G²-F)=0.
E exp(lambda*g_t)|F=exp(-lambda*p_t)*(1+p_t(exp(lambda)-1)).
Hence exponential normalized martingale M_n as in physics46.score_evidence;
a fixed symmetric mixture of six lambda values is nonnegative and mean1.
Use h0 only; arms share seeds, cannot be treated as independent.
Combined older/fresh results descriptive, fresh-only mixture separately
computed. Choice of lambda informed by earlier data, not by calibrated
power. Low e is non-rejection, NOT positive proof of law or PRNG independence.
Reference: Howard et al. Probab. Surveys 17 (2020), doi:10.1214/18-PS321,
exponential nonnegative supermartingales. Wolfram Bernoulli mgf and
second derivative check returned {0,0,p(1-p)}.
Replay every root StdRng uniform with locked rand0.8.5/rand_chacha0.3.1,
seed=Mix(Mix(spec_seed)^0x524f4f54), and evaluator candidate seed
Mix(Mix(replica_seed^Mix(eval_id)^state_hash)^sample_id). FP64 trace
serialization permits <=2e-16; reconstruct actual integer NNUE scores
softmax scale32 with floor1e-6 and reselect using exact Rust PRNG.
Independent Python replay checks chronological reservations, counts,
actual conditional distributions, FP32 score and actual CDF membership.
No production engine change. Existing full analysis validates board/
return provenance and moments. Seed reproducibility does not imply
ideal independence of deterministic PRNG; inferential tests conditional.

For a fixed actual augmented path, changing source from h to h+delta:
log L_delta = sum [delta I_t - log(1+p_t expm1(delta))].
The ideal downstream kernel remains common and all support same within
one law. This is same-path source Radon-Nikodym ratio, not temporal
reversal. Sigma=-log L, conditional KL compensator
D_t=log(1+p expm1(delta))-delta*p >=0, and
Sigma-sum D_t=-delta G. Thus E Sigma=sum E D_t>=0 and
E exp(-Sigma)=1 under ideal path law. Actual finite averages may fail
exact equalities. Never mix different PL/IID supports. Compare each
of six ordered source pairs with actual complete histories, full
softmax counterfactual replay and centered finite LR/direct contrast.
LOO estimator is sample covariance, target E[A D] for finite contrast;
jackknife at whole history level, chosen root equal weights. Derivative
vs finite contrast kept distinct. Do not impose physical FDT or heat.

## B. Discrete projected count dynamics
Use actual last three complete wave4 counts (a,b,c), initial short
waves excluded. For each law fit P(c|b) with Jeffreys .5, and
P(c|a,b) with 4 pseudo-counts of first-order model. PL count domain
0,1; IID/reservation 0..4. Train 4 older roots, test 4 other roots,
16 h0 histories each; no tuning on test. Previous closure used
same opening split, fresh seeds not entirely novel roots.
Declare homogeneous continuation K_(a,b),(b,c)=P(c|a,b), not claim
native time homogeneous. Stationary q defines R (conditional earlier
count given b), E lifts functions of b; R E=I, P=ER, Q=I-P.
Let T_n=R K^n E, A=R K E, B=R K Q, C=Q K E, D=Q K Q.
Set Y_n=Q K^n E. Then T_(n+1)=A T_n+B Y_n,
Y_(n+1)=C T_n+D Y_n, Y_0=0; substituting yields exact
T_(n+1)=A T_n+sum_(j=0)^(n-1) B D^(n-1-j) C T_j.
Check against direct matrix powers to horizon8. This is discrete
Mori-Zwanzig projection of fitted reference, not a native physical
generator or universal RG beta function. Heldout comparison at
actual final wave tests whether additional memory is useful.
Literature context: Woodward et al. arxiv:2301.07203 (data-driven
Mori-Zwanzig, continuous operator projection); discrete derivation
above is direct and finite. Do not transport continuous result
without its assumptions.

## Q. Operational causal-break intervention on prescribed native paths
For each h0 complete actual history, fix first and last complete
wave's binary target marks (8 draws). A declared auxiliary qubit
starts |+X>. U_t=exp(-i alpha xi_t m_t Z), alpha=.5 dimensionless.
Independent sign noise versus single retained sign xi=±1; same
one-step CPTP channels, nonthermal. After first 4 draws measure Y
and apply K_y=|+X><y|; BOTH branches have same prepared state,
the retained classical environment sign is not reset. Final 4
draws and Y measurement. Conditional on same fixed classical
word, independent gives P(y1,y2)=1/4. Static gives
P(y1,y2)=[1+y1*y2*c]/4, c=sin(2alpha*n_p)sin(2alpha*n_f),
n_p,n_f actual target counts. Marginal means0; the conditional
future outcome probability gap = |c|, MI=ln2-H_binary((1+c)/2).
Direct Kraus matrix sequential evolution plus all 16 independent
noise sign strings per half verify formula. If either count zero,
c=0: exact negative control. This is constructive non-Markov
model prediction under reset, not measured physical quantum phase
of MCTS. Conditioning on entire drive removes shared classical
drive as confound; hidden sign is deliberately modelled environment.
Pollock et al., Operational Markov Condition for Quantum Processes
PRL 2018 (SciSpace https://scispace.com/papers/operational-markov-condition-for-quantum-processes-1ubc3qs242)
motivates the causal-break test; no claim of complete process
tomography. Conditional MI not automatically quantum Darwinism/SBS.

R8–R10: assess native fluctuation/weight evidence, memory prediction
and explicit CP reference separately. Preserve original multivariate
action and conditional thermal evidence, no new adaptive 1PI or
physical phase/temperature/hbar, board-size universality or selector
admission. Exact original targets remain local subtasks. Native
independent subagent not exposed; serial and independent code/CAS checks.
