# RESTORE41 — longitudinal commissioning, REPLAN-01
R1/R2 authority: original Q28 six targets and Q30 real game data; RESTORE40
CORE_POLICY and 13-node DAG; source identity ede118256934bbe9947bceb4be60dc332eb44252.
This is a finite CPU 15x15 freestyle commissioning, not an adequately powered
comparison. The previous request deferred generation; the current request to
execute the next full loop authorizes the registered commissioning.
Use the frozen 15-job plan (five schedules x self/Rapfi black/Rapfi white),
one opening and one replicate; separate two matched-leaf self games. No new
large template or arbitrary quantum model. Root-law pilot requires new frozen
actual root list; defer until commissioning data audited.

R3–R5: Completed backup clock k, game ply t, actual rollout samples S, rollout
plies L_r, evaluator calls E, wall time and tree size distinct. All returns
in root player frame. At root per edge n,w,s2: pi=n/k, q=w/n for visited actions,
q=0 FPU only if unvisited. Var(r)=sum(s2)/k-(sum(w)/k)^2, and pooling via
additive moments, not mean Q. For leaf samples averaged within a single
backup, variance of backed-up mark differs from individual sample variance.
Cost ratios S/k, L_r/k, E/k; no equality of costs at equal k.
At quiescent cuts use actual frozen S_a=q_a+c sqrt(k)P_a/(1+n_a).
T: occupancy KL projection beta>=0; separate finite/zero/infinite/flat.
Derivative KL = E_beta[S]-E_pi[S], second derivative Var_beta(S)>=0.
A: exact gradient of frozen objective J(x)=Q dot x+c/sqrt(k) P dot log(x+1/k)
is S(x); KKT gap, active tangent curvature and sign. Not learned loop.
R/C: dyadic same-root cuts, tree/graph prefix contraction and finite budget
crossover. Board size 15 fixed; no universal critical exponents.
D: path entropy, disjoint record or plurality stability; no quantum phase
from diagonal occupancy, repeated-root MI requires repeats.
N: E=-S, delta(pi dot E)=mean(pi) dot delta E+mean(E) dot delta pi. Exact
algebra but not thermodynamic heat/work. Native baseline deterministic
selection, stochastic uniform rollouts; no physical reverse protocol.

R6/R7: Wolfram returns {0,0,0} for variance and score product-rule identity
and gradient. Actual trace checks across all completed cuts, independent
legal and terminal replay, job and evaluator identities. Publication only
after native and Python invariants. Chaslot et al. 2008 distinguishes leaf,
root and shared tree; Grill et al. ICML 2020 relates PUCT to regularized
policy optimization (not a physical action proof).
Sources: https://doi.org/10.1007/978-3-540-87608-3_6 ;
https://proceedings.mlr.press/v119/grill20a.html ;
https://scispace.com/papers/on-effective-parallelization-of-monte-carlo-tree-search-5ayz6abu6j .
R8/R9: Report per-game and per-search descriptive summaries, actual
failures, complete/truncated, achieved cost and uncertainty. No pooled-ply
independent p-values, Elo, physical temperature or selector admission.
One opening replicate does not estimate across-opening variance. Proposed
future paired blocks use N≈ceil(((z_(1-alpha/2)+z_power)*sd(delta)/delta_min)^2);
pilot sd must be estimated from independent blocks (not plies).
R10: Preserve original objectives and negative admissions. Verify backup,
then retire only exact redundant prior exports.

## User-directed correction: common MixNet, no asymmetric random-rollout match
The user rejected Rapfi-vs-uniform rollout as a meaningful comparison and
explicitly requested MixNet. All 17 uniform games and 192 uniform-root jobs
are diagnostic only; preserve bytes and chronology, exclude from scientific
engine comparison. A common actual Rapfi 250615 mix9svq (Evaluation::mixnet)
NNUE model supplies policy and WDL for BOTH sides and all schedules. Model
file, executable and config are pinned by the original Rapfi runtime lock.
Quartz implements actual tree search. This is not full Rapfi alpha-beta on
one side or direct unquantized NNUE API: use published TRACEBOARD integer
policy scores l_a, ScoreScale=32 from evaluator.h, TRACESEARCH rounded WDL
in black perspective. Define common prior/sampling kernel
p_a=(1-eps)*exp((l_a-max(l))/32)/sum_b exp((l_b-max(l))/32)+eps/|A|,
eps=1e-6. Parent logits cover all legal moves. Four independently sampled
one-step candidate actions per nonterminal evaluator call. On a terminal
child return the exact rule outcome, otherwise r=s*(WR-LR)/(WR+LR+DR),
s=+1 black or -1 white at original leaf. This is a dimensionless value
estimate, not a terminal random rollout or a calibrated physical energy.
The evaluator mean (1/4)sum r is a backup mark, its square enters S2.
The root-source tilt changes tree prior only, not this common sampling
kernel. Log probability of sampled candidate is log p_a, exact conditional
support of this declared policy. Equal sample count per evaluation is not
equal total queries because root initialization/terminal hits differ.
Serial and leaf with equal sample count/seed and deterministic backend
should agree in marks and paths; actual eight matched-root pairs did.
Tree and wave can change selection chronology, not the neural model.
Two root-law EOFs were preserved and exact remaining jobs resumed. No
uniform fallback; 48/48 final receipts. These runtime failures remain
unresolved, not a proof of production-scale robustness.
