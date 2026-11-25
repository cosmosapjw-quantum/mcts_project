# RESTORE39 scientific contract (R1–R10)

Authority: MQFT-PLOT-SSOT-20261124 and original mcts_study Q1–3, Q22, Q28, Q30.
Exact source ZIP identities in PROVENANCE.json. Original function spans and
guide outputs in ORIGINAL_TARGET_CATALOG.csv. RESTORE38/37 are prior evidence,
not new observations from this backend.

## Conventions and estimands

A game starts at a declared legal opening, 15x15 freestyle Gomoku, black moves
first. A search starts from a nonterminal state; tree is reset for each game
ply. Edge marks are the actual signed value backed up, transported to the root
frame by negamax parity, with no extra discount. k is completed backups, not
game ply, evaluator samples, board size L, allocated nodes or wall time.
For each edge, N = count, W = sum r, S2 = sum r², Q = W/N if N>0,
pi = N/k. Unvisited Q=0 is the baseline FPU convention, not an empirical value.
Population mark variance S2/k - (W/k)^2 decomposes into within-action and
between-action parts. For independent replicas add N, W, S2, not variances
or Q. Pooled M2 = M2_1 + M2_2 + n1*n2/(n1+n2)*(q1-q2)^2.
The root aggregate is an occupancy summary, not a shared PUCT field.
Each leaf sample is terminal +/-1/0 or an explicitly marked cutoff returning 0.
For leaf mode the backed-up mark is an average of multiple samples; S2 is
the square of that actual mark, not the sum of individual rollout squares.

## Original six axes (definitions and limits)

T: Score S_a = Q_a + c sqrt(k) P_a/(1+N_a), k>0 and quiescent. Minimize
KL(pi || exp(beta S)/Z), beta>=0. Convex derivative E_beta[S]-E_pi[S];
second derivative Var_beta[S]>=0. Finite/zero/infinite/flat solutions kept
separate. H(pi), inverse participation, occupancy ceiling and residuals.
Compare sqrt(k_local)/c separately. Not a native Gibbs law.

A: Freeze Q and P at a cut, lambda=c/sqrt(k), x in simplex:
J(x)=sum Q_a x_a + lambda sum P_a log(x_a+1/k).
KKT x*=max(0,lambda P_a/(mu-Q_a)-1/k), sum x*=1.
Strict concavity on full positive-prior support, hence unique optimum. At
fixed active set D=diag(lambda P/(x*+1/k)^2), orthonormal tangent T,
det(T^T D T)=prod D_i * sum(1/D_i)/m. Constrained source response to a score
field g is dx=D^-1 (g-1*(1^T D^-1 g)/(1^T D^-1 1)).
This fixed-field response excludes adaptive Q changes; do not infer an
effective hbar or loop coefficient from a curvature determinant. Actual
prior tilt P_h ∝ P exp(h 1_{a=a0}), paired independent seed repetitions,
finite difference versus frozen KKT counterpart. Active set may change,
so finite exact KKT is the reference, not extrapolated linear response.
Compare bare/corrected utility only when independent held-out cost/strength
data exist. Original one-loop action and measure remain open.

R: Dyadic k cuts at same root. Separately define connected within-replica
prefix graph and weighted contiguous depth blocks. Q covariance centered
on visited nonroot prefix nodes N>=2; actual pair count and subsampling
recorded. Weighted block variance identity exact, not autonomous RG flow.

D: Root decisions, empirical path entropy, H(path|first action), plurality
stability, duplicate path/root pairs per observed phase. Independent root
replicates provide paired final decision/early disjoint block labels. Root
parallel mode retains tuples by replica, never a fictitious global order.
Classical cumulative I(D;F), disjoint I, increments, permutation null,
H(D)=0 undefined normalization. Overlapping information or final-label
leakage does not establish redundancy/SBS. Declared positive complex states
can instead supply von Neumann entropy, reduced states, QMI and Holevo
separately. No phase/coherence inferred from occupancy.

N: E=-S in declared score units, delta <E>=mean(pi) dot delta E +
mean(E) dot delta pi, exact algebra. Work/heat terminology not physical
without control protocol and Hamiltonian. Optional explicitly supplied
complex memory/bath model with equal Hamiltonians and energy-preserving
partial swap; Q into SM; Sigma=delta S_SM-beta Q = I(SM:B)+D(rho_B||tau_B).
Native deterministic PUCT has no recorded positive stochastic path law;
physical JE/Crooks/FDT need independently specified forward/reverse,
energy and free-energy change. Earlier RESTORE37 verified separate scope;
no new native fluctuation theorem.

C: m=pi_(1)-pi_(2), effective action count, mark fluctuation, graph covariance,
matched prior tilt response. Sweep c, k and parallel mode. L fixed 15 for
this backend; no finite-board-size collapse/critical exponent claimed.

## Schedules and inference

Root: independent trees, global backup budget partitioned deterministically;
merge additive statistics at cuts only. Tree: shared Quartz tree, concurrent
worker selection/evaluation/backup, fixed virtual loss. Leaf: one selected
leaf, independent rollout samples in parallel, mean one backup. Wave:
sequential reservation with VL, concurrent leaf evaluation, ordered commit
at barrier, not GPU batch. Serial: reference. Native selection and finish
sequence counters are observational, not linearized internal atomics.
Equal backups vs equal samples vs equal wall cost are different estimands.
Condition on opening/game and frozen matched position; plies and nested
cuts not independent. Separate empirical descriptive ranges from confidence
intervals. Source +/- uses same seed labels but async order may differ.

Literature anchors: Chaslot, Winands & van den Herik, Parallel Monte-Carlo
Tree Search (2008); Cazenave, Batch Monte Carlo Tree Search; Grill et al.,
Monte-Carlo Tree Search as Regularized Policy Optimization (2020).
These motivate schedule definitions and frozen optimization; no claim of
novel universal physical law. Exact algebra checked with Wolfram and
Python finite differences. Scope-matched data generation explicitly deferred.
