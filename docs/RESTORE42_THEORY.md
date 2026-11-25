# RESTORE42 · common MixNet paired arena and adaptive source response
R1–R3: exact parent main 1ef5edc49979d0a8ffba34f405b5dc9afdd628f0;
RESTORE41 evidence and original Q28/Q30, all 13 nodes preserved. Both colors
and modes use pinned Rapfi 250615 mix9svq policy/WDL through Quartz PUCT,
four one-step samples/evaluation, no uniform rollout. Score/mark convention
unchanged. No new neural model or physical energy. Compare schedules only
at a fixed opening, declared seed, backup budget and actual NNUE costs.
R4–R5: candidate A vs serial S, same opening o, each color once. Let z_B,z_W
be candidate wins/draw/loss coded 1,1/2,0. Pair score X=(z_B+z_W)/2,
contrast D=X-1/2 in [-1/2,1/2]. Five possible pair sums. Color contrast
(z_B-z_W)/2 remains descriptive; pairs sharing openings are not independent.
Opening blocks, not plies or individual games, are analysis units. Common
seed labels do not imply identical stochastic streams once trees diverge.
For fixed root a0, p_h(i)=p_i exp(h 1_i=a0)/Z; p'_0=p_i(1_i=a0-p_a0).
J(x)=q·x+lambda sum p_i log(x_i+1/k), lambda=c/sqrt(k).
At fixed active face, D_i=lambda p_i/(x_i+1/k)^2 and
g_i=lambda p'_i/(x_i+1/k); derivative dx=D^-1(g-1*(1^T D^-1 g)/(1^T D^-1 1)).
Face change => use exact finite KKT at +/-h, not linear derivative.
Actual derivative estimate [pi(+h)-pi(-h)]/(2h) is finite contrast of
adaptive search, not FDT. Also compute q-feedback diagnostic via actual
perturbed q fields substituted into frozen objective: descriptive,
not predictive since uses outcomes. Baseline no-response comparator.
R6–R7: Wolfram verifies normalized tilt derivative and pair contrast.
Use matched independent game blocks and paired fixed-root seed labels;
check actual signs, terminal legality, N/W/S2, NNUE probabilities and
costs. Primary response metrics: absolute error relative zero comparator,
sign, face changes, Q-field shifts; small pilot cannot establish general
response model or one-loop improvement. Exa/Parallel matched-game
pentanomial context: https://chessprogramming.org/Match_Statistics ;
SciSpace CRN Veness/Lanctot/Bowling 2011
https://scispace.com/papers/variance-reduction-in-monte-carlo-tree-search-2zrequxt2m
and Glasserman/Yao 1992 https://scispace.com/papers/some-guidelines-and-guarantees-for-common-random-numbers-inuvd0fs9s .
These motivate pairing, not guaranteed variance reduction.
R8–R10: separate runtime failure, numerical replay, observed contrast,
and original physical admission. Preserve prior negative selector/stopping
evidence. Actual data-driven T/A/R/D/N/C panels, no phase/thermal claim.
Target next pilot effect .10 score fraction per paired opening; use pilot
block SD and normal approximation only if nonzero, no powered conclusion
from four chosen openings. Native independent subagent unavailable.

## Fixed-field sign theorem and exact response decomposition
For fixed Q, positive P, k>0,c>0 and finite h, the shifted objective is
strictly concave on the compact simplex. On a fixed active face, the
target score derivative g_a0>0 (unless support is singleton) while g_b<0
for b!=a0. The constrained derivative is D^-1(g-1*weighted mean g).
If a0 is active, dx_a0/dh>=0; if inactive on an interval, x_a0=0.
Across face changes continuity and local Lipschitz continuity follow from
uniform strong concavity on bounded h intervals (positive P_h bounded
away from zero and x in simplex), so the almost-everywhere nonnegative
derivative implies nondecreasing x_a0(h). No differentiability claim
at switching boundaries; finite KKT is exact there. Hence negative
actual finite occupancy response cannot be represented by that frozen
optimum, but neither identifies a unique Q-feedback mechanism nor
proves a quantum correction. This is a property of the specified
finite variational reference, not a novelty claim or general no-go.

For measured response chi, frozen finite optimum difference chi_F,
and the same optimum evaluated at actual perturbed Q fields chi_A:
chi-chi_F=(chi_A-chi_F)+(chi-chi_A).
First term is a descriptive field feedback contrast, second is
remaining finite-search departure. Not a causal mediation decomposition:
chi_A uses the outcomes. In the present data 2 negative serial
responses occur; all frozen finite source predictions are nonnegative.
