# RESTORE43: crossed arena and prospective finite-field response
R1/R2: RESTORE42 a8f9976ee1e53de0a47d5fe5ffa493ceffeb57dc, original Q28/Q30, same 13-node DAG and policies. Same pinned actual MixNet policy/WDL and four one-step candidates on both sides. No legacy uniform rollout. Grill et al. ICML 2020 Eq.7–8 https://proceedings.mlr.press/v119/grill20a/grill20a.pdf gives reverse KL regularized policy optimization with frozen Q; our finite shifted objective includes the native +1 visit denominator. Finite difference caution: Zazanis/Suri 1993 https://scispace.com/papers/convergence-rates-of-finite-difference-sensitivity-estimates-12f9tth8vb ; Signahl 2004 https://scispace.com/papers/sensitivity-analysis-via-simulation-in-the-presence-of-1f9xfufx58 ; Giles https://people.maths.ox.ac.uk/gilesm/mc/mc/lec7.pdf. Central difference O(h²) requires a smooth expectation, not individual discontinuous greedy histories.
R3: two clocks t=game ply and k=completed backup. All scores in dimensionless reward units, marks in root frame, Q=0 unvisited algorithmic convention. Same B != same actual queries. Color-swapped candidate score X=(z_B+z_W)/2, pair is unit, four predetermined openings and one seed each; no population generalization or Elo.
R4/R5: source prior P_h proportional P exp(h e_a), exact finite KKT J=q·x+lambda P_h·log(x+1/k). For fixed q, x_F(h) monotone at source index (RESTORE42). For true adaptive q(h), fixed active tangent dx/dh = K(g_P + q'), K=diag(D^-1)-vv^T/sum v, v=D^-1 1. Q feedback can reverse sign; discontinuous greedy paths and switching faces prevent using derivative as universal predictor. Decompose actual contrast chi - frozen into prospective field-feedback and remaining finite-search residual, and separately observed-Q diagnostic (post-treatment). Finite count quantization chi is multiple 1/(2hk); changing h changes resolution and possibly trajectory, not evidence of divergence.
R6/R7: choose low-parameter prospective alternatives before held-out generation. On RESTORE42 h=.2 B64 16 paired contrasts, by schedule fit ridge with fixed 0.05 penalty for q-field derivative y=(q(+h)-q(-h))/(2h), features e_a (source indicator) and f=(x_F(+h)-x_F(-h))/(2h), action weights (p0+pi0)/2. Each root/seed equal total weight; per-mode coefficients 2. Prediction qhat±=clip(q0±h*(alpha e_a+beta f),-1,1); f computed solely from h0 and requested h. Re-solve exact KKT on predicted q and perturbed p. Compare to zero, frozen and separately scalar gain times frozen (one ridge coefficient fitted on same training). Target metric equal-weight MAE per root and opening; do not tune after held-out. Record field prediction error versus zero field, occupancy prediction error and actual observed-Q diagnostic. Four held-out roots p12/p24 from two other opening clusters, prior-source selection uses max h0 native prior from previously played game only. No test outcomes in training.
R8: prospective model only if beats both zero and frozen on held-out blocks; no automatic selector change, finite-domain pilot only. Test h=.1,.2,.4 at B64; B32,128 only +/- .2 and zero. One seed per root/mode/budget due cost; no asymptotic convergence claim. Wolfram verifies p'_a=p_a(1-p_a), symmetric cubic central contrast a+c h².
R9/R10: complete missing B32 arena cells, reuse same seed labels by same opening order and base seed. Record actual per-side evaluator queries; no exact query-budget matching yet. Primary six observable figures and new response comparisons. Preserve original quantum/thermal/1PI obligations and prior negative admissions. Native independent subagent NOT_RUN; serial theory and separate numerical checks.

## Query count and finite-resolution checks
Let B be completed backups, R initial replicas and T completed terminal
selection events. In the recorded fixed evaluator protocol the number of
parent requests is B+R-T. Four child candidates per evaluated parent are
attempted; terminal child outcomes are evaluated by exact rules, not NNUE.
Thus C_NNUE=5(B+R-T)-C_terminal_child, counting successful logical
requests rather than transport retries. In the no-terminal regime serial64
and root61 with R=4 both use 325 logical requests, root64 uses 340.
This prediction was registered before the eight root calibration runs;
all four realized matches were exact. Different budget seeds are distinct.
The count lattice step is 1/(2hk), so the smallest nonzero response at
B64,h=.1 is .078125, and at B128,h=.2 is .01953125. Smooth-limit
central-difference extrapolation is not justified from those finite paths.

Prospective closure result: 40 held-out contrasts at four actual roots,
two opening clusters, not independent population sampling. Fit is frozen
and never retuned after evaluation. Although the serial occupancy MAE
improves marginally, Q-field MAE is worse in aggregate and wave occupancy
MAE deteriorates markedly. The candidate is not admitted. The observed-Q
diagnostic is not guaranteed to improve occupancy: wave post-treatment
MAE also exceeds frozen MAE, so dynamics is not represented by a static
optimum even after substituting actual fields. No claim that all possible
closures fail, and no identification of an effective loop coefficient.
