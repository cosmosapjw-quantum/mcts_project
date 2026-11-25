# RESTORE45: grouped physics replication, full tangent saddle and operational probes
Authority: exact user RESTORE32 pasted theory (file identified as Pasted
text(20261125-150626).txt, read sections 1–17), parent RESTORE44
af1028ab0b9b9e9d66aab7337638859fe6e96824, original Q28/Q30 and archives.
Layer A and Q remain active. Existing 13 topic nodes and 121 outputs
are retained; every output has primary node ownership in crosswalk
and is a local subtask, not a new DAG vertex. No independent native
subagent exposed. Same real Quartz/MixNet v3, 4 candidate samples,
15x15 freestyle, no uniform rollout. Four new geometric openings and
4 p8 roots D4-distinct from RESTORE44; 8 repeats per source/law arm,
B16, wave4, beta=1/.3, logit h=±.3,0; prior tilt=0.

## R1–R5: finite sample and exact path response
For each complete history at fixed root/protocol, G is the score,
D=(L(+h)-L(-h))/(2h) a likelihood finite contrast. Under common
downstream law and support, E G=E D=0. For independent histories
with finite second moments, b_i=(sum_j A_j-A_i)/(n-1), independent
of G_i,D_i. E[(A_i-b_i)G_i]=E[A G], likewise D. The estimator is
unbiased sample covariance n/(n-1)*(mean AG-mean A mean G).
The individual centered terms are dependent; use jackknife for
per-root SE, equal-weight root means and descriptive across-root SE.
Compare E[A G] derivative and direct finite contrast separately;
E[A D] targets the SAME finite contrast as direct ±h, not the
derivative. Do not infer zero score from centered-estimator agreement.
Seed pairs within each root/law, complete-history sampling; no
independent plies. More precision required given IID mean score and
weight normalization deviations; no native identity admission.

## R5–R7: multivariate lattice saddle on actual restricted wave law
For fixed total w=4 in d=5 legal restricted actions, drop last count
and denote x in Z^(d-1). P(x) from exact ordered wave enumeration.
K(s)=log sum P(x) exp(s·x); gradient mean, Hessian tilted covariance.
At interior lattice target k/M, solve grad K(s*)=k/M.
I=s*·k/M-K(s*). P_M(k)≈exp(-M I) /
((2π M)^((d-1)/2) sqrt(det C_red)).
A=[I; -1^T] maps integer coordinates to tangent, U orthonormal,
C_tan=U^T A C_red A^T U. det C_tan=d det C_red since
det A^T A=d. Unit lattice fundamental cell covolume sqrt(d) in
orthonormal tangent coordinates, so factor cancels. Wolfram checked
Gram determinant 5 and 3D relation det(C_tan)=3 det(C_red).
The exact DP truncates nonnegative partial counts at target, so
omitted larger indices cannot reach it. Cross-check tilted DP:
log P_M(k)=-M I+log P_(tilt,M)(k). Domain interior all five positive;
for PL all <= M-1 (width4 of 5), M>=8, full span1 lattice. Tested
4D M8/16 and 2D three-sector (source, next, other) M8/16/32/64.
Not a controlled adaptive field-theory loop or physical hbar.
Higher-dimensional saddle approximation is conditional, no imported
continuous-density theorem as discrete proof. Exact DP is the reference.

## R5–R7: nonthermal forward/reverse and CP operational protocol
A frozen snapshot with field h_t ramp from -a to a; reverse comparison
is separately normalized reversed schedule at same initial snapshot
and reversed word. Common support for each same-law comparison.
sigma=log(P_F(word)/P_R(reversed word)). Then <exp(-sigma)>_F=1;
E sigma=D_KL(P_F||P_R∘rev)>=0. For count map C, KL chain rule
D_fine-D_count=E_F D(P_F(word|C)||Q(word|C))>=0.
IID independent product at reversed schedule has exactly zero ratio;
PL/reservation order matters. This is nonthermal relative path entropy,
not physically time-reversed native MCTS or heat. Coarse map discards
information. Prior thermal quench/reset reference retained.

Conditional qubit probe: actual observed target-indicator words m_t
are prescribed classical drive, auxiliary initial (|0>+e^(iπ/4)|1>)/sqrt2.
Noise unitary exp(-i α ξ_t m_t Z), α=.5, ξ=±1. Independent signs versus
one persistent sign have identical single-step maps. Known X pulses
toggle second-half sign (and undo known controls at readout). Offdiag
factor independent prod cos(2αm_t), static cos(2α sum sign_t m_t).
Average over same actual words, check explicit sequential 2x2 density
evolution and Kraus projector readout X,Y vs dephased same populations.
This is explicit CP/instrument experiment in a finite mathematical model,
not a claim that classical CPU MCTS has a quantum state or experimentally
observed physical phase. Echo need not increase trace distance in all
histories; PL width4 has at most one target hit, hence no difference in
this probe, but full 16-draw histories can distinguish memory.
CQ record: binary last-wave target occurrence label, three disjoint
preceding four-draw windows and union; conditional empirical qubit
ensembles. Holevo=S(avg)-sum p S(rho_y), fixed X readout MI <= Holevo <=H(Y).
Do not interpret eight-path plug-in info as population redundancy or SBS.

R8–R10: retain prior negative admissions, reject automatic selector
promotion. Existing history feature model fit on all 4 older opening
roots and tested on 4 new, same law and h0; NLL equal group weights,
no proof of sufficiency or memory effect on every root.
Literature: Glynn score derivation (RESTORE44), [Ohkubo 2009](https://scispace.com/papers/posterior-probability-and-fluctuation-theorem-in-stochastic-1qd6g0g7wz)
and [Chétrite/Marcantoni](https://scispace.com/papers/fluctuation-relations-associated-to-an-arbitrary-bijection-3bc1d8iw8da5)
support path-bijection interpretation. [Process characterization/control](https://www.nature.com/articles/s41467-020-20113-3)
motivates noncommuting interventions; no claim our model is that experiment.
[Saddle point density reference](https://arxiv.org/abs/2510.21545) has
continuous-density hypotheses; our lattice check is separate. Actual
frozen-law and explicit-matrix calculations are derived/numerically checked.
Board size, native phase/hbar, calibrated thermal and adaptive 1PI remain open.
