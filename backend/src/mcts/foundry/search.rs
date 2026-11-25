//! Candidate, allocation, and backend skeletons: A06-A16, A25-A26.

use std::collections::BTreeMap;

use super::types::{
    AxisStatus, CostVector, FoundryAxis, FoundryObservation, MetaAction, MetaProposal,
    ProposalEstimate, FOUNDRY_CONTRACT_SCHEMA_VERSION, SKELETON_EVIDENCE_SCOPE,
};

fn best_q_and_lower(observation: &FoundryObservation<'_>) -> Option<(f32, f32)> {
    observation
        .edges
        .iter()
        .max_by(|a, b| a.q.total_cmp(&b.q))
        .map(|edge| (edge.q, edge.q - edge.sigma_a(4.0)))
}

#[derive(Clone, Debug)]
pub struct A06GumbelSequentialHalving {
    pub candidate_count: u16,
    pub seed: u64,
}

impl Default for A06GumbelSequentialHalving {
    fn default() -> Self {
        Self {
            candidate_count: 16,
            seed: 0,
        }
    }
}

impl FoundryAxis for A06GumbelSequentialHalving {
    fn id(&self) -> &'static str {
        "A06.gumbel_sequential_halving"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::MechanismValid
    }

    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        let count = self
            .candidate_count
            .min(observation.edges.len().try_into().unwrap_or(u16::MAX))
            .max(1);
        let per_arm = (observation.snap.root_visits.max(1) / count as u32).max(1);
        for edge in observation.edges.iter().take(count as usize) {
            out.push(MetaProposal {
                schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
                axis_id: self.id().to_string(),
                action: MetaAction::Sample {
                    edge_pos: edge.idx,
                    visits: per_arm,
                },
                estimate: ProposalEstimate {
                    confidence: 0.25,
                    cost: CostVector {
                        nn_evals: f64::from(per_arm),
                        ..CostVector::default()
                    },
                    ..ProposalEstimate::default()
                },
                activation_guard:
                    "use policy::gumbel_sh implementation; without replacement; tactical reserve preserved"
                        .to_string(),
                explanation: format!("Gumbel/SH bracket candidate edge_pos={}", edge.idx),
                evidence_scope: SKELETON_EVIDENCE_SCOPE.to_string(),
                telemetry: BTreeMap::new(),
            });
        }
    }
}

#[derive(Clone, Debug)]
pub struct A07ResidualEvidenceWidening {
    pub temperature: f32,
    pub max_tail_mass: f32,
    pub widen_count: u16,
}

impl Default for A07ResidualEvidenceWidening {
    fn default() -> Self {
        Self {
            temperature: 0.25,
            max_tail_mass: 0.05,
            widen_count: 4,
        }
    }
}

impl A07ResidualEvidenceWidening {
    pub fn bound(&self, observation: &FoundryObservation<'_>) -> f32 {
        let tau = self.temperature.max(1e-5);
        let mut z_live = 0.0;
        let mut z_out = 0.0;
        for (index, edge) in observation.edges.iter().enumerate() {
            let weight = edge.prior.max(1e-12);
            if index < observation.snap.n_visible as usize {
                z_live += weight * (edge.q / tau).exp();
            } else {
                let upper = edge.q + edge.sigma_a(4.0);
                z_out += weight * (upper / tau).exp();
            }
        }
        if z_live + z_out > 0.0 {
            z_out / (z_live + z_out)
        } else {
            1.0
        }
    }
}

impl FoundryAxis for A07ResidualEvidenceWidening {
    fn id(&self) -> &'static str {
        "A07.residual_evidence_widening"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Seed
    }

    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        let residual = self.bound(observation);
        if residual <= self.max_tail_mass
            || observation.snap.n_visible >= observation.snap.n_children
        {
            return;
        }
        let mut telemetry = BTreeMap::new();
        telemetry.insert(
            "residual_mass_bound".to_string(),
            serde_json::json!(residual),
        );
        out.push(MetaProposal {
            schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
            axis_id: self.id().to_string(),
            action: MetaAction::Widen {
                count: self.widen_count,
            },
            estimate: ProposalEstimate {
                regret_reduction_mean: f64::from(residual),
                regret_reduction_lcb: 0.0,
                confidence: 0.25,
                cost: CostVector {
                    nn_evals: f64::from(self.widen_count),
                    ..CostVector::default()
                },
            },
            activation_guard:
                "calibrated unmaterialized-action upper scores; fresh edge generation".into(),
            explanation: format!("outside posterior mass upper bound={residual:.4}"),
            evidence_scope: SKELETON_EVIDENCE_SCOPE.to_string(),
            telemetry,
        });
    }
}

#[derive(Clone, Debug)]
pub struct A08TacticalProofBackend {
    pub proof_budget: u32,
}

impl Default for A08TacticalProofBackend {
    fn default() -> Self {
        Self { proof_budget: 64 }
    }
}

impl FoundryAxis for A08TacticalProofBackend {
    fn id(&self) -> &'static str {
        "A08.tactical_proof_backend"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Conditional
    }

    fn propose(&self, _observation: &FoundryObservation<'_>, _out: &mut Vec<MetaProposal>) {
        // Game-specific tactical flags are not yet exposed by EdgeView.  Add a
        // typed sentinel channel; do not overload `last_value` or action ids.
    }
}

#[derive(Clone, Debug)]
pub struct A09H3ChangePointRouter {
    pub burst_visits: u32,
}

impl Default for A09H3ChangePointRouter {
    fn default() -> Self {
        Self { burst_visits: 16 }
    }
}

impl FoundryAxis for A09H3ChangePointRouter {
    fn id(&self) -> &'static str {
        "A09.h3_change_point_router"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Shadow
    }

    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        let score =
            observation.extras.entropy_slope.max(0.0) * (-observation.extras.margin_slope).max(0.0);
        if score <= 0.0 {
            return;
        }
        let Some(edge) = observation.edges.iter().max_by(|a, b| a.q.total_cmp(&b.q)) else {
            return;
        };
        out.push(MetaProposal {
            schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
            axis_id: self.id().to_string(),
            action: MetaAction::Deepen {
                edge_pos: edge.idx,
                visits: self.burst_visits,
            },
            estimate: ProposalEstimate {
                regret_reduction_mean: score,
                cost: CostVector {
                    nn_evals: f64::from(self.burst_visits),
                    ..CostVector::default()
                },
                ..ProposalEstimate::default()
            },
            activation_guard:
                "threshold learned from external hard-state labels; no fixed zero floors".into(),
            explanation: format!("entropy-margin change-point score={score:.6}"),
            evidence_scope: SKELETON_EVIDENCE_SCOPE.to_string(),
            telemetry: BTreeMap::new(),
        });
    }
}

#[derive(Clone, Debug)]
pub struct A10PriorRefreshSpecialist {
    pub divergence_threshold: f32,
    pub max_blend: f32,
}

impl Default for A10PriorRefreshSpecialist {
    fn default() -> Self {
        Self {
            divergence_threshold: 0.5,
            max_blend: 0.2,
        }
    }
}

impl FoundryAxis for A10PriorRefreshSpecialist {
    fn id(&self) -> &'static str {
        "A10.prior_refresh_specialist"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Dormant
    }

    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        if observation.extras.prior_visit_js < f64::from(self.divergence_threshold) {
            return;
        }
        out.push(MetaProposal {
            schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
            axis_id: self.id().to_string(),
            action: MetaAction::Noop,
            estimate: ProposalEstimate::default(),
            activation_guard:
                "OOD/weak-evaluator specialist; static anchor retained; never recursive".into(),
            explanation: format!(
                "conditional refresh candidate: JS={:.4}, max_blend={:.3}",
                observation.extras.prior_visit_js, self.max_blend
            ),
            evidence_scope: SKELETON_EVIDENCE_SCOPE.to_string(),
            telemetry: BTreeMap::new(),
        });
    }
}

#[derive(Clone, Debug)]
pub struct A11DynamicLiveSetParticles {
    pub batch: u16,
    pub max_active: usize,
}

impl Default for A11DynamicLiveSetParticles {
    fn default() -> Self {
        Self {
            batch: 4,
            max_active: 8,
        }
    }
}

impl FoundryAxis for A11DynamicLiveSetParticles {
    fn id(&self) -> &'static str {
        "A11.dynamic_live_set_particles"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Seed
    }

    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        let (_, best_lower) = best_q_and_lower(observation).unwrap_or((-1.0, -1.0));
        let mut ranked: Vec<(f32, u16)> = observation
            .edges
            .iter()
            .map(|edge| {
                let upper = edge.q + edge.sigma_a(4.0);
                ((upper - best_lower).max(0.0) + edge.sigma_a(4.0), edge.idx)
            })
            .collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (score, edge_pos) in ranked.into_iter().take(self.max_active) {
            out.push(MetaProposal {
                schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
                axis_id: self.id().to_string(),
                action: MetaAction::ResampleMode {
                    mode_id: edge_pos,
                    count: self.batch,
                },
                estimate: ProposalEstimate {
                    regret_reduction_mean: f64::from(score),
                    cost: CostVector {
                        nn_evals: f64::from(self.batch),
                        ..CostVector::default()
                    },
                    ..ProposalEstimate::default()
                },
                activation_guard: "independent groups; reversible hibernation; resurrection quota"
                    .into(),
                explanation: format!("live-set weight={score:.4} for edge_pos={edge_pos}"),
                evidence_scope: SKELETON_EVIDENCE_SCOPE.to_string(),
                telemetry: BTreeMap::new(),
            });
        }
    }
}

#[derive(Clone, Debug)]
pub struct A12JsdLocallyBalancedSampler {
    pub temperature: f32,
    pub bandwidth: f32,
}

impl Default for A12JsdLocallyBalancedSampler {
    fn default() -> Self {
        Self {
            temperature: 0.25,
            bandwidth: 0.2,
        }
    }
}

impl FoundryAxis for A12JsdLocallyBalancedSampler {
    fn id(&self) -> &'static str {
        "A12.jsd_locally_balanced_sampler"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Seed
    }

    fn propose(&self, _observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        out.push(MetaProposal::noop(
            self.id(),
            "build sibling policy/value geometry on common legal support; JSD is metric, not reward",
        ));
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct A13PendingFlowWuUct;

impl FoundryAxis for A13PendingFlowWuUct {
    fn id(&self) -> &'static str {
        "A13.pending_flow_wu_uct"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Conditional
    }

    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        let pending: u32 = observation.edges.iter().map(|edge| edge.n_virtual).sum();
        let mut telemetry = BTreeMap::new();
        telemetry.insert("pending".to_string(), serde_json::json!(pending));
        out.push(MetaProposal {
            schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
            axis_id: self.id().to_string(),
            action: MetaAction::Noop,
            estimate: ProposalEstimate {
                confidence: 0.5,
                ..ProposalEstimate::default()
            },
            activation_guard:
                "unobserved counts shape selection only; never enter confidence evidence".into(),
            explanation: "separate pending-flow/WU-UCT correction from adaptive virtual value"
                .into(),
            evidence_scope: SKELETON_EVIDENCE_SCOPE.to_string(),
            telemetry,
        });
    }
}

#[derive(Clone, Debug)]
pub struct A14SemanticPathLsh {
    pub min_threads: u16,
    pub overlap_threshold: f32,
}

impl Default for A14SemanticPathLsh {
    fn default() -> Self {
        Self {
            min_threads: 8,
            overlap_threshold: 0.5,
        }
    }
}

impl FoundryAxis for A14SemanticPathLsh {
    fn id(&self) -> &'static str {
        "A14.semantic_path_lsh"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Shadow
    }

    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        let runtime = observation.extras.runtime;
        if runtime.threads < self.min_threads
            || runtime.semantic_path_overlap < f64::from(self.overlap_threshold)
        {
            return;
        }
        out.push(MetaProposal {
            schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
            axis_id: self.id().to_string(),
            action: MetaAction::ResampleMode {
                mode_id: u16::MAX,
                count: runtime.inflight.max(1),
            },
            estimate: ProposalEstimate::default(),
            activation_guard:
                "edge duplication already controlled; high-thread semantic overlap remains".into(),
            explanation: format!("semantic overlap={:.3}", runtime.semantic_path_overlap),
            evidence_scope: SKELETON_EVIDENCE_SCOPE.to_string(),
            telemetry: BTreeMap::new(),
        });
    }
}

#[derive(Clone, Debug, Default)]
pub struct A15ServiceCurveScheduler {
    pub best_batch: u16,
    pub best_inflight: u16,
}

impl FoundryAxis for A15ServiceCurveScheduler {
    fn id(&self) -> &'static str {
        "A15.service_curve_scheduler"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::MechanismValid
    }

    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        if self.best_batch > 0 && self.best_batch != observation.extras.runtime.batch_size {
            out.push(MetaProposal {
                schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
                axis_id: self.id().to_string(),
                action: MetaAction::SetBatch {
                    batch_size: self.best_batch,
                },
                estimate: ProposalEstimate {
                    confidence: 0.5,
                    ..ProposalEstimate::default()
                },
                activation_guard: "service-curve artifact matches hardware/runtime contract".into(),
                explanation: format!("batch target={}", self.best_batch),
                evidence_scope: SKELETON_EVIDENCE_SCOPE.to_string(),
                telemetry: BTreeMap::new(),
            });
        }
        if self.best_inflight > 0 && self.best_inflight != observation.extras.runtime.inflight {
            out.push(MetaProposal {
                schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
                axis_id: self.id().to_string(),
                action: MetaAction::SetInflight {
                    credit: self.best_inflight,
                },
                estimate: ProposalEstimate {
                    confidence: 0.5,
                    ..ProposalEstimate::default()
                },
                activation_guard: "service-curve artifact matches hardware/runtime contract".into(),
                explanation: format!("inflight target={}", self.best_inflight),
                evidence_scope: SKELETON_EVIDENCE_SCOPE.to_string(),
                telemetry: BTreeMap::new(),
            });
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct A16MonteCarloGraphSharing;

impl FoundryAxis for A16MonteCarloGraphSharing {
    fn id(&self) -> &'static str {
        "A16.monte_carlo_graph_sharing"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Seed
    }

    fn propose(&self, _observation: &FoundryObservation<'_>, _out: &mut Vec<MetaProposal>) {
        // Add a state-key channel and explicit share candidates.  State/eval
        // cache sharing lands before graph-wide N/W/Q sharing; parent-edge
        // statistics remain parent-specific by default.
    }
}

#[derive(Clone, Debug)]
pub struct A25MentsSoftBackup {
    pub temperature: f32,
}

impl Default for A25MentsSoftBackup {
    fn default() -> Self {
        Self { temperature: 0.1 }
    }
}

impl FoundryAxis for A25MentsSoftBackup {
    fn id(&self) -> &'static str {
        "A25.ments_soft_backup"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Dormant
    }

    fn propose(&self, _observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        out.push(MetaProposal::noop(
            self.id(),
            "opt-in root/shallow soft-backup ablation; evaluate objective mismatch separately",
        ));
    }
}

#[derive(Clone, Debug)]
pub struct A26NestedContourExactLab {
    pub live_points: u16,
    pub depth: u16,
}

impl Default for A26NestedContourExactLab {
    fn default() -> Self {
        Self {
            live_points: 32,
            depth: 6,
        }
    }
}

impl FoundryAxis for A26NestedContourExactLab {
    fn id(&self) -> &'static str {
        "A26.nested_contour_exact_lab"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::AnalysisOnly
    }

    fn propose(&self, _observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        out.push(MetaProposal::noop(
            self.id(),
            "offline enumerable-tree nested-contour validation; separate from live-set particle search",
        ));
    }
}
