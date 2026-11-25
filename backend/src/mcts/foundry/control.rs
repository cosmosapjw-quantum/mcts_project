//! Control-plane skeletons: A01-A05 and A24.

use std::collections::BTreeMap;

use super::coordinator::EFFECTIVE_PRIOR_SCORE_VECTOR_TELEMETRY_KEY;
use super::types::{
    AxisStatus, CostVector, FoundryAxis, FoundryObservation, MetaAction, MetaProposal,
    ProposalEstimate, UncertaintyChannels, FOUNDRY_CONTRACT_SCHEMA_VERSION,
    SKELETON_EVIDENCE_SCOPE,
};

fn best_two(observation: &FoundryObservation<'_>) -> Option<(usize, Option<usize>)> {
    if observation.edges.is_empty() {
        return None;
    }
    let mut order: Vec<usize> = (0..observation.edges.len()).collect();
    order.sort_by(|&a, &b| observation.edges[b].q.total_cmp(&observation.edges[a].q));
    Some((order[0], order.get(1).copied()))
}

#[derive(Clone, Debug)]
pub struct A01StopCouncil {
    pub risk_limit: f32,
    pub min_visits: u32,
}

impl Default for A01StopCouncil {
    fn default() -> Self {
        Self {
            risk_limit: 0.05,
            min_visits: 16,
        }
    }
}

impl A01StopCouncil {
    pub fn fallback_risk(&self, observation: &FoundryObservation<'_>) -> f64 {
        let h1_risk = 1.0 - observation.extras.h1_stability.unwrap_or(0.0);
        let p_flip = observation.extras.p_flip.unwrap_or(1.0);
        h1_risk
            .max(p_flip)
            .max(observation.extras.omission_bound)
            .clamp(0.0, 1.0)
    }
}

impl FoundryAxis for A01StopCouncil {
    fn id(&self) -> &'static str {
        "A01.stop_council"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Shadow
    }

    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        let risk = self.fallback_risk(observation);
        if observation.snap.root_visits < self.min_visits || risk > f64::from(self.risk_limit) {
            return;
        }
        let Some((best, _)) = best_two(observation) else {
            return;
        };
        let mut telemetry = BTreeMap::new();
        telemetry.insert("risk".to_string(), serde_json::json!(risk));
        out.push(MetaProposal {
            schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
            axis_id: self.id().to_string(),
            action: MetaAction::Stop {
                edge_pos: Some(observation.edges[best].idx),
            },
            estimate: ProposalEstimate {
                confidence: 1.0 - risk,
                ..ProposalEstimate::default()
            },
            activation_guard:
                "fresh snapshot; calibrated risk; arbiter verifies all non-stop net values <= 0"
                    .to_string(),
            explanation: format!("fallback stop-council risk={risk:.4}"),
            evidence_scope: SKELETON_EVIDENCE_SCOPE.to_string(),
            telemetry,
        });
    }
}

#[derive(Clone, Debug)]
pub struct A02StaticAnchorRpo {
    pub temperature: f32,
    pub prior_floor: f32,
    pub use_lower_confidence: bool,
}

impl Default for A02StaticAnchorRpo {
    fn default() -> Self {
        Self {
            temperature: 0.25,
            prior_floor: 1e-8,
            use_lower_confidence: true,
        }
    }
}

impl A02StaticAnchorRpo {
    pub fn solve(&self, observation: &FoundryObservation<'_>, lower: &[f32]) -> Vec<f32> {
        let tau = self.temperature.max(1e-5);
        let mut logits = Vec::with_capacity(observation.edges.len());
        for (i, edge) in observation.edges.iter().enumerate() {
            let score = if self.use_lower_confidence {
                lower.get(i).copied().unwrap_or(edge.q)
            } else {
                edge.q
            };
            logits.push(edge.prior.max(self.prior_floor).ln() + score / tau);
        }
        if logits.is_empty() {
            return logits;
        }
        let shift = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut weights: Vec<f32> = logits.into_iter().map(|x| (x - shift).exp()).collect();
        let total: f32 = weights.iter().sum();
        if total > 0.0 {
            for value in &mut weights {
                *value /= total;
            }
        }
        weights
    }
}

impl FoundryAxis for A02StaticAnchorRpo {
    fn id(&self) -> &'static str {
        "A02.static_anchor_rpo"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Shadow
    }

    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        let lower: Vec<f32> = observation.edges.iter().map(|edge| edge.q).collect();
        let policy = self.solve(observation, &lower);
        let mut telemetry = BTreeMap::new();
        telemetry.insert(
            "temperature".to_string(),
            serde_json::json!(self.temperature),
        );
        telemetry.insert(
            "policy_sum".to_string(),
            serde_json::json!(policy.iter().sum::<f32>()),
        );
        telemetry.insert(
            EFFECTIVE_PRIOR_SCORE_VECTOR_TELEMETRY_KEY.to_string(),
            serde_json::json!(policy),
        );
        out.push(MetaProposal {
            schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
            axis_id: self.id().to_string(),
            action: MetaAction::Noop,
            estimate: ProposalEstimate {
                confidence: 0.5,
                ..ProposalEstimate::default()
            },
            activation_guard: "root-only; frozen NN anchor; no recursive prior mutation".into(),
            explanation: "temporary KL-regularized policy ready for cache/readout ablation".into(),
            evidence_scope: SKELETON_EVIDENCE_SCOPE.to_string(),
            telemetry,
        });
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct A03UncertaintyDecomposition;

impl A03UncertaintyDecomposition {
    pub fn total(&self, channels: UncertaintyChannels) -> f32 {
        channels.conservative_sum()
    }

    pub fn bounds(&self, q: f32, channels: UncertaintyChannels) -> (f32, f32) {
        let radius = self.total(channels);
        ((q - radius).max(-1.0), (q + radius).min(1.0))
    }
}

impl FoundryAxis for A03UncertaintyDecomposition {
    fn id(&self) -> &'static str {
        "A03.uncertainty_decomposition"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Shadow
    }

    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        let mut telemetry = BTreeMap::new();
        telemetry.insert(
            "edge_count".to_string(),
            serde_json::json!(observation.edges.len()),
        );
        out.push(MetaProposal {
            schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
            axis_id: self.id().to_string(),
            action: MetaAction::Noop,
            estimate: ProposalEstimate {
                confidence: 0.5,
                ..ProposalEstimate::default()
            },
            activation_guard: "completed backups only; model version frozen for root epoch".into(),
            explanation: "publish MC/epistemic/drift/bias channels separately".into(),
            evidence_scope: SKELETON_EVIDENCE_SCOPE.to_string(),
            telemetry,
        });
    }
}

#[derive(Clone, Debug)]
pub struct A04KgVocAllocator {
    pub batch: u32,
    pub cost_per_eval_ms: f32,
}

impl Default for A04KgVocAllocator {
    fn default() -> Self {
        Self {
            batch: 8,
            cost_per_eval_ms: 1.0,
        }
    }
}

impl A04KgVocAllocator {
    fn proxy(&self, q: f32, sigma: f32, best_q: f32, best_sigma: f32) -> f32 {
        let gap = (best_q - q).max(0.0);
        let uncertainty = (sigma + best_sigma).max(1e-6);
        uncertainty * (-gap / uncertainty).exp()
    }
}

impl FoundryAxis for A04KgVocAllocator {
    fn id(&self) -> &'static str {
        "A04.kg_voc_allocator"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Shadow
    }

    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        let Some((best_idx, _)) = best_two(observation) else {
            return;
        };
        let best = observation.edges[best_idx];
        let best_sigma = best.sigma_a(4.0);
        for edge in observation.edges {
            let gain = self.proxy(edge.q, edge.sigma_a(4.0), best.q, best_sigma);
            out.push(MetaProposal {
                schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
                axis_id: self.id().to_string(),
                action: MetaAction::Sample {
                    edge_pos: edge.idx,
                    visits: self.batch,
                },
                estimate: ProposalEstimate {
                    regret_reduction_mean: f64::from(gain),
                    regret_reduction_lcb: f64::from(0.5 * gain),
                    confidence: 0.25,
                    cost: CostVector {
                        nn_evals: f64::from(self.batch),
                        cpu_ms: f64::from(self.batch) * f64::from(self.cost_per_eval_ms),
                        ..CostVector::default()
                    },
                },
                activation_guard:
                    "allocation only; measured costs; low-budget KG-stop claim remains closed"
                        .to_string(),
                explanation: format!("KG proxy for edge_pos={}", edge.idx),
                evidence_scope: SKELETON_EVIDENCE_SCOPE.to_string(),
                telemetry: BTreeMap::new(),
            });
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct A05CounterfactualMetaTeacher;

impl FoundryAxis for A05CounterfactualMetaTeacher {
    fn id(&self) -> &'static str {
        "A05.counterfactual_meta_teacher"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Seed
    }

    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        let mut telemetry = BTreeMap::new();
        telemetry.insert(
            "budget".to_string(),
            serde_json::json!(observation.snap.root_visits),
        );
        out.push(MetaProposal {
            schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
            axis_id: self.id().to_string(),
            action: MetaAction::Noop,
            estimate: ProposalEstimate::default(),
            activation_guard: "offline or deterministic resident-session fork only".into(),
            explanation: "serialize/fork identical snapshot for STOP/SAMPLE/WIDEN labels".into(),
            evidence_scope: SKELETON_EVIDENCE_SCOPE.to_string(),
            telemetry,
        });
    }
}

#[derive(Clone, Debug)]
pub struct A24LearnedBudgetGate {
    pub budgets: Vec<u32>,
    pub cost_per_visit_ms: f32,
}

impl Default for A24LearnedBudgetGate {
    fn default() -> Self {
        Self {
            budgets: vec![8, 16, 32, 64, 128],
            cost_per_visit_ms: 0.1,
        }
    }
}

impl FoundryAxis for A24LearnedBudgetGate {
    fn id(&self) -> &'static str {
        "A24.learned_budget_gate"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Seed
    }

    fn propose(&self, _observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        for &visits in &self.budgets {
            let mut telemetry = BTreeMap::new();
            telemetry.insert(
                "offline_root_budget_visits".to_string(),
                serde_json::json!(visits),
            );
            telemetry.insert("offline_only".to_string(), serde_json::json!(true));
            out.push(MetaProposal {
                schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
                axis_id: self.id().to_string(),
                action: MetaAction::Noop,
                estimate: ProposalEstimate {
                    cost: CostVector {
                        nn_evals: f64::from(visits),
                        cpu_ms: f64::from(visits) * f64::from(self.cost_per_visit_ms),
                        ..CostVector::default()
                    },
                    ..ProposalEstimate::default()
                },
                activation_guard:
                    "offline-only; no executable root-budget action exists in contract v1".into(),
                explanation: format!("offline candidate root budget={visits}; NOOP only"),
                evidence_scope: SKELETON_EVIDENCE_SCOPE.to_string(),
                telemetry,
            });
        }
    }
}
