//! Experiment-only SearchPolicy bridge for the A01 stop council.
//!
//! A [`FoundrySearchConfig`] is stored on `MctsConfig`, but a fresh policy is
//! materialized for every `MctsEngine` root.  This is essential: server batch
//! configs are cloned across games, while coordinator decisions and freshness
//! identities are root-local state and may never share a cache.

use parking_lot::Mutex;

use crate::mcts::policy::kg_stop::standard_normal_cdf;
use crate::mcts::policy::{
    ControllerTelemetry, EdgeView, HaltDecision, ScoreAdjustment, SearchPolicy, SearchSnapshot,
};
use crate::mcts::quartz::HaltReason;

use super::control::A01StopCouncil;
use super::coordinator::{
    FoundryCoordinator, GuardedMetaActionExecutor, LiveActionAuthorization, MetaActionExecutor,
};
use super::types::{
    ConservativeArbiter, CostPrices, FoundryObservation, FoundryRootExtras, FreshnessIdentity,
    MetaAction, MetaActionKind, RuntimeSnapshot, FOUNDRY_CONTRACT_SCHEMA_VERSION,
};

pub const A01_AXIS_ID: &str = "A01.stop_council";
pub const A01_LIVE_PFLIP_AXIS_ID: &str = "A01.live_pflip_v1";
pub const A01_TRACE_STABILITY_AXIS_ID: &str = "A01.trace_stability_v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoundryRuntimeMode {
    Shadow,
    Active,
}

impl FoundryRuntimeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Shadow => "shadow_noop",
            Self::Active => "active",
        }
    }
}

#[derive(Clone, Debug)]
pub struct FoundrySearchConfig {
    pub mode: FoundryRuntimeMode,
    pub axis_id: String,
    pub checkpoint_id: String,
    pub evaluator_id: String,
    pub risk_limit: f32,
    pub min_visits: u32,
}

impl FoundrySearchConfig {
    pub fn is_valid(&self) -> bool {
        (self.axis_id == A01_LIVE_PFLIP_AXIS_ID
            || self.axis_id == A01_AXIS_ID
            || self.axis_id == "A01")
            && self.axis_id != A01_TRACE_STABILITY_AXIS_ID
            && !self.checkpoint_id.trim().is_empty()
            && !self.evaluator_id.trim().is_empty()
            && self.risk_limit.is_finite()
            && (0.0..=1.0).contains(&self.risk_limit)
            && self.min_visits > 0
    }
}

#[derive(Clone, Debug, Default)]
struct FoundryPolicyState {
    decisions: u64,
    proposals: u64,
    actions: u64,
    coordination_errors: u64,
    stop_identity: Option<FreshnessIdentity>,
}

#[derive(Default)]
struct StopExecutor;

impl MetaActionExecutor for StopExecutor {
    type Output = bool;
    type Error = MetaActionKind;

    fn execute(&mut self, proposal: &super::types::MetaProposal) -> Result<bool, Self::Error> {
        match proposal.action.base_action() {
            MetaAction::Stop { .. } => Ok(true),
            MetaAction::Noop => Ok(false),
            other => Err(other.kind()),
        }
    }
}

pub struct FoundrySearchPolicy {
    config: FoundrySearchConfig,
    root_hash: u64,
    state: Mutex<FoundryPolicyState>,
}

impl FoundrySearchPolicy {
    pub fn new(config: FoundrySearchConfig, root_hash: u64) -> Self {
        assert!(config.is_valid(), "invalid FoundrySearchConfig");
        Self {
            config,
            root_hash,
            state: Mutex::new(FoundryPolicyState::default()),
        }
    }

    fn edge_set_hash(edges: &[EdgeView<'_>]) -> (String, u64) {
        let mut hash: u64 = 0xcbf29ce484222325;
        for edge in edges {
            for b in (edge.idx as u64).to_le_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            for b in edge.prior.to_bits().to_le_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        (format!("{hash:016x}"), hash)
    }

    fn p_flip(edges: &[EdgeView<'_>]) -> f32 {
        if edges.len() < 2 {
            return 1.0;
        }
        let mut ranked: Vec<&EdgeView<'_>> = edges.iter().collect();
        ranked.sort_by(|a, b| b.q.total_cmp(&a.q));
        let margin = (ranked[0].q - ranked[1].q).max(0.0);
        let sigma = (ranked[0].sigma_a(4.0).powi(2) + ranked[1].sigma_a(4.0).powi(2))
            .sqrt()
            .max(1e-6);
        standard_normal_cdf(-margin / sigma).clamp(0.0, 1.0)
    }

    fn root_statistics(edges: &[EdgeView<'_>]) -> (f64, f64, f64, f64) {
        let total_visits: f64 = edges.iter().map(|edge| f64::from(edge.n)).sum();
        let mut entropy = 0.0;
        let mut prior_visit_js = 0.0;
        let mut prior_total: f64 = edges.iter().map(|edge| f64::from(edge.prior)).sum();
        if prior_total <= 0.0 {
            prior_total = 1.0;
        }
        if total_visits > 0.0 {
            for edge in edges {
                let visit_share = f64::from(edge.n) / total_visits;
                if visit_share > 0.0 {
                    entropy -= visit_share * visit_share.ln();
                }
                let prior = f64::from(edge.prior) / prior_total;
                let midpoint = 0.5 * (visit_share + prior);
                if visit_share > 0.0 && midpoint > 0.0 {
                    prior_visit_js += 0.5 * visit_share * (visit_share / midpoint).ln();
                }
                if prior > 0.0 && midpoint > 0.0 {
                    prior_visit_js += 0.5 * prior * (prior / midpoint).ln();
                }
            }
        }
        let mut q_values: Vec<f32> = edges.iter().map(|edge| edge.q).collect();
        q_values.sort_by(|a, b| b.total_cmp(a));
        let top2_margin = q_values
            .first()
            .zip(q_values.get(1))
            .map_or(0.0, |(best, second)| f64::from(*best - *second));
        (entropy, entropy.exp(), top2_margin, prior_visit_js)
    }

    fn identity_and_extras(
        &self,
        snap: &SearchSnapshot,
        edges: &[EdgeView<'_>],
    ) -> (FreshnessIdentity, FoundryRootExtras) {
        let (edge_set_hash, edge_hash) = Self::edge_set_hash(edges);
        let identity = FreshnessIdentity {
            root_hash: self.root_hash,
            checkpoint_id: self.config.checkpoint_id.clone(),
            evaluator_id: self.config.evaluator_id.clone(),
            edge_set_hash,
            candidate_epoch: edge_hash,
            tt_identity_policy: "exact_tt_hash_v1".to_string(),
            cache_schema_version: 1,
            root_visits: snap.root_visits,
            iteration: snap.iteration,
        };
        let p_flip = Self::p_flip(edges);
        let (entropy, effective_branching, top2_margin, prior_visit_js) =
            Self::root_statistics(edges);
        let extras = FoundryRootExtras {
            schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
            freshness: identity.clone(),
            entropy,
            effective_branching,
            top2_margin,
            margin_slope: 0.0,
            entropy_slope: 0.0,
            h1_stability: Some(f64::from(1.0 - p_flip)),
            p_flip: Some(f64::from(p_flip)),
            prior_visit_js,
            omission_bound: if snap.n_visible >= snap.n_children {
                0.0
            } else {
                1.0
            },
            revision_count: 0,
            runtime: RuntimeSnapshot::default(),
        };
        (identity, extras)
    }
}

impl SearchPolicy for FoundrySearchPolicy {
    fn name(&self) -> &'static str {
        "foundry_a01_stop_v1"
    }

    fn observe(&self, snap: &SearchSnapshot, edges: &[EdgeView<'_>]) {
        let (identity, extras) = self.identity_and_extras(snap, edges);
        let observation = FoundryObservation {
            snap,
            edges,
            extras: &extras,
        };
        let axis = A01StopCouncil {
            risk_limit: self.config.risk_limit,
            min_visits: self.config.min_visits,
        };
        // Both treatments run the same coordinator and arbiter.  The shadow
        // arm discards the selected capability; only the active arm passes it
        // through the freshness guard and concrete STOP executor.
        let coordinator = FoundryCoordinator::with_axes_and_live_authorization(
            ConservativeArbiter,
            vec![Box::new(axis)],
            LiveActionAuthorization::for_axes([A01_AXIS_ID]),
        );
        let outcome = coordinator.coordinate(&observation, CostPrices::default());
        let mut state = self.state.lock();
        state.decisions += 1;
        state.stop_identity = None;
        let Ok(outcome) = outcome else {
            state.coordination_errors += 1;
            return;
        };
        state.proposals += outcome.proposal_count as u64;
        let Some(selected) = outcome.selected else {
            return;
        };
        if self.config.mode == FoundryRuntimeMode::Shadow {
            return;
        }
        let mut executor = GuardedMetaActionExecutor::new(StopExecutor);
        match executor.execute(selected, &identity) {
            Ok(true) => {
                state.actions += 1;
                state.stop_identity = Some(identity);
            }
            Ok(false) => {}
            Err(_) => state.coordination_errors += 1,
        }
    }

    fn score_adjustment(&self, _edge: EdgeView<'_>) -> ScoreAdjustment {
        ScoreAdjustment::default()
    }

    fn should_halt(&self, snap: &SearchSnapshot, _edges: &[EdgeView<'_>]) -> HaltDecision {
        let state = self.state.lock();
        if self.config.mode == FoundryRuntimeMode::Active
            && state.stop_identity.as_ref().is_some_and(|identity| {
                identity.root_visits == snap.root_visits && identity.iteration == snap.iteration
            })
        {
            HaltDecision::Stop(HaltReason::PolicyConverged)
        } else {
            HaltDecision::Continue
        }
    }

    fn telemetry(&self) -> ControllerTelemetry {
        let state = self.state.lock();
        ControllerTelemetry {
            schema_version: 1,
            policy_name: self.name().to_string(),
            halt_reason: (state.actions > 0).then(|| "policy_converged".to_string()),
            foundry_mode: Some(self.config.mode.as_str().to_string()),
            foundry_axis_ids: vec![self.config.axis_id.clone()],
            metacontroller_decisions: state.decisions,
            metacontroller_proposals: state.proposals,
            metacontroller_actions: state.actions,
            metacontroller_coordination_errors: state.coordination_errors,
            ..ControllerTelemetry::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(mode: FoundryRuntimeMode) -> FoundrySearchConfig {
        FoundrySearchConfig {
            mode,
            axis_id: A01_AXIS_ID.to_string(),
            checkpoint_id: "fixture-checkpoint".to_string(),
            evaluator_id: "fixture-evaluator".to_string(),
            risk_limit: 0.50,
            min_visits: 1,
        }
    }

    fn snapshot() -> SearchSnapshot {
        SearchSnapshot {
            root_visits: 64,
            n_children: 2,
            n_visible: 2,
            elapsed_ms: 1,
            depth_max: 1,
            mean_q_root: 0.5,
            sigma_q_root: 0.1,
            sigma_eval: None,
            iteration: 64,
            best_idx: 0,
            second_idx: 1,
        }
    }

    #[test]
    fn shadow_observes_but_never_executes() {
        let policy = FoundrySearchPolicy::new(config(FoundryRuntimeMode::Shadow), 7);
        let snap = snapshot();
        let total = snap.root_visits;
        let edges = [
            EdgeView {
                idx: 0,
                n: 60,
                n_virtual: 0,
                o_a: 0,
                q: 0.9,
                q_sum: 54.0,
                m2: 0.01,
                prior: 0.5,
                depth: 1,
                last_value: 0.9,
                envar_partial: 0.0,
                root_total_n: &total,
                stats: &snap,
            },
            EdgeView {
                idx: 1,
                n: 4,
                n_virtual: 0,
                o_a: 0,
                q: -0.5,
                q_sum: -2.0,
                m2: 0.01,
                prior: 0.5,
                depth: 1,
                last_value: -0.5,
                envar_partial: 0.0,
                root_total_n: &total,
                stats: &snap,
            },
        ];
        policy.observe(&snap, &edges);
        assert!(matches!(
            policy.should_halt(&snap, &edges),
            HaltDecision::Continue
        ));
        let telemetry = policy.telemetry();
        assert_eq!(telemetry.metacontroller_decisions, 1);
        assert_eq!(telemetry.metacontroller_actions, 0);
    }

    #[test]
    fn active_executes_selected_fresh_stop() {
        let policy = FoundrySearchPolicy::new(config(FoundryRuntimeMode::Active), 7);
        let snap = snapshot();
        let total = snap.root_visits;
        let edges = [
            EdgeView {
                idx: 0,
                n: 60,
                n_virtual: 0,
                o_a: 0,
                q: 0.9,
                q_sum: 54.0,
                m2: 0.01,
                prior: 0.5,
                depth: 1,
                last_value: 0.9,
                envar_partial: 0.0,
                root_total_n: &total,
                stats: &snap,
            },
            EdgeView {
                idx: 1,
                n: 4,
                n_virtual: 0,
                o_a: 0,
                q: -0.5,
                q_sum: -2.0,
                m2: 0.01,
                prior: 0.5,
                depth: 1,
                last_value: -0.5,
                envar_partial: 0.0,
                root_total_n: &total,
                stats: &snap,
            },
        ];
        policy.observe(&snap, &edges);
        assert!(matches!(
            policy.should_halt(&snap, &edges),
            HaltDecision::Stop(HaltReason::PolicyConverged)
        ));
        let telemetry = policy.telemetry();
        assert_eq!(telemetry.metacontroller_decisions, 1);
        assert_eq!(telemetry.metacontroller_actions, 1);
        assert_eq!(telemetry.metacontroller_coordination_errors, 0);
    }

    #[test]
    fn trace_stability_variant_rejected_by_live_search_config() {
        let mut cfg = config(FoundryRuntimeMode::Active);
        cfg.axis_id = A01_TRACE_STABILITY_AXIS_ID.to_string();
        assert!(!cfg.is_valid(), "trace stability must not be accepted as live search policy");
        cfg.axis_id = A01_LIVE_PFLIP_AXIS_ID.to_string();
        assert!(cfg.is_valid(), "canonical live_pflip must be accepted");
    }
}
