//! Single-action coordination and guarded execution for Idea Foundry lanes.
//!
//! This module is available only with the crate's `idea-foundry` feature.  It
//! does not hook itself into the production search loop.  Experiment code must
//! explicitly construct a coordinator, obtain at most one decision, and pass
//! that decision through the freshness guard before any executor sees it.

use std::collections::{BTreeMap, BTreeSet};

use crate::mcts::policy::PolicyCachePublisher;

use super::types::{
    AxisStatus, CostPrices, FoundryArbiter, FoundryAxis, FoundryObservation, FreshnessIdentity,
    MetaAction, MetaActionKind, MetaProposal,
};

pub const EFFECTIVE_PRIOR_SCORE_VECTOR_TELEMETRY_KEY: &str = "effective_prior_score_vector";

fn bound_effective_prior(proposal: &MetaProposal) -> Result<Option<Vec<f32>>, ()> {
    let Some(value) = proposal
        .telemetry
        .get(EFFECTIVE_PRIOR_SCORE_VECTOR_TELEMETRY_KEY)
    else {
        return Ok(None);
    };
    let scores: Vec<f32> = serde_json::from_value(value.clone()).map_err(|_| ())?;
    if scores.is_empty()
        || scores
            .iter()
            .any(|score| !score.is_finite() || *score < 0.0)
    {
        return Err(());
    }
    let sum: f32 = scores.iter().sum();
    if (sum - 1.0).abs() > 1e-4 {
        return Err(());
    }
    Ok(Some(scores))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoordinationError {
    ObservationIdentityMismatch,
    DuplicateAxisId(String),
    AxisIdMismatch {
        expected: String,
        actual: String,
    },
    InvalidProposal {
        axis_id: String,
    },
    InvalidScoreVectorBinding {
        axis_id: String,
    },
    InactiveAxisProposedAction {
        axis_id: String,
        status: AxisStatus,
        action: MetaActionKind,
    },
    UnauthorizedActiveAction {
        axis_id: String,
        status: AxisStatus,
        action: MetaActionKind,
    },
}

/// Explicit live-action authority, independent of scientific evidence status.
///
/// The default is deny-all.  Enabling the Cargo feature, constructing a
/// coordinator, or assigning an axis a stronger evidence status never grants
/// execution authority.  The caller must pass an explicit axis allowlist at
/// the live campaign boundary.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LiveActionAuthorization {
    allowed_axis_ids: BTreeSet<String>,
}

impl LiveActionAuthorization {
    pub fn deny_all() -> Self {
        Self::default()
    }

    pub fn for_axes<I, S>(axis_ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            allowed_axis_ids: axis_ids.into_iter().map(Into::into).collect(),
        }
    }

    fn allows(&self, axis_id: &str) -> bool {
        self.allowed_axis_ids.contains(axis_id)
    }
}

/// The sole proposal selected for one checkpoint, bound to the exact evidence
/// identity from which it was produced.  This capability is intentionally
/// non-`Clone`; execution and score publication consume it by value so the
/// same selected decision cannot be replayed.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SelectedDecisionProof;

#[derive(Debug, PartialEq)]
pub struct CoordinatedProposal {
    proposal: MetaProposal,
    evidence_identity: FreshnessIdentity,
    axis_status: AxisStatus,
    live_action_authorized: bool,
    selection_proof: Option<SelectedDecisionProof>,
    bound_effective_prior: Option<Vec<f32>>,
}

impl CoordinatedProposal {
    pub fn proposal(&self) -> &MetaProposal {
        &self.proposal
    }

    pub fn evidence_identity(&self) -> &FreshnessIdentity {
        &self.evidence_identity
    }
}

#[derive(Debug, PartialEq)]
pub struct CoordinationOutcome {
    pub proposal_count: usize,
    pub selected: Option<CoordinatedProposal>,
}

pub struct FoundryCoordinator<A> {
    axes: Vec<Box<dyn FoundryAxis>>,
    arbiter: A,
    live_authorization: LiveActionAuthorization,
}

impl<A: FoundryArbiter> FoundryCoordinator<A> {
    pub fn new(arbiter: A) -> Self {
        Self {
            axes: Vec::new(),
            arbiter,
            live_authorization: LiveActionAuthorization::deny_all(),
        }
    }

    pub fn with_axes(arbiter: A, axes: Vec<Box<dyn FoundryAxis>>) -> Self {
        Self {
            axes,
            arbiter,
            live_authorization: LiveActionAuthorization::deny_all(),
        }
    }

    pub fn with_axes_and_live_authorization(
        arbiter: A,
        axes: Vec<Box<dyn FoundryAxis>>,
        live_authorization: LiveActionAuthorization,
    ) -> Self {
        Self {
            axes,
            arbiter,
            live_authorization,
        }
    }

    pub fn push_axis(&mut self, axis: Box<dyn FoundryAxis>) {
        self.axes.push(axis);
    }

    pub fn coordinate(
        &self,
        observation: &FoundryObservation<'_>,
        prices: CostPrices,
    ) -> Result<CoordinationOutcome, CoordinationError> {
        if !observation
            .extras
            .identity_matches_snapshot(observation.snap)
        {
            return Err(CoordinationError::ObservationIdentityMismatch);
        }

        let mut seen_axis_ids = BTreeSet::new();
        let mut axis_statuses = BTreeMap::new();
        let mut proposals = Vec::new();
        for axis in &self.axes {
            let axis_id = axis.id();
            if !seen_axis_ids.insert(axis_id) {
                return Err(CoordinationError::DuplicateAxisId(axis_id.to_string()));
            }

            let status = axis.status();
            axis_statuses.insert(axis_id, status);
            let start = proposals.len();
            axis.propose(observation, &mut proposals);
            for proposal in &proposals[start..] {
                if proposal.axis_id != axis_id {
                    return Err(CoordinationError::AxisIdMismatch {
                        expected: axis_id.to_string(),
                        actual: proposal.axis_id.clone(),
                    });
                }
                if !proposal.is_contract_valid() {
                    return Err(CoordinationError::InvalidProposal {
                        axis_id: axis_id.to_string(),
                    });
                }
                if bound_effective_prior(proposal).is_err() {
                    return Err(CoordinationError::InvalidScoreVectorBinding {
                        axis_id: axis_id.to_string(),
                    });
                }
                if matches!(status, AxisStatus::AnalysisOnly | AxisStatus::Dormant)
                    && proposal.action.kind() != MetaActionKind::Noop
                {
                    return Err(CoordinationError::InactiveAxisProposedAction {
                        axis_id: axis_id.to_string(),
                        status,
                        action: proposal.action.kind(),
                    });
                }
                if proposal.action.kind() != MetaActionKind::Noop
                    && !self.live_authorization.allows(axis_id)
                {
                    return Err(CoordinationError::UnauthorizedActiveAction {
                        axis_id: axis_id.to_string(),
                        status,
                        action: proposal.action.kind(),
                    });
                }
            }
        }

        let selected = self
            .arbiter
            .choose(&proposals, prices)
            .cloned()
            .map(|proposal| CoordinatedProposal {
                live_action_authorized: proposal.action.kind() == MetaActionKind::Noop
                    || self.live_authorization.allows(&proposal.axis_id),
                axis_status: *axis_statuses
                    .get(proposal.axis_id.as_str())
                    .expect("validated proposal axis has a status"),
                bound_effective_prior: bound_effective_prior(&proposal)
                    .expect("score-vector binding was validated before arbitration"),
                proposal,
                evidence_identity: observation.extras.freshness.clone(),
                selection_proof: Some(SelectedDecisionProof),
            });
        Ok(CoordinationOutcome {
            proposal_count: proposals.len(),
            selected,
        })
    }
}

/// Backend-specific execution boundary.  Implementations translate an
/// accepted proposal to a root-session command, scheduler update, or immutable
/// `PolicyCachePublisher` publish; they must not mutate PUCT from an axis.
pub trait MetaActionExecutor {
    type Output;
    type Error;

    fn execute(&mut self, proposal: &MetaProposal) -> Result<Self::Output, Self::Error>;
}

/// Fail-closed errors from the only supported bridge between a Foundry
/// score-vector experiment and the production policy read path.
#[derive(Clone, Debug, PartialEq)]
pub enum ScoreVectorPublishError {
    UnauthorizedAxis {
        axis_id: String,
    },
    DecisionNotSelected,
    InvalidSelectedDecision,
    DecisionAxisMismatch {
        authorized_axis_id: String,
        selected_axis_id: String,
    },
    SelectedDecisionUnauthorized {
        axis_id: String,
    },
    InactiveSelectedAxis {
        axis_id: String,
        status: AxisStatus,
    },
    SelectedDecisionHasNoScoreVector,
    SelectedScoreVectorBindingMismatch,
    InvalidEvidenceIdentity,
    InvalidCurrentIdentity,
    StaleEvidence,
    EmptyVector,
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidScoreVector,
    NotNormalized {
        sum: f32,
    },
    UnexpectedCacheEpoch {
        expected: u64,
        actual: u64,
    },
    StalePolicyCache {
        cache_root_visits: u32,
        current_root_visits: u32,
        cache_edge_version_hash: u64,
        current_edge_version_hash: u64,
    },
}

/// Feature-gated adapter for publishing a Foundry effective-prior score
/// vector through the existing immutable [`PolicyCachePublisher`].
///
/// This type deliberately exposes no node, edge, or PUCT mutation API.  A
/// caller must first present explicit live authority for the producing axis,
/// then consume the matching one-shot decision selected by the coordinator.
/// The applied vector is the immutable vector bound into that proposal before
/// arbitration; callers cannot substitute bytes after selection.  A live
/// current-identity supplier, observed cache epoch, and numeric edge-version
/// hash are revalidated by a CAS loop. Rejection leaves the winning cache
/// untouched. The vector is edge-position ordered; action identifiers are
/// never accepted as indices.
pub struct PolicyCacheScoreVectorPublisher<'a> {
    publisher: &'a PolicyCachePublisher,
    authorized_axis_id: String,
}

impl<'a> PolicyCacheScoreVectorPublisher<'a> {
    pub fn with_live_authorization(
        publisher: &'a PolicyCachePublisher,
        axis_id: &str,
        live_authorization: &LiveActionAuthorization,
    ) -> Result<Self, ScoreVectorPublishError> {
        if !live_authorization.allows(axis_id) {
            return Err(ScoreVectorPublishError::UnauthorizedAxis {
                axis_id: axis_id.to_string(),
            });
        }
        Ok(Self {
            publisher,
            authorized_axis_id: axis_id.to_string(),
        })
    }

    /// Atomically replace only `PolicyCache::p_eff`, preserving every other
    /// cache channel.  The score vector is interpreted as a probability
    /// vector and therefore must be finite, non-negative, and normalized.
    pub fn publish_effective_prior<F>(
        &self,
        selected_decision: CoordinatedProposal,
        current_identity: F,
        expected_cache_epoch: u64,
        current_edge_version_hash: u64,
        expected_edge_count: usize,
    ) -> Result<u64, ScoreVectorPublishError>
    where
        F: FnMut() -> FreshnessIdentity,
    {
        self.publish_effective_prior_with_hook(
            selected_decision,
            current_identity,
            expected_cache_epoch,
            current_edge_version_hash,
            expected_edge_count,
            |_| {},
        )
    }

    fn publish_effective_prior_with_hook<F, H>(
        &self,
        selected_decision: CoordinatedProposal,
        mut current_identity: F,
        expected_cache_epoch: u64,
        current_edge_version_hash: u64,
        expected_edge_count: usize,
        mut before_compare_exchange: H,
    ) -> Result<u64, ScoreVectorPublishError>
    where
        F: FnMut() -> FreshnessIdentity,
        H: FnMut(usize),
    {
        if selected_decision.selection_proof.is_none() {
            return Err(ScoreVectorPublishError::DecisionNotSelected);
        }
        if !selected_decision.proposal.is_contract_valid() {
            return Err(ScoreVectorPublishError::InvalidSelectedDecision);
        }
        if selected_decision.proposal.axis_id != self.authorized_axis_id {
            return Err(ScoreVectorPublishError::DecisionAxisMismatch {
                authorized_axis_id: self.authorized_axis_id.clone(),
                selected_axis_id: selected_decision.proposal.axis_id.clone(),
            });
        }
        if matches!(
            selected_decision.axis_status,
            AxisStatus::AnalysisOnly | AxisStatus::Dormant
        ) {
            return Err(ScoreVectorPublishError::InactiveSelectedAxis {
                axis_id: selected_decision.proposal.axis_id.clone(),
                status: selected_decision.axis_status,
            });
        }
        if selected_decision.proposal.action.kind() != MetaActionKind::Noop
            && !selected_decision.live_action_authorized
        {
            return Err(ScoreVectorPublishError::SelectedDecisionUnauthorized {
                axis_id: selected_decision.proposal.axis_id.clone(),
            });
        }
        let rebound = bound_effective_prior(&selected_decision.proposal)
            .map_err(|_| ScoreVectorPublishError::SelectedScoreVectorBindingMismatch)?;
        if rebound != selected_decision.bound_effective_prior {
            return Err(ScoreVectorPublishError::SelectedScoreVectorBindingMismatch);
        }
        let scores_by_edge_pos = selected_decision
            .bound_effective_prior
            .as_deref()
            .ok_or(ScoreVectorPublishError::SelectedDecisionHasNoScoreVector)?;
        let evidence_identity = &selected_decision.evidence_identity;
        if expected_edge_count == 0 {
            return Err(ScoreVectorPublishError::EmptyVector);
        }
        if scores_by_edge_pos.len() != expected_edge_count {
            return Err(ScoreVectorPublishError::DimensionMismatch {
                expected: expected_edge_count,
                actual: scores_by_edge_pos.len(),
            });
        }
        if scores_by_edge_pos
            .iter()
            .any(|score| !score.is_finite() || *score < 0.0)
        {
            return Err(ScoreVectorPublishError::InvalidScoreVector);
        }
        let sum: f32 = scores_by_edge_pos.iter().sum();
        if (sum - 1.0).abs() > 1e-4 {
            return Err(ScoreVectorPublishError::NotNormalized { sum });
        }

        let mut attempt = 0;
        loop {
            // Revalidate identity and the observed epoch on every CAS attempt.
            let current_identity = current_identity();
            if !evidence_identity.is_well_formed() {
                return Err(ScoreVectorPublishError::InvalidEvidenceIdentity);
            }
            if !current_identity.is_well_formed() {
                return Err(ScoreVectorPublishError::InvalidCurrentIdentity);
            }
            if evidence_identity != &current_identity {
                return Err(ScoreVectorPublishError::StaleEvidence);
            }

            let observed = self.publisher.snapshot();
            if observed.epoch != expected_cache_epoch {
                return Err(ScoreVectorPublishError::UnexpectedCacheEpoch {
                    expected: expected_cache_epoch,
                    actual: observed.epoch,
                });
            }
            if observed.root_visits != current_identity.root_visits
                || observed.edge_version_hash != current_edge_version_hash
            {
                return Err(ScoreVectorPublishError::StalePolicyCache {
                    cache_root_visits: observed.root_visits,
                    current_root_visits: current_identity.root_visits,
                    cache_edge_version_hash: observed.edge_version_hash,
                    current_edge_version_hash,
                });
            }
            if !observed.p_eff.is_empty() && observed.p_eff.len() != expected_edge_count {
                return Err(ScoreVectorPublishError::DimensionMismatch {
                    expected: expected_edge_count,
                    actual: observed.p_eff.len(),
                });
            }

            let mut replacement = (*observed).clone();
            replacement.p_eff = scores_by_edge_pos.iter().copied().collect();
            before_compare_exchange(attempt);
            attempt += 1;
            match self.publisher.compare_and_store(&observed, replacement) {
                Ok(epoch) => return Ok(epoch),
                Err(_newer_cache) => continue,
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GuardedExecutionError<E> {
    DecisionNotSelected,
    InvalidProposal,
    UnauthorizedAction {
        action: MetaActionKind,
    },
    InvalidCurrentIdentity,
    InvalidEvidenceIdentity,
    StaleEvidence {
        action: MetaActionKind,
        evidence: FreshnessIdentity,
        current: FreshnessIdentity,
    },
    Executor(E),
}

/// Freshness-checking wrapper for any concrete experimental executor.
pub struct GuardedMetaActionExecutor<E> {
    inner: E,
}

impl<E> GuardedMetaActionExecutor<E> {
    pub fn new(inner: E) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &E {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }

    pub fn into_inner(self) -> E {
        self.inner
    }
}

impl<E: MetaActionExecutor> GuardedMetaActionExecutor<E> {
    pub fn execute(
        &mut self,
        decision: CoordinatedProposal,
        current_identity: &FreshnessIdentity,
    ) -> Result<E::Output, GuardedExecutionError<E::Error>> {
        if decision.selection_proof.is_none() {
            return Err(GuardedExecutionError::DecisionNotSelected);
        }
        if !decision.proposal.is_contract_valid() {
            return Err(GuardedExecutionError::InvalidProposal);
        }
        if decision.proposal.action.kind() != MetaActionKind::Noop
            && !decision.live_action_authorized
        {
            return Err(GuardedExecutionError::UnauthorizedAction {
                action: decision.proposal.action.kind(),
            });
        }
        if !current_identity.is_well_formed() {
            return Err(GuardedExecutionError::InvalidCurrentIdentity);
        }
        if !decision.evidence_identity.is_well_formed() {
            return Err(GuardedExecutionError::InvalidEvidenceIdentity);
        }
        if decision.proposal.action.requires_exact_freshness()
            && decision.evidence_identity != *current_identity
        {
            return Err(GuardedExecutionError::StaleEvidence {
                action: decision.proposal.action.kind(),
                evidence: decision.evidence_identity.clone(),
                current: current_identity.clone(),
            });
        }
        self.inner
            .execute(&decision.proposal)
            .map_err(GuardedExecutionError::Executor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::foundry::control::A24LearnedBudgetGate;
    use crate::mcts::foundry::types::{
        ConservativeArbiter, FoundryRootExtras, ProposalEstimate, RuntimeSnapshot,
        FOUNDRY_CONTRACT_SCHEMA_VERSION,
    };
    use crate::mcts::policy::{PolicyCache, PolicyCachePublisher, SearchSnapshot};
    use smallvec::SmallVec;
    use std::cell::Cell;
    use std::sync::{Arc, Barrier};

    fn identity() -> FreshnessIdentity {
        FreshnessIdentity {
            root_hash: 11,
            checkpoint_id: "seed_1/gen_1".into(),
            evaluator_id: "model-sha256:abc".into(),
            edge_set_hash: "edge-sha256:def".into(),
            candidate_epoch: 3,
            tt_identity_policy: "state-cache-only-v1".into(),
            cache_schema_version: 1,
            root_visits: 24,
            iteration: 25,
        }
    }

    fn snapshot() -> SearchSnapshot {
        SearchSnapshot {
            root_visits: 24,
            n_children: 0,
            n_visible: 0,
            elapsed_ms: 12,
            depth_max: 0,
            mean_q_root: 0.0,
            sigma_q_root: 0.1,
            sigma_eval: None,
            iteration: 25,
            best_idx: 0,
            second_idx: 0,
        }
    }

    fn extras() -> FoundryRootExtras {
        FoundryRootExtras {
            schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
            freshness: identity(),
            runtime: RuntimeSnapshot::default(),
            ..FoundryRootExtras::default()
        }
    }

    fn publisher_with_fresh_cache() -> PolicyCachePublisher {
        let publisher = PolicyCachePublisher::new();
        let mut cache = PolicyCache::empty();
        cache.root_visits = identity().root_visits;
        cache.edge_version_hash = 77;
        cache.p_eff = SmallVec::from_slice(&[0.5, 0.5]);
        cache.q_ctrl = SmallVec::from_slice(&[0.1, 0.2]);
        cache.penalty = SmallVec::from_slice(&[-0.1, 0.0]);
        publisher.store(cache);
        publisher
    }

    fn authorized_score_bridge(
        publisher: &PolicyCachePublisher,
    ) -> PolicyCacheScoreVectorPublisher<'_> {
        authorized_score_bridge_for(publisher, "A02.live")
    }

    fn authorized_score_bridge_for<'a>(
        publisher: &'a PolicyCachePublisher,
        axis_id: &str,
    ) -> PolicyCacheScoreVectorPublisher<'a> {
        PolicyCacheScoreVectorPublisher::with_live_authorization(
            publisher,
            axis_id,
            &LiveActionAuthorization::for_axes([axis_id]),
        )
        .expect("explicit live authority")
    }

    struct FixedAxis {
        id: &'static str,
        status: AxisStatus,
        action: MetaAction,
        net_lcb: f64,
        effective_prior: Option<Vec<f32>>,
    }

    impl FoundryAxis for FixedAxis {
        fn id(&self) -> &'static str {
            self.id
        }

        fn status(&self) -> AxisStatus {
            self.status
        }

        fn propose(&self, _observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
            let mut proposal = MetaProposal::new(
                self.id,
                self.action.clone(),
                ProposalEstimate {
                    regret_reduction_lcb: self.net_lcb,
                    confidence: 0.5,
                    ..ProposalEstimate::default()
                },
                "unit-test guard",
                "unit-test proposal",
            );
            if let Some(scores) = &self.effective_prior {
                proposal.telemetry.insert(
                    EFFECTIVE_PRIOR_SCORE_VECTOR_TELEMETRY_KEY.to_string(),
                    serde_json::json!(scores),
                );
            }
            out.push(proposal);
        }
    }

    fn selected_score_decision(axis_id: &'static str) -> CoordinatedProposal {
        selected_score_decision_with_vector(axis_id, AxisStatus::Shadow, vec![0.25, 0.75])
    }

    fn selected_score_decision_with_status(
        axis_id: &'static str,
        status: AxisStatus,
    ) -> CoordinatedProposal {
        selected_score_decision_with_vector(axis_id, status, vec![0.25, 0.75])
    }

    fn selected_score_decision_with_vector(
        axis_id: &'static str,
        status: AxisStatus,
        effective_prior: Vec<f32>,
    ) -> CoordinatedProposal {
        let snap = snapshot();
        let extras = extras();
        let observation = FoundryObservation {
            snap: &snap,
            edges: &[],
            extras: &extras,
        };
        FoundryCoordinator::with_axes(
            ConservativeArbiter,
            vec![Box::new(FixedAxis {
                id: axis_id,
                status,
                action: MetaAction::Noop,
                net_lcb: 1.0,
                effective_prior: Some(effective_prior),
            })],
        )
        .coordinate(&observation, CostPrices::default())
        .expect("coordinator accepts selected score proposal")
        .selected
        .expect("arbiter selects positive score proposal")
    }

    #[test]
    fn coordinator_selects_at_most_one_proposal() {
        let snap = snapshot();
        let extras = extras();
        let observation = FoundryObservation {
            snap: &snap,
            edges: &[],
            extras: &extras,
        };
        let coordinator = FoundryCoordinator::with_axes_and_live_authorization(
            ConservativeArbiter,
            vec![
                Box::new(FixedAxis {
                    id: "A04.low",
                    status: AxisStatus::Shadow,
                    action: MetaAction::Sample {
                        edge_pos: 0,
                        visits: 1,
                    },
                    net_lcb: 1.0,
                    effective_prior: None,
                }),
                Box::new(FixedAxis {
                    id: "A06.high",
                    status: AxisStatus::MechanismValid,
                    action: MetaAction::Sample {
                        edge_pos: 1,
                        visits: 1,
                    },
                    net_lcb: 2.0,
                    effective_prior: None,
                }),
            ],
            LiveActionAuthorization::for_axes(["A04.low", "A06.high"]),
        );

        let outcome = coordinator
            .coordinate(&observation, CostPrices::default())
            .expect("coordinate");
        assert_eq!(outcome.proposal_count, 2);
        let selected = outcome.selected.expect("one selection");
        assert_eq!(selected.proposal.axis_id, "A06.high");
        assert_eq!(selected.evidence_identity, identity());
    }

    #[test]
    fn coordinator_rejects_stale_observation_extras_before_axes_run() {
        let snap = snapshot();
        let mut extras = extras();
        extras.freshness.root_visits += 1;
        let observation = FoundryObservation {
            snap: &snap,
            edges: &[],
            extras: &extras,
        };
        let coordinator = FoundryCoordinator::new(ConservativeArbiter);
        assert_eq!(
            coordinator.coordinate(&observation, CostPrices::default()),
            Err(CoordinationError::ObservationIdentityMismatch)
        );
    }

    #[test]
    fn coordinator_rejects_active_action_from_inactive_axis_even_when_authorized() {
        let snap = snapshot();
        let extras = extras();
        let observation = FoundryObservation {
            snap: &snap,
            edges: &[],
            extras: &extras,
        };
        for status in [AxisStatus::AnalysisOnly, AxisStatus::Dormant] {
            let coordinator = FoundryCoordinator::with_axes_and_live_authorization(
                ConservativeArbiter,
                vec![Box::new(FixedAxis {
                    id: "A21.inactive",
                    status,
                    action: MetaAction::Stop { edge_pos: None },
                    net_lcb: 1.0,
                    effective_prior: None,
                })],
                LiveActionAuthorization::for_axes(["A21.inactive"]),
            );
            assert!(matches!(
                coordinator.coordinate(&observation, CostPrices::default()),
                Err(CoordinationError::InactiveAxisProposedAction { .. })
            ));
        }
    }

    #[test]
    fn evidence_status_never_grants_live_action_authority() {
        let snap = snapshot();
        let extras = extras();
        let observation = FoundryObservation {
            snap: &snap,
            edges: &[],
            extras: &extras,
        };
        for status in [
            AxisStatus::Seed,
            AxisStatus::MechanismValid,
            AxisStatus::Shadow,
            AxisStatus::Conditional,
            AxisStatus::ActiveExperimental,
            AxisStatus::DeploymentCandidate,
        ] {
            let coordinator = FoundryCoordinator::with_axes(
                ConservativeArbiter,
                vec![Box::new(FixedAxis {
                    id: "A04.unauthorized",
                    status,
                    action: MetaAction::Sample {
                        edge_pos: 0,
                        visits: 1,
                    },
                    net_lcb: 1.0,
                    effective_prior: None,
                })],
            );
            assert!(matches!(
                coordinator.coordinate(&observation, CostPrices::default()),
                Err(CoordinationError::UnauthorizedActiveAction { .. })
            ));
        }
    }

    #[test]
    fn noop_requires_no_live_authorization() {
        let snap = snapshot();
        let extras = extras();
        let observation = FoundryObservation {
            snap: &snap,
            edges: &[],
            extras: &extras,
        };
        let coordinator = FoundryCoordinator::with_axes(
            ConservativeArbiter,
            vec![Box::new(FixedAxis {
                id: "A03.trace",
                status: AxisStatus::Seed,
                action: MetaAction::Noop,
                net_lcb: 0.0,
                effective_prior: None,
            })],
        );
        let outcome = coordinator
            .coordinate(&observation, CostPrices::default())
            .expect("NOOP is not a live control action");
        assert_eq!(outcome.proposal_count, 1);
        assert!(outcome.selected.is_none());
    }

    #[test]
    fn a24_root_budget_candidates_are_strict_offline_noops() {
        let snap = snapshot();
        let extras = extras();
        let observation = FoundryObservation {
            snap: &snap,
            edges: &[],
            extras: &extras,
        };
        let axis = A24LearnedBudgetGate {
            budgets: vec![8, 32],
            cost_per_visit_ms: 0.1,
        };
        let mut proposals = Vec::new();
        axis.propose(&observation, &mut proposals);
        assert_eq!(proposals.len(), 2);
        for (proposal, expected_budget) in proposals.iter().zip([8_u32, 32]) {
            assert_eq!(proposal.action.kind(), MetaActionKind::Noop);
            assert_eq!(
                proposal.telemetry["offline_root_budget_visits"],
                expected_budget
            );
            assert_eq!(proposal.telemetry["offline_only"], true);
        }
    }

    #[derive(Default)]
    struct RecordingExecutor {
        calls: usize,
    }

    impl MetaActionExecutor for RecordingExecutor {
        type Output = MetaActionKind;
        type Error = ();

        fn execute(&mut self, proposal: &MetaProposal) -> Result<Self::Output, Self::Error> {
            self.calls += 1;
            Ok(proposal.action.kind())
        }
    }

    fn decision(action: MetaAction) -> CoordinatedProposal {
        CoordinatedProposal {
            proposal: MetaProposal::new(
                "A08.test",
                action,
                ProposalEstimate::default(),
                "freshness required",
                "test",
            ),
            evidence_identity: identity(),
            axis_status: AxisStatus::Conditional,
            live_action_authorized: true,
            selection_proof: Some(SelectedDecisionProof),
            bound_effective_prior: None,
        }
    }

    #[test]
    fn executor_rejects_stale_stop_and_prove_without_dispatch() {
        for action in [
            MetaAction::Stop { edge_pos: Some(0) },
            MetaAction::Prove {
                edge_pos: 0,
                budget: 64,
            },
        ] {
            let mut guarded = GuardedMetaActionExecutor::new(RecordingExecutor::default());
            let mut current = identity();
            current.evaluator_id = "model-sha256:different".into();
            let error = guarded
                .execute(decision(action), &current)
                .expect_err("stale certificate action must fail closed");
            assert!(matches!(error, GuardedExecutionError::StaleEvidence { .. }));
            assert_eq!(guarded.inner().calls, 0);
        }
    }

    #[test]
    fn stop_freshness_compares_every_identity_dimension() {
        let baseline = identity();
        let mut variants = Vec::new();
        let mut value = baseline.clone();
        value.root_hash += 1;
        variants.push(value);
        let mut value = baseline.clone();
        value.checkpoint_id = "seed_2/gen_1".into();
        variants.push(value);
        let mut value = baseline.clone();
        value.evaluator_id = "model-sha256:different".into();
        variants.push(value);
        let mut value = baseline.clone();
        value.edge_set_hash = "edge-sha256:different".into();
        variants.push(value);
        let mut value = baseline.clone();
        value.candidate_epoch += 1;
        variants.push(value);
        let mut value = baseline.clone();
        value.tt_identity_policy = "graph-stat-sharing-v1".into();
        variants.push(value);
        let mut value = baseline.clone();
        value.cache_schema_version += 1;
        variants.push(value);
        let mut value = baseline.clone();
        value.root_visits += 1;
        variants.push(value);
        let mut value = baseline;
        value.iteration += 1;
        variants.push(value);

        for current in variants {
            let mut guarded = GuardedMetaActionExecutor::new(RecordingExecutor::default());
            assert!(matches!(
                guarded.execute(decision(MetaAction::Stop { edge_pos: Some(0) }), &current,),
                Err(GuardedExecutionError::StaleEvidence { .. })
            ));
            assert_eq!(guarded.inner().calls, 0);
        }
    }

    #[test]
    fn executor_dispatches_fresh_stop_once() {
        let mut guarded = GuardedMetaActionExecutor::new(RecordingExecutor::default());
        let kind = guarded
            .execute(
                decision(MetaAction::Stop { edge_pos: Some(0) }),
                &identity(),
            )
            .expect("fresh stop");
        assert_eq!(kind, MetaActionKind::Stop);
        assert_eq!(guarded.inner().calls, 1);
    }

    #[test]
    fn executor_api_consumes_one_shot_selected_capability() {
        let _consuming_signature: fn(
            &mut GuardedMetaActionExecutor<RecordingExecutor>,
            CoordinatedProposal,
            &FreshnessIdentity,
        ) -> Result<MetaActionKind, GuardedExecutionError<()>> =
            GuardedMetaActionExecutor::<RecordingExecutor>::execute;

        let mut guarded = GuardedMetaActionExecutor::new(RecordingExecutor::default());
        let one_shot = decision(MetaAction::Stop { edge_pos: Some(0) });
        guarded
            .execute(one_shot, &identity())
            .expect("owned capability dispatches once");
        // A second call with `one_shot` is a compile-time use-after-move and
        // CoordinatedProposal intentionally does not implement Clone.
        assert_eq!(guarded.inner().calls, 1);
    }

    #[test]
    fn executor_rejects_missing_selection_proof() {
        let mut guarded = GuardedMetaActionExecutor::new(RecordingExecutor::default());
        let mut unselected = decision(MetaAction::Stop { edge_pos: Some(0) });
        unselected.selection_proof = None;
        assert_eq!(
            guarded.execute(unselected, &identity()),
            Err(GuardedExecutionError::DecisionNotSelected)
        );
        assert_eq!(guarded.inner().calls, 0);
    }

    #[test]
    fn executor_rejects_invalid_manually_constructed_proposal() {
        let mut guarded = GuardedMetaActionExecutor::new(RecordingExecutor::default());
        let mut invalid = decision(MetaAction::Stop { edge_pos: Some(0) });
        invalid.proposal.evidence_scope.clear();
        assert_eq!(
            guarded.execute(invalid, &identity()),
            Err(GuardedExecutionError::InvalidProposal)
        );
        assert_eq!(guarded.inner().calls, 0);
    }

    #[test]
    fn executor_rejects_active_decision_without_coordinator_authority() {
        let mut guarded = GuardedMetaActionExecutor::new(RecordingExecutor::default());
        let mut unauthorized = decision(MetaAction::Sample {
            edge_pos: 0,
            visits: 1,
        });
        unauthorized.live_action_authorized = false;
        assert_eq!(
            guarded.execute(unauthorized, &identity()),
            Err(GuardedExecutionError::UnauthorizedAction {
                action: MetaActionKind::Sample,
            })
        );
        assert_eq!(guarded.inner().calls, 0);
    }

    #[test]
    fn executor_rejects_stale_sample_without_dispatch() {
        let mut guarded = GuardedMetaActionExecutor::new(RecordingExecutor::default());
        let mut current = identity();
        current.candidate_epoch += 1;
        let error = guarded
            .execute(
                decision(MetaAction::Sample {
                    edge_pos: 0,
                    visits: 1,
                }),
                &current,
            )
            .expect_err("SAMPLE is root-bound and may not be rebased");
        assert!(matches!(error, GuardedExecutionError::StaleEvidence { .. }));
        assert_eq!(guarded.inner().calls, 0);
    }

    #[test]
    fn explicitly_identity_independent_system_actions_allow_stale_root() {
        for action in [
            MetaAction::SetBatch { batch_size: 8 },
            MetaAction::SetInflight { credit: 4 },
            MetaAction::SetThreads { threads: 2 },
        ] {
            let mut guarded = GuardedMetaActionExecutor::new(RecordingExecutor::default());
            let mut current = identity();
            current.candidate_epoch += 1;
            let expected_kind = action.kind();
            let kind = guarded
                .execute(decision(action), &current)
                .expect("scheduler-wide action is identity-independent");
            assert_eq!(kind, expected_kind);
            assert_eq!(guarded.inner().calls, 1);
        }
    }

    #[test]
    fn every_non_system_control_action_requires_exact_freshness() {
        let root_bound = [
            MetaAction::Stop { edge_pos: None },
            MetaAction::Sample {
                edge_pos: 0,
                visits: 1,
            },
            MetaAction::Challenge {
                best_pos: 0,
                challenger_pos: 1,
                visits: 1,
            },
            MetaAction::Widen { count: 1 },
            MetaAction::Deepen {
                edge_pos: 0,
                visits: 1,
            },
            MetaAction::Prove {
                edge_pos: 0,
                budget: 1,
            },
            MetaAction::ResampleMode {
                mode_id: 0,
                count: 1,
            },
            MetaAction::MergeOrShare { state_key: 1 },
            MetaAction::Reanalyse { state_key: 1 },
            MetaAction::ArchiveState { priority: 1.0 },
        ];
        assert!(root_bound.iter().all(MetaAction::requires_exact_freshness));

        let scheduler_wide = [
            MetaAction::SetBatch { batch_size: 1 },
            MetaAction::SetInflight { credit: 1 },
            MetaAction::SetThreads { threads: 1 },
        ];
        assert!(scheduler_wide
            .iter()
            .all(MetaAction::is_identity_independent_system_action));
        assert!(scheduler_wide
            .iter()
            .all(|action| !action.requires_exact_freshness()));
        assert!(!MetaAction::Noop.requires_exact_freshness());
    }

    #[test]
    fn score_vector_bridge_requires_explicit_live_axis_authority() {
        let publisher = publisher_with_fresh_cache();
        assert!(matches!(
            PolicyCacheScoreVectorPublisher::with_live_authorization(
                &publisher,
                "A02.live",
                &LiveActionAuthorization::deny_all(),
            ),
            Err(ScoreVectorPublishError::UnauthorizedAxis { .. })
        ));
    }

    #[test]
    fn score_vector_bridge_rejects_selected_decision_from_other_axis() {
        let publisher = publisher_with_fresh_cache();
        let before = publisher.snapshot();
        let bridge = authorized_score_bridge(&publisher);
        let selected = selected_score_decision("A03.trace");
        assert!(matches!(
            bridge.publish_effective_prior(selected, identity, before.epoch, 77, 2),
            Err(ScoreVectorPublishError::DecisionAxisMismatch { .. })
        ));
        assert_eq!(publisher.snapshot().epoch, before.epoch);
    }

    #[test]
    fn score_vector_bridge_rejects_analysis_axis_even_with_explicit_allowlist() {
        let publisher = publisher_with_fresh_cache();
        let before = publisher.snapshot();
        let bridge = authorized_score_bridge_for(&publisher, "A21.analysis");
        let selected =
            selected_score_decision_with_status("A21.analysis", AxisStatus::AnalysisOnly);
        assert_eq!(
            bridge.publish_effective_prior(selected, identity, before.epoch, 77, 2),
            Err(ScoreVectorPublishError::InactiveSelectedAxis {
                axis_id: "A21.analysis".into(),
                status: AxisStatus::AnalysisOnly,
            })
        );
        assert_eq!(publisher.snapshot().epoch, before.epoch);
    }

    #[test]
    fn score_vector_bridge_rejects_unselected_direct_publish() {
        let publisher = publisher_with_fresh_cache();
        let before = publisher.snapshot();
        let bridge = authorized_score_bridge(&publisher);
        let mut unselected = selected_score_decision("A02.live");
        unselected.selection_proof = None;
        assert_eq!(
            bridge.publish_effective_prior(unselected, identity, before.epoch, 77, 2),
            Err(ScoreVectorPublishError::DecisionNotSelected)
        );
        assert_eq!(publisher.snapshot().epoch, before.epoch);
    }

    #[test]
    fn score_vector_bridge_rejects_post_selection_vector_substitution() {
        let publisher = publisher_with_fresh_cache();
        let before = publisher.snapshot();
        let bridge = authorized_score_bridge(&publisher);
        let mut substituted = selected_score_decision("A02.live");
        substituted.proposal.telemetry.insert(
            EFFECTIVE_PRIOR_SCORE_VECTOR_TELEMETRY_KEY.to_string(),
            serde_json::json!([0.9, 0.1]),
        );
        assert_eq!(
            bridge.publish_effective_prior(substituted, identity, before.epoch, 77, 2),
            Err(ScoreVectorPublishError::SelectedScoreVectorBindingMismatch)
        );
        assert_eq!(publisher.snapshot().epoch, before.epoch);
        assert_eq!(publisher.snapshot().p_eff, before.p_eff);
    }

    #[test]
    fn coordinator_rejects_invalid_score_vector_before_arbitration() {
        let snap = snapshot();
        let extras = extras();
        let observation = FoundryObservation {
            snap: &snap,
            edges: &[],
            extras: &extras,
        };
        let coordinator = FoundryCoordinator::with_axes(
            ConservativeArbiter,
            vec![Box::new(FixedAxis {
                id: "A02.invalid",
                status: AxisStatus::Shadow,
                action: MetaAction::Noop,
                net_lcb: 1.0,
                effective_prior: Some(vec![0.2, 0.2]),
            })],
        );
        assert!(matches!(
            coordinator.coordinate(&observation, CostPrices::default()),
            Err(CoordinationError::InvalidScoreVectorBinding { .. })
        ));
    }

    #[test]
    fn score_vector_bridge_publishes_only_through_immutable_policy_cache() {
        let publisher = publisher_with_fresh_cache();
        let before = publisher.snapshot();
        let bridge = authorized_score_bridge(&publisher);
        let selected = selected_score_decision("A02.live");

        let epoch = bridge
            .publish_effective_prior(selected, identity, before.epoch, 77, 2)
            .expect("fresh normalized score vector");
        let after = publisher.snapshot();

        assert_eq!(epoch, before.epoch + 1);
        assert_eq!(after.epoch, epoch);
        assert_eq!(after.p_eff.as_slice(), &[0.25, 0.75]);
        assert_eq!(after.q_ctrl, before.q_ctrl);
        assert_eq!(after.penalty, before.penalty);
        assert_eq!(after.root_visits, before.root_visits);
        assert_eq!(after.edge_version_hash, before.edge_version_hash);
    }

    #[test]
    fn score_vector_bridge_rejects_stale_identity_without_publish() {
        let publisher = publisher_with_fresh_cache();
        let before = publisher.snapshot();
        let bridge = authorized_score_bridge(&publisher);
        let selected = selected_score_decision("A02.live");
        let mut current = identity();
        current.candidate_epoch += 1;

        assert_eq!(
            bridge.publish_effective_prior(selected, || current.clone(), before.epoch, 77, 2,),
            Err(ScoreVectorPublishError::StaleEvidence)
        );
        let after = publisher.snapshot();
        assert_eq!(after.epoch, before.epoch);
        assert_eq!(after.p_eff, before.p_eff);
    }

    #[test]
    fn score_vector_bridge_rejects_invalid_shape_values_and_stale_cache() {
        let publisher = publisher_with_fresh_cache();
        let before_epoch = publisher.snapshot().epoch;
        let bridge = authorized_score_bridge(&publisher);
        let selected =
            selected_score_decision_with_vector("A02.live", AxisStatus::Shadow, vec![1.0]);

        assert!(matches!(
            bridge.publish_effective_prior(selected, identity, before_epoch, 77, 2),
            Err(ScoreVectorPublishError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            bridge.publish_effective_prior(
                selected_score_decision("A02.live"),
                identity,
                before_epoch,
                78,
                2,
            ),
            Err(ScoreVectorPublishError::StalePolicyCache { .. })
        ));
        assert_eq!(publisher.snapshot().epoch, before_epoch);
    }

    #[test]
    fn score_vector_bridge_cas_preserves_concurrent_observe_publish() {
        let publisher = publisher_with_fresh_cache();
        let before = publisher.snapshot();
        let bridge = authorized_score_bridge(&publisher);
        let selected = selected_score_decision("A02.live");
        let mut hook_calls = 0;

        let result = bridge.publish_effective_prior_with_hook(
            selected,
            identity,
            before.epoch,
            77,
            2,
            |_| {
                hook_calls += 1;
                let mut concurrent = (*publisher.snapshot()).clone();
                concurrent.p_eff = SmallVec::from_slice(&[0.9, 0.1]);
                concurrent.q_ctrl = SmallVec::from_slice(&[0.8, 0.2]);
                publisher.store(concurrent);
            },
        );

        assert_eq!(hook_calls, 1);
        assert_eq!(
            result,
            Err(ScoreVectorPublishError::UnexpectedCacheEpoch {
                expected: before.epoch,
                actual: before.epoch + 1,
            })
        );
        let after = publisher.snapshot();
        assert_eq!(after.epoch, before.epoch + 1);
        assert_eq!(after.p_eff.as_slice(), &[0.9, 0.1]);
        assert_eq!(after.q_ctrl.as_slice(), &[0.8, 0.2]);
    }

    #[test]
    fn score_vector_bridge_reloads_live_identity_after_cas_interleaving() {
        let publisher = publisher_with_fresh_cache();
        let before = publisher.snapshot();
        let bridge = authorized_score_bridge(&publisher);
        let selected = selected_score_decision("A02.live");
        let identity_drifted = Cell::new(false);

        let result = bridge.publish_effective_prior_with_hook(
            selected,
            || {
                let mut current = identity();
                if identity_drifted.get() {
                    current.evaluator_id = "model-sha256:replacement".into();
                }
                current
            },
            before.epoch,
            77,
            2,
            |_| {
                identity_drifted.set(true);
                let concurrent = (*publisher.snapshot()).clone();
                publisher.store(concurrent);
            },
        );

        assert_eq!(result, Err(ScoreVectorPublishError::StaleEvidence));
        assert_eq!(publisher.snapshot().epoch, before.epoch + 1);
        assert_eq!(publisher.snapshot().p_eff, before.p_eff);
    }

    #[test]
    fn normal_store_reservation_cannot_overwrite_foundry_cas_or_regress_epoch() {
        let publisher = Arc::new(publisher_with_fresh_cache());
        let before = publisher.snapshot();
        let expected_epoch = before.epoch;
        let normal_reserved = Arc::new(Barrier::new(2));
        let release_normal = Arc::new(Barrier::new(2));
        let cas_snapshot_ready = Arc::new(Barrier::new(2));

        let result = std::thread::scope(|scope| {
            let normal_publisher = Arc::clone(&publisher);
            let normal_reserved_thread = Arc::clone(&normal_reserved);
            let release_normal_thread = Arc::clone(&release_normal);
            let mut normal_cache = (*before).clone();
            normal_cache.p_eff = SmallVec::from_slice(&[0.6, 0.4]);
            let normal_writer = scope.spawn(move || {
                normal_publisher.store_with_before_publish_hook(normal_cache, |epoch| {
                    assert_eq!(epoch, expected_epoch + 1);
                    normal_reserved_thread.wait();
                    release_normal_thread.wait();
                });
            });

            // The normal writer has reserved epoch+1 while still holding the
            // writer lock, but has not yet published its replacement.
            normal_reserved.wait();
            let cas_publisher = Arc::clone(&publisher);
            let cas_snapshot_ready_thread = Arc::clone(&cas_snapshot_ready);
            let foundry_writer = scope.spawn(move || {
                let bridge = authorized_score_bridge(&cas_publisher);
                let selected = selected_score_decision("A02.live");
                bridge.publish_effective_prior_with_hook(
                    selected,
                    identity,
                    expected_epoch,
                    77,
                    2,
                    |_| {
                        // Foundry has cloned the old epoch and is about to CAS.
                        cas_snapshot_ready_thread.wait();
                    },
                )
            });

            cas_snapshot_ready.wait();
            release_normal.wait();
            normal_writer.join().expect("normal writer");
            foundry_writer.join().expect("foundry writer")
        });

        assert_eq!(
            result,
            Err(ScoreVectorPublishError::UnexpectedCacheEpoch {
                expected: expected_epoch,
                actual: expected_epoch + 1,
            })
        );
        let after = publisher.snapshot();
        assert_eq!(after.epoch, expected_epoch + 1);
        assert_eq!(after.p_eff.as_slice(), &[0.6, 0.4]);
    }
}
