//! Shared, feature-gated contracts for the QUARTZ Idea Foundry.
//!
//! The serializable types in this module are the Rust half of schema version
//! 1.  They intentionally keep `edge_pos` distinct from a game's `action_id`,
//! use the plural `nn_evals` cost key, and carry an explicit evidence scope.
//! Live axes still observe the existing read-only `SearchSnapshot`/`EdgeView`
//! boundary; no type here mutates the production search tree.

use std::collections::{BTreeMap, BTreeSet};

use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

use crate::mcts::policy::{EdgeView, SearchSnapshot};

pub const FOUNDRY_CONTRACT_SCHEMA_VERSION: u16 = 1;
pub const SKELETON_EVIDENCE_SCOPE: &str = "skeleton_only";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AxisStatus {
    Seed,
    MechanismValid,
    Shadow,
    Conditional,
    ActiveExperimental,
    DeploymentCandidate,
    Dormant,
    AnalysisOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetaActionKind {
    Stop,
    Sample,
    Challenge,
    Widen,
    Deepen,
    Prove,
    ResampleMode,
    MergeOrShare,
    SetBatch,
    SetInflight,
    SetThreads,
    Reanalyse,
    ArchiveState,
    Noop,
}

/// Typed in-process representation of one explicit computation action.
///
/// Serialization uses the language-neutral Python contract shape
/// `{kind, primary, secondary, amount, value, label}`.  The generic wire
/// fields never leak into the Rust execution boundary, where position/count
/// widths remain checked by the enum variants below.
#[derive(Clone, Debug, PartialEq)]
pub enum MetaAction {
    Stop {
        edge_pos: Option<u16>,
    },
    Sample {
        edge_pos: u16,
        visits: u32,
    },
    Challenge {
        best_pos: u16,
        challenger_pos: u16,
        visits: u32,
    },
    Widen {
        count: u16,
    },
    Deepen {
        edge_pos: u16,
        visits: u32,
    },
    Prove {
        edge_pos: u16,
        budget: u32,
    },
    ResampleMode {
        mode_id: u16,
        count: u16,
    },
    MergeOrShare {
        state_key: u64,
    },
    SetBatch {
        batch_size: u16,
    },
    SetInflight {
        credit: u16,
    },
    SetThreads {
        threads: u16,
    },
    Reanalyse {
        state_key: u64,
    },
    ArchiveState {
        priority: f64,
    },
    Noop,
    /// Optional human-readable wire label.  This wrapper preserves the typed
    /// action while round-tripping Python's generic schema without teaching
    /// the executor to interpret labels as control inputs.
    Labeled {
        action: Box<MetaAction>,
        label: String,
    },
}

impl MetaAction {
    /// Return the typed action while treating an optional wire label as
    /// non-semantic metadata.
    pub fn base_action(&self) -> &MetaAction {
        match self {
            Self::Labeled { action, .. } => action.base_action(),
            other => other,
        }
    }

    pub fn kind(&self) -> MetaActionKind {
        match self {
            Self::Stop { .. } => MetaActionKind::Stop,
            Self::Sample { .. } => MetaActionKind::Sample,
            Self::Challenge { .. } => MetaActionKind::Challenge,
            Self::Widen { .. } => MetaActionKind::Widen,
            Self::Deepen { .. } => MetaActionKind::Deepen,
            Self::Prove { .. } => MetaActionKind::Prove,
            Self::ResampleMode { .. } => MetaActionKind::ResampleMode,
            Self::MergeOrShare { .. } => MetaActionKind::MergeOrShare,
            Self::SetBatch { .. } => MetaActionKind::SetBatch,
            Self::SetInflight { .. } => MetaActionKind::SetInflight,
            Self::SetThreads { .. } => MetaActionKind::SetThreads,
            Self::Reanalyse { .. } => MetaActionKind::Reanalyse,
            Self::ArchiveState { .. } => MetaActionKind::ArchiveState,
            Self::Noop => MetaActionKind::Noop,
            Self::Labeled { action, .. } => action.kind(),
        }
    }

    /// Return true only for scheduler-wide actions whose target does not
    /// depend on the observed root, evaluator, candidate set, TT, or cache.
    /// Every other non-NOOP action is root-bound and must retain exact
    /// freshness through execution.
    pub fn is_identity_independent_system_action(&self) -> bool {
        match self.base_action() {
            Self::SetBatch { .. } | Self::SetInflight { .. } | Self::SetThreads { .. } => true,
            _ => false,
        }
    }

    /// All root-bound actions require the exact evidence identity.  NOOP is
    /// harmless, and the three explicitly enumerated scheduler-wide actions
    /// are the only identity-independent execution exceptions.
    pub fn requires_exact_freshness(&self) -> bool {
        !matches!(self.base_action(), Self::Noop) && !self.is_identity_independent_system_action()
    }

    fn is_contract_valid(&self) -> bool {
        match self {
            Self::ArchiveState { priority } => priority.is_finite(),
            Self::Labeled { action, label } => {
                !label.trim().is_empty()
                    && !matches!(action.as_ref(), Self::Labeled { .. })
                    && action.is_contract_valid()
            }
            _ => true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct MetaActionWire {
    kind: MetaActionKind,
    #[serde(default)]
    primary: Option<u64>,
    #[serde(default)]
    secondary: Option<u64>,
    #[serde(default)]
    amount: u64,
    #[serde(default)]
    value: Option<f64>,
    #[serde(default)]
    label: Option<String>,
}

impl From<&MetaAction> for MetaActionWire {
    fn from(action: &MetaAction) -> Self {
        let (action, label) = match action {
            MetaAction::Labeled { action, label } => (action.as_ref(), Some(label.clone())),
            other => (other, None),
        };
        let mut wire = Self {
            kind: action.kind(),
            primary: None,
            secondary: None,
            amount: 0,
            value: None,
            label,
        };
        match action {
            MetaAction::Stop { edge_pos } => wire.primary = edge_pos.map(u64::from),
            MetaAction::Sample { edge_pos, visits } | MetaAction::Deepen { edge_pos, visits } => {
                wire.primary = Some(u64::from(*edge_pos));
                wire.amount = u64::from(*visits);
            }
            MetaAction::Challenge {
                best_pos,
                challenger_pos,
                visits,
            } => {
                wire.primary = Some(u64::from(*best_pos));
                wire.secondary = Some(u64::from(*challenger_pos));
                wire.amount = u64::from(*visits);
            }
            MetaAction::Widen { count } => wire.amount = u64::from(*count),
            MetaAction::Prove { edge_pos, budget } => {
                wire.primary = Some(u64::from(*edge_pos));
                wire.amount = u64::from(*budget);
            }
            MetaAction::ResampleMode { mode_id, count } => {
                wire.primary = Some(u64::from(*mode_id));
                wire.amount = u64::from(*count);
            }
            MetaAction::MergeOrShare { state_key } | MetaAction::Reanalyse { state_key } => {
                wire.primary = Some(*state_key);
            }
            MetaAction::SetBatch { batch_size } => wire.amount = u64::from(*batch_size),
            MetaAction::SetInflight { credit } => wire.amount = u64::from(*credit),
            MetaAction::SetThreads { threads } => wire.amount = u64::from(*threads),
            MetaAction::ArchiveState { priority } => wire.value = Some(*priority),
            MetaAction::Noop => {}
            MetaAction::Labeled { .. } => unreachable!("labels are unwrapped above"),
        }
        wire
    }
}

fn required<T, E: de::Error>(value: Option<T>, field: &'static str) -> Result<T, E> {
    value.ok_or_else(|| E::missing_field(field))
}

fn as_u16<E: de::Error>(value: u64, field: &'static str) -> Result<u16, E> {
    value
        .try_into()
        .map_err(|_| E::custom(format!("{field} does not fit u16")))
}

fn as_u32<E: de::Error>(value: u64, field: &'static str) -> Result<u32, E> {
    value
        .try_into()
        .map_err(|_| E::custom(format!("{field} does not fit u32")))
}

impl MetaActionWire {
    fn into_action<E: de::Error>(self) -> Result<MetaAction, E> {
        let original = self.clone();
        let label = self.label.clone();
        let mut action = match self.kind {
            MetaActionKind::Stop => MetaAction::Stop {
                edge_pos: self
                    .primary
                    .map(|v| as_u16::<E>(v, "primary"))
                    .transpose()?,
            },
            MetaActionKind::Sample => MetaAction::Sample {
                edge_pos: as_u16(required::<_, E>(self.primary, "primary")?, "primary")?,
                visits: as_u32(self.amount, "amount")?,
            },
            MetaActionKind::Challenge => MetaAction::Challenge {
                best_pos: as_u16(required::<_, E>(self.primary, "primary")?, "primary")?,
                challenger_pos: as_u16(
                    required::<_, E>(self.secondary, "secondary")?,
                    "secondary",
                )?,
                visits: as_u32(self.amount, "amount")?,
            },
            MetaActionKind::Widen => MetaAction::Widen {
                count: as_u16(self.amount, "amount")?,
            },
            MetaActionKind::Deepen => MetaAction::Deepen {
                edge_pos: as_u16(required::<_, E>(self.primary, "primary")?, "primary")?,
                visits: as_u32(self.amount, "amount")?,
            },
            MetaActionKind::Prove => MetaAction::Prove {
                edge_pos: as_u16(required::<_, E>(self.primary, "primary")?, "primary")?,
                budget: as_u32(self.amount, "amount")?,
            },
            MetaActionKind::ResampleMode => MetaAction::ResampleMode {
                mode_id: as_u16(required::<_, E>(self.primary, "primary")?, "primary")?,
                count: as_u16(self.amount, "amount")?,
            },
            MetaActionKind::MergeOrShare => MetaAction::MergeOrShare {
                state_key: required::<_, E>(self.primary, "primary")?,
            },
            MetaActionKind::SetBatch => MetaAction::SetBatch {
                batch_size: as_u16(self.amount, "amount")?,
            },
            MetaActionKind::SetInflight => MetaAction::SetInflight {
                credit: as_u16(self.amount, "amount")?,
            },
            MetaActionKind::SetThreads => MetaAction::SetThreads {
                threads: as_u16(self.amount, "amount")?,
            },
            MetaActionKind::Reanalyse => MetaAction::Reanalyse {
                state_key: required::<_, E>(self.primary, "primary")?,
            },
            MetaActionKind::ArchiveState => MetaAction::ArchiveState {
                priority: required::<_, E>(self.value, "value")?,
            },
            MetaActionKind::Noop => MetaAction::Noop,
        };
        if let Some(label) = label {
            if label.trim().is_empty() {
                return Err(E::custom("action label must be non-empty when present"));
            }
            action = MetaAction::Labeled {
                action: Box::new(action),
                label,
            };
        }
        if MetaActionWire::from(&action) != original {
            return Err(E::custom(
                "action contains non-canonical fields for its kind",
            ));
        }
        Ok(action)
    }
}

impl Serialize for MetaAction {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        MetaActionWire::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MetaAction {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        MetaActionWire::deserialize(deserializer)?.into_action::<D::Error>()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CostVector {
    pub nn_evals: f64,
    pub cpu_ms: f64,
    pub gpu_ms: f64,
    pub energy_proxy: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CostPrices {
    /// Price per neural-network evaluation.  The plural spelling is the
    /// canonical cross-language key and matches `CostVector::nn_evals`.
    pub nn_evals: f64,
    pub cpu_ms: f64,
    pub gpu_ms: f64,
    pub energy_proxy: f64,
}

impl CostVector {
    pub fn weighted(self, prices: CostPrices) -> f64 {
        self.nn_evals * prices.nn_evals
            + self.cpu_ms * prices.cpu_ms
            + self.gpu_ms * prices.gpu_ms
            + self.energy_proxy * prices.energy_proxy
    }

    fn is_valid(self) -> bool {
        self.nn_evals.is_finite()
            && self.nn_evals >= 0.0
            && self.cpu_ms.is_finite()
            && self.cpu_ms >= 0.0
            && self.gpu_ms.is_finite()
            && self.gpu_ms >= 0.0
            && self.energy_proxy.is_finite()
            && self.energy_proxy >= 0.0
    }
}

impl CostPrices {
    pub fn is_valid(self) -> bool {
        CostVector {
            nn_evals: self.nn_evals,
            cpu_ms: self.cpu_ms,
            gpu_ms: self.gpu_ms,
            energy_proxy: self.energy_proxy,
        }
        .is_valid()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProposalEstimate {
    pub regret_reduction_mean: f64,
    pub regret_reduction_lcb: f64,
    pub confidence: f64,
    pub cost: CostVector,
}

impl ProposalEstimate {
    pub fn net_lcb(self, prices: CostPrices) -> f64 {
        self.regret_reduction_lcb - self.cost.weighted(prices)
    }

    fn is_finite(self) -> bool {
        self.regret_reduction_mean.is_finite()
            && self.regret_reduction_lcb.is_finite()
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
            && self.cost.is_valid()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetaProposal {
    pub schema_version: u16,
    pub axis_id: String,
    pub action: MetaAction,
    pub estimate: ProposalEstimate,
    pub activation_guard: String,
    pub explanation: String,
    pub evidence_scope: String,
    #[serde(default)]
    pub telemetry: BTreeMap<String, serde_json::Value>,
}

impl MetaProposal {
    pub fn new(
        axis_id: impl Into<String>,
        action: MetaAction,
        estimate: ProposalEstimate,
        activation_guard: impl Into<String>,
        explanation: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
            axis_id: axis_id.into(),
            action,
            estimate,
            activation_guard: activation_guard.into(),
            explanation: explanation.into(),
            evidence_scope: SKELETON_EVIDENCE_SCOPE.to_string(),
            telemetry: BTreeMap::new(),
        }
    }

    pub fn noop(axis_id: &'static str, explanation: impl Into<String>) -> Self {
        Self::new(
            axis_id,
            MetaAction::Noop,
            ProposalEstimate::default(),
            "skeleton-only",
            explanation,
        )
    }

    /// Invalid or non-finite proposals fail closed before arbitration.
    pub fn is_contract_valid(&self) -> bool {
        self.schema_version == FOUNDRY_CONTRACT_SCHEMA_VERSION
            && valid_axis_id(&self.axis_id)
            && !self.activation_guard.trim().is_empty()
            && !self.explanation.trim().is_empty()
            && !self.evidence_scope.trim().is_empty()
            && self.action.is_contract_valid()
            && self.estimate.is_finite()
    }

    pub fn to_payload(&self) -> Result<MetaProposalPayload, ContractWireError> {
        if !self.is_contract_valid() {
            return Err(ContractWireError::InvalidProposal);
        }
        Ok(MetaProposalPayload {
            schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
            kind: "meta_proposal".to_string(),
            proposal: MetaProposalBody::from(self),
        })
    }
}

fn valid_axis_id(axis_id: &str) -> bool {
    let mut parts = axis_id.splitn(2, '.');
    let prefix = parts.next().unwrap_or_default();
    let valid_prefix = prefix.len() == 3
        && prefix.starts_with('A')
        && match prefix[1..].parse::<u8>() {
            Ok(number) => (1..=26).contains(&number),
            Err(_) => false,
        };
    let valid_suffix = match parts.next() {
        None => true,
        Some(suffix) => {
            !suffix.is_empty()
                && suffix
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        }
    };
    valid_prefix && valid_suffix
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContractWireError {
    UnsupportedSchemaVersion(u16),
    UnexpectedKind(String),
    InvalidProposal,
    InvalidRootObservation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetaProposalBody {
    pub axis_id: String,
    pub action: MetaAction,
    pub estimate: ProposalEstimate,
    pub activation_guard: String,
    pub explanation: String,
    pub evidence_scope: String,
    #[serde(default)]
    pub telemetry: BTreeMap<String, serde_json::Value>,
}

impl From<&MetaProposal> for MetaProposalBody {
    fn from(proposal: &MetaProposal) -> Self {
        Self {
            axis_id: proposal.axis_id.clone(),
            action: proposal.action.clone(),
            estimate: proposal.estimate,
            activation_guard: proposal.activation_guard.clone(),
            explanation: proposal.explanation.clone(),
            evidence_scope: proposal.evidence_scope.clone(),
            telemetry: proposal.telemetry.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetaProposalPayload {
    pub schema_version: u16,
    pub kind: String,
    pub proposal: MetaProposalBody,
}

impl MetaProposalPayload {
    pub fn into_proposal(self) -> Result<MetaProposal, ContractWireError> {
        if self.schema_version != FOUNDRY_CONTRACT_SCHEMA_VERSION {
            return Err(ContractWireError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        if self.kind != "meta_proposal" {
            return Err(ContractWireError::UnexpectedKind(self.kind));
        }
        let proposal = MetaProposal {
            schema_version: self.schema_version,
            axis_id: self.proposal.axis_id,
            action: self.proposal.action,
            estimate: self.proposal.estimate,
            activation_guard: self.proposal.activation_guard,
            explanation: self.proposal.explanation,
            evidence_scope: self.proposal.evidence_scope,
            telemetry: self.proposal.telemetry,
        };
        if !proposal.is_contract_valid() {
            return Err(ContractWireError::InvalidProposal);
        }
        Ok(proposal)
    }
}

/// Complete identity of the evidence used to produce a proposal.
///
/// Exact equality is intentionally strict.  In particular, a stable root hash
/// does not permit a STOP generated by another evaluator, candidate epoch,
/// transposition-table policy, or cache schema to pass as fresh.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FreshnessIdentity {
    pub root_hash: u64,
    pub checkpoint_id: String,
    pub evaluator_id: String,
    pub edge_set_hash: String,
    pub candidate_epoch: u64,
    pub tt_identity_policy: String,
    pub cache_schema_version: u16,
    pub root_visits: u32,
    pub iteration: u64,
}

impl Default for FreshnessIdentity {
    fn default() -> Self {
        Self {
            root_hash: 0,
            checkpoint_id: String::new(),
            evaluator_id: String::new(),
            edge_set_hash: String::new(),
            candidate_epoch: 0,
            tt_identity_policy: String::new(),
            cache_schema_version: 0,
            root_visits: 0,
            iteration: 0,
        }
    }
}

impl FreshnessIdentity {
    pub fn is_well_formed(&self) -> bool {
        !self.checkpoint_id.trim().is_empty()
            && !self.evaluator_id.trim().is_empty()
            && !self.edge_set_hash.trim().is_empty()
            && !self.tt_identity_policy.trim().is_empty()
            && self.cache_schema_version > 0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyChannels {
    pub mc: f32,
    pub epistemic: f32,
    pub drift: f32,
    pub bias: f32,
}

impl UncertaintyChannels {
    pub fn conservative_sum(self) -> f32 {
        self.mc.max(0.0) + self.epistemic.max(0.0) + self.drift.max(0.0) + self.bias.max(0.0)
    }

    pub fn rss(self) -> f32 {
        (self.mc.max(0.0).powi(2)
            + self.epistemic.max(0.0).powi(2)
            + self.drift.max(0.0).powi(2)
            + self.bias.max(0.0).powi(2))
        .sqrt()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    pub threads: u16,
    pub batch_size: u16,
    pub inflight: u16,
    pub queue_wait_ms: f64,
    pub eval_latency_ms: f64,
    pub nps: f64,
    pub edge_duplicate_rate: f64,
    pub semantic_path_overlap: f64,
    pub max_pending: u16,
    pub tt_wait_ns: u64,
}

impl Default for RuntimeSnapshot {
    fn default() -> Self {
        Self {
            threads: 1,
            batch_size: 1,
            inflight: 1,
            queue_wait_ms: 0.0,
            eval_latency_ms: 0.0,
            nps: 0.0,
            edge_duplicate_rate: 0.0,
            semantic_path_overlap: 0.0,
            max_pending: 0,
            tt_wait_ns: 0,
        }
    }
}

impl RuntimeSnapshot {
    fn is_contract_valid(self) -> bool {
        self.queue_wait_ms.is_finite()
            && self.eval_latency_ms.is_finite()
            && self.nps.is_finite()
            && self.edge_duplicate_rate.is_finite()
            && self.semantic_path_overlap.is_finite()
    }
}

/// Owned edge representation used only at serialization/replay boundaries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EdgeObservation {
    pub edge_pos: u16,
    pub action_id: u32,
    pub visible: bool,
    pub prior_anchor: f64,
    pub prior_current: f64,
    pub visits: u32,
    pub virtual_visits: u32,
    pub pending: u32,
    pub q_mean: f64,
    pub q_sum: f64,
    pub m2: f64,
    #[serde(default)]
    pub last_value: f64,
    #[serde(default)]
    pub mc_radius: f64,
    #[serde(default)]
    pub epistemic_radius: f64,
    #[serde(default)]
    pub drift_radius: f64,
    #[serde(default)]
    pub bias_radius: f64,
    #[serde(default = "negative_one")]
    pub lower: f64,
    #[serde(default = "positive_one")]
    pub upper: f64,
    #[serde(default)]
    pub tactical_flags: Vec<String>,
}

impl EdgeObservation {
    fn is_contract_valid(&self) -> bool {
        self.prior_anchor.is_finite()
            && self.prior_anchor >= 0.0
            && self.prior_current.is_finite()
            && self.prior_current >= 0.0
            && self.q_mean.is_finite()
            && self.q_sum.is_finite()
            && self.m2.is_finite()
            && self.last_value.is_finite()
            && self.mc_radius.is_finite()
            && self.epistemic_radius.is_finite()
            && self.drift_radius.is_finite()
            && self.bias_radius.is_finite()
            && self.lower.is_finite()
            && self.upper.is_finite()
            && self.lower <= self.upper
    }
}

fn negative_one() -> f64 {
    -1.0
}

fn positive_one() -> f64 {
    1.0
}

/// Owned root observation used for JSON fixtures and replay.  Versioning lives
/// on [`RootObservationPayload`], matching the Python envelope.
/// Live axes use [`FoundryObservation`] so this type adds no production copy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RootObservation {
    pub root_hash: u64,
    pub checkpoint_id: String,
    pub position_id: String,
    pub game: String,
    pub root_visits: u32,
    pub iteration: u64,
    pub elapsed_ms: u64,
    pub remaining_visits: u32,
    pub n_children: u16,
    pub n_visible: u16,
    pub entropy: f64,
    pub effective_branching: f64,
    pub top2_margin: f64,
    pub margin_slope: f64,
    pub entropy_slope: f64,
    pub h1_stability: Option<f64>,
    pub p_flip: Option<f64>,
    pub prior_visit_js: f64,
    pub candidate_omission_bound: f64,
    pub revision_count: u16,
    pub edges: Vec<EdgeObservation>,
    #[serde(default)]
    pub runtime: RuntimeSnapshot,
    #[serde(default)]
    pub extras: BTreeMap<String, serde_json::Value>,
    pub freshness: FreshnessIdentity,
}

impl RootObservation {
    pub fn is_contract_valid(&self) -> bool {
        let mut positions = BTreeSet::new();
        self.freshness.is_well_formed()
            && self.freshness.root_hash == self.root_hash
            && self.freshness.checkpoint_id == self.checkpoint_id
            && self.freshness.root_visits == self.root_visits
            && self.freshness.iteration == self.iteration
            && self.n_visible <= self.n_children
            && self.entropy.is_finite()
            && self.effective_branching.is_finite()
            && self.top2_margin.is_finite()
            && self.margin_slope.is_finite()
            && self.entropy_slope.is_finite()
            && self.h1_stability.map_or(true, f64::is_finite)
            && self.p_flip.map_or(true, f64::is_finite)
            && self.prior_visit_js.is_finite()
            && self.candidate_omission_bound.is_finite()
            && self.runtime.is_contract_valid()
            && self.edges.iter().all(|edge| {
                edge.edge_pos < self.n_children
                    && positions.insert(edge.edge_pos)
                    && edge.is_contract_valid()
            })
    }

    pub fn to_payload(&self) -> Result<RootObservationPayload, ContractWireError> {
        if !self.is_contract_valid() {
            return Err(ContractWireError::InvalidRootObservation);
        }
        Ok(RootObservationPayload {
            schema_version: FOUNDRY_CONTRACT_SCHEMA_VERSION,
            kind: "root_observation".to_string(),
            observation: self.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RootObservationPayload {
    pub schema_version: u16,
    pub kind: String,
    pub observation: RootObservation,
}

impl RootObservationPayload {
    pub fn into_observation(self) -> Result<RootObservation, ContractWireError> {
        if self.schema_version != FOUNDRY_CONTRACT_SCHEMA_VERSION {
            return Err(ContractWireError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        if self.kind != "root_observation" {
            return Err(ContractWireError::UnexpectedKind(self.kind));
        }
        if !self.observation.is_contract_valid() {
            return Err(ContractWireError::InvalidRootObservation);
        }
        Ok(self.observation)
    }
}

/// Extra root observations not present in the production `SearchSnapshot`.
///
/// The identity is supplied beside the existing snapshot rather than
/// repurposing any legacy field.  `FoundryCoordinator` verifies that the
/// snapshot counters agree before it stamps a selected proposal.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FoundryRootExtras {
    pub schema_version: u16,
    pub freshness: FreshnessIdentity,
    pub entropy: f64,
    pub effective_branching: f64,
    pub top2_margin: f64,
    pub margin_slope: f64,
    pub entropy_slope: f64,
    pub h1_stability: Option<f64>,
    pub p_flip: Option<f64>,
    pub prior_visit_js: f64,
    pub omission_bound: f64,
    pub revision_count: u16,
    pub runtime: RuntimeSnapshot,
}

impl FoundryRootExtras {
    pub fn is_contract_valid(&self) -> bool {
        self.schema_version == FOUNDRY_CONTRACT_SCHEMA_VERSION
            && self.freshness.is_well_formed()
            && self.entropy.is_finite()
            && self.effective_branching.is_finite()
            && self.top2_margin.is_finite()
            && self.margin_slope.is_finite()
            && self.entropy_slope.is_finite()
            && self.h1_stability.map_or(true, f64::is_finite)
            && self.p_flip.map_or(true, f64::is_finite)
            && self.prior_visit_js.is_finite()
            && self.omission_bound.is_finite()
            && self.runtime.is_contract_valid()
    }

    pub fn identity_matches_snapshot(&self, snap: &SearchSnapshot) -> bool {
        self.is_contract_valid()
            && self.freshness.root_visits == snap.root_visits
            && self.freshness.iteration == snap.iteration
    }
}

pub struct FoundryObservation<'a> {
    pub snap: &'a SearchSnapshot,
    pub edges: &'a [EdgeView<'a>],
    pub extras: &'a FoundryRootExtras,
}

pub trait FoundryAxis: Send + Sync {
    fn id(&self) -> &'static str;
    fn status(&self) -> AxisStatus;
    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>);
}

pub trait FoundryArbiter: Send + Sync {
    fn choose<'a>(
        &self,
        proposals: &'a [MetaProposal],
        prices: CostPrices,
    ) -> Option<&'a MetaProposal>;
}

#[derive(Default)]
pub struct ConservativeArbiter;

impl FoundryArbiter for ConservativeArbiter {
    fn choose<'a>(
        &self,
        proposals: &'a [MetaProposal],
        prices: CostPrices,
    ) -> Option<&'a MetaProposal> {
        proposals
            .iter()
            .filter(|proposal| {
                prices.is_valid()
                    && proposal.is_contract_valid()
                    && (proposal.action.kind() == MetaActionKind::Stop
                        || proposal.estimate.net_lcb(prices) > 0.0)
            })
            .max_by(|a, b| {
                a.estimate
                    .net_lcb(prices)
                    .total_cmp(&b.estimate.net_lcb(prices))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_schema_matches_generic_cross_language_shape() {
        let action = MetaAction::Challenge {
            best_pos: 1,
            challenger_pos: 481,
            visits: 32,
        };
        let value = serde_json::to_value(&action).expect("serialize action");
        assert_eq!(value["kind"], "challenge");
        assert_eq!(value["primary"], 1);
        assert_eq!(value["secondary"], 481);
        assert_eq!(value["amount"], 32);
        assert!(value["value"].is_null());
        assert!(value["label"].is_null());
        let decoded: MetaAction = serde_json::from_value(value).expect("deserialize action");
        assert_eq!(decoded, action);
    }

    #[test]
    fn labeled_action_round_trips_without_becoming_control_input() {
        let action = MetaAction::Labeled {
            action: Box::new(MetaAction::Sample {
                edge_pos: 1,
                visits: 8,
            }),
            label: "golden_sample".into(),
        };
        assert_eq!(action.kind(), MetaActionKind::Sample);
        assert!(matches!(action.base_action(), MetaAction::Sample { .. }));
        let value = serde_json::to_value(&action).expect("serialize labeled action");
        assert_eq!(value["label"], "golden_sample");
        let decoded: MetaAction = serde_json::from_value(value).expect("deserialize action");
        assert_eq!(decoded, action);
    }

    #[test]
    fn action_schema_rejects_position_width_overflow() {
        let value = serde_json::json!({
            "kind": "sample",
            "primary": 70_000,
            "secondary": null,
            "amount": 1,
            "value": null,
            "label": null
        });
        assert!(serde_json::from_value::<MetaAction>(value).is_err());
    }

    #[test]
    fn action_schema_rejects_non_canonical_unused_fields() {
        let value = serde_json::json!({
            "kind": "noop",
            "primary": 1,
            "secondary": null,
            "amount": 0,
            "value": null,
            "label": null
        });
        assert!(serde_json::from_value::<MetaAction>(value).is_err());
    }

    #[test]
    fn cost_prices_use_plural_nn_evals_key() {
        let prices = CostPrices {
            nn_evals: 2.0,
            cpu_ms: 3.0,
            gpu_ms: 5.0,
            energy_proxy: 7.0,
        };
        let value = serde_json::to_value(prices).expect("serialize prices");
        assert_eq!(value["nn_evals"], 2.0);
        assert!(value.get("nn_eval").is_none());
        assert_eq!(
            CostVector {
                nn_evals: 1.0,
                cpu_ms: 1.0,
                gpu_ms: 1.0,
                energy_proxy: 1.0,
            }
            .weighted(prices),
            17.0
        );
    }

    #[test]
    fn arbiter_rejects_negative_prices_and_costs() {
        let proposal = MetaProposal::new(
            "A04.test",
            MetaAction::Sample {
                edge_pos: 0,
                visits: 1,
            },
            ProposalEstimate {
                regret_reduction_lcb: 1.0,
                cost: CostVector {
                    nn_evals: -1.0,
                    ..CostVector::default()
                },
                ..ProposalEstimate::default()
            },
            "unit test",
            "unit test",
        );
        assert!(!proposal.is_contract_valid());
        assert!(ConservativeArbiter
            .choose(
                &[MetaProposal::new(
                    "A04.test",
                    MetaAction::Sample {
                        edge_pos: 0,
                        visits: 1,
                    },
                    ProposalEstimate {
                        regret_reduction_lcb: 1.0,
                        ..ProposalEstimate::default()
                    },
                    "unit test",
                    "unit test",
                )],
                CostPrices {
                    nn_evals: -1.0,
                    ..CostPrices::default()
                },
            )
            .is_none());
    }

    #[test]
    fn edge_position_and_action_id_remain_distinct_in_wire_contract() {
        let edge = EdgeObservation {
            edge_pos: 1,
            action_id: 481,
            visible: true,
            prior_anchor: 0.3,
            prior_current: 0.3,
            visits: 8,
            virtual_visits: 0,
            pending: 0,
            q_mean: 0.2,
            q_sum: 1.6,
            m2: 0.6,
            last_value: 0.1,
            mc_radius: 0.0,
            epistemic_radius: 0.0,
            drift_radius: 0.0,
            bias_radius: 0.0,
            lower: -1.0,
            upper: 1.0,
            tactical_flags: Vec::new(),
        };
        let value = serde_json::to_value(edge).expect("serialize edge");
        assert_eq!(value["edge_pos"], 1);
        assert_eq!(value["action_id"], 481);
    }

    #[test]
    fn cross_language_golden_fixture_round_trips_exactly() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/idea_foundry_contract_v1.json"
        ))
        .expect("parse fixture");
        let freshness: FreshnessIdentity = serde_json::from_value(fixture["freshness"].clone())
            .expect("decode freshness identity");
        assert!(freshness.is_well_formed());
        assert_eq!(freshness.root_hash, 123_456_789);
        assert_eq!(freshness.edge_set_hash.len(), 64);

        let payload: MetaProposalPayload =
            serde_json::from_value(fixture["meta_proposal_payload"].clone())
                .expect("decode proposal payload");
        let proposal = payload.into_proposal().expect("validate proposal payload");
        assert_eq!(proposal.action.kind(), MetaActionKind::Sample);
        assert_eq!(proposal.evidence_scope, "contract_fixture_only");
        let roundtrip = proposal.to_payload().expect("encode proposal payload");
        assert_eq!(
            serde_json::to_value(roundtrip).expect("proposal to JSON"),
            fixture["meta_proposal_payload"]
        );

        let root_payload: RootObservationPayload =
            serde_json::from_value(fixture["root_observation_payload"].clone())
                .expect("decode root observation payload");
        let root_observation = root_payload
            .into_observation()
            .expect("validate root observation payload");
        assert_eq!(root_observation.root_hash, 123);
        assert_eq!(root_observation.checkpoint_id, "seed_1/gen_1");
        assert_eq!(
            serde_json::to_value(
                root_observation
                    .to_payload()
                    .expect("encode root observation payload")
            )
            .expect("root observation to JSON"),
            fixture["root_observation_payload"]
        );

        let root_extras: FoundryRootExtras =
            serde_json::from_value(fixture["foundry_root_extras"].clone())
                .expect("decode foundry root extras");
        assert_eq!(root_extras.schema_version, FOUNDRY_CONTRACT_SCHEMA_VERSION);
        assert!(root_extras.is_contract_valid());
        assert_eq!(
            serde_json::to_value(root_extras).expect("foundry root extras to JSON"),
            fixture["foundry_root_extras"]
        );
    }

    #[test]
    fn root_observation_uses_versioned_python_compatible_envelope() {
        let freshness = FreshnessIdentity {
            root_hash: 123,
            checkpoint_id: "seed_1/gen_1".into(),
            evaluator_id: "model-sha256:abc".into(),
            edge_set_hash: "edge-sha256:def".into(),
            candidate_epoch: 1,
            tt_identity_policy: "state_eval_cache_only".into(),
            cache_schema_version: 1,
            root_visits: 24,
            iteration: 25,
        };
        let observation = RootObservation {
            root_hash: 123,
            checkpoint_id: "seed_1/gen_1".into(),
            position_id: "p1".into(),
            game: "gomoku7".into(),
            root_visits: 24,
            iteration: 25,
            elapsed_ms: 12,
            remaining_visits: 40,
            n_children: 1,
            n_visible: 1,
            entropy: 0.1,
            effective_branching: 1.0,
            top2_margin: 0.2,
            margin_slope: -0.01,
            entropy_slope: 0.02,
            h1_stability: Some(0.99),
            p_flip: Some(0.01),
            prior_visit_js: 0.2,
            candidate_omission_bound: 0.01,
            revision_count: 0,
            edges: vec![EdgeObservation {
                edge_pos: 0,
                action_id: 481,
                visible: true,
                prior_anchor: 1.0,
                prior_current: 1.0,
                visits: 24,
                virtual_visits: 0,
                pending: 0,
                q_mean: 0.2,
                q_sum: 4.8,
                m2: 0.6,
                last_value: 0.1,
                mc_radius: 0.0,
                epistemic_radius: 0.0,
                drift_radius: 0.0,
                bias_radius: 0.0,
                lower: -1.0,
                upper: 1.0,
                tactical_flags: Vec::new(),
            }],
            runtime: RuntimeSnapshot::default(),
            extras: BTreeMap::new(),
            freshness,
        };
        let payload = observation.to_payload().expect("valid root payload");
        let value = serde_json::to_value(&payload).expect("serialize root payload");
        assert_eq!(value["schema_version"], FOUNDRY_CONTRACT_SCHEMA_VERSION);
        assert_eq!(value["kind"], "root_observation");
        assert_eq!(value["observation"]["root_hash"], 123);
        assert_eq!(value["observation"]["freshness"]["root_visits"], 24);
        let decoded: RootObservationPayload =
            serde_json::from_value(value).expect("deserialize root payload");
        assert_eq!(
            decoded.into_observation().expect("validate root"),
            observation
        );

        let mut invalid = observation;
        invalid.entropy = f64::NAN;
        assert_eq!(
            invalid.to_payload(),
            Err(ContractWireError::InvalidRootObservation)
        );
    }
}
