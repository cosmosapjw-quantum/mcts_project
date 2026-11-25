//! SearchController 트레이트 + root_entropy 헬퍼

#[cfg(test)]
use crate::mcts::node::{ArenaRef, MctsNode};

// ─────────────────────────────────────────────
// § StopReason — PR-0B: 정지 의미론 명시화
// ─────────────────────────────────────────────

/// Why did the search terminate?
/// Every search MUST produce a StopReason so downstream calibration
/// analysis can stratify results by halt category.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    /// P_flip converged below threshold for FLIP_STABLE_N consecutive checks
    Converged { p_flip: f32, stable_count: u32 },
    /// Reached maximum iteration / visit budget
    BudgetExhausted { iterations: u32 },
    /// Exceeded time cap (ctm_budget_ms × safety_factor)
    TimeCapHit { elapsed_ms: u64 },
    /// Transposition table or node count cap hit
    MaxNodesHit { nodes: u32 },
    /// All VOC channels ≤ 0 (no computation worth doing)
    VocNonPositive { max_gvoc: f32 },
    /// Not yet determined (search still running or controller doesn't track)
    Unknown,
}

impl Default for StopReason {
    fn default() -> Self {
        StopReason::Unknown
    }
}

impl StopReason {
    /// Short string for JSON serialization
    pub fn tag(&self) -> &'static str {
        match self {
            StopReason::Converged { .. } => "Converged",
            StopReason::BudgetExhausted { .. } => "BudgetExhausted",
            StopReason::TimeCapHit { .. } => "TimeCapHit",
            StopReason::MaxNodesHit { .. } => "MaxNodesHit",
            StopReason::VocNonPositive { .. } => "VocNonPositive",
            StopReason::Unknown => "Unknown",
        }
    }

    pub fn is_set(&self) -> bool {
        !matches!(self, StopReason::Unknown)
    }
}

// ─────────────────────────────────────────────
// § SearchStats
// ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SearchStats {
    pub iterations: u32,
    pub elapsed_ms: u64,
    pub nps: f64,
    pub tt_hit_rate: f64,
    pub tt_size: usize,
    pub root_visits: u32,
    pub stop_reason: StopReason,
}

// ─────────────────────────────────────────────
// § SearchController 트레이트
// ─────────────────────────────────────────────

pub trait SearchController: Send + Sync {
    fn should_stop(&self, root_visits: u32, elapsed_ms: u64) -> bool;
    fn reset(&mut self) {}
    /// Whether the caller must compute wall-clock elapsed time before polling.
    /// Controllers that stop only on visit counts can return false to avoid
    /// per-iteration `Instant::elapsed()` overhead.
    fn needs_elapsed_ms(&self) -> bool {
        true
    }
    /// Optional exact visit-budget hint for controllers whose stop condition is
    /// purely `root_visits >= limit`.
    fn visit_limit_hint(&self) -> Option<u32> {
        None
    }
    /// Return the reason for the most recent stop decision.
    /// Default: Unknown. Implementors should override to provide specifics.
    fn stop_reason(&self) -> StopReason {
        StopReason::Unknown
    }
}

// ─────────────────────────────────────────────
// § FixedIterations
// ─────────────────────────────────────────────

pub struct FixedIterations {
    pub limit: u32,
}

impl FixedIterations {
    pub fn new(limit: u32) -> Self {
        FixedIterations { limit }
    }
}

impl SearchController for FixedIterations {
    fn should_stop(&self, root_visits: u32, _elapsed_ms: u64) -> bool {
        root_visits >= self.limit
    }
    fn needs_elapsed_ms(&self) -> bool {
        false
    }
    fn visit_limit_hint(&self) -> Option<u32> {
        Some(self.limit)
    }
    fn stop_reason(&self) -> StopReason {
        StopReason::BudgetExhausted {
            iterations: self.limit,
        }
    }
}

// ─────────────────────────────────────────────
// § TimeManager
// ─────────────────────────────────────────────

#[cfg(test)]
pub struct TimeManager {
    budget_ms: u64,
    effective_budget_ms: u64,
    last_elapsed_ms: std::sync::atomic::AtomicU64,
}

#[cfg(test)]
impl TimeManager {
    pub fn new(budget_ms: u64) -> Self {
        TimeManager {
            budget_ms,
            effective_budget_ms: budget_ms,
            last_elapsed_ms: std::sync::atomic::AtomicU64::new(0),
        }
    }
    pub fn with_hard_cap(mut self, cap_ms: u64) -> Self {
        self.effective_budget_ms = self.budget_ms.min(cap_ms);
        self
    }
}

#[cfg(test)]
impl SearchController for TimeManager {
    fn should_stop(&self, _root_visits: u32, elapsed_ms: u64) -> bool {
        self.last_elapsed_ms
            .store(elapsed_ms, std::sync::atomic::Ordering::Relaxed);
        elapsed_ms >= self.effective_budget_ms
    }
    fn stop_reason(&self) -> StopReason {
        StopReason::TimeCapHit {
            elapsed_ms: self
                .last_elapsed_ms
                .load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

// ─────────────────────────────────────────────
// § root_entropy
// ─────────────────────────────────────────────

/// 루트 방문 분포의 Shannon entropy H(π)
/// visit_counts: 각 자식의 N
#[cfg(test)]
pub fn root_entropy(visit_counts: &[u32]) -> f32 {
    let total: u32 = visit_counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let tf = total as f32;
    visit_counts
        .iter()
        .filter(|&&n| n > 0)
        .map(|&n| {
            let p = n as f32 / tf;
            -p * p.ln()
        })
        .sum()
}

/// 루트 노드에서 직접 entropy 계산
#[cfg(test)]
pub fn root_entropy_from_node<M: Copy + Send + Sync + 'static>(
    node: &ArenaRef<MctsNode<M>>,
) -> f32 {
    let n_mat = node.materialized_count();
    let edge_arcs = node.edge_snapshot(n_mat);
    let counts: Vec<u32> = edge_arcs.iter().map(|e| e.n).collect();
    root_entropy(&counts)
}

// ─────────────────────────────────────────────
// § PR-0B 단위 테스트
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // TEST-0B-1 (partial): StopReason default is set on construction
    #[test]
    fn test_0b1_stop_reason_default() {
        let reason = StopReason::default();
        assert!(!reason.is_set(), "default should be Unknown (not set)");
    }

    #[test]
    fn test_0b1_stop_reason_always_set_after_stop() {
        let ctrl = FixedIterations::new(100);
        // Before stop triggers
        assert!(
            ctrl.stop_reason().is_set(),
            "FixedIterations always knows its reason"
        );
        assert_eq!(ctrl.stop_reason().tag(), "BudgetExhausted");
    }

    #[test]
    fn test_fixed_iterations_hints() {
        let ctrl = FixedIterations::new(100);
        assert!(!ctrl.needs_elapsed_ms());
        assert_eq!(ctrl.visit_limit_hint(), Some(100));
    }

    // TEST-0B-2: Category coverage
    #[test]
    fn test_0b2_stop_reason_categories() {
        // Converged
        let r = StopReason::Converged {
            p_flip: 0.05,
            stable_count: 3,
        };
        assert_eq!(r.tag(), "Converged");
        assert!(r.is_set());

        // BudgetExhausted
        let r = StopReason::BudgetExhausted { iterations: 200 };
        assert_eq!(r.tag(), "BudgetExhausted");
        assert!(r.is_set());

        // TimeCapHit
        let r = StopReason::TimeCapHit { elapsed_ms: 5000 };
        assert_eq!(r.tag(), "TimeCapHit");
        assert!(r.is_set());

        // MaxNodesHit
        let r = StopReason::MaxNodesHit { nodes: 10000 };
        assert_eq!(r.tag(), "MaxNodesHit");
        assert!(r.is_set());

        // VocNonPositive
        let r = StopReason::VocNonPositive { max_gvoc: -0.001 };
        assert_eq!(r.tag(), "VocNonPositive");
        assert!(r.is_set());

        // Unknown
        let r = StopReason::Unknown;
        assert_eq!(r.tag(), "Unknown");
        assert!(!r.is_set());
    }

    // TEST-0B-2 (extended): FixedIterations reports BudgetExhausted
    #[test]
    fn test_0b2_fixed_iterations_reason() {
        let ctrl = FixedIterations::new(500);
        assert!(matches!(
            ctrl.stop_reason(),
            StopReason::BudgetExhausted { iterations: 500 }
        ));
    }

    // TEST-0B-2 (extended): TimeManager reports TimeCapHit with actual elapsed
    #[test]
    fn test_0b2_time_manager_reason() {
        let ctrl = TimeManager::new(3000);
        // Before should_stop is called, elapsed is 0
        assert!(matches!(
            ctrl.stop_reason(),
            StopReason::TimeCapHit { elapsed_ms: 0 }
        ));
        // After should_stop observes elapsed time, it records the actual value
        ctrl.should_stop(0, 3100);
        assert!(matches!(
            ctrl.stop_reason(),
            StopReason::TimeCapHit { elapsed_ms: 3100 }
        ));
    }

    // TEST-0B-4: Determinism — same reason for same conditions
    #[test]
    fn test_0b4_stop_reason_deterministic() {
        let ctrl = FixedIterations::new(100);
        let r1 = ctrl.stop_reason();
        let r2 = ctrl.stop_reason();
        assert_eq!(r1, r2, "same controller should return same stop reason");
    }

    // SearchStats includes stop_reason
    #[test]
    fn test_search_stats_has_stop_reason() {
        let stats = SearchStats {
            iterations: 100,
            elapsed_ms: 50,
            nps: 2000.0,
            tt_hit_rate: 0.5,
            tt_size: 1000,
            root_visits: 100,
            stop_reason: StopReason::Converged {
                p_flip: 0.1,
                stable_count: 3,
            },
        };
        assert!(stats.stop_reason.is_set());
        assert_eq!(stats.stop_reason.tag(), "Converged");
    }
}
