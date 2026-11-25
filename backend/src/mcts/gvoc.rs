//! GVOC Scheduler — Greedy Value of Computation
//!
//! QUARTZ의 핵심 기여: 고정 iteration budget 대신 **VOC 기반 동적 할당**
//!
//! # 설계 (QUARTZ 원문 + Doc4 요약)
//!
//! 기본 MCTS: `N_total` 회 반복 → 고정 비용
//! QUARTZ GVOC:
//!   - VOC(s) = P_flip(s) × |Q₁(s)−Q₂(s)| 가 충분히 크면 "이 노드를 더 탐색할 가치가 있다"
//!   - 작으면 조기 중단 또는 다른 노드로 예산 이동
//!   - `InsideProposals`: 현재 top-k 후보 안에서 정밀화
//!   - `OutsideProposals`: WL bias로 탐색되지 않은 basin 시도
//!
//! # 구현 (CPU-first 근사)
//!
//! 1. 루트 VOC 기반 전체 탐색 정지 (QuartzController에 이미 구현)
//! 2. **노드 수준 PW 확장 폭 동적 조정**:
//!    - VOC(s) > expand_thresh → n_visible 상향 (더 많은 후보 열기)
//!    - VOC(s) < contract_thresh → n_visible 하향 (이미 충분히 확인됨)
//! 3. **예산 재배분 지표**:
//!    - `gvoc_score(s)` = VOC(s) / (N(s)+1): 방문당 기대 정보이득
//!    - 높은 노드에 더 많은 simulations 집중 (root에서 제어)

use crate::mcts::node::{ArenaRef, MctsNode};
use crate::mcts::quartz::{compute_quartz_stats, QuartzConfig, QuartzStats};

// ─────────────────────────────────────────────
// § GvocConfig
// ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GvocConfig {
    /// VOC > 이 값이면 PW 확장 폭 증가
    pub expand_thresh: f32,
    /// VOC < 이 값이면 PW 확장 폭 감소 (수렴)
    pub contract_thresh: f32,
    /// 확장 시 n_visible 증가량
    pub expand_delta: usize,
    /// 최대 n_visible (total candidates 초과 불가)
    pub max_visible: usize,
    /// 최소 n_visible
    pub min_visible: usize,
    /// gvoc_score 계산 주기
    pub score_interval: u32,
}

impl Default for GvocConfig {
    fn default() -> Self {
        GvocConfig {
            expand_thresh: 0.02,
            contract_thresh: 0.002,
            expand_delta: 4,
            max_visible: 64,
            min_visible: 1,
            score_interval: 200,
        }
    }
}

// ─────────────────────────────────────────────
// § GvocState — 루트 노드의 동적 PW 상태
// ─────────────────────────────────────────────

/// 루트 VOC 추적 + PW 확장 폭 동적 관리
pub struct GvocState {
    pub cfg: GvocConfig,
    /// 현재 루트 effective n_visible (PW override)
    pub n_visible_eff: usize,
    /// 마지막 계산된 VOC
    pub last_voc: f32,
    /// 누적 expand 횟수
    pub expand_count: u32,
    /// 누적 contract 횟수
    pub contract_count: u32,
    /// 총 iterations
    pub iterations: u32,
}

impl GvocState {
    pub fn new(cfg: GvocConfig, initial_visible: usize) -> Self {
        GvocState {
            n_visible_eff: initial_visible,
            last_voc: 0.0,
            expand_count: 0,
            contract_count: 0,
            iterations: 0,
            cfg,
        }
    }

    /// 매 iteration 후 호출 — VOC 기반 n_visible 조정
    pub fn update<M: Copy + Send + Sync + 'static>(
        &mut self,
        root: &ArenaRef<MctsNode<M>>,
        n_total: usize,
        qcfg: &QuartzConfig,
    ) {
        self.iterations += 1;
        if self.iterations % self.cfg.score_interval != 0 {
            return;
        }

        let mut s0 = crate::mcts::quartz::RunningMedian::new(0.05);
        let stats = compute_quartz_stats(root, None, &mut s0, 0.0, 0, 0, qcfg);
        if stats.n_visible == 0 {
            return;
        }

        // Unified VOC: EXPAND channel → expand PW, STOP → contract
        let voc_signal = stats.unified.voc_total;
        self.last_voc = voc_signal;

        let should_expand = matches!(
            stats.unified.action,
            crate::mcts::quartz::ComputeAction::Expand
        ) && voc_signal > self.cfg.expand_thresh;

        let should_contract = matches!(
            stats.unified.action,
            crate::mcts::quartz::ComputeAction::Stop
        ) || stats.unified.voc_expand < self.cfg.contract_thresh;

        if should_expand {
            let new_vis = (self.n_visible_eff + self.cfg.expand_delta)
                .min(self.cfg.max_visible)
                .min(n_total);
            if new_vis > self.n_visible_eff {
                self.n_visible_eff = new_vis;
                self.expand_count += 1;
            }
        } else if should_contract && self.n_visible_eff > self.cfg.min_visible {
            let new_vis = self
                .n_visible_eff
                .saturating_sub(1)
                .max(self.cfg.min_visible);
            if new_vis < self.n_visible_eff {
                self.n_visible_eff = new_vis;
                self.contract_count += 1;
            }
        }
    }

    /// GVOC score: 방문당 기대 정보이득
    pub fn gvoc_score(&self, root_visits: u32) -> f32 {
        self.last_voc / (root_visits + 1) as f32
    }

    pub fn print(&self, label: &str) {
        println!("╔══ GVOC State: {} ══", label);
        println!(
            "║  n_visible_eff={}  last_voc={:.4}",
            self.n_visible_eff, self.last_voc
        );
        println!(
            "║  expand×{}  contract×{}  iters={}",
            self.expand_count, self.contract_count, self.iterations
        );
        println!("╚══");
    }
}

// ─────────────────────────────────────────────
// § inside / outside proposal routing
// ─────────────────────────────────────────────

/// 현재 탐색 모드
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposalMode {
    /// 현재 top-k 후보 정밀화 (일반 MCTS)
    Inside,
    /// WL bias로 탐색 안 된 basin 시도
    Outside,
}

/// iteration당 proposal 모드 결정
///
/// 기준:
///   - p_hidden < hidden_thresh AND voc < expand_thresh → Inside (이미 충분)
///   - p_hidden ≥ hidden_thresh → Outside (hidden mode 탐사 필요)
///   - voc ≥ expand_thresh     → Inside  (현재 후보 더 정밀화)
pub fn routing_mode(stats: &QuartzStats, _qcfg: &QuartzConfig) -> ProposalMode {
    if stats.p_envar >= 0.2 {
        ProposalMode::Outside
    } else {
        ProposalMode::Inside
    }
}

// ─────────────────────────────────────────────
// § 단위 테스트
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::quartz::QuartzStats;

    #[test]
    fn test_gvoc_expand_on_high_voc() {
        let mut state = GvocState::new(GvocConfig::default(), 10);
        // VOC > expand_thresh → expand
        state.last_voc = 0.05;
        // 직접 simulate update effect
        if state.last_voc > state.cfg.expand_thresh {
            let new = (state.n_visible_eff + state.cfg.expand_delta)
                .min(64)
                .min(100);
            state.n_visible_eff = new;
            state.expand_count += 1;
        }
        assert!(state.n_visible_eff > 10, "should have expanded");
        assert_eq!(state.expand_count, 1);
    }

    #[test]
    fn test_gvoc_contract_on_low_voc() {
        let mut state = GvocState::new(GvocConfig::default(), 10);
        state.last_voc = 0.0001; // < contract_thresh
        if state.last_voc < state.cfg.contract_thresh && state.n_visible_eff > state.cfg.min_visible
        {
            state.n_visible_eff = state
                .n_visible_eff
                .saturating_sub(1)
                .max(state.cfg.min_visible);
            state.contract_count += 1;
        }
        assert!(state.n_visible_eff < 10);
        assert_eq!(state.contract_count, 1);
    }

    #[test]
    fn test_routing_mode() {
        let qcfg = QuartzConfig::default();

        let mut s = QuartzStats::default();
        s.p_envar = 0.1;
        s.voc_legacy = 0.001;
        assert_eq!(routing_mode(&s, &qcfg), ProposalMode::Inside);

        s.p_envar = 0.8;
        assert_eq!(routing_mode(&s, &qcfg), ProposalMode::Outside);
    }

    // Q6 (audit_codex_20260428.md W'8): integration test that calls
    // `GvocState::update()` with a real MCTS root rather than simulating
    // the body inline. The pre-Q6 tests at
    // `test_gvoc_expand_on_high_voc` / `test_gvoc_contract_on_low_voc`
    // re-implemented the threshold logic in test code, so a refactor of
    // `update()` could regress without tripping CI. This test plugs the
    // PW scheduler into a TicTacToe search with a `score_interval` of 1
    // and asserts (a) `iterations` is incremented per call and (b) the
    // `n_visible_eff` knob remains within its declared bounds.

    use crate::game::{Evaluator, GameState};
    use crate::games::tictactoe::TicTacToe;
    use crate::mcts::eval::UniformEval;
    use crate::mcts::quartz::QuartzController;
    use crate::mcts::{MctsConfig, MctsEngine, PwConfig};
    use std::sync::Arc;

    #[test]
    fn test_q6_gvoc_update_with_real_root_respects_bounds_and_advances_iters() {
        let cfg = MctsConfig::evaluation_with_pw(2.0, PwConfig::new(2.0, 0.5))
            .with_quartz(QuartzConfig::default());
        let eval: Arc<dyn Evaluator<TicTacToe> + Send + Sync> = Arc::new(UniformEval);
        let engine = MctsEngine::new(TicTacToe::initial(), eval, cfg);
        let mut ctrl = QuartzController::new(64, QuartzConfig::default());
        engine.run_quartz(&mut ctrl);

        let n_cands = engine.root.candidate_count();
        let qcfg = ctrl.cfg.clone();
        let gvoc_cfg = GvocConfig {
            score_interval: 1,
            min_visible: 2,
            max_visible: n_cands.max(2),
            ..Default::default()
        };
        let mut state = GvocState::new(gvoc_cfg.clone(), n_cands.max(2).min(4));

        let initial_visible = state.n_visible_eff;
        for _ in 0..3 {
            state.update(&engine.root, n_cands, &qcfg);
        }

        // (a) iterations counter advances by one per update call regardless
        //     of whether the threshold logic fires.
        assert_eq!(
            state.iterations, 3,
            "GvocState::update must increment iterations on every call"
        );
        // (b) n_visible_eff stays in [min_visible, max_visible] at all
        //     times — this is the bounds invariant the inline tests
        //     never exercised.
        assert!(state.n_visible_eff >= gvoc_cfg.min_visible);
        assert!(state.n_visible_eff <= gvoc_cfg.max_visible);
        // (c) every change must be reflected in either expand_count or
        //     contract_count; if neither fired, n_visible_eff must equal
        //     its starting value. Catches a class of off-by-one bugs in
        //     the side-effect bookkeeping.
        let changed = state.n_visible_eff != initial_visible;
        let counted = state.expand_count + state.contract_count > 0;
        assert_eq!(
            changed, counted,
            "n_visible_eff change must be matched by exactly one bookkeeping counter increment"
        );
    }

    #[test]
    fn test_q6_gvoc_disabled_when_below_score_interval() {
        // Q6: when score_interval is large relative to the number of
        // update() calls, the scheduler must NOT mutate any of its state
        // beyond the iterations counter. Verifies that the early-return
        // path at `if self.iterations % self.cfg.score_interval != 0`
        // is honored under real-engine inputs (not just inline mocks).
        let cfg = MctsConfig::evaluation_with_pw(2.0, PwConfig::new(2.0, 0.5))
            .with_quartz(QuartzConfig::default());
        let eval: Arc<dyn Evaluator<TicTacToe> + Send + Sync> = Arc::new(UniformEval);
        let engine = MctsEngine::new(TicTacToe::initial(), eval, cfg);
        let mut ctrl = QuartzController::new(64, QuartzConfig::default());
        engine.run_quartz(&mut ctrl);

        let n_cands = engine.root.candidate_count();
        let qcfg = ctrl.cfg.clone();
        // Large score_interval — only the call where `iterations %
        // score_interval == 0` does any work. With iterations=1..=3 and
        // score_interval=200, no real update body should ever execute.
        let mut state = GvocState::new(
            GvocConfig {
                score_interval: 200,
                ..Default::default()
            },
            8,
        );
        let initial_visible = state.n_visible_eff;
        for _ in 0..3 {
            state.update(&engine.root, n_cands, &qcfg);
        }
        assert_eq!(state.iterations, 3);
        assert_eq!(state.n_visible_eff, initial_visible);
        assert_eq!(state.expand_count, 0);
        assert_eq!(state.contract_count, 0);
    }
}
