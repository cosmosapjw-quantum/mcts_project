//! QUARTZ Prior Refresh Ablation Study
//!
//! ABLATION_PLAN_PRIOR_REFRESH_v1.md 기반 단계별 실험.
//! Phase 0: 통계적 검정력 확인 (Bootstrap CI)
//! Phase 1A: Penalty-only ablation (refresh off)
//! Phase 1B: GatedRefresh 구현 + 비교 (조건부)
//!
//! 실행: cargo test --release -- ablation_refresh --ignored --nocapture

use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{seq::SliceRandom, Rng, SeedableRng};

use crate::game::{Evaluator, GameState};
use crate::games::tictactoe::TicTacToe;
use crate::games::Gomoku;
use crate::mcts::eval::{ShortRollout, UniformEval};
use crate::mcts::mod_types::PwConfig;
use crate::mcts::quartz::{HaltMode, PenaltyMode, QuartzConfig, QuartzController};
use crate::mcts::{MctsConfig, MctsEngine};

// ═══════════════════════════════════════════════════════
// § Bootstrap CI
// ═══════════════════════════════════════════════════════

/// Bootstrap confidence interval for flip rate.
/// Returns (lower, upper) for given significance level alpha.
pub fn bootstrap_ci(flips: &[bool], n_boot: usize, alpha: f32) -> (f32, f32) {
    let n = flips.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut rng = StdRng::seed_from_u64(42);
    let mut boot_rates = Vec::with_capacity(n_boot);

    for _ in 0..n_boot {
        let mut count = 0u32;
        for _ in 0..n {
            if flips[rng.gen_range(0..n)] {
                count += 1;
            }
        }
        boot_rates.push(count as f32 / n as f32);
    }

    boot_rates.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let lo_idx = ((alpha / 2.0) * n_boot as f32) as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * n_boot as f32) as usize;
    (boot_rates[lo_idx], boot_rates[hi_idx.min(n_boot - 1)])
}

/// Test whether two flip rate CIs overlap.
/// Non-overlapping → statistically significant difference.
pub fn cis_overlap(ci_a: (f32, f32), ci_b: (f32, f32)) -> bool {
    ci_a.0 <= ci_b.1 && ci_b.0 <= ci_a.1
}

// ═══════════════════════════════════════════════════════
// § Flip rate measurement core
// ═══════════════════════════════════════════════════════

pub struct FlipResult {
    pub label: String,
    pub flips: Vec<bool>,
    pub rate: f32,
    pub ci: (f32, f32),
    pub avg_pflip: f32,
    pub avg_sigma_q: f32,
    pub ms: u128,
}

/// Run flip rate experiment: for each position, search with budget1 and budget2,
/// record whether best move changed (flip).
pub fn measure_flips<G, E>(
    label: &str,
    positions: &[G],
    eval: &Arc<E>,
    budget1: u32,
    budget2: u32,
    qcfg: QuartzConfig,
    base_config: MctsConfig,
) -> FlipResult
where
    G: GameState + Clone,
    E: Evaluator<G> + Send + Sync + 'static,
{
    let mut flips = Vec::with_capacity(positions.len());
    let mut total_pflip = 0.0f32;
    let mut total_sq = 0.0f32;
    let t0 = Instant::now();

    for state in positions {
        // Search 1 (low budget)
        let cfg1 = base_config.clone().with_quartz(QuartzConfig {
            halt_mode: HaltMode::Fixed { budget: budget1 },
            ..qcfg.clone()
        });
        let eng1 = MctsEngine::new(state.clone(), eval.clone(), cfg1);
        let mut ctrl1 = QuartzController::new(
            budget1,
            QuartzConfig {
                halt_mode: HaltMode::Fixed { budget: budget1 },
                ..qcfg.clone()
            },
        );
        eng1.run_quartz(&mut ctrl1);
        let best1 = eng1.best_move();
        let s1 = ctrl1.last_stats();
        total_pflip += s1.p_flip;
        total_sq += s1.sigma_q;

        // Search 2 (high budget)
        let cfg2 = base_config.clone().with_quartz(QuartzConfig {
            halt_mode: HaltMode::Fixed { budget: budget2 },
            ..qcfg.clone()
        });
        let eng2 = MctsEngine::new(state.clone(), eval.clone(), cfg2);
        let mut ctrl2 = QuartzController::new(
            budget2,
            QuartzConfig {
                halt_mode: HaltMode::Fixed { budget: budget2 },
                ..qcfg.clone()
            },
        );
        eng2.run_quartz(&mut ctrl2);
        let best2 = eng2.best_move();

        flips.push(best1 != best2);
    }

    let n = positions.len() as f32;
    let rate = flips.iter().filter(|&&f| f).count() as f32 / n;
    let ci = bootstrap_ci(&flips, 10_000, 0.05);

    FlipResult {
        label: label.to_string(),
        flips,
        rate,
        ci,
        avg_pflip: total_pflip / n,
        avg_sigma_q: total_sq / n,
        ms: t0.elapsed().as_millis(),
    }
}

pub fn print_result(r: &FlipResult) {
    let n = r.flips.len();
    let nf = r.flips.iter().filter(|&&f| f).count();
    eprintln!(
        "  {:<28} flip={:.3} [{:.3},{:.3}]  P_flip={:.3}  σ_Q={:.4}  ({}/{})  {}ms",
        r.label, r.rate, r.ci.0, r.ci.1, r.avg_pflip, r.avg_sigma_q, nf, n, r.ms
    );
}

// ═══════════════════════════════════════════════════════
// § Position generators
// ═══════════════════════════════════════════════════════

pub fn gen_ttt_positions(n: usize, seed: u64) -> Vec<TicTacToe> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut positions = Vec::new();
    for _ in 0..n * 10 {
        let n_moves = 1 + rng.gen::<usize>() % 5;
        let mut s = TicTacToe::initial();
        for _ in 0..n_moves {
            if s.is_terminal() {
                break;
            }
            let legal = s.legal_moves();
            s = s.apply_move(legal[rng.gen::<usize>() % legal.len()]);
        }
        if !s.is_terminal() && s.legal_moves().len() >= 2 {
            positions.push(s);
            if positions.len() >= n {
                break;
            }
        }
    }
    positions
}

pub fn gen_gomoku_positions(n: usize, seed: u64) -> Vec<Gomoku> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut positions = Vec::new();
    for _ in 0..n * 10 {
        let n_moves = 4 + rng.gen::<usize>() % 12;
        let mut mvs: Vec<usize> = (0..49).collect();
        mvs.shuffle(&mut rng);
        mvs.truncate(n_moves);
        let mut s = Gomoku::new_with_win(7, 4);
        let mut ok = true;
        for &mv in &mvs {
            if s.is_terminal() {
                ok = false;
                break;
            }
            s = s.apply_move(mv);
        }
        if ok && !s.is_terminal() && s.legal_moves().len() >= 2 {
            positions.push(s);
            if positions.len() >= n {
                break;
            }
        }
    }
    positions
}

// ═══════════════════════════════════════════════════════
// § Phase 0: Statistical power verification
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Bootstrap CI unit tests ──

    #[test]
    fn test_bootstrap_ci_all_true() {
        let all = vec![true; 50];
        let (lo, hi) = bootstrap_ci(&all, 10_000, 0.05);
        assert!((lo - 1.0).abs() < 0.01);
        assert!((hi - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_bootstrap_ci_half() {
        let half: Vec<bool> = (0..50).map(|i| i < 25).collect();
        let (lo, hi) = bootstrap_ci(&half, 10_000, 0.05);
        assert!(lo > 0.30 && lo < 0.45, "lo={}", lo);
        assert!(hi > 0.55 && hi < 0.70, "hi={}", hi);
    }

    #[test]
    fn test_bootstrap_ci_none() {
        let none = vec![false; 50];
        let (lo, hi) = bootstrap_ci(&none, 10_000, 0.05);
        assert!(lo.abs() < 0.01);
        assert!(hi.abs() < 0.05);
    }

    #[test]
    fn test_ci_overlap() {
        assert!(cis_overlap((0.2, 0.4), (0.3, 0.5))); // overlap
        assert!(!cis_overlap((0.2, 0.3), (0.4, 0.5))); // no overlap
        assert!(cis_overlap((0.2, 0.4), (0.4, 0.5))); // edge touch
    }

    // ── Phase 0A: TTT + UniformEval ──

    #[test]
    #[ignore]
    fn phase_0a_ttt_ci() {
        eprintln!("\n{}", "═".repeat(70));
        eprintln!("  Phase 0A: TTT + UniformEval — Flip Rate CI (seed=42, N=50)");
        eprintln!("  Reproduces Exp-6A configs with bootstrap 95% CI.");
        eprintln!("{}\n", "═".repeat(70));

        let positions = gen_ttt_positions(50, 42);
        eprintln!("  {} TTT positions generated\n", positions.len());

        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        let base = MctsConfig::evaluation(2.0);
        let b1 = 2000u32;
        let b2 = 10000u32;

        let configs: Vec<(&str, QuartzConfig)> = vec![
            (
                "NoPenalty",
                QuartzConfig {
                    penalty_mode: PenaltyMode::None,
                    min_visits: 30,
                    check_interval: 50,
                    prior_refresh_rate: 0.0,
                    ..Default::default()
                },
            ),
            (
                "Legacy_0.3",
                QuartzConfig {
                    penalty_mode: PenaltyMode::Legacy,
                    hbar_penalty_cap: 0.3,
                    min_visits: 30,
                    check_interval: 50,
                    prior_refresh_rate: 0.0,
                    ..Default::default()
                },
            ),
            (
                "EffV2_0.3",
                QuartzConfig {
                    penalty_mode: PenaltyMode::EffectiveV2,
                    hbar_penalty_cap: 0.3,
                    min_visits: 30,
                    check_interval: 50,
                    prior_refresh_rate: 0.0,
                    ..Default::default()
                },
            ),
            (
                "NoPen+Refresh",
                QuartzConfig {
                    penalty_mode: PenaltyMode::None,
                    min_visits: 30,
                    check_interval: 50,
                    prior_refresh_rate: 0.3,
                    prior_refresh_temp: 0.5,
                    ..Default::default()
                },
            ),
            (
                "EffV2+Refresh",
                QuartzConfig {
                    penalty_mode: PenaltyMode::EffectiveV2,
                    hbar_penalty_cap: 0.3,
                    min_visits: 30,
                    check_interval: 50,
                    prior_refresh_rate: 0.3,
                    prior_refresh_temp: 0.5,
                    ..Default::default()
                },
            ),
        ];

        let mut results = Vec::new();
        for (label, qcfg) in &configs {
            let r = measure_flips(label, &positions, &eval, b1, b2, qcfg.clone(), base.clone());
            print_result(&r);
            results.push(r);
        }

        // ── CI overlap analysis ──
        eprintln!("\n  ── CI Overlap Analysis ──");
        let baseline = &results[0]; // NoPenalty
        for r in &results[1..] {
            let overlap = cis_overlap(baseline.ci, r.ci);
            let sig = if overlap {
                "NOT significant"
            } else {
                "SIGNIFICANT"
            };
            eprintln!(
                "  {} vs {}: {} (overlap={})",
                baseline.label, r.label, sig, overlap
            );
        }

        // Key comparison: NoPen+Refresh vs NoPenalty
        let refresh_idx = results
            .iter()
            .position(|r| r.label == "NoPen+Refresh")
            .unwrap();
        let overlap_key = cis_overlap(baseline.ci, results[refresh_idx].ci);
        eprintln!("\n  ── DECISION (Phase 0A) ──");
        if overlap_key {
            eprintln!("  CIs OVERLAP → Refresh effect NOT statistically significant at 95%.");
            eprintln!("  → Proceed to Phase 1A (penalty-only ablation) to confirm.");
        } else {
            eprintln!("  CIs DO NOT OVERLAP → Refresh effect IS statistically significant.");
            eprintln!("  → Proceed to Phase 1A+1B.");
        }
        eprintln!();
    }

    // ── Phase 0B: Gomoku 7×7 + ShortRollout ──

    #[test]
    #[ignore]
    fn phase_0b_gomoku_ci() {
        eprintln!("\n{}", "═".repeat(70));
        eprintln!("  Phase 0B: Gomoku 7×7 + ShortRollout — Flip Rate CI (seed=123, N=20)");
        eprintln!("{}\n", "═".repeat(70));

        let positions = gen_gomoku_positions(20, 123);
        eprintln!("  {} Gomoku positions generated\n", positions.len());

        let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(12));
        let base = MctsConfig::evaluation_with_pw(2.0, PwConfig::default_gomoku());
        let b1 = 150u32;
        let b2 = 600u32;

        let configs: Vec<(&str, QuartzConfig)> = vec![
            (
                "NoPenalty",
                QuartzConfig {
                    penalty_mode: PenaltyMode::None,
                    min_visits: 15,
                    check_interval: 20,
                    prior_refresh_rate: 0.0,
                    ..Default::default()
                },
            ),
            (
                "Legacy_0.3",
                QuartzConfig {
                    penalty_mode: PenaltyMode::Legacy,
                    hbar_penalty_cap: 0.3,
                    min_visits: 15,
                    check_interval: 20,
                    prior_refresh_rate: 0.0,
                    ..Default::default()
                },
            ),
            (
                "NoPen+Refresh",
                QuartzConfig {
                    penalty_mode: PenaltyMode::None,
                    min_visits: 15,
                    check_interval: 20,
                    prior_refresh_rate: 0.3,
                    prior_refresh_temp: 0.5,
                    ..Default::default()
                },
            ),
            (
                "Leg+Refresh",
                QuartzConfig {
                    penalty_mode: PenaltyMode::Legacy,
                    hbar_penalty_cap: 0.3,
                    min_visits: 15,
                    check_interval: 20,
                    prior_refresh_rate: 0.3,
                    prior_refresh_temp: 0.5,
                    ..Default::default()
                },
            ),
        ];

        let mut results = Vec::new();
        for (label, qcfg) in &configs {
            let r = measure_flips(label, &positions, &eval, b1, b2, qcfg.clone(), base.clone());
            print_result(&r);
            results.push(r);
        }

        eprintln!("\n  ── CI Overlap Analysis ──");
        let baseline = &results[0];
        for r in &results[1..] {
            let overlap = cis_overlap(baseline.ci, r.ci);
            let sig = if overlap { "NOT sig" } else { "SIG" };
            eprintln!(
                "  {} vs {}: {} [{:.3},{:.3}] vs [{:.3},{:.3}]",
                baseline.label, r.label, sig, baseline.ci.0, baseline.ci.1, r.ci.0, r.ci.1
            );
        }
        eprintln!();
    }

    // ═══════════════════════════════════════════════════
    // § Phase 1A: Penalty-only ablation
    // ═══════════════════════════════════════════════════

    #[test]
    #[ignore]
    fn phase_1a_penalty_only() {
        eprintln!("\n{}", "═".repeat(70));
        eprintln!("  Phase 1A: TTT Penalty-Only Ablation (refresh=off, N=50)");
        eprintln!("  Question: Which penalty mode best reduces flip rate WITHOUT refresh?");
        eprintln!("{}\n", "═".repeat(70));

        let positions = gen_ttt_positions(50, 42);
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        let base = MctsConfig::evaluation(2.0);
        let b1 = 2000u32;
        let b2 = 10000u32;

        let configs: Vec<(&str, QuartzConfig)> = vec![
            (
                "NoPenalty",
                QuartzConfig {
                    penalty_mode: PenaltyMode::None,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
            (
                "Legacy_0.1",
                QuartzConfig {
                    penalty_mode: PenaltyMode::Legacy,
                    hbar_penalty_cap: 0.1,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
            (
                "Legacy_0.3",
                QuartzConfig {
                    penalty_mode: PenaltyMode::Legacy,
                    hbar_penalty_cap: 0.3,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
            (
                "Legacy_0.5",
                QuartzConfig {
                    penalty_mode: PenaltyMode::Legacy,
                    hbar_penalty_cap: 0.5,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
            (
                "EffV2_0.1",
                QuartzConfig {
                    penalty_mode: PenaltyMode::EffectiveV2,
                    hbar_penalty_cap: 0.1,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
            (
                "EffV2_0.3",
                QuartzConfig {
                    penalty_mode: PenaltyMode::EffectiveV2,
                    hbar_penalty_cap: 0.3,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
            (
                "EffV2_0.5",
                QuartzConfig {
                    penalty_mode: PenaltyMode::EffectiveV2,
                    hbar_penalty_cap: 0.5,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
            (
                "SelfAdaptive",
                QuartzConfig {
                    penalty_mode: PenaltyMode::SelfAdaptive,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
        ];

        let mut results = Vec::new();
        for (label, qcfg) in &configs {
            let r = measure_flips(label, &positions, &eval, b1, b2, qcfg.clone(), base.clone());
            print_result(&r);
            results.push(r);
        }

        // Find best penalty-only config
        let best = results
            .iter()
            .min_by(|a, b| a.rate.partial_cmp(&b.rate).unwrap())
            .unwrap();

        eprintln!(
            "\n  ── Best Penalty-Only: {} (flip={:.3} [{:.3},{:.3}]) ──",
            best.label, best.rate, best.ci.0, best.ci.1
        );

        // Compare with NoPen+Refresh from Phase 0A
        eprintln!("  → Compare with Phase 0A NoPen+Refresh to decide Phase 1B.\n");
    }

    #[test]
    #[ignore]
    fn phase_1a_extended_n200() {
        eprintln!("\n{}", "═".repeat(70));
        eprintln!("  Phase 1A-ext: TTT N=200 — SelfAdaptive vs NoPenalty vs Refresh");
        eprintln!("{}\n", "═".repeat(70));
        let positions = gen_ttt_positions(200, 42);
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        let base = MctsConfig::evaluation(2.0);
        let (b1, b2) = (2000u32, 10000u32);
        let configs: Vec<(&str, QuartzConfig)> = vec![
            (
                "NoPenalty",
                QuartzConfig {
                    penalty_mode: PenaltyMode::None,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
            (
                "SelfAdaptive",
                QuartzConfig {
                    penalty_mode: PenaltyMode::SelfAdaptive,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
            (
                "NoPen+Refresh",
                QuartzConfig {
                    penalty_mode: PenaltyMode::None,
                    min_visits: 30,
                    check_interval: 50,
                    prior_refresh_rate: 0.3,
                    prior_refresh_temp: 0.5,
                    ..Default::default()
                },
            ),
            (
                "SA+Refresh",
                QuartzConfig {
                    penalty_mode: PenaltyMode::SelfAdaptive,
                    min_visits: 30,
                    check_interval: 50,
                    prior_refresh_rate: 0.3,
                    prior_refresh_temp: 0.5,
                    ..Default::default()
                },
            ),
        ];
        let mut results = Vec::new();
        for (label, qcfg) in &configs {
            let r = measure_flips(label, &positions, &eval, b1, b2, qcfg.clone(), base.clone());
            print_result(&r);
            results.push(r);
        }
        eprintln!("\n  ── Pairwise CI ──");
        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                let ov = cis_overlap(results[i].ci, results[j].ci);
                eprintln!(
                    "  {} vs {}: {} (Δ={:.3})",
                    results[i].label,
                    results[j].label,
                    if ov { "overlap" } else { "*** SIG ***" },
                    (results[i].rate - results[j].rate).abs()
                );
            }
        }
        let sa = &results[1];
        let refresh = &results[2];
        eprintln!("\n  ── DECISION ──");
        if sa.rate < refresh.rate && !cis_overlap(sa.ci, refresh.ci) {
            eprintln!("  SCENARIO A: SelfAdaptive < Refresh, SIGNIFICANT. Refresh 불필요.");
        } else if sa.rate <= refresh.rate {
            eprintln!("  SCENARIO A 경향: SA ≤ Refresh, CIs overlap. Default=SelfAdaptive.");
        } else {
            eprintln!("  Proceed to Phase 1B.");
        }
        eprintln!();
    }
}
