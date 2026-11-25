//! Standalone research transport using the extracted Quartz native search.
//! Every completed traversal is observed; no generation is implicit in serve().
use crate::game::{EvalResult, Evaluator, GameState};
use crate::games::Gomoku15;
use crate::mcts::node::atomic_f64_load;
use crate::mcts::parallel::VlMode;
use crate::mcts::search::FixedIterations;
use crate::mcts::{ImmediateReason, MctsConfig, MctsEngine, PreparedIteration};
use crate::mqft_trace::{self, Event, Trace};
use parking_lot::Mutex;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::Instant;

pub const SCHEMA: &str = "mqft.quartz.trace.v2";
const QUARTZ: &str = "65e7d3e9596ca259ebfe523b0c74058692b6a755";
fn default_c() -> f32 {
    1.4
}
fn default_depth() -> usize {
    225
}
fn default_samples() -> usize {
    1
}
fn default_evaluator() -> String {
    "uniform_rollout".into()
}
fn default_prior() -> String {
    "local".into()
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub mode: String,
    pub workers: usize,
    pub budget: u32,
    pub cuts: Vec<u32>,
    pub seed: u64,
    #[serde(default = "default_c")]
    pub c_puct: f32,
    #[serde(default = "default_depth")]
    pub rollout_depth: usize,
    #[serde(default = "default_samples")]
    pub samples_per_eval: usize,
    #[serde(default = "default_prior")]
    pub prior: String,
    #[serde(default)]
    pub source_action: Option<usize>,
    #[serde(default)]
    pub source_h: f64,
    #[serde(default)]
    pub root_law: Option<crate::mqft_law::LawSpec>,
    #[serde(default)]
    pub record_rollouts: bool,
    #[serde(default = "default_evaluator")]
    pub evaluator: String,
}
impl Spec {
    fn validate(&self) -> Result<(), String> {
        if !["serial", "root", "tree", "leaf", "wave"].contains(&self.mode.as_str()) {
            return Err("unknown mode".into());
        }
        if self.workers == 0 || self.workers > 64 || self.budget == 0 || self.budget > 65536 {
            return Err("workers 1..64; budget 1..65536".into());
        }
        if self.mode == "serial" && self.workers != 1 {
            return Err("serial requires workers=1".into());
        }
        if self.mode == "leaf" && self.samples_per_eval < 2 {
            return Err("leaf requires >=2 samples_per_eval".into());
        }
        if self.samples_per_eval == 0
            || self.samples_per_eval > 64
            || self.rollout_depth == 0
            || self.rollout_depth > 225
        {
            return Err("sample or rollout depth bounds".into());
        }
        if !self.c_puct.is_finite()
            || self.c_puct < 1e-6
            || self.c_puct > 1000.
            || !self.source_h.is_finite()
            || self.source_h.abs() > 20.
            || !["local", "uniform"].contains(&self.prior.as_str())
        {
            return Err("invalid c_puct, source or prior".into());
        }
        if self.cuts.is_empty()
            || self.cuts[0] == 0
            || self.cuts.last() != Some(&self.budget)
            || self.cuts.windows(2).any(|x| x[0] >= x[1])
        {
            return Err("cuts must be positive increasing, ending at budget".into());
        }
        if self.source_h != 0. && self.source_action.is_none() {
            return Err("source_h needs source_action".into());
        }
        if self.mode == "root" && self.workers > self.budget as usize {
            return Err("root budget must cover every replica".into());
        }
        if !["uniform_rollout", crate::mqft_rapfi::ID].contains(&self.evaluator.as_str()) {
            return Err("unknown evaluator".into());
        }
        if self.evaluator == crate::mqft_rapfi::ID
            && (self.rollout_depth != 1 || !self.record_rollouts || self.prior != "local")
        {
            return Err("NNUE lookahead requires depth=1, recorded actions, local prior placeholder (actual NNUE prior)".into());
        }
        if let Some(law) = &self.root_law {
            law.validate(self.budget, &self.mode, self.source_action)?;
            // Conservative representable-full-support contract, including pending VL.
            let max_beta = law
                .beta_p
                .iter()
                .chain(&law.beta_q)
                .copied()
                .fold(0., f64::max);
            let span =
                2. + self.c_puct as f64 * ((self.budget as f64 + self.workers as f64).sqrt());
            if max_beta * span + law.logit_h.abs() > 640. {
                return Err("root-law full support may underflow: reduce beta/c/budget".into());
            }
        }
        Ok(())
    }
}
fn from_moves(moves: &[usize]) -> Result<Gomoku15, String> {
    let mut s = Gomoku15::freestyle();
    for &a in moves {
        if s.is_terminal() {
            return Err("moves after terminal".into());
        }
        let m = s
            .idx_to_move(a)
            .filter(|m| s.legal_moves().contains(m))
            .ok_or_else(|| format!("illegal action {a}"))?;
        s.apply_move_in_place_no_undo(m);
    }
    Ok(s)
}
fn state_value(s: &Gomoku15) -> Value {
    let terminal = s.is_terminal();
    let winner = if terminal && s.outcome() != 0. {
        Some(if s.outcome() > 0. {
            s.current_player()
        } else {
            -s.current_player()
        })
    } else {
        None
    };
    json!({"board":s.board_state_record(),"side":s.current_player(),"terminal":terminal,"winner":winner,"legal":s.legal_moves(),"state_hash":format!("{:016x}",s.tt_hash())})
}
fn mix(x: u64) -> u64 {
    crate::game::tt_mix64(x.wrapping_add(0x9e3779b97f4a7c15))
}
#[derive(Debug, Clone, Serialize)]
pub struct Sample {
    seed: String,
    value: f32,
    plies: usize,
    terminal: bool,
    actions: Option<Vec<usize>>,
    log_path_prob: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    nnue_wdl: Option<[f64; 3]>,
}
#[derive(Debug, Clone, Serialize)]
pub struct Evaluation {
    id: u64,
    state_hash: String,
    side: i8,
    value: f32,
    samples: Vec<Sample>,
    elapsed_ns: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    nnue_scores: Option<Vec<(u16, i32)>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    nnue_wdl: Option<[f64; 3]>,
}
struct ResearchEval {
    spec: Spec,
    root_hash: u64,
    seed: u64,
    calls: AtomicU64,
    records: Mutex<Vec<Evaluation>>,
}
impl ResearchEval {
    fn new(spec: &Spec, root_hash: u64, seed: u64) -> Self {
        Self {
            spec: spec.clone(),
            root_hash,
            seed,
            calls: AtomicU64::new(0),
            records: Mutex::new(Vec::new()),
        }
    }
    fn sample(&self, state: &Gomoku15, seed: u64) -> Sample {
        let mut s = state.clone();
        let side = s.current_player();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut plies = 0;
        let mut actions = self.spec.record_rollouts.then(Vec::new);
        let mut log_path_prob = 0.;
        while !s.is_terminal() && plies < self.spec.rollout_depth {
            // Same rollout policy in all schedules, independent draw per rollout.
            let legal = s.legal_moves();
            if legal.is_empty() {
                break;
            }
            let m = legal[rng.gen_range(0..legal.len())];
            log_path_prob -= (legal.len() as f64).ln();
            if let Some(ref mut a) = actions {
                a.push(m as usize);
            }
            s.apply_move_in_place_no_undo(m);
            plies += 1;
        }
        let terminal = s.is_terminal();
        // A truncated rollout is explicitly marked; its chosen cutoff value is zero.
        let value = if terminal {
            s.outcome() * if s.current_player() == side { 1. } else { -1. }
        } else {
            0.
        };
        Sample {
            seed: format!("{seed:016x}"),
            value,
            plies,
            terminal,
            actions,
            log_path_prob,
            nnue_wdl: None,
        }
    }
    fn nnue_sample(&self, state: &Gomoku15, p: &[(u16, f64)], seed: u64) -> Sample {
        let mut rng = StdRng::seed_from_u64(seed);
        let u = rng.gen::<f64>();
        let mut sum = 0.;
        let mut selected = p.last().expect("nonempty NNUE support");
        for item in p {
            sum += item.1;
            if u < sum {
                selected = item;
                break;
            }
        }
        let child = state.apply_move(selected.0);
        let terminal = child.is_terminal();
        let wdl = if terminal {
            None
        } else {
            Some(
                crate::mqft_rapfi::query(&child, false)
                    .expect("actual MixNet child evaluation failed")
                    .1,
            )
        };
        let value = if terminal {
            -child.outcome()
        } else {
            -crate::mqft_rapfi::value(wdl.unwrap(), child.current_player())
        };
        Sample {
            seed: format!("{seed:016x}"),
            value,
            plies: 1,
            terminal,
            actions: Some(vec![selected.0 as usize]),
            log_path_prob: selected.1.ln(),
            nnue_wdl: wdl,
        }
    }
    fn priors(&self, state: &Gomoku15) -> Vec<(u16, f32)> {
        let board = state.board_state_record();
        let legal = state.legal_moves();
        let mut p: Vec<(u16, f64)> = legal
            .into_iter()
            .map(|m| {
                let a = m as usize;
                let row = a / 15;
                let col = a % 15;
                let mut w = 1.;
                if self.spec.prior == "local" {
                    let mut near = 0;
                    for dr in -1i32..=1 {
                        for dc in -1i32..=1 {
                            let y = row as i32 + dr;
                            let x = col as i32 + dc;
                            if (0..15).contains(&y)
                                && (0..15).contains(&x)
                                && board[(y * 15 + x) as usize] != 0
                            {
                                near += 1
                            }
                        }
                    }
                    w += 0.35 * near as f64 + 0.1 * (14 - row.abs_diff(7) - col.abs_diff(7)) as f64;
                }
                if state.tt_hash() == self.root_hash && self.spec.source_action == Some(a) {
                    w *= self.spec.source_h.exp();
                }
                (m, w)
            })
            .collect();
        let z: f64 = p.iter().map(|x| x.1).sum();
        p.sort_by_key(|x| x.0);
        p.into_iter().map(|(m, w)| (m, (w / z) as f32)).collect()
    }
}
impl Evaluator<Gomoku15> for ResearchEval {
    fn evaluate(&self, state: &Gomoku15) -> EvalResult<u16> {
        let t = Instant::now();
        let actual_call = self.calls.fetch_add(1, Ordering::Relaxed);
        let id = mqft_trace::take_eval_ticket().unwrap_or(actual_call);
        let seed = mix(self.seed ^ mix(id) ^ state.tt_hash());
        let n = self.spec.samples_per_eval;
        let (nnue_scores, nnue_wdl, nnue_p) = if self.spec.evaluator == crate::mqft_rapfi::ID {
            let (scores, wdl) = crate::mqft_rapfi::query(state, true)
                .expect("actual MixNet parent evaluation failed");
            let scores = scores.expect("missing NNUE policy");
            let p = crate::mqft_rapfi::probabilities(&scores);
            (Some(scores), Some(wdl), Some(p))
        } else {
            (None, None, None)
        };
        let sample = |j: usize| {
            if let Some(p) = &nnue_p {
                self.nnue_sample(state, p, mix(seed ^ j as u64))
            } else {
                self.sample(state, mix(seed ^ j as u64))
            }
        };
        let samples: Vec<_> = if self.spec.mode == "leaf" {
            (0..n).into_par_iter().map(sample).collect()
        } else {
            (0..n).map(sample).collect()
        };
        let value = samples.iter().map(|x| x.value as f64).sum::<f64>() / n as f64;
        let value = value as f32;
        let policy = if let Some(p) = nnue_p {
            // Prior source changes tree selection only. Evaluator sampling law remains fixed.
            let mut p = p;
            if state.tt_hash() == self.root_hash {
                if let Some(a) = self.spec.source_action {
                    for (m, w) in &mut p {
                        if *m as usize == a {
                            *w *= self.spec.source_h.exp();
                        }
                    }
                }
            }
            let z: f64 = p.iter().map(|x| x.1).sum();
            p.into_iter().map(|(m, w)| (m, (w / z) as f32)).collect()
        } else {
            self.priors(state)
        };
        self.records.lock().push(Evaluation {
            id,
            state_hash: format!("{:016x}", state.tt_hash()),
            side: state.current_player(),
            value,
            samples,
            elapsed_ns: t.elapsed().as_nanos() as u64,
            nnue_scores,
            nnue_wdl,
        });
        mqft_trace::set_eval(id);
        EvalResult { policy, value }
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct Edge {
    action: usize,
    prior: f32,
    n: u32,
    w: f64,
    s2: f64,
    pending: i32,
    pending_value: f64,
}
#[derive(Debug, Clone, Serialize)]
pub struct Cut {
    global_k: u32,
    local_k: u32,
    edges: Vec<Edge>,
    root_w: f64,
    allocated_nodes: usize,
    eval_calls: u64,
    rollout_samples: u64,
    rollout_plies: u64,
    elapsed_ns: u64,
}
#[derive(Debug, Serialize)]
pub struct Replica {
    replica: usize,
    seed: String,
    cuts: Vec<Cut>,
    root_order: Vec<usize>,
    events: Vec<Event>,
    evaluations: Vec<Evaluation>,
}
struct Work {
    engine: MctsEngine<Gomoku15>,
    eval: Arc<ResearchEval>,
    trace: Arc<Trace>,
    seed: u64,
    cuts: Vec<Cut>,
    wave_seq: u32,
}
impl Work {
    fn new(state: &Gomoku15, spec: &Spec, seed: u64) -> Self {
        let eval = Arc::new(ResearchEval::new(spec, state.tt_hash(), seed));
        let mut cfg = MctsConfig::evaluation(spec.c_puct);
        // Fixed baseline semantics; TT-off uses the repaired arena ownership, full legal support.
        cfg.tt_enabled = false;
        cfg.root_forced_win = false;
        cfg.exact_terminal_value = false;
        cfg.fpu_reduction = 0.;
        cfg.vl_mode = if ["tree", "wave"].contains(&spec.mode.as_str()) {
            VlMode::Fixed
        } else {
            VlMode::Disabled
        };
        cfg.seed = Some(seed);
        cfg.max_depth = 0;
        cfg.pw = None;
        let mut engine = MctsEngine::new(state.clone(), eval.clone(), cfg);
        mqft_trace::clear_eval();
        let trace = Arc::new(Trace::default());
        engine.mqft_trace = Some(trace.clone());
        engine.par_ctrl.mqft_record = true;
        if let Some(law) = &spec.root_law {
            engine.par_ctrl.mqft_sampler = Some(Mutex::new(crate::mqft_law::Sampler::new(
                law.clone(),
                spec.source_action,
                mix(seed ^ 0x52_4f_4f_54),
            )));
        }
        Self {
            engine,
            eval,
            trace,
            seed,
            cuts: Vec::new(),
            wave_seq: 0,
        }
    }
    fn n(&self) -> u32 {
        self.engine.root.n_total.load(Ordering::Acquire)
    }
    fn step_wave(&mut self, width: usize) -> Result<(), String> {
        // All paths reserved before any evaluation or backup, synchronous barrier.
        self.wave_seq += 1;
        self.trace.set_phase(self.wave_seq);
        self.engine.par_ctrl.set_n_threads(width as u32);
        let mut prepared = Vec::with_capacity(width);
        for _ in 0..width {
            let p = self.engine.prepare_iteration_async();
            let path = match &p {
                PreparedIteration::Immediate { path, .. } => path,
                PreparedIteration::Pending(p) => &p.path,
            };
            if path.is_empty() {
                return Err("empty selected path".into());
            }
            prepared.push((self.trace.begin(self.engine.root_state(), path), p));
        }
        // Rayon CPU evaluation across distinct selected paths; no fake NN batch/GPU.
        let mut next_id = self.eval.calls.load(Ordering::Acquire);
        let tickets: Vec<_> = prepared
            .iter()
            .map(|(_, p)| match p {
                PreparedIteration::Pending(_) => {
                    let id = next_id;
                    next_id += 1;
                    Some(id)
                }
                _ => None,
            })
            .collect();
        let evals: Vec<_> = prepared
            .par_iter()
            .zip(tickets.par_iter())
            .map(|((_, p), ticket)| {
                mqft_trace::clear_eval();
                match p {
                    PreparedIteration::Pending(p) => {
                        mqft_trace::set_eval_ticket(ticket.expect("pending ticket"));
                        let v = self.engine.evaluator.evaluate(&p.leaf_state);
                        (Some(v), mqft_trace::take_eval())
                    }
                    _ => (None, None),
                }
            })
            .collect();
        // Commit in selection order, after the whole evaluation barrier.
        for ((sel, p), (ev, id)) in prepared.into_iter().zip(evals) {
            match p {
                PreparedIteration::Immediate {
                    path,
                    value,
                    reason,
                } => {
                    if reason != ImmediateReason::TerminalNode {
                        return Err("unexpected capacity truncation".into());
                    }
                    let clock = self.engine.apply_iteration_value_async(path, value);
                    self.trace.finish(sel, value, None, true, clock);
                }
                PreparedIteration::Pending(p) => {
                    let value = crate::mcts::expand::expand_with_result(
                        &p.leaf,
                        &p.leaf_state,
                        ev.expect("pending eval"),
                        &self.engine.tt,
                        self.engine.config.pw.as_ref(),
                    );
                    let clock = self.engine.apply_iteration_value_async(p.path, value);
                    self.trace.finish(sel, value, id, false, clock);
                }
            }
        }
        Ok(())
    }
    fn run_to(&mut self, target: u32, global: u32, spec: &Spec) -> Result<(), String> {
        if target < self.n() {
            return Err("decreasing budget".into());
        }
        let start = Instant::now();
        match spec.mode.as_str() {
            "tree" => {
                self.trace.set_phase(global);
                self.engine
                    .run_par(&FixedIterations::new(target), spec.workers);
            }
            "wave" => {
                while self.n() < target {
                    let width = spec.workers.min((target - self.n()) as usize);
                    let width = if spec.root_law.as_ref().is_some_and(|x| x.kind == "pl") {
                        width.min(self.engine.root_state().legal_moves().len())
                    } else {
                        width
                    };
                    self.step_wave(width)?;
                }
            }
            _ => {
                self.trace.set_phase(global);
                self.engine.run(&mut FixedIterations::new(target));
            }
        }
        let n = self.n();
        let es = self.trace.events.lock();
        if n != target || es.len() != n as usize {
            return Err(format!(
                "actual count mismatch {n} target {target} events {}",
                es.len()
            ));
        }
        let mut sums = vec![(0u32, 0f64, 0f64); 225];
        for e in es.iter() {
            let a = *e.selection.actions.first().ok_or("empty path")?;
            let z = e.root_value as f64;
            sums[a].0 += 1;
            sums[a].1 += z;
            sums[a].2 += z * z;
        }
        // Prefix-indexed replay, not TT hashes (TT is disabled). Compare actual
        // native sums at every visited edge, including deeper pending counters.
        let mut prefix = std::collections::HashMap::<Vec<usize>, (u32, f64, f64)>::new();
        for event in es.iter() {
            let mut key = Vec::new();
            for (d, a) in event.selection.actions.iter().enumerate() {
                key.push(*a);
                let mark = if d % 2 == 0 {
                    event.root_value as f64
                } else {
                    -(event.root_value as f64)
                };
                let z = prefix.entry(key.clone()).or_insert((0, 0., 0.));
                z.0 += 1;
                z.1 += mark;
                z.2 += mark * mark;
            }
        }
        let mut stack = vec![(self.engine.root, Vec::<usize>::new())];
        while let Some((node, path)) = stack.pop() {
            let mut total = 0;
            let mut wtot = 0.;
            for e in node.read_edges() {
                let mut key = path.clone();
                key.push(e.mv as usize);
                let expected = prefix.remove(&key).unwrap_or((0, 0., 0.));
                let n = e.n.load(Ordering::Acquire);
                let w = atomic_f64_load(&e.w);
                let s2 = atomic_f64_load(&e.s2);
                let m2 = atomic_f64_load(&e.m2);
                let m2_ref = if n > 0 { s2 - w * w / n as f64 } else { 0. };
                let tol = 1e-9 * (1. + n as f64);
                if n != expected.0
                    || (w - expected.1).abs() > tol
                    || (s2 - expected.2).abs() > tol
                    || (m2 - m2_ref).abs() > tol
                    || e.virtual_losses.load(Ordering::Acquire) != 0
                    || atomic_f64_load(&e.virtual_value).abs() > tol
                {
                    return Err("deep edge moment or reservation audit failed".into());
                }
                total += n;
                wtot += w;
                if n > 0 {
                    stack.push((e.child, key));
                }
            }
            if total != node.n_total.load(Ordering::Acquire)
                || (wtot - atomic_f64_load(&node.w_total)).abs() > 1e-9 * (1. + total as f64)
            {
                return Err("deep node conservation failed".into());
            }
        }
        if !prefix.is_empty() {
            return Err("path replay has unmatched edges".into());
        }
        let mut edges = Vec::new();
        for e in self.engine.root.read_edges() {
            let a = e.mv as usize;
            let count = e.n.load(Ordering::Acquire);
            let w = atomic_f64_load(&e.w);
            let s2 = atomic_f64_load(&e.s2);
            let m2 = atomic_f64_load(&e.m2);
            let m2_ref = if count > 0 {
                s2 - w * w / count as f64
            } else {
                0.
            };
            let pending = e.virtual_losses.load(Ordering::Acquire);
            let pv = atomic_f64_load(&e.virtual_value);
            if count != sums[a].0
                || (w - sums[a].1).abs() > 1e-9 * (1. + count as f64)
                || (s2 - sums[a].2).abs() > 1e-9 * (1. + count as f64)
                || (m2 - m2_ref).abs() > 1e-9 * (1. + count as f64)
                || pending != 0
                || pv.abs() > 1e-9
            {
                return Err(format!("edge audit failed action {a}"));
            }
            edges.push(Edge {
                action: a,
                prior: e.p,
                n: count,
                w,
                s2,
                pending,
                pending_value: pv,
            });
        }
        edges.sort_by_key(|e| e.action);
        let legal = self.engine.root_state().legal_moves();
        let mut a = legal.clone();
        a.sort_unstable();
        if edges.iter().map(|e| e.action).collect::<Vec<_>>()
            != a.iter().map(|&x| x as usize).collect::<Vec<_>>()
            || edges.iter().map(|e| e.n).sum::<u32>() != n
        {
            return Err("root legal support/count mismatch".into());
        }
        let root_w = atomic_f64_load(&self.engine.root.w_total);
        if (root_w - edges.iter().map(|e| e.w).sum::<f64>()).abs() > 1e-9 * (1. + n as f64) {
            return Err("root value conservation".into());
        }
        let ev = self.eval.records.lock();
        let eval_calls = ev.len() as u64;
        let rollout_samples = ev.iter().map(|e| e.samples.len() as u64).sum();
        let rollout_plies = ev
            .iter()
            .flat_map(|e| e.samples.iter())
            .map(|s| s.plies as u64)
            .sum();
        if self.engine.evaluator_call_count() != eval_calls {
            return Err("evaluator accounting".into());
        }
        self.cuts.push(Cut {
            global_k: global,
            local_k: n,
            edges,
            root_w,
            allocated_nodes: self.engine.tt.allocated_nodes(),
            eval_calls,
            rollout_samples,
            rollout_plies,
            elapsed_ns: start.elapsed().as_nanos() as u64,
        });
        Ok(())
    }
    fn into_replica(self, id: usize) -> Replica {
        let events = self.trace.events.lock().clone();
        let mut evaluations = self.eval.records.lock().clone();
        evaluations.sort_by_key(|x| x.id);
        Replica {
            replica: id,
            seed: format!("{:016x}", self.seed),
            cuts: self.cuts,
            root_order: self
                .engine
                .root
                .read_edges()
                .iter()
                .map(|e| e.mv as usize)
                .collect(),
            events,
            evaluations,
        }
    }
}
#[derive(Debug, Serialize)]
pub struct SearchResult {
    schema: &'static str,
    quartz_commit: &'static str,
    spec: Spec,
    root: Value,
    replicas: Vec<Replica>,
    wall_ns: u64,
    move_counts: Vec<(usize, u32)>,
}
pub fn search(moves: &[usize], spec: Spec) -> Result<SearchResult, String> {
    spec.validate()?;
    let state = from_moves(moves)?;
    if state.is_terminal() {
        return Err("terminal root".into());
    }
    if let Some(a) = spec.source_action {
        if !state
            .idx_to_move(a)
            .is_some_and(|m| state.legal_moves().contains(&m))
        {
            return Err("source action not legal".into());
        }
    }
    let start = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(spec.workers)
        .build()
        .map_err(|e| e.to_string())?;
    let nr = if spec.mode == "root" { spec.workers } else { 1 };
    let mut works: Vec<_> = pool.install(|| {
        (0..nr)
            .into_par_iter()
            .map(|i| Work::new(&state, &spec, mix(spec.seed ^ i as u64)))
            .collect()
    });
    for &k in &spec.cuts {
        if spec.mode == "root" {
            pool.install(|| {
                works.par_iter_mut().enumerate().try_for_each(|(i, w)| {
                    w.run_to(k / nr as u32 + u32::from(i < (k as usize % nr)), k, &spec)
                })
            })?;
        } else {
            pool.install(|| works[0].run_to(k, k, &spec))?;
        }
    }
    let replicas: Vec<_> = works
        .into_iter()
        .enumerate()
        .map(|(i, w)| w.into_replica(i))
        .collect();
    let mut counts = vec![0u32; 225];
    for r in &replicas {
        for e in &r.cuts.last().unwrap().edges {
            counts[e.action] += e.n;
        }
    }
    if counts.iter().sum::<u32>() != spec.budget {
        return Err("global budget mismatch".into());
    }
    let move_counts = state
        .legal_moves()
        .into_iter()
        .map(|m| (m as usize, counts[m as usize]))
        .collect();
    Ok(SearchResult {
        schema: SCHEMA,
        quartz_commit: QUARTZ,
        spec,
        root: state_value(&state),
        replicas,
        wall_ns: start.elapsed().as_nanos() as u64,
        move_counts,
    })
}
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Describe,
    State { moves: Vec<usize> },
    Search { moves: Vec<usize>, spec: Spec },
}
fn request(line: &str) -> Result<Value, String> {
    let r: Request = serde_json::from_str(line).map_err(|e| e.to_string())?;
    match r {
        Request::Describe => Ok(
            json!({"schema":SCHEMA,"quartz_commit":QUARTZ,"board_size":15,"rules":"freestyle","modes":["serial","root","tree","leaf","wave"],"evaluator":"uniform_rollout (legacy commissioning); explicit rapfi_nnue_lookahead_v1 (MixNet)","evaluators":["uniform_rollout","rapfi_nnue_lookahead_v1"],"tt":"disabled_owned_arena","virtual_loss":"fixed tree/wave","native_phase":"root select/reserve and root backup linearized; descendant event completion order observational","root_laws":["iid","pl","reserve"],"root_proposals":["p","q","local","path"],"record_rollouts":"opt-in exact augmented uniform rollout path"}),
        ),
        Request::State { moves } => Ok(state_value(&from_moves(&moves)?)),
        Request::Search { moves, spec } => {
            serde_json::to_value(search(&moves, spec)?).map_err(|e| e.to_string())
        }
    }
}
pub fn serve() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        match line {
            Ok(line) => {
                let reply = match request(&line) {
                    Ok(v) => json!({"ok":true,"result":v}),
                    Err(e) => json!({"ok":false,"error":e}),
                };
                if serde_json::to_writer(&mut stdout, &reply).is_err()
                    || writeln!(&mut stdout).is_err()
                    || stdout.flush().is_err()
                {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    crate::mqft_rapfi::close();
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn board_legal_terminal() {
        let moves = [0, 15, 1, 16, 2, 17, 3, 18, 4];
        let s = from_moves(&moves).unwrap();
        assert!(s.is_terminal());
        assert_eq!(s.outcome(), -1.);
        assert!(from_moves(&[0, 0]).is_err());
        let mut m = moves.to_vec();
        m.push(30);
        assert!(from_moves(&m).is_err());
        assert!(from_moves(&[225]).is_err());
    }
    #[test]
    fn all_schedules_tiny_contract() {
        // In-memory tiny verification only; no self-play or research dataset.
        for mode in ["serial", "root", "tree", "leaf", "wave"] {
            let spec = Spec {
                mode: mode.into(),
                workers: if mode == "serial" { 1 } else { 2 },
                budget: 5,
                cuts: vec![1, 3, 5],
                seed: 7,
                c_puct: 1.4,
                rollout_depth: 2,
                samples_per_eval: if mode == "leaf" { 2 } else { 1 },
                prior: "uniform".into(),
                source_action: None,
                source_h: 0.,
                root_law: None,
                record_rollouts: false,
                evaluator: default_evaluator(),
            };
            let r = search(&[112, 113], spec).unwrap();
            assert_eq!(r.replicas.iter().map(|r| r.events.len()).sum::<usize>(), 5);
            for replica in &r.replicas {
                assert_eq!(
                    replica.cuts.last().unwrap().local_k as usize,
                    replica.events.len()
                );
                let ids: std::collections::HashSet<_> =
                    replica.evaluations.iter().map(|e| e.id).collect();
                for e in &replica.events {
                    assert_eq!(
                        e.root_value,
                        if e.selection.actions.len() % 2 == 0 {
                            e.leaf_value
                        } else {
                            -e.leaf_value
                        }
                    );
                    if let Some(id) = e.eval_id {
                        assert!(ids.contains(&id));
                        let v = replica.evaluations.iter().find(|x| x.id == id).unwrap();
                        assert_eq!(v.value, e.leaf_value);
                        assert_eq!(v.state_hash, e.selection.leaf_hash);
                    }
                }
            }
        }
    }
    #[test]
    fn contention_and_wave_replay() {
        let mut spec = Spec {
            mode: "wave".into(),
            workers: 4,
            budget: 37,
            cuts: vec![1, 4, 9, 37],
            seed: 42,
            c_puct: 1.4,
            rollout_depth: 225,
            samples_per_eval: 1,
            prior: "local".into(),
            source_action: None,
            source_h: 0.,
            root_law: None,
            record_rollouts: false,
            evaluator: default_evaluator(),
        };
        let moves = [112, 113, 97, 127];
        let a = search(&moves, spec.clone()).unwrap();
        let b = search(&moves, spec.clone()).unwrap();
        let sig = |r: SearchResult| {
            r.replicas
                .into_iter()
                .flat_map(|r| {
                    r.events
                        .into_iter()
                        .map(|e| (e.selection.actions, e.eval_id, e.root_value))
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(sig(a), sig(b));
        spec.mode = "tree".into();
        spec.workers = 8;
        spec.budget = 96;
        spec.cuts = vec![1, 13, 96];
        for seed in 0..4 {
            spec.seed = seed;
            let r = search(&moves, spec.clone()).unwrap();
            assert_eq!(r.replicas[0].events.len(), 96);
        }
    }
    #[test]
    fn full_board_draw_and_terminal_marks() {
        let mut black = Vec::new();
        let mut white = Vec::new();
        for y in 0..15 {
            for x in 0..15 {
                if (x / 2 + y) % 2 == 0 {
                    black.push(y * 15 + x)
                } else {
                    white.push(y * 15 + x)
                }
            }
        }
        let mut moves = Vec::new();
        for i in 0..white.len() {
            moves.push(black[i]);
            moves.push(white[i]);
        }
        assert_eq!(moves.len(), 224);
        let s = from_moves(&moves).unwrap();
        assert!(!s.is_terminal());
        assert_eq!(s.legal_moves().len(), 1);
        let spec = Spec {
            mode: "wave".into(),
            workers: 4,
            budget: 5,
            cuts: vec![1, 5],
            seed: 7,
            c_puct: 1.4,
            rollout_depth: 225,
            samples_per_eval: 1,
            prior: "uniform".into(),
            source_action: None,
            source_h: 0.,
            root_law: None,
            record_rollouts: false,
            evaluator: default_evaluator(),
        };
        let r = search(&moves, spec).unwrap();
        assert!(r.replicas[0]
            .events
            .iter()
            .all(|e| e.terminal && e.eval_id.is_none() && e.root_value == 0.));
        moves.push(*black.last().unwrap());
        let st = from_moves(&moves).unwrap();
        assert!(st.is_terminal());
        assert_eq!(st.outcome(), 0.);
    }
    #[test]
    fn root_source_support() {
        let spec = Spec {
            mode: "serial".into(),
            workers: 1,
            budget: 1,
            cuts: vec![1],
            seed: 1,
            c_puct: 1.4,
            rollout_depth: 1,
            samples_per_eval: 1,
            prior: "uniform".into(),
            source_action: Some(111),
            source_h: 2.,
            root_law: None,
            record_rollouts: false,
            evaluator: default_evaluator(),
        };
        let r = search(&[112, 113], spec).unwrap();
        let e = &r.replicas[0].cuts[0].edges;
        assert!(e.iter().all(|x| x.prior > 0.));
        assert!((e.iter().map(|x| x.prior as f64).sum::<f64>() - 1.).abs() < 1e-6);
        let a = e.iter().find(|x| x.action == 111).unwrap().prior;
        let b = e.iter().find(|x| x.action == 110).unwrap().prior;
        assert!((((a / b) as f64) - 2f64.exp()).abs() < 1e-5);
    }
}
