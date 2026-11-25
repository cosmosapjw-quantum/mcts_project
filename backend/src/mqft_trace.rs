//! MQFT opt-in observation. No change to PUCT, backup sign, or edge statistics.
use crate::game::GameState;
use crate::mcts::node::PathEdge;
use parking_lot::Mutex;
use serde::Serialize;
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

thread_local! { static LAST_EVAL: Cell<Option<u64>> = const { Cell::new(None) }; static EVAL_TICKET: Cell<Option<u64>> = const { Cell::new(None) }; }
pub fn set_eval_ticket(i: u64) {
    EVAL_TICKET.with(|x| x.set(Some(i)));
}
pub fn take_eval_ticket() -> Option<u64> {
    EVAL_TICKET.with(|x| x.take())
}
pub fn clear_eval() {
    LAST_EVAL.with(|x| x.set(None));
}
pub fn set_eval(i: u64) {
    LAST_EVAL.with(|x| x.set(Some(i)));
}
pub fn take_eval() -> Option<u64> {
    LAST_EVAL.with(|x| x.take())
}

#[derive(Clone, Debug, Serialize)]
pub struct RootSelection {
    pub clock: u64,
    pub score: f32,
    pub edge_index: usize,
    pub law: Option<crate::mqft_law::Draw>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Selected {
    pub select_seq: u64,
    pub phase: u32,
    pub worker: Option<usize>,
    pub actions: Vec<usize>,
    pub edge_priors: Vec<f32>,
    pub parent_hashes: Vec<String>,
    pub leaf_hash: String,
    pub vl_visit: Vec<f32>,
    pub vl_value: Vec<f32>,
    pub root_selection: Option<RootSelection>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Event {
    #[serde(flatten)]
    pub selection: Selected,
    pub complete_seq: u64,
    pub root_backup_clock: u64,
    pub leaf_value: f32,
    pub root_value: f32,
    pub eval_id: Option<u64>,
    pub terminal: bool,
}
#[derive(Default)]
pub struct Trace {
    seq: AtomicU64,
    phase: AtomicU64,
    pub events: Mutex<Vec<Event>>,
}
impl Trace {
    pub fn set_phase(&self, phase: u32) {
        self.phase.store(phase as u64, Ordering::Relaxed);
    }
    pub fn begin<G: GameState>(&self, root: &G, path: &[PathEdge<G::Move>]) -> Selected {
        let mut s = Selected {
            select_seq: self.seq.fetch_add(1, Ordering::Relaxed),
            phase: self.phase.load(Ordering::Relaxed) as u32,
            worker: rayon::current_thread_index(),
            actions: Vec::with_capacity(path.len()),
            edge_priors: Vec::with_capacity(path.len()),
            parent_hashes: Vec::with_capacity(path.len()),
            leaf_hash: format!("{:016x}", root.tt_hash()),
            vl_visit: Vec::with_capacity(path.len()),
            vl_value: Vec::with_capacity(path.len()),
            root_selection: path.first().and_then(|e| e.root_selection.clone()),
        };
        for e in path {
            let edge = &e.parent.read_edges()[e.edge_idx];
            s.actions.push(root.move_to_idx(edge.mv));
            s.edge_priors.push(edge.p);
            s.parent_hashes.push(format!("{:016x}", e.parent.hash));
            s.leaf_hash = format!("{:016x}", edge.child.hash);
            s.vl_visit.push(e.applied_vl.0);
            s.vl_value.push(e.applied_vl.1);
        }
        s
    }
    pub fn finish(
        &self,
        s: Selected,
        leaf_value: f32,
        eval_id: Option<u64>,
        terminal: bool,
        root_backup_clock: u64,
    ) {
        // Sequence is observation order, not a claim of linearizable internal atomic writes.
        let mut a = self.events.lock();
        let v = if s.actions.len() % 2 == 0 {
            leaf_value
        } else {
            -leaf_value
        };
        let complete_seq = a.len() as u64;
        a.push(Event {
            selection: s,
            complete_seq,
            root_backup_clock,
            leaf_value,
            root_value: v,
            eval_id,
            terminal,
        });
    }
}
