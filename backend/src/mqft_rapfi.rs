//! Actual pinned Rapfi release NNUE protocol, shared policy/value backend.
//! The driver sets the verified directory. No uniform evaluator fallback.
use crate::{game::GameState, games::Gomoku15};
use parking_lot::{Condvar, Mutex};
use std::{
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{Arc, OnceLock},
};

pub const ID: &str = "rapfi_nnue_lookahead_v1";
pub const SCALE: f64 = 32.;
pub const FLOOR: f64 = 1e-6;
pub type Scores = Vec<(u16, i32)>;
struct Session {
    child: Child,
    input: ChildStdin,
    out: BufReader<ChildStdout>,
    loaded: bool,
}
impl Session {
    fn start(dir: &PathBuf) -> Result<Self, String> {
        let mut child = Command::new(dir.join("pbrain-rapfi-linux-clang-avx2"))
            .current_dir(dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| e.to_string())?;
        let input = child.stdin.take().ok_or("missing input")?;
        let out = BufReader::new(child.stdout.take().ok_or("missing output")?);
        let mut s = Self {
            child,
            input,
            out,
            loaded: false,
        };
        let lines = s.exchange("START 15\nINFO rule 0\nINFO thread_num 1\nABOUT\n")?;
        if !lines.iter().any(|l| l.trim() == "OK") {
            return Err("Rapfi start failed".into());
        }
        Ok(s)
    }
    fn exchange(&mut self, cmd: &str) -> Result<Vec<String>, String> {
        self.input
            .write_all(cmd.as_bytes())
            .and_then(|_| self.input.flush())
            .map_err(|e| e.to_string())?;
        let mut out = Vec::new();
        let mut bytes = 0;
        loop {
            let mut l = String::new();
            if self.out.read_line(&mut l).map_err(|e| e.to_string())? == 0 {
                return Err("Rapfi EOF".into());
            }
            bytes += l.len();
            if bytes > 262144 {
                return Err("Rapfi response exceeds limit".into());
            }
            let l = l.trim_end().to_string();
            if l.starts_with("ERROR") || l.starts_with("UNKNOWN") {
                return Err(l);
            }
            if l.contains("nnue: load weight from mix9svqfreestyle_bsmix.bin.lz4") {
                self.loaded = true;
            }
            if l.starts_with("name=\"Rapfi\"") {
                break;
            }
            out.push(l);
        }
        Ok(out)
    }
    fn query(
        &mut self,
        state: &Gomoku15,
        policy: bool,
    ) -> Result<(Option<Scores>, [f64; 3]), String> {
        if state.is_terminal() {
            return Err("terminal NNUE query".into());
        }
        let board = state.board_state_record();
        let side = state.current_player();
        let black: Vec<_> = board
            .iter()
            .enumerate()
            .filter(|(_, x)| **x == 1)
            .map(|(i, _)| i)
            .collect();
        let white: Vec<_> = board
            .iter()
            .enumerate()
            .filter(|(_, x)| **x == 2)
            .map(|(i, _)| i)
            .collect();
        if !(black.len() == white.len() || black.len() == white.len() + 1) {
            return Err("board parity".into());
        }
        // Canonical alternating placement: NNUE board input, not search history.
        let mut cmd = "YXBOARD\n".to_string();
        for i in 0..black.len() {
            let a = black[i];
            cmd += &format!("{},{},{}\n", a % 15, a / 15, if side == 1 { 1 } else { 2 });
            if let Some(a) = white.get(i) {
                cmd += &format!("{},{},{}\n", a % 15, a / 15, if side == -1 { 1 } else { 2 });
            }
        }
        cmd += "DONE\n";
        if policy {
            cmd += "TRACEBOARD\n";
        }
        cmd += "TRACESEARCH\nABOUT\n";
        let lines = self.exchange(&cmd)?;
        if !self.loaded {
            return Err("pinned freestyle NNUE was not loaded".into());
        }
        let line = lines
            .iter()
            .find(|l| l.starts_with("MESSAGE Evaluator Eval[Black]:"))
            .ok_or("missing Rapfi WDL")?;
        let tokens: Vec<_> = line.split_whitespace().collect();
        let parse = |key: &str| -> Result<f64, String> {
            let i = tokens
                .iter()
                .position(|&x| x == key)
                .ok_or(format!("missing {}", key))?;
            tokens
                .get(i + 1)
                .ok_or("missing WDL field")?
                .parse::<f64>()
                .map_err(|e| e.to_string())
        };
        let wdl = [parse("WR:")?, parse("LR:")?, parse("DR:")?];
        let sum: f64 = wdl.iter().sum();
        if wdl.iter().any(|x| !x.is_finite() || *x < 0. || *x > 1.) || (sum - 1.).abs() > 0.001 {
            return Err("invalid rounded WDL".into());
        }
        let scores = if policy {
            let ply = lines
                .iter()
                .find(|l| l.starts_with("MESSAGE Ply: "))
                .ok_or("missing ply")?;
            if ply
                .split_whitespace()
                .last()
                .and_then(|x| x.parse::<usize>().ok())
                != Some(black.len() + white.len())
            {
                return Err("Rapfi reconstructed ply mismatch".into());
            }
            let side_line = lines
                .iter()
                .find(|l| l.starts_with("MESSAGE SideToMove: "))
                .ok_or("missing side")?;
            if !side_line.ends_with(if side == 1 { "Black" } else { "White" }) {
                return Err("Rapfi side mismatch".into());
            }
            let i = lines
                .iter()
                .position(|l| l.contains("Policy------Self"))
                .ok_or("missing NNUE policy")?;
            let mut out = Vec::new();
            for y in 0..15 {
                let row = lines.get(i + 1 + y).ok_or("missing policy row")?;
                let cells: Vec<_> = row.split_whitespace().collect();
                if cells.len() != 17
                    || cells[0] != "MESSAGE"
                    || cells[16].parse::<usize>().ok() != Some(y + 1)
                {
                    return Err("policy grid shape".into());
                }
                for x in 0..15 {
                    let a = y * 15 + x;
                    if board[a] == 0 {
                        out.push((
                            a as u16,
                            cells[x + 1].parse::<i32>().map_err(|e| e.to_string())?,
                        ));
                    } else if cells[x + 1] != (if board[a] == 1 { "[X]" } else { "[O]" }) {
                        return Err("Rapfi reconstructed board mismatch".into());
                    }
                }
            }
            if out.iter().map(|x| x.0).collect::<Vec<_>>() != state.legal_moves() {
                return Err("NNUE legal support mismatch".into());
            }
            Some(out)
        } else {
            None
        };
        Ok((scores, wdl))
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
struct Pool {
    dir: PathBuf,
    idle: Mutex<Vec<Session>>,
    cv: Condvar,
    created: Mutex<usize>,
    capacity: usize,
}
impl Pool {
    fn new() -> Result<Self, String> {
        let dir = std::env::var_os("MQFT_RAPFI_DIR")
            .ok_or("MQFT_RAPFI_DIR required for verified Rapfi NNUE")?;
        let dir = std::fs::canonicalize(dir).map_err(|e| e.to_string())?;
        let capacity = std::env::var("MQFT_RAPFI_WORKERS")
            .unwrap_or("1".into())
            .parse::<usize>()
            .map_err(|e| e.to_string())?;
        if !(1..=64).contains(&capacity) {
            return Err("invalid Rapfi evaluator pool width".into());
        }
        Ok(Self {
            dir,
            idle: Mutex::new(Vec::new()),
            cv: Condvar::new(),
            created: Mutex::new(0),
            capacity,
        })
    }
    fn query(&self, state: &Gomoku15, policy: bool) -> Result<(Option<Scores>, [f64; 3]), String> {
        let mut idle = self.idle.lock();
        let mut s = loop {
            if let Some(s) = idle.pop() {
                drop(idle);
                break s;
            }
            let mut n = self.created.lock();
            if *n < self.capacity {
                *n += 1;
                drop(n);
                drop(idle);
                match Session::start(&self.dir) {
                    Ok(s) => break s,
                    Err(e) => {
                        let mut idle = self.idle.lock();
                        *self.created.lock() -= 1;
                        self.cv.notify_one();
                        drop(idle);
                        return Err(e);
                    }
                }
            }
            drop(n);
            self.cv.wait(&mut idle);
        };
        let result = s.query(state, policy);
        let mut idle = self.idle.lock();
        if result.is_ok() {
            idle.push(s)
        } else {
            *self.created.lock() -= 1;
            drop(s)
        }
        self.cv.notify_one();
        result
    }
}
static POOL: OnceLock<Mutex<Option<Arc<Pool>>>> = OnceLock::new();
pub fn query(state: &Gomoku15, policy: bool) -> Result<(Option<Scores>, [f64; 3]), String> {
    let pool = {
        let mut p = POOL.get_or_init(|| Mutex::new(None)).lock();
        if p.is_none() {
            *p = Some(Arc::new(Pool::new()?))
        }
        p.as_ref().unwrap().clone()
    };
    pool.query(state, policy)
}
pub fn close() {
    if let Some(p) = POOL.get() {
        p.lock().take();
    }
}
pub fn probabilities(scores: &Scores) -> Vec<(u16, f64)> {
    let max = scores.iter().map(|x| x.1).max().unwrap() as f64;
    let z: Vec<_> = scores
        .iter()
        .map(|(a, s)| (*a, ((*s as f64 - max) / SCALE).exp()))
        .collect();
    let sum: f64 = z.iter().map(|x| x.1).sum();
    let n = z.len() as f64;
    z.into_iter()
        .map(|(a, w)| (a, (1. - FLOOR) * w / sum + FLOOR / n))
        .collect()
}
pub fn value(wdl: [f64; 3], side: i8) -> f32 {
    (side as f64 * (wdl[0] - wdl[1]) / wdl.iter().sum::<f64>()) as f32
}
