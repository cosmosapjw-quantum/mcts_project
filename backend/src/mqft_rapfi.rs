//! Actual pinned Rapfi release NNUE protocol, shared policy/value backend.
//! The driver sets the verified directory. No uniform evaluator fallback.
use crate::{game::GameState, games::Gomoku15};
use parking_lot::{Condvar, Mutex};
use std::{
    io::{BufRead, BufReader, Read, Write},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{mpsc, Arc, OnceLock},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

pub const ID: &str = "rapfi_nnue_lookahead_v1";
pub const SCALE: f64 = 32.;
pub const FLOOR: f64 = 1e-6;
pub type Scores = Vec<(u16, i32)>;
struct Session {
    child: Child,
    input: ChildStdin,
    out: mpsc::Receiver<Result<String, String>>,
    reader: Option<JoinHandle<()>>,
    err_reader: Option<JoinHandle<()>>,
    stderr: Arc<Mutex<Vec<u8>>>,
    loaded: bool,
}
impl Session {
    fn start(dir: &PathBuf) -> Result<Self, String> {
        let mut child = Command::new(dir.join("pbrain-rapfi-linux-clang-avx2"))
            .current_dir(dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| e.to_string())?;
        let input = child.stdin.take().ok_or("missing input")?;
        let stdout = child.stdout.take().ok_or("missing output")?;
        let mut stderr_pipe = child.stderr.take().ok_or("missing stderr")?;
        let (tx, out) = mpsc::channel();
        let reader = thread::spawn(move || {
            let mut r = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                match r.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if tx.send(Ok(line)).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e.to_string()));
                        break;
                    }
                }
            }
        });
        let stderr = Arc::new(Mutex::new(Vec::new()));
        let ring = stderr.clone();
        let err_reader = thread::spawn(move || {
            let mut b = [0u8; 4096];
            while let Ok(n) = stderr_pipe.read(&mut b) {
                if n == 0 {
                    break;
                }
                let mut v = ring.lock();
                v.extend_from_slice(&b[..n]);
                let excess = v.len().saturating_sub(8192);
                if excess > 0 {
                    v.drain(..excess);
                }
            }
        });
        let mut s = Self {
            child,
            input,
            out,
            reader: Some(reader),
            err_reader: Some(err_reader),
            stderr,
            loaded: false,
        };
        let result = s
            .exchange("START 15\nINFO rule 0\nINFO thread_num 1\nABOUT\n")
            .and_then(|lines| {
                if lines.iter().any(|l| l.trim() == "OK") {
                    Ok(())
                } else {
                    Err("Rapfi start failed".into())
                }
            });
        if let Err(ref e) = result {
            let pid = s.child.id();
            let status = s.child.try_wait();
            s.shutdown();
            eprintln!(
                "MQFT_RAPFI_START_FAILURE pid={} status={:?} error={} stderr={:?}",
                pid,
                status,
                e,
                String::from_utf8_lossy(&s.stderr.lock())
            );
        }
        result.map(|_| s)
    }
    fn exchange(&mut self, cmd: &str) -> Result<Vec<String>, String> {
        self.input
            .write_all(cmd.as_bytes())
            .and_then(|_| self.input.flush())
            .map_err(|e| format!("transport: write: {}", e))?;
        let mut out = Vec::new();
        let mut bytes = 0;
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let l = match self.out.recv_timeout(left) {
                Ok(Ok(l)) => l,
                Ok(Err(e)) => return Err(format!("transport: read: {}", e)),
                Err(e) => {
                    return Err(format!(
                        "transport: receive {:?}; pid={} status={:?}",
                        e,
                        self.child.id(),
                        self.child.try_wait()
                    ))
                }
            };
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
impl Session {
    fn shutdown(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(t) = self.reader.take() {
            let _ = t.join();
        }
        if let Some(t) = self.err_reader.take() {
            let _ = t.join();
        }
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        self.shutdown();
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
        if let Err(ref err) = result {
            let pid = s.child.id();
            let status = s.child.try_wait();
            s.shutdown();
            eprintln!("MQFT_RAPFI_FAILURE state={:016x} side={} policy={} pid={} status={:?} error={} stderr={:?}",state.tt_hash(),state.current_player(),policy,pid,status,err,String::from_utf8_lossy(&s.stderr.lock()));
        }
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
    // Retry transport only, once, with same state and pinned model. Never
    // substitute an evaluator. Record failed attempt and retry outcome.
    match pool.query(state, policy) {
        Err(e) if e.starts_with("transport:") => {
            let r = pool.query(state, policy);
            eprintln!(
                "MQFT_RAPFI_RETRY state={:016x} policy={} original_error={:?} success={}",
                state.tt_hash(),
                policy,
                e,
                r.is_ok()
            );
            r
        }
        r => r,
    }
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

// Controlled protocol-only subprocess fixtures, not neural engine observations.
#[cfg(all(test, unix))]
mod transport_tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::time::{SystemTime, UNIX_EPOCH};
    fn fixture() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mqft-transport-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let exe = dir.join("pbrain-rapfi-linux-clang-avx2");
        std::fs::write(&exe, "#!/bin/sh\nwhile IFS= read -r l; do\ncase \"$l\" in\nSTART*) echo OK;;\nABOUT) echo 'name=\"Rapfi\"';;\nTEST_EOF) echo fixture-eof >&2; exit 19;;\nTEST_ERROR) echo 'ERROR fixture semantic';;\nesac\ndone\n").unwrap();
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o700)).unwrap();
        dir
    }
    #[test]
    fn transport_eof_and_stderr() {
        let dir = fixture();
        let mut s = Session::start(&dir).unwrap();
        let err = s.exchange("TEST_EOF\n").unwrap_err();
        assert!(err.starts_with("transport:"));
        let status = s.child.wait().unwrap();
        s.shutdown();
        assert_eq!(status.code(), Some(19));
        assert!(String::from_utf8_lossy(&s.stderr.lock()).contains("fixture-eof"));
        drop(s);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn semantic_error_not_transport() {
        let dir = fixture();
        let mut s = Session::start(&dir).unwrap();
        let err = s.exchange("TEST_ERROR\n").unwrap_err();
        assert_eq!(err, "ERROR fixture semantic");
        assert!(!err.starts_with("transport:"));
        drop(s);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
