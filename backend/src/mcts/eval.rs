//! Evaluator 구현체 (v0.4.2)
//!
//! UniformEval     — 균등 prior, value=0.0
//! ShortRollout    — max_depth 제한 랜덤 플레이아웃 (Gomoku용 실질 signal)
//! RandomPlayout   — 완전 랜덤 플레이아웃 (TicTacToe용)
//! PythonIpcEval   — eval_server.py JSON-line IPC (center_preference prior)

use crossbeam_channel as channel;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Read, Write};
use std::mem;
use std::os::fd::AsRawFd;
use std::os::raw::{c_int, c_void};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

use crate::game::{tt_combine, EvalResult, Evaluator, GameState};
use crate::simd_utils::normalize_nonnegative_in_place;
use std::fmt::Debug;
use std::hash::Hash;

// ─────────────────────────────────────────────
// § UniformEval
// ─────────────────────────────────────────────

pub struct UniformEval;

impl<G: GameState> Evaluator<G> for UniformEval {
    fn evaluate(&self, state: &G) -> EvalResult<G::Move> {
        state.uniform_eval(0.0)
    }
}

// ─────────────────────────────────────────────
// § ShortRollout — max_depth 제한 플레이아웃
//
// Gomoku/Chess처럼 playout이 긴 게임에서 실질 value signal을 생성하기 위해
// 최대 `max_depth` 수까지만 플레이아웃하고 terminal이 아니면 `draw_value`를 반환.
//
// 결과:
//   - Q에 실제 ±1 역전파 → σ_Q > 0 → QUARTZ 신호 활성화
//   - 완전 플레이아웃보다 빠르고 메모리 safe
// ─────────────────────────────────────────────

pub struct ShortRollout {
    /// 최대 플레이아웃 깊이 (0 = immediate eval only)
    pub max_depth: usize,
    /// terminal에 도달하지 못한 경우 반환 값
    pub draw_value: f32,
    /// 재현성 시드 (None = 비결정적)
    pub seed: Option<u64>,
}

impl ShortRollout {
    pub fn new(max_depth: usize) -> Self {
        ShortRollout {
            max_depth,
            draw_value: 0.0,
            seed: None,
        }
    }

    #[cfg(test)]
    pub fn seeded(max_depth: usize, seed: u64) -> Self {
        ShortRollout {
            max_depth,
            draw_value: 0.0,
            seed: Some(seed),
        }
    }
}

impl<G: GameState> Evaluator<G> for ShortRollout {
    fn evaluate(&self, state: &G) -> EvalResult<G::Move> {
        let value = if state.is_terminal() {
            state.outcome()
        } else {
            short_playout(state, self.max_depth, self.draw_value, self.seed)
        };
        state.uniform_eval(value)
    }
}

fn short_playout<G: GameState>(
    start: &G,
    max_depth: usize,
    draw_value: f32,
    seed: Option<u64>,
) -> f32 {
    let mut state = start.clone();
    let root_player = start.current_player();

    // [OPT] Use fast PCG-style RNG to avoid thread_rng() per depth + Vec allocation
    let mut rng_state: u64 = seed.unwrap_or_else(|| {
        start
            .hash()
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1)
    });

    for _depth in 0..max_depth.max(1) {
        if state.is_terminal() {
            let raw = state.outcome();
            let flip = if state.current_player() == root_player {
                1.0
            } else {
                -1.0
            };
            return raw * flip;
        }

        // [OPT] Use random_legal_move() to avoid Vec allocation for legal_moves().
        // First check winning moves via legal_moves only if needed.
        // Trade-off: skip is_winning_move scan in deeper playout to save time.
        // is_winning_move is O(1) per move but iterating all legal moves is O(n).
        // Instead, use random_legal_move (O(n) scan, zero alloc).
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let rand_idx = (rng_state >> 33) as usize;

        let mv = match state.random_legal_move(rand_idx) {
            Some(m) => m,
            None => break,
        };
        state = state.apply_move(mv);
    }

    if state.is_terminal() {
        let raw = state.outcome();
        let flip = if state.current_player() == root_player {
            1.0
        } else {
            -1.0
        };
        raw * flip
    } else {
        draw_value
    }
}

// ─────────────────────────────────────────────
// § RandomPlayout — 완전 플레이아웃
// ─────────────────────────────────────────────

pub struct RandomPlayout;

impl<G: GameState> Evaluator<G> for RandomPlayout {
    fn evaluate(&self, state: &G) -> EvalResult<G::Move> {
        let value = full_playout(state);
        state.uniform_eval(value)
    }
}

fn full_playout<G: GameState>(start: &G) -> f32 {
    let mut state = start.clone();
    let root_player = start.current_player();
    let mut rng = thread_rng();
    loop {
        if state.is_terminal() {
            let raw = state.outcome();
            let flip = if state.current_player() == root_player {
                1.0
            } else {
                -1.0
            };
            return raw * flip;
        }
        let legal = state.legal_moves();
        if legal.is_empty() {
            return state.outcome();
        }
        let mv = if let Some(w) = immediate_win(&state, &legal) {
            w
        } else {
            *legal.choose(&mut rng).unwrap()
        };
        state = state.apply_move(mv);
    }
}

fn immediate_win<G: GameState>(state: &G, legal: &[G::Move]) -> Option<G::Move> {
    for &mv in legal {
        let next = state.apply_move(mv);
        if next.is_terminal() && next.outcome() < 0.0 {
            return Some(mv);
        }
    }
    None
}

// ─────────────────────────────────────────────
// § PythonIpcEval — eval_server.py IPC
// ─────────────────────────────────────────────

struct IpcInner {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
}

pub struct PythonIpcEval {
    inner: Mutex<Option<IpcInner>>,
    server_path: String,
    /// 보드 크기 힌트 (eval_server.py의 center_preference용)
    pub board_size: usize,
}

impl PythonIpcEval {
    pub fn new(server_path: &str) -> std::io::Result<Self> {
        Ok(PythonIpcEval {
            inner: Mutex::new(None),
            server_path: server_path.to_string(),
            board_size: 0,
        })
    }

    #[cfg(test)]
    pub fn with_board_size(mut self, size: usize) -> Self {
        self.board_size = size;
        self
    }

    fn ensure_started(&self) -> Result<(), String> {
        let mut lock = self.inner.lock().unwrap();
        if lock.is_some() {
            return Ok(());
        }
        let mut child = Command::new("python3")
            .arg(&self.server_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to start eval server: {e}"))?;
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        *lock = Some(IpcInner {
            child,
            stdin,
            reader: BufReader::new(stdout),
        });
        Ok(())
    }

    fn call<G: GameState>(&self, state: &G) -> EvalResult<G::Move> {
        let legal = state.legal_moves();
        let n_act = state.num_actions();
        let mut mask = vec![0u8; n_act];
        for &mv in &legal {
            mask[state.move_to_idx(mv)] = 1;
        }

        let bs = if self.board_size > 0 {
            self.board_size
        } else {
            // 자동 감지: num_actions가 제곱수면 sqrt 사용
            let sq = (n_act as f64).sqrt() as usize;
            if sq * sq == n_act {
                sq
            } else {
                0
            }
        };

        let req = serde_json::json!({
            "num_actions": n_act,
            "action_mask": mask,
            "board_size":  bs,
            "features":    state.encode_planes(),
        });

        let mut lock = self.inner.lock().unwrap();
        let inner = lock.as_mut().expect("eval server not started");

        if inner
            .stdin
            .write_all((req.to_string() + "\n").as_bytes())
            .is_err()
            || inner.stdin.flush().is_err()
        {
            return EvalResult::uniform(&legal, 0.0);
        }

        let mut line = String::new();
        if inner.reader.read_line(&mut line).unwrap_or(0) == 0 {
            return EvalResult::uniform(&legal, 0.0);
        }

        let resp: serde_json::Value = match serde_json::from_str(line.trim()) {
            Ok(v) => v,
            Err(_) => return EvalResult::uniform(&legal, 0.0),
        };
        if resp["status"].as_str() != Some("ok") {
            return EvalResult::uniform(&legal, 0.0);
        }

        let policy_raw: Vec<f32> = resp["policy"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
            .unwrap_or_default();
        let value = resp["value"].as_f64().unwrap_or(0.0) as f32;

        let policy: Vec<(G::Move, f32)> = legal
            .iter()
            .filter_map(|&mv| {
                policy_raw
                    .get(state.move_to_idx(mv))
                    .copied()
                    .map(|p| (mv, p.max(0.0)))
            })
            .collect();

        let total: f32 = policy.iter().map(|(_, p)| p).sum();
        let policy = if total > 1e-8 {
            policy.into_iter().map(|(m, p)| (m, p / total)).collect()
        } else {
            EvalResult::uniform(&legal, value).policy
        };
        EvalResult { policy, value }
    }
}

impl<G: GameState> Evaluator<G> for PythonIpcEval {
    fn evaluate(&self, state: &G) -> EvalResult<G::Move> {
        if self.ensure_started().is_err() {
            return EvalResult::uniform(&state.legal_moves(), 0.0);
        }
        self.call(state)
    }
}

impl Drop for PythonIpcEval {
    fn drop(&mut self) {
        let mut lock = self.inner.lock().unwrap();
        if let Some(mut inner) = lock.take() {
            let _ = inner.stdin.write_all(b"{\"cmd\":\"quit\"}\n");
            let _ = inner.child.wait();
        }
    }
}

unsafe impl Send for PythonIpcEval {}
unsafe impl Sync for PythonIpcEval {}

// ─── StdioCallbackEval: bidirectional NN evaluation via server stdin/stdout ───

const QIPC_MAGIC: [u8; 4] = *b"QIPC";
const QIPC_EVAL_REQ: u8 = 1;
const QIPC_EVAL_RESP: u8 = 2;
const QIPC_BATCH_EVAL_REQ: u8 = 3;
const QIPC_BATCH_EVAL_RESP: u8 = 4;
const QIPC_EVAL_REQ_SHM: u8 = 5;
const QIPC_EVAL_RESP_SHM: u8 = 6;
const QIPC_BATCH_EVAL_REQ_SHM: u8 = 7;
const QIPC_BATCH_EVAL_RESP_SHM: u8 = 8;

const PROT_READ: c_int = 0x1;
const PROT_WRITE: c_int = 0x2;
const MAP_SHARED: c_int = 0x01;

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: isize,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
}

struct SharedMemRegion {
    ptr: *mut u8,
    len: usize,
}

impl SharedMemRegion {
    fn open(name: &str, len: usize) -> Option<Self> {
        if name.trim().is_empty() || len == 0 {
            return None;
        }
        let path = if name.starts_with('/') {
            name.to_string()
        } else {
            format!("/dev/shm/{name}")
        };
        let file = OpenOptions::new().read(true).write(true).open(path).ok()?;
        let fd = file.as_raw_fd();
        let mapped = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                0,
            )
        };
        if mapped == (-1isize as *mut c_void) {
            return None;
        }
        drop(file);
        Some(Self {
            ptr: mapped as *mut u8,
            len,
        })
    }

    #[inline]
    fn write(&self, payload: &[u8]) -> bool {
        if payload.len() > self.len {
            return false;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(payload.as_ptr(), self.ptr, payload.len());
        }
        true
    }

    #[inline]
    fn as_slice(&self, n_bytes: usize) -> Option<&[u8]> {
        if n_bytes > self.len {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(self.ptr as *const u8, n_bytes) })
    }
}

impl Drop for SharedMemRegion {
    fn drop(&mut self) {
        let _ = unsafe { munmap(self.ptr as *mut c_void, self.len) };
    }
}

unsafe impl Send for SharedMemRegion {}
unsafe impl Sync for SharedMemRegion {}

struct SharedMemTransport {
    req: SharedMemRegion,
    resp: SharedMemRegion,
}

impl SharedMemTransport {
    fn load_from_env() -> Option<Self> {
        let req_name = std::env::var("QUARTZ_QIPC_REQ_SHM_NAME").ok()?;
        let resp_name = std::env::var("QUARTZ_QIPC_RESP_SHM_NAME").ok()?;
        let req_size = std::env::var("QUARTZ_QIPC_REQ_SHM_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())?;
        let resp_size = std::env::var("QUARTZ_QIPC_RESP_SHM_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())?;
        Some(Self {
            req: SharedMemRegion::open(&req_name, req_size)?,
            resp: SharedMemRegion::open(&resp_name, resp_size)?,
        })
    }

    #[inline]
    fn write_request(&self, payload: &[u8]) -> bool {
        self.req.write(payload)
    }

    #[inline]
    fn write_response(&self, payload: &[u8]) -> bool {
        self.resp.write(payload)
    }

    #[inline]
    fn read_response(&self, n_bytes: usize) -> Option<&[u8]> {
        self.resp.as_slice(n_bytes)
    }
}

// ─── SHM Ring Buffer for lock-free eval pipeline ───

/// SHM ring buffer layout constants.
const SHM_RING_MAGIC: u32 = 0x51524E47; // "QRNG"
const SHM_RING_VERSION: u32 = 1;
const SHM_RING_HEADER_SIZE: usize = 256;
const SHM_RING_SLOT_HEADER: usize = 16; // state(1)+type(1)+dir(1)+pad(1)+len(4)+epoch(4)+seq(4)

// Slot states
const SHM_SLOT_EMPTY: u8 = 0;
const SHM_SLOT_WRITTEN: u8 = 1;
const SHM_SLOT_DONE: u8 = 2;

// Message types
pub(crate) const SHM_MSG_EVAL_BATCH_REQ: u8 = 1;
pub(crate) const SHM_MSG_EVAL_BATCH_RESP: u8 = 2;
pub(crate) const SHM_MSG_JSON: u8 = 3;
pub(crate) const SHM_MSG_SEARCH_RESP: u8 = 4;

// Direction
const SHM_DIR_TO_PYTHON: u8 = 0;
const SHM_DIR_TO_RUST: u8 = 1;

/// Lock-free ring buffer in shared memory for Rust↔Python eval communication.
/// Replaces stdin/stdout signaling so broker doesn't need to touch pipes.
pub(crate) struct ShmRingBuffer {
    region: SharedMemRegion,
    r2p_slot_count: u32,
    p2r_slot_count: u32,
    slot_data_size: u32,
    r2p_base: usize, // byte offset to first r2p slot
    p2r_base: usize, // byte offset to first p2r slot
}

impl ShmRingBuffer {
    /// Open an existing ring buffer SHM region (Rust side — Python creates it).
    pub fn open(name: &str, expected_size: usize) -> Option<Self> {
        let region = SharedMemRegion::open(name, expected_size)?;
        // Validate magic and version
        let magic = Self::read_u32(&region, 0);
        let version = Self::read_u32(&region, 4);
        if magic != SHM_RING_MAGIC || version != SHM_RING_VERSION {
            return None;
        }
        let r2p_slot_count = Self::read_u32(&region, 8);
        let p2r_slot_count = Self::read_u32(&region, 12);
        let slot_data_size = Self::read_u32(&region, 16);
        if r2p_slot_count == 0 || p2r_slot_count == 0 || slot_data_size < 1024 {
            return None;
        }
        let r2p_base = SHM_RING_HEADER_SIZE;
        let p2r_base = SHM_RING_HEADER_SIZE + (r2p_slot_count as usize) * (slot_data_size as usize);
        let needed = p2r_base + (p2r_slot_count as usize) * (slot_data_size as usize);
        if needed > expected_size {
            return None;
        }
        Some(Self {
            region,
            r2p_slot_count,
            p2r_slot_count,
            slot_data_size,
            r2p_base,
            p2r_base,
        })
    }

    // --- Header accessors ---

    pub fn epoch(&self) -> u32 {
        let ptr = unsafe { (self.region.ptr as *const u8).add(20) as *const AtomicU32 };
        unsafe { (*ptr).load(Ordering::Acquire) }
    }

    pub fn bump_epoch(&self) -> u32 {
        let ptr = unsafe { (self.region.ptr as *const u8).add(20) as *const AtomicU32 };
        let new_epoch = unsafe { (*ptr).fetch_add(1, Ordering::AcqRel) } + 1;
        // Reset cmd_done and cancel
        self.set_cmd_done(false);
        self.clear_cancel();
        // Reset all slot states to EMPTY
        for i in 0..self.r2p_slot_count {
            self.set_slot_state(self.r2p_slot_offset(i), SHM_SLOT_EMPTY);
        }
        for i in 0..self.p2r_slot_count {
            self.set_slot_state(self.p2r_slot_offset(i), SHM_SLOT_EMPTY);
        }
        new_epoch
    }

    pub fn set_cmd_done(&self, done: bool) {
        let ptr = unsafe { (self.region.ptr as *const u8).add(24) as *const AtomicU8 };
        unsafe { (*ptr).store(if done { 1 } else { 0 }, Ordering::Release) };
    }

    /// Check if Python has requested cancellation of the current command.
    /// Stored at header offset 25 as AtomicU8. Set by Python, read by Rust.
    pub fn cancel_requested(&self) -> bool {
        let ptr = unsafe { (self.region.ptr as *const u8).add(25) as *const AtomicU8 };
        let val = unsafe { (*ptr).load(Ordering::Acquire) };
        val != 0
    }

    /// Clear the cancel flag (called by bump_epoch at command start).
    pub fn clear_cancel(&self) {
        let ptr = unsafe { (self.region.ptr as *const u8).add(25) as *const AtomicU8 };
        unsafe { (*ptr).store(0, Ordering::Release) };
    }

    // --- Slot offset helpers ---

    fn r2p_slot_offset(&self, idx: u32) -> usize {
        self.r2p_base + (idx as usize) * (self.slot_data_size as usize)
    }

    fn p2r_slot_offset(&self, idx: u32) -> usize {
        self.p2r_base + (idx as usize) * (self.slot_data_size as usize)
    }

    // --- Atomic slot state ---

    fn slot_state(&self, slot_offset: usize) -> u8 {
        let ptr = unsafe { (self.region.ptr as *const u8).add(slot_offset) as *const AtomicU8 };
        unsafe { (*ptr).load(Ordering::Acquire) }
    }

    fn set_slot_state(&self, slot_offset: usize, state: u8) {
        let ptr = unsafe { (self.region.ptr as *const u8).add(slot_offset) as *const AtomicU8 };
        unsafe { (*ptr).store(state, Ordering::Release) };
    }

    // --- Slot read/write ---

    fn write_slot(
        &self,
        slot_offset: usize,
        msg_type: u8,
        direction: u8,
        epoch: u32,
        seq: u32,
        payload: &[u8],
    ) -> bool {
        let max_payload = self.slot_data_size as usize - SHM_RING_SLOT_HEADER;
        if payload.len() > max_payload {
            return false;
        }
        let base = unsafe { self.region.ptr.add(slot_offset) };
        unsafe {
            // Write metadata (after state byte)
            *base.add(1) = msg_type;
            *base.add(2) = direction;
            *base.add(3) = 0; // reserved
            std::ptr::copy_nonoverlapping(
                (payload.len() as u32).to_le_bytes().as_ptr(),
                base.add(4),
                4,
            );
            std::ptr::copy_nonoverlapping(epoch.to_le_bytes().as_ptr(), base.add(8), 4);
            std::ptr::copy_nonoverlapping(seq.to_le_bytes().as_ptr(), base.add(12), 4);
            // Write payload
            std::ptr::copy_nonoverlapping(
                payload.as_ptr(),
                base.add(SHM_RING_SLOT_HEADER),
                payload.len(),
            );
        }
        // Set state to WRITTEN (Release — ensures all writes above are visible)
        self.set_slot_state(slot_offset, SHM_SLOT_WRITTEN);
        true
    }

    fn read_slot_meta(&self, slot_offset: usize) -> (u8, u8, u32, u32) {
        let base = unsafe { self.region.ptr.add(slot_offset) };
        unsafe {
            let msg_type = *base.add(1);
            let direction = *base.add(2);
            let mut len_buf = [0u8; 4];
            std::ptr::copy_nonoverlapping(base.add(4), len_buf.as_mut_ptr(), 4);
            let payload_len = u32::from_le_bytes(len_buf);
            let mut epoch_buf = [0u8; 4];
            std::ptr::copy_nonoverlapping(base.add(8), epoch_buf.as_mut_ptr(), 4);
            let epoch = u32::from_le_bytes(epoch_buf);
            (msg_type, direction, payload_len, epoch)
        }
    }

    fn read_slot_payload(&self, slot_offset: usize, payload_len: u32) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.region.ptr.add(slot_offset + SHM_RING_SLOT_HEADER),
                payload_len as usize,
            )
        }
    }

    // --- High-level Rust→Python write ---

    /// Write a message to the next available r2p slot. Returns false if no slot is EMPTY.
    pub fn r2p_try_write(&self, msg_type: u8, payload: &[u8], epoch: u32, seq: u32) -> bool {
        for i in 0..self.r2p_slot_count {
            let off = self.r2p_slot_offset(i);
            if self.slot_state(off) == SHM_SLOT_EMPTY {
                return self.write_slot(off, msg_type, SHM_DIR_TO_PYTHON, epoch, seq, payload);
            }
        }
        false
    }

    /// Reclaim DONE r2p slots (set back to EMPTY).
    pub fn r2p_reclaim(&self) {
        for i in 0..self.r2p_slot_count {
            let off = self.r2p_slot_offset(i);
            if self.slot_state(off) == SHM_SLOT_DONE {
                self.set_slot_state(off, SHM_SLOT_EMPTY);
            }
        }
    }

    // --- High-level Python→Rust read ---

    /// Try to read a WRITTEN p2r slot, returning metadata for validation.
    /// Returns (msg_type, epoch, seq, payload) or None.
    pub fn p2r_try_read_meta(&self) -> Option<(u8, u32, u32, &[u8])> {
        for i in 0..self.p2r_slot_count {
            let off = self.p2r_slot_offset(i);
            if self.slot_state(off) == SHM_SLOT_WRITTEN {
                let (msg_type, _dir, payload_len, epoch) = self.read_slot_meta(off);
                // Read seq from slot header offset 12
                let seq = {
                    let mut buf = [0u8; 4];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            self.region.ptr.add(off + 12),
                            buf.as_mut_ptr(),
                            4,
                        );
                    }
                    u32::from_le_bytes(buf)
                };
                let payload = self.read_slot_payload(off, payload_len);
                self.set_slot_state(off, SHM_SLOT_DONE);
                return Some((msg_type, epoch, seq, payload));
            }
        }
        None
    }

    // --- Utility ---

    fn read_u32(region: &SharedMemRegion, offset: usize) -> u32 {
        let mut buf = [0u8; 4];
        unsafe {
            std::ptr::copy_nonoverlapping(region.ptr.add(offset), buf.as_mut_ptr(), 4);
        }
        u32::from_le_bytes(buf)
    }
}

unsafe impl Send for ShmRingBuffer {}
unsafe impl Sync for ShmRingBuffer {}

/// Global ring buffer singleton — initialized once from environment variables.
/// Used by broker_loop_shm, emit_json_message, and serve() epoch management.
pub(crate) fn global_ring_buffer() -> Option<&'static ShmRingBuffer> {
    static RING: std::sync::OnceLock<Option<ShmRingBuffer>> = std::sync::OnceLock::new();
    RING.get_or_init(|| {
        let name = std::env::var("QUARTZ_QIPC_RING_SHM_NAME").ok()?;
        let size: usize = std::env::var("QUARTZ_QIPC_RING_SHM_SIZE")
            .ok()?
            .parse()
            .ok()?;
        ShmRingBuffer::open(&name, size)
    })
    .as_ref()
}

fn write_qipc_frame<W: Write>(
    writer: &mut W,
    frame_kind: u8,
    payload: &[u8],
) -> std::io::Result<()> {
    let mut header = [0u8; 9];
    header[0..4].copy_from_slice(&QIPC_MAGIC);
    header[4] = frame_kind;
    header[5..9].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    writer.write_all(&header)?;
    writer.write_all(payload)?;
    writer.flush()
}

fn read_qipc_frame<R: Read>(reader: &mut R) -> std::io::Result<(u8, Vec<u8>)> {
    let mut header = [0u8; 9];
    reader.read_exact(&mut header)?;
    if header[0..4] != QIPC_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid QIPC frame magic",
        ));
    }
    let frame_kind = header[4];
    let payload_len = u32::from_le_bytes(header[5..9].try_into().unwrap()) as usize;
    let mut payload = vec![0u8; payload_len];
    reader.read_exact(&mut payload)?;
    Ok((frame_kind, payload))
}

#[inline]
fn qipc_shm_meta_len(payload: &[u8]) -> Option<usize> {
    if payload.len() != 4 {
        return None;
    }
    Some(u32::from_le_bytes(payload.try_into().ok()?) as usize)
}

fn write_qipc_eval_frame<W: Write>(
    writer: &mut W,
    shm: Option<&SharedMemTransport>,
    frame_kind: u8,
    payload: &[u8],
) -> std::io::Result<()> {
    if let Some(shm) = shm {
        let shm_kind = match frame_kind {
            QIPC_EVAL_REQ => Some(QIPC_EVAL_REQ_SHM),
            QIPC_EVAL_RESP => Some(QIPC_EVAL_RESP_SHM),
            QIPC_BATCH_EVAL_REQ => Some(QIPC_BATCH_EVAL_REQ_SHM),
            QIPC_BATCH_EVAL_RESP => Some(QIPC_BATCH_EVAL_RESP_SHM),
            _ => None,
        };
        let wrote = match frame_kind {
            QIPC_EVAL_REQ | QIPC_BATCH_EVAL_REQ => shm.write_request(payload),
            QIPC_EVAL_RESP | QIPC_BATCH_EVAL_RESP => shm.write_response(payload),
            _ => false,
        };
        if let (Some(shm_kind), true) = (shm_kind, wrote) {
            return write_qipc_frame(writer, shm_kind, &(payload.len() as u32).to_le_bytes());
        }
    }
    write_qipc_frame(writer, frame_kind, payload)
}

fn push_u32_le(buf: &mut Vec<u8>, value: usize) {
    buf.extend_from_slice(&(value as u32).to_le_bytes());
}

fn push_u64_le(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_f32_slice_le(buf: &mut Vec<u8>, values: &[f32]) {
    if cfg!(target_endian = "little") {
        let byte_len = mem::size_of_val(values);
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, byte_len) };
        buf.extend_from_slice(bytes);
    } else {
        for value in values {
            buf.extend_from_slice(&value.to_le_bytes());
        }
    }
}

fn read_u32_le(payload: &[u8], offset: &mut usize) -> Option<usize> {
    if *offset + 4 > payload.len() {
        return None;
    }
    let value = u32::from_le_bytes(payload[*offset..*offset + 4].try_into().ok()?);
    *offset += 4;
    Some(value as usize)
}

fn read_f32_le(payload: &[u8], offset: &mut usize) -> Option<f32> {
    if *offset + 4 > payload.len() {
        return None;
    }
    let value = f32::from_le_bytes(payload[*offset..*offset + 4].try_into().ok()?);
    *offset += 4;
    Some(value)
}

fn read_f32_bytes<'a>(payload: &'a [u8], offset: &mut usize, len: usize) -> Option<&'a [u8]> {
    let byte_len = len.checked_mul(4)?;
    if *offset + byte_len > payload.len() {
        return None;
    }
    let out = &payload[*offset..*offset + byte_len];
    *offset += byte_len;
    Some(out)
}

#[inline]
fn dense_prob_at(probs_bytes: &[u8], policy_len: usize, idx: usize) -> f32 {
    if idx >= policy_len {
        return 0.0;
    }
    let base = idx * 4;
    let raw = [
        probs_bytes[base],
        probs_bytes[base + 1],
        probs_bytes[base + 2],
        probs_bytes[base + 3],
    ];
    f32::from_le_bytes(raw)
}

fn eval_cache_fingerprint<G: GameState>(
    state: &G,
    feature_len: usize,
    n_actions: usize,
) -> (u64, u64, u32) {
    let encoder_rev = state.eval_encoder_revision();
    let fp_lo = state.tt_hash();
    let mut fp_hi = tt_combine(0x4556_414c_5f46_5031, fp_lo);
    fp_hi = tt_combine(fp_hi, n_actions as u64);
    fp_hi = tt_combine(fp_hi, feature_len as u64);
    fp_hi = tt_combine(fp_hi, encoder_rev as u64);
    (fp_lo, fp_hi, encoder_rev)
}

fn encode_eval_req_payload(
    features: &[f32],
    n_actions: usize,
    model_tag: u32,
    feature_fp_lo: u64,
    feature_fp_hi: u64,
    encoder_rev: u32,
) -> Vec<u8> {
    let mut payload = Vec::with_capacity(32 + features.len() * 4);
    push_u32_le(&mut payload, model_tag as usize);
    push_u32_le(&mut payload, n_actions);
    push_u32_le(&mut payload, features.len());
    push_u64_le(&mut payload, feature_fp_lo);
    push_u64_le(&mut payload, feature_fp_hi);
    push_u32_le(&mut payload, encoder_rev as usize);
    push_f32_slice_le(&mut payload, features);
    payload
}

fn encode_batch_eval_req_payload<M: Copy + Send + 'static>(batch: &[BatchRequest<M>]) -> Vec<u8> {
    let total_floats: usize = batch.iter().map(|req| req.features.len()).sum();
    let mut payload = Vec::with_capacity(4 + batch.len() * 32 + total_floats * 4);
    push_u32_le(&mut payload, batch.len());
    for req in batch {
        push_u32_le(&mut payload, req.model_tag as usize);
        push_u32_le(&mut payload, req.n_actions);
        push_u32_le(&mut payload, req.features.len());
        push_u64_le(&mut payload, req.feature_fp_lo);
        push_u64_le(&mut payload, req.feature_fp_hi);
        push_u32_le(&mut payload, req.encoder_rev as usize);
        push_f32_slice_le(&mut payload, &req.features);
    }
    payload
}

fn reclaim_batch_features<M: Copy + Send + 'static>(batch: &mut [BatchRequest<M>]) {
    for req in batch {
        if let Some(pool) = req.feature_pool.as_ref() {
            pool.give_back(mem::take(&mut req.features));
        } else {
            req.features.clear();
        }
    }
}

#[cfg(test)]
fn build_policy_from_dense<M: Copy>(
    legal_moves_idx: &[(M, usize)],
    probs: &[f32],
) -> Vec<(M, f32)> {
    let mut selected = Vec::with_capacity(legal_moves_idx.len());
    for &(_, idx) in legal_moves_idx {
        selected.push(if idx < probs.len() { probs[idx] } else { 0.0 });
    }
    normalize_nonnegative_in_place(&mut selected);
    legal_moves_idx
        .iter()
        .zip(selected.into_iter())
        .map(|(&(mv, _), p)| (mv, p))
        .collect()
}

fn build_policy_from_dense_bytes<M: Copy>(
    legal_moves_idx: &[(M, usize)],
    probs_bytes: &[u8],
    policy_len: usize,
) -> Vec<(M, f32)> {
    let mut selected = Vec::with_capacity(legal_moves_idx.len());
    for &(_, idx) in legal_moves_idx {
        selected.push(dense_prob_at(probs_bytes, policy_len, idx));
    }
    normalize_nonnegative_in_place(&mut selected);
    legal_moves_idx
        .iter()
        .zip(selected.into_iter())
        .map(|(&(mv, _), p)| (mv, p))
        .collect()
}

#[derive(Default)]
struct BatchCollectorProfile {
    batches: usize,
    requests: usize,
    full_batches: usize,
    partial_batches: usize,
    singleton_batches: usize,
    low_concurrency_flushes: usize,
    max_batch: usize,
    batch_sum: usize,
    adaptive_timeout_min_us: f64,
    adaptive_timeout_max_us: f64,
    adaptive_timeout_last_us: f64,
    payload_out_bytes: usize,
    payload_in_bytes: usize,
    idle_wait_s: f64,
    collect_s: f64,
    encode_s: f64,
    write_s: f64,
    read_s: f64,
    decode_s: f64,
    fallback_batches: usize,
}

impl BatchCollectorProfile {
    fn load_path() -> Option<String> {
        std::env::var("QUARTZ_RUST_QIPC_PROFILE")
            .ok()
            .filter(|s| !s.trim().is_empty())
    }

    fn note_batch(&mut self, batch_len: usize, max_batch_size: usize) {
        self.batches += 1;
        self.requests += batch_len;
        self.batch_sum += batch_len;
        self.max_batch = self.max_batch.max(batch_len);
        if batch_len == 1 {
            self.singleton_batches += 1;
        }
        if batch_len > 0 {
            if batch_len >= max_batch_size {
                self.full_batches += 1;
            } else {
                self.partial_batches += 1;
            }
        }
    }

    fn record_fallback(&mut self) {
        self.fallback_batches += 1;
    }

    fn note_adaptive_timeout(&mut self, timeout_us: f64) {
        if self.batches <= 1 {
            self.adaptive_timeout_min_us = timeout_us;
            self.adaptive_timeout_max_us = timeout_us;
        } else {
            self.adaptive_timeout_min_us = self.adaptive_timeout_min_us.min(timeout_us);
            self.adaptive_timeout_max_us = self.adaptive_timeout_max_us.max(timeout_us);
        }
        self.adaptive_timeout_last_us = timeout_us;
    }

    fn note_low_concurrency_flush(&mut self) {
        self.low_concurrency_flushes += 1;
    }

    fn flush_jsonl(
        &self,
        path: &str,
        cfg: &BatchConfig,
        transport: &str,
        broker_stats: Option<&BatchBrokerStats>,
    ) {
        let mean_batch = if self.batches > 0 {
            self.batch_sum as f64 / self.batches as f64
        } else {
            0.0
        };
        let mut obj = serde_json::json!({
            "pid": std::process::id(),
            "kind": "batch",
            "transport": transport,
            "max_batch_size": cfg.max_batch_size,
            "timeout_us": cfg.timeout_us,
            "batches": self.batches,
            "requests": self.requests,
            "full_batches": self.full_batches,
            "partial_batches": self.partial_batches,
            "singleton_batches": self.singleton_batches,
            "low_concurrency_flushes": self.low_concurrency_flushes,
            "fallback_batches": self.fallback_batches,
            "max_batch": self.max_batch,
            "mean_batch": mean_batch,
            "adaptive_timeout_min_us": self.adaptive_timeout_min_us,
            "adaptive_timeout_max_us": self.adaptive_timeout_max_us,
            "adaptive_timeout_last_us": self.adaptive_timeout_last_us,
            "payload_out_bytes": self.payload_out_bytes,
            "payload_in_bytes": self.payload_in_bytes,
            "idle_wait_s": self.idle_wait_s,
            "collect_s": self.collect_s,
            "encode_s": self.encode_s,
            "write_s": self.write_s,
            "read_s": self.read_s,
            "decode_s": self.decode_s,
        });
        if let (Some(broker), serde_json::Value::Object(map)) = (broker_stats, &mut obj) {
            if let serde_json::Value::Object(broker_map) = broker.snapshot_json() {
                for (k, v) in broker_map {
                    map.insert(k, v);
                }
            }
        }
        let line = obj.to_string();
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
            let mut rec = line.into_bytes();
            rec.push(b'\n');
            let _ = file.write_all(&rec);
        }
    }
}

fn retune_adaptive_timeout_us(
    adaptive_timeout_us: f64,
    base_timeout_us: f64,
    min_timeout_us: f64,
    max_timeout_us: f64,
    batch_len: usize,
    max_batch_size: usize,
    queue_depth: usize,
    active_waiters: usize,
) -> (f64, bool) {
    if max_batch_size <= 1 {
        return (adaptive_timeout_us, false);
    }
    let fill_ratio = batch_len as f64 / max_batch_size as f64;
    let low_concurrency = batch_len <= 2 && queue_depth == 0 && active_waiters <= 2;
    if low_concurrency {
        let target = (adaptive_timeout_us * 0.6).min(base_timeout_us * 0.75);
        return (target.max(min_timeout_us), true);
    }
    if fill_ratio < 0.45 {
        ((adaptive_timeout_us * 1.12).min(max_timeout_us), false)
    } else if fill_ratio > 0.80 {
        ((adaptive_timeout_us * 0.88).max(min_timeout_us), false)
    } else if adaptive_timeout_us > base_timeout_us {
        ((adaptive_timeout_us * 0.96).max(base_timeout_us), false)
    } else if adaptive_timeout_us < base_timeout_us {
        ((adaptive_timeout_us * 1.04).min(base_timeout_us), false)
    } else {
        (adaptive_timeout_us, false)
    }
}

#[derive(Default)]
struct SingleEvalProfile {
    calls: usize,
    payload_out_bytes: usize,
    payload_in_bytes: usize,
    encode_s: f64,
    write_s: f64,
    read_s: f64,
    decode_s: f64,
}

impl SingleEvalProfile {
    fn flush_jsonl(&self, path: &str, n_actions: usize, transport: &str) {
        let line = serde_json::json!({
            "pid": std::process::id(),
            "kind": "single",
            "transport": transport,
            "n_actions": n_actions,
            "calls": self.calls,
            "payload_out_bytes": self.payload_out_bytes,
            "payload_in_bytes": self.payload_in_bytes,
            "encode_s": self.encode_s,
            "write_s": self.write_s,
            "read_s": self.read_s,
            "decode_s": self.decode_s,
        })
        .to_string();
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
            let mut rec = line.into_bytes();
            rec.push(b'\n');
            let _ = file.write_all(&rec);
        }
    }
}

/// Evaluator that sends eval requests to stdout and reads responses from stdin.
/// Used by `search_nn` server command for NN-backed MCTS.
///
/// Protocol:
///   Rust → Python: QIPC binary frame (eval_req)
///   Python → Rust: QIPC binary frame (eval_resp)
pub struct StdioCallbackEval {
    n_actions: usize,
    io: std::sync::Mutex<()>,
    shm: Option<std::sync::Arc<SharedMemTransport>>,
    profile_path: Option<String>,
    profile: std::sync::Mutex<SingleEvalProfile>,
    feature_scratch: std::sync::Mutex<Vec<f32>>,
}

impl StdioCallbackEval {
    pub fn new(n_actions: usize) -> Self {
        StdioCallbackEval {
            n_actions,
            io: std::sync::Mutex::new(()),
            shm: SharedMemTransport::load_from_env().map(std::sync::Arc::new),
            profile_path: BatchCollectorProfile::load_path(),
            profile: std::sync::Mutex::new(SingleEvalProfile::default()),
            feature_scratch: std::sync::Mutex::new(Vec::new()),
        }
    }
}

impl<G: GameState> Evaluator<G> for StdioCallbackEval {
    fn evaluate(&self, state: &G) -> EvalResult<G::Move> {
        let legal = state.legal_moves();
        if legal.is_empty() {
            return EvalResult {
                policy: vec![],
                value: 0.0,
            };
        }

        let mut planes = self.feature_scratch.lock().unwrap();
        state.encode_planes_into(&mut planes);
        let (feature_fp_lo, feature_fp_hi, encoder_rev) =
            eval_cache_fingerprint(state, planes.len(), self.n_actions);
        let mut legal_moves_idx = Vec::with_capacity(legal.len());
        for &mv in &legal {
            legal_moves_idx.push((mv, state.move_to_idx(mv)));
        }

        let encode_t0 = Instant::now();
        let req = encode_eval_req_payload(
            &planes,
            self.n_actions,
            0,
            feature_fp_lo,
            feature_fp_hi,
            encoder_rev,
        );
        planes.clear();
        drop(planes);

        // Mutex serializes ALL evaluate() calls — critical for bidirectional I/O.
        // Only one thread can write eval_req + read eval_resp at a time.
        let _guard = self.io.lock().unwrap();
        if let Ok(mut stats) = self.profile.lock() {
            stats.calls += 1;
            stats.encode_s += encode_t0.elapsed().as_secs_f64();
            stats.payload_out_bytes += req.len();
        }

        // Write eval_req
        let write_t0 = Instant::now();
        {
            let mut out = std::io::stdout().lock();
            if write_qipc_eval_frame(&mut out, self.shm.as_deref(), QIPC_EVAL_REQ, &req).is_err() {
                return EvalResult::uniform(&legal, 0.0);
            }
        }
        if let Ok(mut stats) = self.profile.lock() {
            stats.write_s += write_t0.elapsed().as_secs_f64();
        }

        // Read eval_resp
        let read_t0 = Instant::now();
        let response = {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            read_qipc_frame(&mut reader).ok()
        };
        if let Ok(mut stats) = self.profile.lock() {
            stats.read_s += read_t0.elapsed().as_secs_f64();
        }

        let (mut frame_kind, payload) = match response {
            Some(frame) => frame,
            None => return EvalResult::uniform(&legal, 0.0),
        };
        let payload_bytes: &[u8] = match frame_kind {
            QIPC_EVAL_RESP_SHM => {
                let Some(shm) = self.shm.as_deref() else {
                    return EvalResult::uniform(&legal, 0.0);
                };
                let Some(n_bytes) = qipc_shm_meta_len(&payload) else {
                    return EvalResult::uniform(&legal, 0.0);
                };
                frame_kind = QIPC_EVAL_RESP;
                match shm.read_response(n_bytes) {
                    Some(bytes) => bytes,
                    None => return EvalResult::uniform(&legal, 0.0),
                }
            }
            _ => payload.as_slice(),
        };
        if frame_kind != QIPC_EVAL_RESP {
            return EvalResult::uniform(&legal, 0.0);
        }

        if let Ok(mut stats) = self.profile.lock() {
            stats.payload_in_bytes += payload_bytes.len();
        }
        let decode_t0 = Instant::now();
        let mut offset = 0;
        let decoded = read_u32_le(payload_bytes, &mut offset).and_then(|policy_len| {
            let probs_bytes = read_f32_bytes(payload_bytes, &mut offset, policy_len)?;
            let value = read_f32_le(payload_bytes, &mut offset)?;
            if offset != payload_bytes.len() {
                return None;
            }
            Some((policy_len, probs_bytes, value))
        });
        if let Some((policy_len, probs_bytes, value)) = decoded {
            let policy = build_policy_from_dense_bytes(&legal_moves_idx, probs_bytes, policy_len);
            if let Ok(mut stats) = self.profile.lock() {
                stats.decode_s += decode_t0.elapsed().as_secs_f64();
            }
            return EvalResult { policy, value };
        }

        EvalResult::uniform(&legal, 0.0)
    }
}

impl Drop for StdioCallbackEval {
    fn drop(&mut self) {
        if let Some(path) = self.profile_path.as_ref() {
            if let Ok(stats) = self.profile.lock() {
                stats.flush_jsonl(
                    path,
                    self.n_actions,
                    if self.shm.is_some() { "shm" } else { "stdio" },
                );
            }
        }
    }
}

unsafe impl Send for StdioCallbackEval {}
unsafe impl Sync for StdioCallbackEval {}

// ─── BatchStdioEval: batched NN evaluation via producer-consumer channels ───

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for batched NN evaluation.
pub struct BatchConfig {
    /// Maximum requests to batch into one GPU forward pass.
    pub max_batch_size: usize,
    /// Microseconds to wait for the batch to fill before dispatching.
    pub timeout_us: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        BatchConfig {
            max_batch_size: 8,
            timeout_us: 1500,
        }
    }
}

struct FeatureVecPool {
    buffers: Mutex<Vec<Vec<f32>>>,
    max_cached: usize,
}

impl FeatureVecPool {
    fn new(max_cached: usize) -> Self {
        Self {
            buffers: Mutex::new(Vec::new()),
            max_cached,
        }
    }

    #[inline]
    fn checkout(&self) -> Vec<f32> {
        self.buffers.lock().unwrap().pop().unwrap_or_default()
    }

    #[inline]
    fn give_back(&self, mut buffer: Vec<f32>) {
        if self.max_cached == 0 {
            return;
        }
        buffer.clear();
        let mut guard = self.buffers.lock().unwrap();
        if guard.len() < self.max_cached {
            guard.push(buffer);
        }
    }
}

/// Internal request sent from MCTS worker threads to the collector.
struct BatchRequest<M: Copy + Send + 'static> {
    features: Vec<f32>,
    /// Legal moves with their action-space indices for policy lookup.
    legal_moves_idx: Vec<(M, usize)>,
    n_actions: usize,
    model_tag: u32,
    feature_fp_lo: u64,
    feature_fp_hi: u64,
    encoder_rev: u32,
    enqueued_at: Instant,
    result_tx: channel::Sender<EvalResult<M>>,
    feature_pool: Option<Arc<FeatureVecPool>>,
}

#[derive(Default)]
struct BatchBrokerStats {
    submitted_requests: AtomicUsize,
    dequeued_requests: AtomicUsize,
    completed_requests: AtomicUsize,
    queue_depth: AtomicUsize,
    max_queue_depth: AtomicUsize,
    active_waiters: AtomicUsize,
    max_active_waiters: AtomicUsize,
    flush_target_batch: AtomicUsize,
    flush_timeout: AtomicUsize,
    flush_low_concurrency: AtomicUsize,
    flush_fallback: AtomicUsize,
    queue_wait_micros: AtomicU64,
    result_wait_micros: AtomicU64,
}

impl BatchBrokerStats {
    fn update_max(slot: &AtomicUsize, value: usize) {
        let mut prev = slot.load(Ordering::Relaxed);
        while value > prev {
            match slot.compare_exchange(prev, value, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(cur) => prev = cur,
            }
        }
    }

    fn on_submit(&self) {
        self.submitted_requests.fetch_add(1, Ordering::Relaxed);
        let depth = self.queue_depth.fetch_add(1, Ordering::Relaxed) + 1;
        Self::update_max(&self.max_queue_depth, depth);
    }

    fn on_send_failed(&self) {
        self.queue_depth.fetch_sub(1, Ordering::Relaxed);
    }

    fn on_dequeue(&self, enqueued_at: Instant) {
        self.dequeued_requests.fetch_add(1, Ordering::Relaxed);
        self.queue_depth.fetch_sub(1, Ordering::Relaxed);
        self.queue_wait_micros
            .fetch_add(enqueued_at.elapsed().as_micros() as u64, Ordering::Relaxed);
    }

    fn waiter_enter(&self) {
        let waiters = self.active_waiters.fetch_add(1, Ordering::Relaxed) + 1;
        Self::update_max(&self.max_active_waiters, waiters);
    }

    fn waiter_exit(&self, wait_started_at: Instant) {
        self.active_waiters.fetch_sub(1, Ordering::Relaxed);
        self.completed_requests.fetch_add(1, Ordering::Relaxed);
        self.result_wait_micros.fetch_add(
            wait_started_at.elapsed().as_micros() as u64,
            Ordering::Relaxed,
        );
    }

    fn record_flush_reason(&self, reason: &str) {
        match reason {
            "target_batch_reached" => {
                self.flush_target_batch.fetch_add(1, Ordering::Relaxed);
            }
            "max_wait_reached" => {
                self.flush_timeout.fetch_add(1, Ordering::Relaxed);
            }
            "low_concurrency" => {
                self.flush_low_concurrency.fetch_add(1, Ordering::Relaxed);
            }
            "fallback" => {
                self.flush_fallback.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    fn snapshot_json(&self) -> serde_json::Value {
        serde_json::json!({
            "submitted_requests": self.submitted_requests.load(Ordering::Relaxed),
            "dequeued_requests": self.dequeued_requests.load(Ordering::Relaxed),
            "completed_requests": self.completed_requests.load(Ordering::Relaxed),
            "queue_depth": self.queue_depth.load(Ordering::Relaxed),
            "max_queue_depth": self.max_queue_depth.load(Ordering::Relaxed),
            "active_waiters": self.active_waiters.load(Ordering::Relaxed),
            "max_active_waiters": self.max_active_waiters.load(Ordering::Relaxed),
            "flush_reason_counts": {
                "target_batch_reached": self.flush_target_batch.load(Ordering::Relaxed),
                "max_wait_reached": self.flush_timeout.load(Ordering::Relaxed),
                "low_concurrency": self.flush_low_concurrency.load(Ordering::Relaxed),
                "fallback": self.flush_fallback.load(Ordering::Relaxed),
            },
            "queue_wait_s": self.queue_wait_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            "result_wait_s": self.result_wait_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        })
    }
}

fn rust_eval_trace_path() -> Option<&'static str> {
    static TRACE_PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    TRACE_PATH
        .get_or_init(|| {
            std::env::var("QUARTZ_RUST_SERVER_TRACE")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .as_deref()
}

fn rust_eval_trace(event: &str, fields: serde_json::Value) {
    let Some(path) = rust_eval_trace_path() else {
        return;
    };
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let mut obj = serde_json::Map::new();
    obj.insert("ts".to_string(), serde_json::json!(ts));
    obj.insert("pid".to_string(), serde_json::json!(std::process::id()));
    obj.insert("event".to_string(), serde_json::json!(event));
    if let serde_json::Value::Object(map) = fields {
        for (k, v) in map {
            obj.insert(k, v);
        }
    }
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let mut line = serde_json::Value::Object(obj).to_string().into_bytes();
        line.push(b'\n');
        let _ = file.write_all(&line);
    }
}

// ─── GlobalBroker: single process-wide eval I/O owner ───

/// Shared state backing the GlobalBroker, referenced by all `BatchStdioEval` instances.
pub(crate) struct GlobalBrokerShared<M: Copy + Eq + Hash + Debug + Send + 'static> {
    request_tx: channel::Sender<BatchRequest<M>>,
    shutdown: Arc<AtomicBool>,
    io_handle: Mutex<Option<std::thread::JoinHandle<()>>>,
    stats: Arc<BatchBrokerStats>,
    feature_pool: Arc<FeatureVecPool>,
}

/// Single process-wide inference broker. Owns the collector thread that performs
/// all QIPC I/O (stdin/stdout). All `BatchStdioEval` instances share one broker.
pub struct GlobalBroker<M: Copy + Eq + Hash + Debug + Send + 'static> {
    shared: Arc<GlobalBrokerShared<M>>,
}

impl<M: Copy + Eq + Hash + Debug + Send + 'static> GlobalBroker<M> {
    pub fn new(n_actions: usize, config: BatchConfig) -> Self {
        let (request_tx, request_rx) = channel::unbounded::<BatchRequest<M>>();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        let stats = Arc::new(BatchBrokerStats::default());
        let stats_for_thread = stats.clone();
        let feature_pool = Arc::new(FeatureVecPool::new(config.max_batch_size * 4));
        let shm = SharedMemTransport::load_from_env().map(Arc::new);
        let use_ring = global_ring_buffer().is_some();

        let handle = std::thread::Builder::new()
            .name("global-broker".into())
            .spawn(move || {
                if use_ring {
                    if let Some(ring) = global_ring_buffer() {
                        broker_loop_shm::<M>(
                            request_rx,
                            &config,
                            &shutdown_clone,
                            &stats_for_thread,
                            ring,
                        );
                        return;
                    }
                }
                broker_loop::<M>(
                    request_rx,
                    n_actions,
                    &config,
                    &shutdown_clone,
                    &stats_for_thread,
                    shm,
                );
            })
            .expect("failed to spawn global-broker thread");

        GlobalBroker {
            shared: Arc::new(GlobalBrokerShared {
                request_tx,
                shutdown,
                io_handle: Mutex::new(Some(handle)),
                stats,
                feature_pool,
            }),
        }
    }

    pub fn shared(&self) -> &Arc<GlobalBrokerShared<M>> {
        &self.shared
    }
}

impl<M: Copy + Eq + Hash + Debug + Send + 'static> Drop for GlobalBroker<M> {
    fn drop(&mut self) {
        // Only shut down the broker thread when no other holders remain.
        // BatchStdioEval instances (and from_broker callers) hold their own
        // Arc<GlobalBrokerShared>, so the broker must stay alive while they exist.
        if Arc::strong_count(&self.shared) <= 1 {
            self.shared.shutdown.store(true, Ordering::Relaxed);
            if let Some(handle) = self.shared.io_handle.lock().unwrap().take() {
                let _ = handle.join();
            }
        }
    }
}

/// Batched NN evaluator for multi-threaded MCTS (thin wrapper around GlobalBroker).
///
/// Architecture:
/// - MCTS worker threads call `evaluate()` → encode state → push to channel → block for result
/// - A single GlobalBroker thread drains the channel, builds batch payload, does one stdout/stdin
///   round-trip for the whole batch, then distributes results via per-request response channels.
///
/// Protocol:
///   Rust → Python: QIPC binary frame (batch_eval_req)
///   Python → Rust: QIPC binary frame (batch_eval_resp)

pub struct AsyncEvalTicket<M: Copy + Eq + Hash + Debug + Send + 'static> {
    legal_moves: Vec<M>,
    result_rx: channel::Receiver<EvalResult<M>>,
    wait_started_at: Instant,
    stats: Arc<BatchBrokerStats>,
    accounted: bool,
}

impl<M: Copy + Eq + Hash + Debug + Send + 'static> AsyncEvalTicket<M> {
    fn from_parts(
        legal_moves: Vec<M>,
        result_rx: channel::Receiver<EvalResult<M>>,
        wait_started_at: Instant,
        stats: Arc<BatchBrokerStats>,
    ) -> Self {
        Self {
            legal_moves,
            result_rx,
            wait_started_at,
            stats,
            accounted: false,
        }
    }

    pub fn try_take(&mut self) -> Option<EvalResult<M>> {
        match self.result_rx.try_recv() {
            Ok(result) => {
                self.accounted = true;
                self.stats.waiter_exit(self.wait_started_at);
                Some(result)
            }
            Err(channel::TryRecvError::Empty) => None,
            Err(channel::TryRecvError::Disconnected) => {
                self.accounted = true;
                self.stats.waiter_exit(self.wait_started_at);
                Some(EvalResult::uniform(&self.legal_moves, 0.0))
            }
        }
    }

    pub fn recv_blocking(mut self) -> EvalResult<M> {
        match self.result_rx.recv() {
            Ok(result) => {
                self.accounted = true;
                self.stats.waiter_exit(self.wait_started_at);
                result
            }
            Err(_) => {
                self.accounted = true;
                self.stats.waiter_exit(self.wait_started_at);
                EvalResult::uniform(&self.legal_moves, 0.0)
            }
        }
    }
}

impl<M: Copy + Eq + Hash + Debug + Send + 'static> Drop for AsyncEvalTicket<M> {
    fn drop(&mut self) {
        if !self.accounted {
            self.accounted = true;
            self.stats.waiter_exit(self.wait_started_at);
        }
    }
}

pub struct BatchStdioEval<M: Copy + Eq + Hash + Debug + Send + 'static> {
    broker: Arc<GlobalBrokerShared<M>>,
    model_tag: u32,
}

impl<M: Copy + Eq + Hash + Debug + Send + 'static> BatchStdioEval<M> {
    /// Create a new evaluator backed by its own broker (convenience for single-use).
    pub fn new(n_actions: usize, config: BatchConfig) -> Self {
        let broker = GlobalBroker::<M>::new(n_actions, config);
        Self {
            broker: broker.shared().clone(),
            model_tag: 0,
        }
        // GlobalBroker drops here but skips shutdown because strong_count > 1.
        // The broker thread stays alive until this eval (and all clones) are dropped.
    }

    /// Create a pair of evaluators sharing a single broker (convenience).
    pub fn new_shared_pair(
        n_actions: usize,
        config: BatchConfig,
        tag_a: u32,
        tag_b: u32,
    ) -> (Self, Self) {
        let broker = GlobalBroker::<M>::new(n_actions, config);
        let shared = broker.shared().clone();
        (
            Self {
                broker: shared.clone(),
                model_tag: tag_a,
            },
            Self {
                broker: shared,
                model_tag: tag_b,
            },
        )
    }

    /// Create an evaluator from an existing GlobalBroker.
    pub fn from_broker(broker: &GlobalBroker<M>, model_tag: u32) -> Self {
        Self {
            broker: broker.shared().clone(),
            model_tag,
        }
    }
}

/// Broker thread main loop: drain requests → batch I/O → distribute results.
/// Single-threaded: owns both stdin (read response) and stdout (write request)
/// exclusively during each batch cycle. This avoids stdin contention with serve().
fn broker_loop<M: Copy + Eq + Hash + Debug + Send + 'static>(
    rx: channel::Receiver<BatchRequest<M>>,
    _n_actions: usize,
    config: &BatchConfig,
    shutdown: &AtomicBool,
    stats: &BatchBrokerStats,
    shm: Option<Arc<SharedMemTransport>>,
) {
    let base_timeout_us = config.timeout_us.max(250) as f64;
    let min_timeout_us = (base_timeout_us * 0.5).max(250.0);
    let max_timeout_us = (base_timeout_us * 4.0).min(12000.0);
    let mut adaptive_timeout_us = base_timeout_us;
    let profile_path = BatchCollectorProfile::load_path();
    let mut profile = profile_path
        .as_ref()
        .map(|_| BatchCollectorProfile::default());
    loop {
        let idle_t0 = Instant::now();
        let first = match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(req) => req,
            Err(channel::RecvTimeoutError::Timeout) => {
                if let Some(s) = profile.as_mut() {
                    s.idle_wait_s += idle_t0.elapsed().as_secs_f64();
                }
                if shutdown.load(Ordering::Relaxed) {
                    if let (Some(path), Some(ps)) = (profile_path.as_ref(), profile.as_ref()) {
                        ps.flush_jsonl(
                            path,
                            config,
                            if shm.is_some() { "shm" } else { "stdio" },
                            Some(stats),
                        );
                    }
                    return;
                }
                continue;
            }
            Err(channel::RecvTimeoutError::Disconnected) => {
                if let (Some(path), Some(ps)) = (profile_path.as_ref(), profile.as_ref()) {
                    ps.flush_jsonl(
                        path,
                        config,
                        if shm.is_some() { "shm" } else { "stdio" },
                        Some(stats),
                    );
                }
                return;
            }
        };
        stats.on_dequeue(first.enqueued_at);
        if let Some(s) = profile.as_mut() {
            s.idle_wait_s += idle_t0.elapsed().as_secs_f64();
        }

        let mut batch: Vec<BatchRequest<M>> = Vec::with_capacity(config.max_batch_size);
        batch.push(first);
        let timeout = Duration::from_micros(adaptive_timeout_us.round() as u64);
        let deadline = Instant::now() + timeout;
        let collect_t0 = Instant::now();
        while batch.len() < config.max_batch_size {
            while batch.len() < config.max_batch_size {
                match rx.try_recv() {
                    Ok(req) => {
                        stats.on_dequeue(req.enqueued_at);
                        batch.push(req)
                    }
                    Err(channel::TryRecvError::Empty) => break,
                    Err(channel::TryRecvError::Disconnected) => break,
                }
            }
            if batch.len() >= config.max_batch_size {
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match rx.recv_timeout(remaining) {
                Ok(req) => {
                    stats.on_dequeue(req.enqueued_at);
                    batch.push(req)
                }
                Err(channel::RecvTimeoutError::Timeout) => break,
                Err(channel::RecvTimeoutError::Disconnected) => break,
            }
        }
        if let Some(s) = profile.as_mut() {
            s.collect_s += collect_t0.elapsed().as_secs_f64();
            s.note_batch(batch.len(), config.max_batch_size);
        }
        let queue_depth_now = stats.queue_depth.load(Ordering::Relaxed);
        let active_waiters_now = stats.active_waiters.load(Ordering::Relaxed);
        let (next_timeout_us, low_concurrency_flush) = retune_adaptive_timeout_us(
            adaptive_timeout_us,
            base_timeout_us,
            min_timeout_us,
            max_timeout_us,
            batch.len(),
            config.max_batch_size,
            queue_depth_now,
            active_waiters_now,
        );
        let flush_reason = if batch.len() >= config.max_batch_size {
            "target_batch_reached"
        } else if low_concurrency_flush {
            "low_concurrency"
        } else {
            "max_wait_reached"
        };
        stats.record_flush_reason(flush_reason);
        adaptive_timeout_us = next_timeout_us;
        if let Some(s) = profile.as_mut() {
            s.note_adaptive_timeout(adaptive_timeout_us);
            if low_concurrency_flush {
                s.note_low_concurrency_flush();
            }
        }

        let encode_t0 = Instant::now();
        let payload = encode_batch_eval_req_payload(&batch);
        if let Some(s) = profile.as_mut() {
            s.encode_s += encode_t0.elapsed().as_secs_f64();
            s.payload_out_bytes += payload.len();
        }

        let write_t0 = Instant::now();
        let write_ok = {
            let stdout = std::io::stdout();
            let mut writer = stdout.lock();
            write_qipc_eval_frame(&mut writer, shm.as_deref(), QIPC_BATCH_EVAL_REQ, &payload)
                .is_ok()
        };
        if let Some(s) = profile.as_mut() {
            s.write_s += write_t0.elapsed().as_secs_f64();
        }

        if !write_ok {
            send_uniform_fallback(&batch);
            reclaim_batch_features(&mut batch);
            stats.record_flush_reason("fallback");
            if let Some(s) = profile.as_mut() {
                s.record_fallback();
            }
            if shutdown.load(Ordering::Relaxed) {
                if let (Some(path), Some(ps)) = (profile_path.as_ref(), profile.as_ref()) {
                    ps.flush_jsonl(
                        path,
                        config,
                        if shm.is_some() { "shm" } else { "stdio" },
                        Some(stats),
                    );
                }
                return;
            }
            continue;
        }

        let read_t0 = Instant::now();
        let response = {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            read_qipc_frame(&mut reader).ok()
        };
        if let Some(s) = profile.as_mut() {
            s.read_s += read_t0.elapsed().as_secs_f64();
        }

        let Some((mut frame_kind, resp_payload)) = response else {
            send_uniform_fallback(&batch);
            reclaim_batch_features(&mut batch);
            stats.record_flush_reason("fallback");
            if let Some(s) = profile.as_mut() {
                s.record_fallback();
            }
            if shutdown.load(Ordering::Relaxed) {
                if let (Some(path), Some(ps)) = (profile_path.as_ref(), profile.as_ref()) {
                    ps.flush_jsonl(
                        path,
                        config,
                        if shm.is_some() { "shm" } else { "stdio" },
                        Some(stats),
                    );
                }
                return;
            }
            continue;
        };
        let payload_bytes: &[u8] = match frame_kind {
            QIPC_BATCH_EVAL_RESP_SHM => {
                let Some(shm) = shm.as_deref() else {
                    send_uniform_fallback(&batch);
                    reclaim_batch_features(&mut batch);
                    stats.record_flush_reason("fallback");
                    if let Some(s) = profile.as_mut() {
                        s.record_fallback();
                    }
                    continue;
                };
                let Some(n_bytes) = qipc_shm_meta_len(&resp_payload) else {
                    send_uniform_fallback(&batch);
                    reclaim_batch_features(&mut batch);
                    stats.record_flush_reason("fallback");
                    if let Some(s) = profile.as_mut() {
                        s.record_fallback();
                    }
                    continue;
                };
                frame_kind = QIPC_BATCH_EVAL_RESP;
                match shm.read_response(n_bytes) {
                    Some(bytes) => bytes,
                    None => {
                        send_uniform_fallback(&batch);
                        reclaim_batch_features(&mut batch);
                        stats.record_flush_reason("fallback");
                        if let Some(s) = profile.as_mut() {
                            s.record_fallback();
                        }
                        continue;
                    }
                }
            }
            _ => resp_payload.as_slice(),
        };
        if frame_kind != QIPC_BATCH_EVAL_RESP {
            send_uniform_fallback(&batch);
            reclaim_batch_features(&mut batch);
            stats.record_flush_reason("fallback");
            if let Some(s) = profile.as_mut() {
                s.record_fallback();
            }
            continue;
        }

        if let Some(s) = profile.as_mut() {
            s.payload_in_bytes += payload_bytes.len();
        }
        let decode_t0 = Instant::now();
        distribute_binary_batch(payload_bytes, &batch);
        reclaim_batch_features(&mut batch);
        if let Some(s) = profile.as_mut() {
            s.decode_s += decode_t0.elapsed().as_secs_f64();
        }
        if let Some(ps) = profile.as_ref() {
            if ps.batches % 64 == 0 {
                rust_eval_trace(
                    "batch_broker_snapshot",
                    serde_json::json!({
                        "transport": if shm.is_some() { "shm" } else { "stdio" },
                        "max_batch_size": config.max_batch_size,
                        "adaptive_timeout_us": adaptive_timeout_us,
                        "queue_depth": queue_depth_now,
                        "active_waiters": active_waiters_now,
                        "low_concurrency_flush": low_concurrency_flush,
                        "broker": stats.snapshot_json(),
                    }),
                );
            }
        }
    }
}

/// SHM ring buffer variant of broker_loop.
/// Writes eval batch requests to r2p slots and reads responses from p2r slots.
/// Never touches stdout/stdin — all IPC goes through shared memory.
fn broker_loop_shm<M: Copy + Eq + Hash + Debug + Send + 'static>(
    rx: channel::Receiver<BatchRequest<M>>,
    config: &BatchConfig,
    shutdown: &AtomicBool,
    stats: &BatchBrokerStats,
    ring: &ShmRingBuffer,
) {
    let base_timeout_us = config.timeout_us.max(250) as f64;
    let min_timeout_us = (base_timeout_us * 0.5).max(250.0);
    let max_timeout_us = (base_timeout_us * 4.0).min(12000.0);
    let mut adaptive_timeout_us = base_timeout_us;
    let profile_path = BatchCollectorProfile::load_path();
    let mut profile = profile_path
        .as_ref()
        .map(|_| BatchCollectorProfile::default());
    let mut seq: u32 = 0;

    loop {
        // --- Idle wait for first request ---
        let idle_t0 = Instant::now();
        let first = match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(req) => req,
            Err(channel::RecvTimeoutError::Timeout) => {
                if let Some(s) = profile.as_mut() {
                    s.idle_wait_s += idle_t0.elapsed().as_secs_f64();
                }
                if shutdown.load(Ordering::Relaxed) {
                    if let (Some(path), Some(ps)) = (profile_path.as_ref(), profile.as_ref()) {
                        ps.flush_jsonl(path, config, "shm_ring", Some(stats));
                    }
                    return;
                }
                continue;
            }
            Err(channel::RecvTimeoutError::Disconnected) => {
                if let (Some(path), Some(ps)) = (profile_path.as_ref(), profile.as_ref()) {
                    ps.flush_jsonl(path, config, "shm_ring", Some(stats));
                }
                return;
            }
        };
        stats.on_dequeue(first.enqueued_at);
        if let Some(s) = profile.as_mut() {
            s.idle_wait_s += idle_t0.elapsed().as_secs_f64();
        }

        // --- Batch collection (identical to broker_loop) ---
        let mut batch: Vec<BatchRequest<M>> = Vec::with_capacity(config.max_batch_size);
        batch.push(first);
        let timeout = Duration::from_micros(adaptive_timeout_us.round() as u64);
        let deadline = Instant::now() + timeout;
        let collect_t0 = Instant::now();
        while batch.len() < config.max_batch_size {
            while batch.len() < config.max_batch_size {
                match rx.try_recv() {
                    Ok(req) => {
                        stats.on_dequeue(req.enqueued_at);
                        batch.push(req)
                    }
                    Err(channel::TryRecvError::Empty) => break,
                    Err(channel::TryRecvError::Disconnected) => break,
                }
            }
            if batch.len() >= config.max_batch_size {
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match rx.recv_timeout(remaining) {
                Ok(req) => {
                    stats.on_dequeue(req.enqueued_at);
                    batch.push(req)
                }
                Err(channel::RecvTimeoutError::Timeout) => break,
                Err(channel::RecvTimeoutError::Disconnected) => break,
            }
        }
        if let Some(s) = profile.as_mut() {
            s.collect_s += collect_t0.elapsed().as_secs_f64();
            s.note_batch(batch.len(), config.max_batch_size);
        }
        let queue_depth_now = stats.queue_depth.load(Ordering::Relaxed);
        let active_waiters_now = stats.active_waiters.load(Ordering::Relaxed);
        let (next_timeout_us, low_concurrency_flush) = retune_adaptive_timeout_us(
            adaptive_timeout_us,
            base_timeout_us,
            min_timeout_us,
            max_timeout_us,
            batch.len(),
            config.max_batch_size,
            queue_depth_now,
            active_waiters_now,
        );
        let flush_reason = if batch.len() >= config.max_batch_size {
            "target_batch_reached"
        } else if low_concurrency_flush {
            "low_concurrency"
        } else {
            "max_wait_reached"
        };
        stats.record_flush_reason(flush_reason);
        adaptive_timeout_us = next_timeout_us;
        if let Some(s) = profile.as_mut() {
            s.note_adaptive_timeout(adaptive_timeout_us);
            if low_concurrency_flush {
                s.note_low_concurrency_flush();
            }
        }

        // --- Encode batch ---
        let encode_t0 = Instant::now();
        let payload = encode_batch_eval_req_payload(&batch);
        if let Some(s) = profile.as_mut() {
            s.encode_s += encode_t0.elapsed().as_secs_f64();
            s.payload_out_bytes += payload.len();
        }

        // --- Write to r2p ring slot ---
        let write_t0 = Instant::now();
        let epoch = ring.epoch();
        seq = seq.wrapping_add(1);

        // Spin-wait for an empty r2p slot, reclaiming DONE slots
        let mut write_ok = false;
        let write_deadline = Instant::now() + Duration::from_secs(5);
        let mut spin = 0u32;
        while Instant::now() < write_deadline {
            ring.r2p_reclaim();
            if ring.r2p_try_write(SHM_MSG_EVAL_BATCH_REQ, &payload, epoch, seq) {
                write_ok = true;
                break;
            }
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
            spin += 1;
            if spin < 64 {
                std::hint::spin_loop();
            } else if spin < 512 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(Duration::from_micros(10));
            }
        }
        if let Some(s) = profile.as_mut() {
            s.write_s += write_t0.elapsed().as_secs_f64();
        }

        if !write_ok {
            send_uniform_fallback(&batch);
            reclaim_batch_features(&mut batch);
            stats.record_flush_reason("fallback");
            if let Some(s) = profile.as_mut() {
                s.record_fallback();
            }
            if shutdown.load(Ordering::Relaxed) {
                if let (Some(path), Some(ps)) = (profile_path.as_ref(), profile.as_ref()) {
                    ps.flush_jsonl(path, config, "shm_ring", Some(stats));
                }
                return;
            }
            continue;
        }

        // --- Read response from p2r ring slot (spin-wait) ---
        // Validate epoch/seq to reject stale responses from timed-out batches.
        let read_t0 = Instant::now();
        let mut resp_payload: Option<Vec<u8>> = None;
        let read_deadline = Instant::now() + Duration::from_secs(30);
        spin = 0;
        while Instant::now() < read_deadline {
            if let Some((msg_type, resp_epoch, resp_seq, data)) = ring.p2r_try_read_meta() {
                if msg_type == SHM_MSG_EVAL_BATCH_RESP && resp_epoch == epoch && resp_seq == seq {
                    resp_payload = Some(data.to_vec());
                    break;
                }
                // Stale or mismatched response — discard and keep waiting
            }
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
            spin += 1;
            if spin < 64 {
                std::hint::spin_loop();
            } else if spin < 512 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(Duration::from_micros(10));
            }
        }
        if let Some(s) = profile.as_mut() {
            s.read_s += read_t0.elapsed().as_secs_f64();
        }

        let Some(payload_bytes) = resp_payload else {
            send_uniform_fallback(&batch);
            reclaim_batch_features(&mut batch);
            stats.record_flush_reason("fallback");
            if let Some(s) = profile.as_mut() {
                s.record_fallback();
            }
            if shutdown.load(Ordering::Relaxed) {
                if let (Some(path), Some(ps)) = (profile_path.as_ref(), profile.as_ref()) {
                    ps.flush_jsonl(path, config, "shm_ring", Some(stats));
                }
                return;
            }
            continue;
        };

        // --- Distribute results ---
        if let Some(s) = profile.as_mut() {
            s.payload_in_bytes += payload_bytes.len();
        }
        let decode_t0 = Instant::now();
        distribute_binary_batch(&payload_bytes, &batch);
        reclaim_batch_features(&mut batch);
        if let Some(s) = profile.as_mut() {
            s.decode_s += decode_t0.elapsed().as_secs_f64();
        }
        if let Some(ps) = profile.as_ref() {
            if ps.batches % 64 == 0 {
                rust_eval_trace(
                    "batch_broker_snapshot",
                    serde_json::json!({
                        "transport": "shm_ring",
                        "max_batch_size": config.max_batch_size,
                        "adaptive_timeout_us": adaptive_timeout_us,
                        "queue_depth": queue_depth_now,
                        "active_waiters": active_waiters_now,
                        "low_concurrency_flush": low_concurrency_flush,
                        "broker": stats.snapshot_json(),
                    }),
                );
            }
        }
    }
}

fn distribute_binary_batch<M: Copy + Eq + Hash + Debug + Send + 'static>(
    payload: &[u8],
    batch: &[BatchRequest<M>],
) {
    let mut offset = 0;
    let Some(batch_size) = read_u32_le(payload, &mut offset) else {
        send_uniform_fallback(batch);
        return;
    };
    if batch_size != batch.len() {
        send_uniform_fallback(batch);
        return;
    }

    let mut results = Vec::with_capacity(batch.len());
    for req in batch.iter() {
        let Some(policy_len) = read_u32_le(payload, &mut offset) else {
            send_uniform_fallback(batch);
            return;
        };
        let Some(probs_bytes) = read_f32_bytes(payload, &mut offset, policy_len) else {
            send_uniform_fallback(batch);
            return;
        };
        let Some(value) = read_f32_le(payload, &mut offset) else {
            send_uniform_fallback(batch);
            return;
        };
        results.push(EvalResult {
            policy: build_policy_from_dense_bytes(&req.legal_moves_idx, probs_bytes, policy_len),
            value,
        });
    }
    if offset != payload.len() {
        send_uniform_fallback(batch);
        return;
    }
    for (req, result) in batch.iter().zip(results.into_iter()) {
        let _ = req.result_tx.send(result);
    }
}

fn send_uniform_fallback<M: Copy + Send + 'static>(batch: &[BatchRequest<M>]) {
    for req in batch {
        send_single_uniform(req);
    }
}

fn send_single_uniform<M: Copy + Send + 'static>(req: &BatchRequest<M>) {
    let n = req.legal_moves_idx.len();
    let p = if n > 0 { 1.0 / n as f32 } else { 0.0 };
    let policy = req.legal_moves_idx.iter().map(|&(m, _)| (m, p)).collect();
    let _ = req.result_tx.send(EvalResult { policy, value: 0.0 });
}

/// Parse batch_eval_resp JSON and send results to waiting threads (test-only).
#[cfg(test)]
fn parse_and_distribute<M: Copy + Eq + Hash + Debug + Send + 'static>(
    resp_line: &str,
    batch: &[BatchRequest<M>],
) {
    let line = resp_line.trim();

    // Find "responses":[ array
    let responses_key = "\"responses\":[";
    let resp_start = match line.find(responses_key) {
        Some(pos) => pos + responses_key.len(),
        None => {
            send_uniform_fallback(batch);
            return;
        }
    };

    // Parse each {"policy":[...],"value":V} entry sequentially
    let mut cursor = resp_start;
    for req in batch.iter() {
        let pol_key = "\"policy\":[";
        let pol_start = match line[cursor..].find(pol_key) {
            Some(pos) => cursor + pos + pol_key.len(),
            None => {
                send_single_uniform(req);
                continue;
            }
        };
        let pol_end = match line[pol_start..].find(']') {
            Some(pos) => pol_start + pos,
            None => {
                send_single_uniform(req);
                continue;
            }
        };
        let probs: Vec<f32> = line[pol_start..pol_end]
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        let val_key = "\"value\":";
        let value = if let Some(vs) = line[pol_end..].find(val_key) {
            let vstart = pol_end + vs + val_key.len();
            let vrest = line[vstart..].trim_start();
            let vend = vrest
                .find(|c: char| !c.is_ascii_digit() && c != '-' && c != '.')
                .unwrap_or(vrest.len());
            vrest[..vend].parse::<f32>().unwrap_or(0.0)
        } else {
            0.0
        };

        // Build policy: legal_moves_idx stores (move, action_index) pairs,
        // so we directly index into the flat probs array.
        let mut policy = Vec::with_capacity(req.legal_moves_idx.len());
        let mut sum = 0.0f32;
        for &(mv, idx) in &req.legal_moves_idx {
            let p = if idx < probs.len() {
                probs[idx].max(0.0)
            } else {
                0.0
            };
            sum += p;
            policy.push((mv, p));
        }
        if sum > 0.0 {
            for (_, p) in &mut policy {
                *p /= sum;
            }
        }

        let _ = req.result_tx.send(EvalResult { policy, value });
        cursor = pol_end;
    }
}

impl<M: Copy + Eq + Hash + Debug + Send + 'static> BatchStdioEval<M> {
    pub fn submit<G>(&self, state: &G) -> AsyncEvalTicket<M>
    where
        G: GameState<Move = M>,
    {
        let legal = state.legal_moves();
        if legal.is_empty() {
            let (result_tx, result_rx) = channel::bounded(1);
            let _ = result_tx.send(EvalResult {
                policy: vec![],
                value: 0.0,
            });
            let started = Instant::now();
            self.broker.stats.waiter_enter();
            return AsyncEvalTicket::from_parts(
                legal,
                result_rx,
                started,
                self.broker.stats.clone(),
            );
        }

        let mut planes = self.broker.feature_pool.checkout();
        state.encode_planes_into(&mut planes);
        let n_actions = state.num_actions();
        let (feature_fp_lo, feature_fp_hi, encoder_rev) =
            eval_cache_fingerprint(state, planes.len(), n_actions);
        let mut legal_moves_idx = Vec::with_capacity(legal.len());
        for &mv in &legal {
            let idx = state.move_to_idx(mv);
            legal_moves_idx.push((mv, idx));
        }

        let (result_tx, result_rx) = channel::bounded(1);
        let wait_started_at = Instant::now();
        let request = BatchRequest {
            features: planes,
            legal_moves_idx,
            n_actions,
            model_tag: self.model_tag,
            feature_fp_lo,
            feature_fp_hi,
            encoder_rev,
            enqueued_at: Instant::now(),
            result_tx,
            feature_pool: Some(self.broker.feature_pool.clone()),
        };

        self.broker.stats.on_submit();
        self.broker.stats.waiter_enter();
        if self.broker.request_tx.send(request).is_err() {
            self.broker.stats.on_send_failed();
            let (fallback_tx, fallback_rx) = channel::bounded(1);
            let _ = fallback_tx.send(EvalResult::uniform(&legal, 0.0));
            return AsyncEvalTicket::from_parts(
                legal,
                fallback_rx,
                wait_started_at,
                self.broker.stats.clone(),
            );
        }

        AsyncEvalTicket::from_parts(legal, result_rx, wait_started_at, self.broker.stats.clone())
    }
}

impl<M: Copy + Eq + Hash + Debug + Send + 'static> Clone for BatchStdioEval<M> {
    fn clone(&self) -> Self {
        Self {
            broker: self.broker.clone(),
            model_tag: self.model_tag,
        }
    }
}

impl<G: GameState> Evaluator<G> for BatchStdioEval<G::Move> {
    fn evaluate(&self, state: &G) -> EvalResult<G::Move> {
        self.submit::<G>(state).recv_blocking()
    }
}

impl<M: Copy + Eq + Hash + Debug + Send + 'static> Drop for BatchStdioEval<M> {
    fn drop(&mut self) {
        // Evaluators no longer own the broker thread. When the last reference to
        // GlobalBrokerShared is dropped, the broker thread will exit on its own
        // (channel disconnects → recv returns Disconnected).
        // For backward-compat with new() which creates an internal broker,
        // we still signal shutdown when we're the last holder.
        if Arc::strong_count(&self.broker) == 1 {
            self.broker.shutdown.store(true, Ordering::Relaxed);
            if let Some(handle) = self.broker.io_handle.lock().unwrap().take() {
                let _ = handle.join();
            }
        }
    }
}

unsafe impl<M: Copy + Eq + Hash + Debug + Send + 'static> Send for BatchStdioEval<M> {}
unsafe impl<M: Copy + Eq + Hash + Debug + Send + 'static> Sync for BatchStdioEval<M> {}

// ─── Tests ───

#[cfg(test)]
mod tests {
    use super::*;

    // Simple move type for testing (action index = move value)
    type TestMove = usize;

    // ── parse_and_distribute tests ──

    fn make_request(
        moves: &[(usize, usize)],
        tx: channel::Sender<EvalResult<TestMove>>,
    ) -> BatchRequest<TestMove> {
        let n_actions = 9; // 3x3 board
        let mut legal_moves_idx = Vec::new();
        for &(mv, idx) in moves {
            legal_moves_idx.push((mv, idx));
        }
        BatchRequest {
            features: vec![0.0; n_actions],
            legal_moves_idx,
            n_actions,
            model_tag: 0,
            feature_fp_lo: 11,
            feature_fp_hi: 22,
            encoder_rev: 1,
            enqueued_at: Instant::now(),
            result_tx: tx,
            feature_pool: None,
        }
    }

    #[test]
    fn test_encode_eval_req_payload_includes_fingerprint_header() {
        let payload = encode_eval_req_payload(&[0.25, -0.5], 9, 7, 11, 22, 3);

        assert_eq!(payload.len(), 32 + 8);
        assert_eq!(u32::from_le_bytes(payload[0..4].try_into().unwrap()), 7);
        assert_eq!(u32::from_le_bytes(payload[4..8].try_into().unwrap()), 9);
        assert_eq!(u32::from_le_bytes(payload[8..12].try_into().unwrap()), 2);
        assert_eq!(u64::from_le_bytes(payload[12..20].try_into().unwrap()), 11);
        assert_eq!(u64::from_le_bytes(payload[20..28].try_into().unwrap()), 22);
        assert_eq!(u32::from_le_bytes(payload[28..32].try_into().unwrap()), 3);
    }

    #[test]
    fn test_parse_single_response() {
        let (tx, rx) = channel::bounded(1);
        // Moves at indices 0, 4, 8 (corners + center of 3x3)
        let req = make_request(&[(0, 0), (4, 4), (8, 8)], tx);
        let batch = vec![req];

        let resp = r#"{"batch_eval_resp":{"responses":[{"policy":[0.1,0.0,0.0,0.0,0.5,0.0,0.0,0.0,0.4],"value":0.42}]}}"#;
        parse_and_distribute(resp, &batch);

        let result = rx.recv().unwrap();
        assert_eq!(result.policy.len(), 3);
        assert!((result.value - 0.42).abs() < 1e-4);
        // Policy should be normalized: 0.1+0.5+0.4 = 1.0
        let sum: f32 = result.policy.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-4);
        // Check individual priors
        assert!((result.policy[0].1 - 0.1).abs() < 1e-4); // move 0 → idx 0 → 0.1
        assert!((result.policy[1].1 - 0.5).abs() < 1e-4); // move 4 → idx 4 → 0.5
        assert!((result.policy[2].1 - 0.4).abs() < 1e-4); // move 8 → idx 8 → 0.4
    }

    #[test]
    fn test_parse_multi_response() {
        let (tx1, rx1) = channel::bounded(1);
        let (tx2, rx2) = channel::bounded(1);
        let req1 = make_request(&[(0, 0), (1, 1)], tx1);
        let req2 = make_request(&[(3, 3), (5, 5)], tx2);
        let batch = vec![req1, req2];

        let resp = r#"{"batch_eval_resp":{"responses":[{"policy":[0.6,0.4,0,0,0,0,0,0,0],"value":0.1},{"policy":[0,0,0,0.3,0,0.7,0,0,0],"value":-0.5}]}}"#;
        parse_and_distribute(resp, &batch);

        let r1 = rx1.recv().unwrap();
        assert_eq!(r1.policy.len(), 2);
        assert!((r1.value - 0.1).abs() < 1e-4);
        assert!((r1.policy[0].1 - 0.6).abs() < 1e-4);
        assert!((r1.policy[1].1 - 0.4).abs() < 1e-4);

        let r2 = rx2.recv().unwrap();
        assert_eq!(r2.policy.len(), 2);
        assert!((r2.value - (-0.5)).abs() < 1e-4);
        // 0.3 + 0.7 = 1.0
        assert!((r2.policy[0].1 - 0.3).abs() < 1e-4);
        assert!((r2.policy[1].1 - 0.7).abs() < 1e-4);
    }

    #[test]
    fn test_parse_malformed_response_fallback() {
        let (tx, rx) = channel::bounded(1);
        let req = make_request(&[(0, 0), (1, 1), (2, 2)], tx);
        let batch = vec![req];

        // Malformed JSON: missing "responses" key
        let resp = r#"{"garbage": true}"#;
        parse_and_distribute(resp, &batch);

        let result = rx.recv().unwrap();
        // Should get uniform fallback: 3 moves, each with 1/3 probability
        assert_eq!(result.policy.len(), 3);
        assert!((result.value - 0.0).abs() < 1e-6);
        for (_, p) in &result.policy {
            assert!((p - 1.0 / 3.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_parse_empty_policy_fallback() {
        let (tx, rx) = channel::bounded(1);
        let req = make_request(&[(0, 0)], tx);
        let batch = vec![req];

        // Valid structure but policy missing for this response
        let resp = r#"{"batch_eval_resp":{"responses":[{"value":0.5}]}}"#;
        parse_and_distribute(resp, &batch);

        let result = rx.recv().unwrap();
        // Fallback: uniform with 1 move
        assert_eq!(result.policy.len(), 1);
        assert!((result.policy[0].1 - 1.0).abs() < 1e-4);
    }

    // ── Channel mechanics tests ──

    #[test]
    fn test_channel_concurrent_requests() {
        // Test that multiple threads can submit requests concurrently
        // and all receive results.
        use std::thread;
        let (request_tx, request_rx) = channel::bounded::<BatchRequest<TestMove>>(16);

        // Spawn a mock collector that returns uniform results
        let collector = thread::spawn(move || {
            let mut received = 0;
            while let Ok(req) = request_rx.recv_timeout(Duration::from_secs(2)) {
                let n = req.legal_moves_idx.len();
                let p = if n > 0 { 1.0 / n as f32 } else { 0.0 };
                let policy = req.legal_moves_idx.iter().map(|&(m, _)| (m, p)).collect();
                let _ = req.result_tx.send(EvalResult { policy, value: 0.0 });
                received += 1;
                if received >= 8 {
                    break;
                }
            }
            received
        });

        // Spawn 8 worker threads, each submitting one request
        let mut handles = Vec::new();
        for i in 0..8 {
            let tx = request_tx.clone();
            handles.push(thread::spawn(move || {
                let (result_tx, result_rx) = channel::bounded(1);
                let req = BatchRequest {
                    features: vec![0.0; 9],
                    legal_moves_idx: vec![(i, 1)],
                    n_actions: 9,
                    model_tag: 0,
                    feature_fp_lo: i as u64,
                    feature_fp_hi: i as u64 + 100,
                    encoder_rev: 1,
                    enqueued_at: Instant::now(),
                    result_tx,
                    feature_pool: None,
                };
                tx.send(req).unwrap();
                let result = result_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                assert_eq!(result.policy.len(), 1);
                assert_eq!(result.policy[0].0, i);
            }));
        }

        drop(request_tx); // Drop sender so collector can exit
        for h in handles {
            h.join().unwrap();
        }
        let received = collector.join().unwrap();
        assert_eq!(received, 8);
    }

    #[test]
    fn test_send_uniform_fallback() {
        let (tx, rx) = channel::bounded(1);
        let req = make_request(&[(0, 0), (1, 1), (2, 2), (3, 3)], tx);

        send_single_uniform(&req);

        let result = rx.recv().unwrap();
        assert_eq!(result.policy.len(), 4);
        for (_, p) in &result.policy {
            assert!((p - 0.25).abs() < 1e-6);
        }
        assert!((result.value - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_config_default() {
        let cfg = BatchConfig::default();
        assert_eq!(cfg.max_batch_size, 8);
        assert_eq!(cfg.timeout_us, 1500);
    }

    #[test]
    fn test_parse_negative_value() {
        let (tx, rx) = channel::bounded(1);
        let req = make_request(&[(4, 4)], tx);
        let batch = vec![req];

        let resp =
            r#"{"batch_eval_resp":{"responses":[{"policy":[0,0,0,0,1.0,0,0,0,0],"value":-0.99}]}}"#;
        parse_and_distribute(resp, &batch);

        let result = rx.recv().unwrap();
        assert!((result.value - (-0.99)).abs() < 1e-4);
        assert!((result.policy[0].1 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_parse_zero_sum_policy_normalized() {
        // All zero probs → should still return zeros (no division by zero)
        let (tx, rx) = channel::bounded(1);
        let req = make_request(&[(0, 0), (1, 1)], tx);
        let batch = vec![req];

        let resp =
            r#"{"batch_eval_resp":{"responses":[{"policy":[0,0,0,0,0,0,0,0,0],"value":0.0}]}}"#;
        parse_and_distribute(resp, &batch);

        let result = rx.recv().unwrap();
        assert_eq!(result.policy.len(), 2);
        // sum is 0, so no normalization occurs — values remain 0
        for (_, p) in &result.policy {
            assert!((p - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_build_policy_from_dense_bytes_matches_dense() {
        let legal = vec![(0usize, 0usize), (4usize, 4usize), (8usize, 8usize)];
        let probs = vec![0.1f32, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.4];
        let mut bytes = Vec::new();
        push_f32_slice_le(&mut bytes, &probs);

        let dense = build_policy_from_dense(&legal, &probs);
        let from_bytes = build_policy_from_dense_bytes(&legal, &bytes, probs.len());

        assert_eq!(dense.len(), from_bytes.len());
        for ((mv_a, p_a), (mv_b, p_b)) in dense.iter().zip(from_bytes.iter()) {
            assert_eq!(mv_a, mv_b);
            assert!((p_a - p_b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_distribute_binary_batch_parses_dense_payload_without_intermediate_batch_vec() {
        let (tx1, rx1) = channel::bounded(1);
        let (tx2, rx2) = channel::bounded(1);
        let req1 = make_request(&[(0, 0), (4, 4)], tx1);
        let req2 = make_request(&[(3, 3), (8, 8)], tx2);
        let batch = vec![req1, req2];

        let payload = pack_qipc_batch_eval_resp_for_test(&[
            (&[0.6, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0], 0.25f32),
            (&[0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.7], -0.5f32),
        ]);

        distribute_binary_batch(&payload, &batch);

        let r1 = rx1.recv().unwrap();
        assert_eq!(r1.policy.len(), 2);
        assert!((r1.policy[0].1 - 0.6).abs() < 1e-6);
        assert!((r1.policy[1].1 - 0.4).abs() < 1e-6);
        assert!((r1.value - 0.25).abs() < 1e-6);

        let r2 = rx2.recv().unwrap();
        assert_eq!(r2.policy.len(), 2);
        assert!((r2.policy[0].1 - 0.3).abs() < 1e-6);
        assert!((r2.policy[1].1 - 0.7).abs() < 1e-6);
        assert!((r2.value + 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_distribute_binary_batch_falls_back_uniform_on_truncated_payload() {
        let (tx1, rx1) = channel::bounded(1);
        let (tx2, rx2) = channel::bounded(1);
        let req1 = make_request(&[(0, 0), (4, 4)], tx1);
        let req2 = make_request(&[(3, 3), (8, 8)], tx2);
        let batch = vec![req1, req2];

        let mut payload = pack_qipc_batch_eval_resp_for_test(&[
            (&[0.6, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0], 0.25f32),
            (&[0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.7], -0.5f32),
        ]);
        payload.pop();

        distribute_binary_batch(&payload, &batch);

        let r1 = rx1.recv().unwrap();
        assert_eq!(r1.policy.len(), 2);
        for (_, p) in &r1.policy {
            assert!((p - 0.5).abs() < 1e-6);
        }
        assert_eq!(r1.value, 0.0);

        let r2 = rx2.recv().unwrap();
        assert_eq!(r2.policy.len(), 2);
        for (_, p) in &r2.policy {
            assert!((p - 0.5).abs() < 1e-6);
        }
        assert_eq!(r2.value, 0.0);
    }

    #[test]
    fn test_retune_adaptive_timeout_prefers_lower_latency_for_low_concurrency() {
        let (next, low_concurrency) =
            retune_adaptive_timeout_us(2000.0, 2000.0, 250.0, 8000.0, 1, 8, 0, 1);
        assert!(low_concurrency);
        assert!(next < 2000.0);
        assert!(next >= 250.0);
    }

    fn pack_qipc_batch_eval_resp_for_test(entries: &[(&[f32], f32)]) -> Vec<u8> {
        let mut payload = Vec::new();
        push_u32_le(&mut payload, entries.len());
        for (policy, value) in entries {
            push_u32_le(&mut payload, policy.len());
            push_f32_slice_le(&mut payload, policy);
            payload.extend_from_slice(&value.to_le_bytes());
        }
        payload
    }
}
