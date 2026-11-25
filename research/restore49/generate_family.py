"""Compile-time specialized family from exact RESTORE48 sources.
No modification to the old 15x15 backend or Python package. All generated
versions share unchanged Quartz MCTS, draw and backup algorithms.
The archived parent is the authority, not a dirty working copy.
"""
from pathlib import Path
import sys,subprocess,tarfile,io,hashlib,json,re
P=Path(__file__).resolve().parents[2]
BASE="b448c0257934edf832a8b4b2f1978a27b2c68180"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def patch(p,pairs):
 s=p.read_text()
 for a,b,n in pairs:
  if s.count(a)!=n:raise ValueError((str(p),a,s.count(a),n))
  s=s.replace(a,b)
 p.write_text(s)
def generate(out,L):
 if L not in (13,15,17,19,21):raise ValueError("registered size family only")
 out.mkdir(parents=True,exist_ok=False)
 blob=subprocess.check_output(["git","-C",str(P),"archive",BASE,"backend","python/mqft"])
 with tarfile.open(fileobj=io.BytesIO(blob)) as t:t.extractall(out,filter="data")
 b=out/"backend/src";py=out/"python/mqft";n=L*L
 # Unique Cargo package identity prevents cache reuse across specialized sizes.
 patch(out/"backend/Cargo.toml",[('name = "mcts_demo"',f'name = "mcts_demo_l{L}"',1)])
 patch(out/"backend/Cargo.lock",[('name = "mcts_demo"',f'name = "mcts_demo_l{L}"',1)])
 patch(b/"main.rs",[('mcts_demo::mqft',f'mcts_demo_l{L}::mqft',1)])

 # Change ONLY the production state constants and bitmask capacity.
 # Existing state implementation, including undo, feature planes, and
 # five-in-a-row, is preserved; legacy tests are kept out of this build.
 f=b/"games/gomoku15.rs"
 patch(f,[("const N: usize = 15;",f"const N: usize = {L};",1),
   ("[u64; 4]",f"[u64; {(n+63)//64}]",2),("[0u64; 4]",f"[0u64; {(n+63)//64}]",1),
   ("0xA0B0_C015_DEAD_BEEF",f"0xA0B0_C0{L:02d}_DEAD_BEEF",1)])
 # L is explicit even for the empty state, whose bare hash is zero.
 f=b/"mqft.rs"
 s=f.read_text();k=s.index("#[cfg(test)]\nmod tests");s=s[:k]
 s=s.replace('"mqft.quartz.trace.v2"',f'"mqft.quartz.trace.v3.L{L}"')
 # Restrict to explicit actual NNUE, do not enable uniform rollout fallback.
 s=s.replace('if !["uniform_rollout", crate::mqft_rapfi::ID].contains(&self.evaluator.as_str()) {','if self.evaluator != crate::mqft_rapfi::ID {')
 s=re.sub(r'\b225\b',str(n),s)
 for a,c in [('a / 15',f'a / {L}'),('a % 15',f'a % {L}'),('0..15',f'0..{L}'),('y * 15 + x',f'y * {L} + x'),('"board_size":15',f'"board_size":{L}'),('json!({"board":',f'json!({{"board_size":{L},"board":')]:
  assert a in s,a;s=s.replace(a,c)
 s=s.replace('law.logit_h.abs()', 'law.logit_schedule.iter().map(|x|x.abs()).fold(law.logit_h.abs(),f64::max)')
 f.write_text(s)
 f=b/"mqft_rapfi.rs";s=f.read_text()
 s=s.replace("START 15\\n",f"START {L}\\n")
 for a,c in [('a % 15',f'a % {L}'),('a / 15',f'a / {L}'),('0..15',f'0..{L}'),('cells.len() != 17',f'cells.len() != {L+2}'),('cells[16]',f'cells[{L+1}]'),('y * 15 + x',f'y * {L} + x')]:
  assert a in s,a;s=s.replace(a,c)
 # Explicit reconstructed board/ply/side verification even on WDL-only child calls.
 s=s.replace('if policy {\n            cmd += "TRACEBOARD\\n";\n        }','cmd += "TRACEBOARD\\n";')
 s=s.replace('let scores = if policy {','let scores = {').replace('            Some(out)\n        } else {\n            None\n        };','            if policy { Some(out) } else { None }\n        };')
 assert 'cmd += "TRACEBOARD\\n";' in s and 'if policy { Some(out) }' in s
 f.write_text(s)
 f=b/"mqft_law.rs";s=f.read_text()
 # Optional time-dependent logit field, one entry per root selection.
 s=s.replace('    pub logit_h: f64,\n}', '    pub logit_h: f64,\n    #[serde(default, skip_serializing_if = "Vec::is_empty")]\n    pub logit_schedule: Vec<f64>,\n}',1)
 s=s.replace('        if !self.logit_h.is_finite()', '''        if !self.logit_schedule.is_empty() &&
            (self.logit_schedule.len()!=budget as usize || self.logit_h!=0. ||
             self.logit_schedule.iter().any(|h|!h.is_finite() || h.abs()>20.) ||
             source.is_none()) { return Err("logit schedule requires budget entries, legal source, scalar h=0".into()); }
        if !self.logit_h.is_finite()''')
 s=s.replace('        let t = self.t;','        let t = self.t;\n        let h = if self.spec.logit_schedule.is_empty() { self.spec.logit_h } else { self.spec.logit_schedule[t] };')
 s=s.replace('            self.spec.logit_h,','            h,').replace('logit_h: self.spec.logit_h,','logit_h: h,')
 f.write_text(s)
 # Python transport/analysis: compile-time size-scoped schema in a separate package tree.
 patch(py/"__init__.py",[('"mqft.quartz.trace.v2"',f'"mqft.quartz.trace.v3.L{L}"',1)])
 patch(py/"client.py",[('self.identity["board_size"]!=15',f'self.identity["board_size"]!={L}',1)])
 f=py/"campaign.py";s=f.read_text().replace('"mqft.campaign.v2"',f'"mqft.campaign.v3.L{L}"')
 s=re.sub(r'\b225\b',str(n),s)
 # tokenize exact geometry occurrences, not unrelated numeric constants.
 for a,c in [('divmod(a,15)',f'divmod(a,{L})'),('yy<15',f'yy<{L}'),('xx<15',f'xx<{L}'),('yy*15+xx',f'yy*{L}+xx')]:
  assert a in s,a;s=s.replace(a,c)
 s=s.replace('if state["board"]!=b',f'if state["board_size"]!={L} or state["board"]!=b')
 s=s.replace('"proposal","logit_h"}','"proposal","logit_h","logit_schedule"}')
 s=s.replace('            law={**law,"logit_h":hh}', '''            law={**law,"logit_h":hh}
            sched=law.get("logit_schedule")
            if sched is not None:
                if not isinstance(sched,list) or len(sched)!=budget or hh!=0 or pos.get("source_action") is None:raise ValueError("invalid logit schedule")
                sched=[finite(v,"logit schedule",-20,20) for v in sched]
                law["logit_schedule"]=sched
            else:sched=[]''')
 s=s.replace('+abs(hh)>640','+max([abs(hh)]+[abs(v) for v in sched])>640')
 # lock size in seed identity via position IDs and plan schema. No external alpha-beta opponents.
 s=s.replace('    for pos in positions:\n        b=valid_moves', '    if evaluator!="rapfi_nnue_lookahead_v1" or any(o.startswith("rapfi_") for o in opponents):raise ValueError("size family requires same actual NNUE on all sides")\n    for pos in positions:\n        b=valid_moves')
 f.write_text(s)
 f=py/"analysis.py";s=f.read_text();s=re.sub(r'\b225\b',str(n),s);s=s.replace('"mqft.analysis.v2"',f'"mqft.analysis.v3.L{L}"');f.write_text(s)
 patch(py/"path_law.py",[('h=law.get("logit_h",0.);source=spec.get("source_action")','h=law["logit_schedule"][t] if law.get("logit_schedule") else law.get("logit_h",0.);source=spec.get("source_action")',1)])
 patch(py/"rapfi.py",[('rules="freestyle15"',f'rules="freestyle{L}"',1)])
 # Reject old fixed-size external move API; root/game plan explicitly prohibits it.
 # Child/evaluator identity audited independently on every call.
 files={str(x.relative_to(out)):sha(x) for x in sorted(out.rglob("*")) if x.is_file()}
 (out/"SOURCE_MANIFEST.json").write_text(json.dumps(dict(parent=BASE,L=L,files=files),indent=2)+"\n")
if __name__=="__main__":generate(Path(sys.argv[1]),int(sys.argv[2]))
