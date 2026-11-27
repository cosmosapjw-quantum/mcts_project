"""Isolated replay-only binary; the production/pulse executable is unchanged."""
from base61 import *
import subprocess,shutil,os,difflib
P=B/'mqft_restore39_20261125/repo';A=B/'mqft_restore74_20261126/pulse_src';D=S/'replay_src';parent='149e0bd757e0244ded64b96149c7faab40cf77ed'
assert subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip()==parent
shutil.copytree(A,D);patches={}
def change(f,pairs):
 p=D/'src'/f;old=p.read_text();new=old
 for a,b in pairs:
  assert new.count(a)==1,(f,a,new.count(a));new=new.replace(a,b)
 p.write_text(new);patches[f]=dict(original_sha=sha(A/'src'/f),modified_sha=sha(p),diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+f,tofile='b/'+f)))
# Replay-only opt-in forced sequence, thread-local per independent job.
change('mqft_law.rs',[
 ('pub struct Sampler {','thread_local! { static FORCE80: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) }; }\npub fn force80(v:Vec<usize>) { FORCE80.with(|x| *x.borrow_mut()=v); }\npub struct Sampler {'),
 ('let i = picked.unwrap_or_else(|| b.iter().rposition(|&x| x > 0.).unwrap());','let i = FORCE80.with(|x| x.borrow().get(t).copied()).map(|a|actions.iter().position(|v|*v==a).expect("forced legal action")).unwrap_or_else(||picked.unwrap_or_else(|| b.iter().rposition(|&x| x > 0.).unwrap()));')
])
change('mqft_train_schedule66.rs',[
 ('rep:usize,arm:String}', 'rep:usize,arm:String,#[serde(default)] forced80:Vec<usize>}'),
 ('let raw=execute_schedule(&old,net,None,None,None,Some(hs))?;', 'if !j.forced80.is_empty() && (j.forced80.len()!=128 || j.arm!="zero74" || j.forced80.iter().any(|a|!legal.contains(a))){return Err("invalid forced replay".into())}\n crate::mqft_law::force80(j.forced80.clone());let rr=execute_schedule(&old,net,None,None,None,Some(hs));crate::mqft_law::force80(vec![]);let raw=rr?;')
])
change('mqft_arena68.rs',[
 ('pulse74:Option<f64>}', 'pulse74:Option<f64>,#[serde(default)] forced80:Vec<usize>}'),
 ('let t0=Instant::now();let ev=', 'if !j.forced80.is_empty() && (j.forced80.len()!=128 || j.arm!="clean" || j.pulse74!=Some(0.) || j.forced80.iter().any(|a| !s.idx_to_move(*a).is_some_and(|m|s.legal_moves().contains(&m)))) {return Err("invalid forced replay".into())}\n crate::mqft_law::force80(j.forced80.clone());\n let t0=Instant::now();let ev='),
 ('let es=tr.events.lock();let rec=ev.records.lock();','crate::mqft_law::force80(vec![]);let es=tr.events.lock();let rec=ev.records.lock();')
])
# Include exact source identity of all reused Rust source, including clean selection and scheduling.
orig={str(f.relative_to(A)):sha(f) for f in sorted(A.rglob('*')) if f.is_file()}
dump(S/'REPLAY_PATCH80.json',dict(parent=parent,base_source_files=orig,patches=patches,scope='Forced replay test only, altered root draws not native probability samples.'))
(S/'replay80.patch').write_text(''.join(x['diff'] for x in patches.values()))
env=dict(os.environ,CARGO_TARGET_DIR=str(P/'backend/target'),CARGO_PROFILE_RELEASE_LTO='false',CARGO_PROFILE_RELEASE_CODEGEN_UNITS='8',CARGO_BUILD_JOBS='8')
cmd=['cargo','build','--offline','--release','--features','mqft-train','--bin','mqft_train','--manifest-path',str(D/'Cargo.toml')]
with (S/'logs/build80.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
shutil.copy2(Path(env['CARGO_TARGET_DIR'])/'release/mqft_train',S/'mqft_replay80')
print('BUILD',sha(S/'mqft_replay80'),flush=True)
