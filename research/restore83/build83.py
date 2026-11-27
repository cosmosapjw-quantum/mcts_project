"""Isolated force-prefix diagnostic patch to R81 source."""
from base61 import *
import os,shutil,subprocess,difflib
P=B/'mqft_restore39_20261125/repo';A=B/'mqft_restore81_20261127/phase_src';D=S/'branch_src';parent='fda7af649e105cd14cdc5099f0ea654c83267cb4'
assert subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip()==parent
shutil.copytree(A,D);patches={}
def change(f,pairs):
 p=D/'src'/f;old=p.read_text();new=old
 for a,b in pairs:
  assert new.count(a)==1,(f,a,new.count(a));new=new.replace(a,b)
 p.write_text(new);patches[f]=dict(original_sha=sha(A/'src'/f),modified_sha=sha(p),diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+f,tofile='b/'+f)))
change('mqft_law.rs',[
 ('pub struct Sampler {','thread_local! { static FORCE83: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) }; }\npub fn force83(v:Vec<usize>) { FORCE83.with(|x| *x.borrow_mut()=v); }\npub struct Sampler {'),
 ('let i = picked.unwrap_or_else(|| b.iter().rposition(|&x| x > 0.).unwrap());','let i = FORCE83.with(|x| x.borrow().get(t).copied()).map(|a|actions.iter().position(|v|*v==a).expect("forced legal action")).unwrap_or_else(||picked.unwrap_or_else(|| b.iter().rposition(|&x| x > 0.).unwrap()));')
])
change('mqft_train_persistent65.rs',[
 ('rep:usize,arm:String}', 'rep:usize,arm:String,#[serde(default)] forced83:Vec<usize>}'),
 ('else{execute_persistent(&old,net,None,None,Some(Phase65::new(j.sources,j.arm.clone(),j.width,j.budget as usize)))?};','else{if !j.forced83.is_empty() && (j.forced83.len()!=44 || j.arm!="plus81" || j.forced83.iter().any(|a|!legal.contains(a))){return Err("invalid forced83 prefix".into())} crate::mqft_law::force83(j.forced83.clone());let rr=execute_persistent(&old,net,None,None,Some(Phase65::new(j.sources,j.arm.clone(),j.width,j.budget as usize)));crate::mqft_law::force83(vec![]);rr?};')
])
change('mqft_arena68.rs',[
 ('pair81:Option<[usize;2]>}', 'pair81:Option<[usize;2]>,#[serde(default)] forced83:Vec<usize>}'),
 ('let t0=Instant::now();let ev=', 'if !j.forced83.is_empty() && (j.forced83.len()!=44 || j.arm!="clean" || j.phase81.as_deref()!=Some("plus81") || j.forced83.iter().any(|a| !s.idx_to_move(*a).is_some_and(|m|s.legal_moves().contains(&m)))) {return Err("invalid forced83 prefix".into())}\n crate::mqft_law::force83(j.forced83.clone());\n let t0=Instant::now();let ev='),
 ('let es=tr.events.lock();let rec=ev.records.lock();','crate::mqft_law::force83(vec![]);let es=tr.events.lock();let rec=ev.records.lock();')
])
dump(S/'BRANCH_PATCH83.json',dict(parent=parent,base_source_files={str(f.relative_to(A)):sha(f) for f in sorted(A.rglob('*')) if f.is_file()},patches=patches,scope='Forced diagnostic prefix 44 only; no default production changes; recorded q for actual selected path, not forced sampling probability.'))
(S/'branch83.patch').write_text(''.join(x['diff'] for x in patches.values()))
env=dict(os.environ,CARGO_TARGET_DIR=str(P/'backend/target'),CARGO_PROFILE_RELEASE_LTO='false',CARGO_PROFILE_RELEASE_CODEGEN_UNITS='8',CARGO_BUILD_JOBS='8')
cmd=['cargo','build','--offline','--release','--features','mqft-train','--bin','mqft_train','--manifest-path',str(D/'Cargo.toml')]
with (S/'logs/build83.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
shutil.copy2(Path(env['CARGO_TARGET_DIR'])/'release/mqft_train',S/'mqft_branch83')
test=['cargo','test','--offline','--release','--features','mqft-train','--lib','--manifest-path',str(D/'Cargo.toml'),'--','--test-threads=4']
with (S/'logs/test83.log').open('w') as f:subprocess.run(test,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
dump(S/'BUILD83.json',dict(status='PASS_BUILD_AND_LIB_SUITE',binary_sha=sha(S/'mqft_branch83'),patch_sha=sha(S/'branch83.patch'),build=cmd,test=test,build_log_sha=sha(S/'logs/build83.log'),test_log_sha=sha(S/'logs/test83.log')))
print('BUILD',sha(S/'mqft_branch83'),flush=True)
