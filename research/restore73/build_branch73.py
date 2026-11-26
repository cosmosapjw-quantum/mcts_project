"""Build isolated experimental forced-prefix backend; canonical Git source unchanged."""
from base61 import *
import subprocess,shutil,os,difflib
P=B/'mqft_restore39_20261125/repo';parent='31277d04b65d37931ad35686c5ae660fb573563c'
assert subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip()==parent
D=S/'branch_src';D.mkdir(exist_ok=False);shutil.copytree(P/'backend/src',D/'src');shutil.copy2(P/'backend/Cargo.toml',D);shutil.copy2(P/'backend/Cargo.lock',D)
patches={}
def change(f,pairs):
 p=D/'src'/f;old=p.read_text();new=old
 for a,b in pairs:
  assert new.count(a)==1,(f,a,new.count(a));new=new.replace(a,b)
 p.write_text(new);patches[f]=dict(original_sha=sha(P/'backend/src'/f),modified_sha=sha(p),diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/backend/src/'+f,tofile='b/backend/src/'+f)))
change('mqft_law.rs',[
 ('    state:[f64;3], exposure:f64,previous_gap:f64,','    state:[f64;3], exposure:f64,previous_gap:f64,') ] if False else [
 ('    state:[f64;3], exposure:f64,','    state:[f64;3], exposure:f64,') ] if False else [
 ('previous_gap:f64,\n}', 'previous_gap:f64,\n    forced73:Vec<usize>,\n    stop73:Option<usize>,\n}'),
 ('state:[0.,y,0.],exposure:0.,previous_gap:0.}', 'state:[0.,y,0.],exposure:0.,previous_gap:0.,forced73:vec![],stop73:None}'),
 ('    fn rotate_x(v:[f64;3],angle:f64)', '    pub fn branch73(mut self, prefix:Vec<usize>, stop:usize)->Self { assert!(prefix.len()<=16 && stop==32 && self.budget==128);self.forced73=prefix;self.stop73=Some(stop);self }\n    pub fn stop73(&self)->Option<usize>{self.stop73}\n    fn rotate_x(v:[f64;3],angle:f64)'),
 ('        let i = picked.unwrap_or_else(|| b.iter().rposition(|&x| x > 0.).unwrap());','        let i = if let Some(a)=self.phase65.as_ref().and_then(|p|p.forced73.get(t)) { actions.iter().position(|v|v==a).expect("forced legal root action") } else {picked.unwrap_or_else(|| b.iter().rposition(|&x| x > 0.).unwrap())};')
 ])
change('mqft_train_controlled61.rs',[
 ('    let s=state(&j.moves)?;if s.is_terminal()||!s.legal_moves().contains(&j.source)','    let run_budget=persistent.as_ref().and_then(|ph|ph.stop73()).unwrap_or(j.budget as usize);\n    let s=state(&j.moves)?;if s.is_terminal()||!s.legal_moves().contains(&j.source)'),
 ('    for phase in 1..=j.budget as usize/j.width{','    for phase in 1..=run_budget/j.width{'),
 ('    assert_eq!(es.len(),j.budget as usize);assert_eq!(rec.len() as u64,eng.evaluator_call_count());','    assert_eq!(es.len(),run_budget);assert_eq!(rec.len() as u64,eng.evaluator_call_count());')
 ])
change('mqft_train_persistent65.rs',[
 ('rep:usize,arm:String}', 'rep:usize,arm:String,#[serde(default,skip_serializing_if="Vec::is_empty")] forced73:Vec<usize>}'),
 (' let old=ControlJob', ' if !j.forced73.is_empty() && (j.forced73.len()!=16 || j.budget!=128 || !["persist_plus","persist_minus"].contains(&j.arm.as_str()) || j.forced73.iter().any(|a|!legal.contains(a))){return Err("invalid R73 prefix".into())}\n let old=ControlJob'),
 (' else{execute_persistent(&old,net,None,None,Some(Phase65::new(j.sources,j.arm.clone(),j.width,j.budget as usize)))?};',' else{let mut ph=Phase65::new(j.sources,j.arm.clone(),j.width,j.budget as usize);if !j.forced73.is_empty(){ph=ph.branch73(j.forced73.clone(),32);}execute_persistent(&old,net,None,None,Some(ph))?};')
 ])
dump(S/'BRANCH_PATCH73.json',dict(parent=parent,patches=patches,scope='forced diagnostic prefix 16, stop after 32 completed backups but preserve 128 time normalization; original RNG still advanced; no native trajectory sample claim'))
(S/'branch73.patch').write_text(''.join(x['diff'] for x in patches.values()))
env=dict(os.environ,CARGO_TARGET_DIR=str(B/'mqft_restore39_20261125/repo/backend/target'),CARGO_PROFILE_RELEASE_LTO='false',CARGO_PROFILE_RELEASE_CODEGEN_UNITS='8',CARGO_BUILD_JOBS='8')
cmd=['cargo','build','--offline','--release','--features','mqft-train','--bin','mqft_train','--manifest-path',str(D/'Cargo.toml')]
with (S/'logs/build73.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
shutil.copy2(Path(env['CARGO_TARGET_DIR'])/'release/mqft_train',S/'mqft_branch73')
print('BRANCH BUILD',sha(S/'mqft_branch73'),flush=True)
