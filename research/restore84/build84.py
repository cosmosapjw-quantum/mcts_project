from base61 import *
import os,shutil,subprocess,difflib
P=B/'mqft_restore39_20261125/repo';A=B/'mqft_restore74_20261126/pulse_src';D=S/'landscape_src';parent='68c0249a9a5991731f8bec4ee70c98d16268f00a'
assert subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip()==parent
shutil.copytree(A,D);p=D/'src/mqft_train.rs';old=p.read_text()
new=r'''
fn landscape84(weights:&str,fixture:&str,workers:&str)->Result<(),String>{
 let ms:Vec<Vec<usize>>=serde_json::from_str(&fs::read_to_string(fixture).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 let workers:usize=workers.parse().map_err(|_|"worker parse")?;
 if workers==0||workers>12||ms.is_empty(){return Err("bad landscape workers".into())}
 let n=if weights=="rapfi"{None}else{Some(Net::load(weights)?)};
 let pool=rayon::ThreadPoolBuilder::new().num_threads(workers).build().map_err(|e|e.to_string())?;
 let out:Result<Vec<Value>,String>=pool.install(||ms.par_iter().map(|m|->Result<Value,String>{
 let t=Instant::now();
 if let Some(net)=&n{
  let s=state(m)?;if s.is_terminal(){return Err("terminal root".into())}
  let(p,v)=net.forward(&s);let mut children=vec![];let mut calls=1;
  for a in s.legal_moves(){let mut z=s.clone();z.apply_move_in_place_no_undo(a);let term=z.is_terminal();
   let child=if term{z.outcome()}else{calls+=1;net.forward(&z).1};
   children.push(json!({"a":a,"side":z.current_player(),"hash":format!("{:016x}",z.tt_hash()),"terminal":term,"child_v":child,"root_v":-child}));
  }
  Ok(json!({"moves":m,"p":p,"v":v,"side":s.current_player(),"hash":format!("{:016x}",s.tt_hash()),"children":children,"calls":calls,"seconds":t.elapsed().as_secs_f64()}))
 }else{
  use crate::games::Gomoku15;
  let mut s=Gomoku15::freestyle();for &a in m{if s.is_terminal(){return Err("terminal prefix".into())}let mv=s.idx_to_move(a).filter(|mv|s.legal_moves().contains(mv)).ok_or("illegal move")?;s.apply_move_in_place_no_undo(mv);}
  if s.is_terminal(){return Err("terminal root".into())}
  let(sc,wdl)=crate::mqft_rapfi::query(&s,true)?;let sc=sc.ok_or("no policy")?;let mut p=vec![0f32;225];for(a,w)in crate::mqft_rapfi::probabilities(&sc){p[a as usize]=w as f32;}
  let v=crate::mqft_rapfi::value(wdl,s.current_player());let mut children=vec![];let mut calls=1;
  for a in s.legal_moves(){let mut z=s.clone();z.apply_move_in_place_no_undo(a);let term=z.is_terminal();let mut w=None;
   let child=if term{z.outcome()}else{calls+=1;let(_,wd)=crate::mqft_rapfi::query(&z,true)?;w=Some(wd);crate::mqft_rapfi::value(wd,z.current_player())};
   children.push(json!({"a":a,"side":z.current_player(),"hash":format!("{:016x}",z.tt_hash()),"terminal":term,"child_v":child,"root_v":-child,"wdl":w}));
  }
  Ok(json!({"moves":m,"p":p,"v":v,"side":s.current_player(),"hash":format!("{:016x}",s.tt_hash()),"wdl":wdl,"scores":sc,"children":children,"calls":calls,"seconds":t.elapsed().as_secs_f64()}))
 }
 }).collect());
 crate::mqft_rapfi::close();println!("{}",serde_json::to_string(&out?).unwrap());Ok(())
}
'''
s=old.replace('pub fn main_cli()->Result<(),String>{',new+'\npub fn main_cli()->Result<(),String>{').replace('Some("arena68") if a.len()==2=>','Some("landscape84") if a.len()==5=>landscape84(&a[2],&a[3],&a[4]),\n      Some("arena68") if a.len()==2=>')
assert s!=old;p.write_text(s)
patch=''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/mqft_train.rs',tofile='b/mqft_train.rs'))
(S/'landscape84.patch').write_text(patch)
dump(S/'LANDSCAPE_PATCH84.json',dict(parent=parent,base_source_files={str(f.relative_to(A)):sha(f) for f in sorted(A.rglob('*')) if f.is_file()},modified_sha=sha(p),original_sha=sha(A/'src/mqft_train.rs'),patch_sha=sha(S/'landscape84.patch'),scope='Opt-in deterministic direct eval only, no change to stochastic backend.'))
env=dict(os.environ,CARGO_TARGET_DIR=str(P/'backend/target'),CARGO_PROFILE_RELEASE_LTO='false',CARGO_PROFILE_RELEASE_CODEGEN_UNITS='8',CARGO_BUILD_JOBS='8')
cmd=['cargo','build','--offline','--release','--features','mqft-train','--bin','mqft_train','--manifest-path',str(D/'Cargo.toml')]
with (S/'logs/build84.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
shutil.copy2(Path(env['CARGO_TARGET_DIR'])/'release/mqft_train',S/'mqft_landscape84')
dump(S/'BUILD84.json',dict(status='PASS_BUILD',binary_sha=sha(S/'mqft_landscape84'),patch_sha=sha(S/'landscape84.patch'),build=cmd,log_sha=sha(S/'logs/build84.log')))
print('BUILD',sha(S/'mqft_landscape84'),flush=True)
