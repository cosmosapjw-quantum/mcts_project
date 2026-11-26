//! Opt-in 15x15 native direct MixNet WDL/policy evaluator and auditable sign transport.
//! The prior R67 9x9 BoardNet is not resized. No default production change.
use crate::{game::{GameState,Evaluator,EvalResult},games::Gomoku15,mcts::{MctsEngine,MctsConfig,PreparedIteration,ImmediateReason},mcts::parallel::VlMode,mcts::node::atomic_f64_load,mqft_trace::{self,Trace},mqft_law::{LawSpec,Sampler}};
use serde::{Deserialize,Serialize};
use serde_json::{json,Value};
use parking_lot::Mutex;
use std::{sync::{Arc,atomic::Ordering},io::{self,BufRead,Write},time::Instant};
#[derive(Deserialize,Debug)]
#[serde(deny_unknown_fields)]
struct Request {moves:Vec<usize>,seed:u64,budget:u32,arm:String}
#[derive(Serialize)]
struct EvalRec {id:u64,state_hash:String,side:i8,wdl:[f64;3],scores:Vec<(u16,i32)>,clean:f32,delivered:f32,mask:[u8;3],syndrome:[u8;2],corrected:usize}
struct Eval {seed:u64,arm:String,records:Mutex<Vec<EvalRec>>}
fn mix(mut x:u64)->u64{x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
fn u(seed:u64,id:u64,k:u64)->f64{let x=(seed^0x67f00dca12345678).wrapping_add(0x9e3779b97f4a7c15u64.wrapping_mul(id*5+k+1));(mix(x)>>11) as f64/9007199254740992.}
fn fault(seed:u64,id:u64,arm:&str,v:f32)->(f32,[u8;3],[u8;2],usize){
 let mut m=[0;3];if id>0&&arm!="clean"{for k in 0..3{m[k]=u8::from(u(seed,id,k as u64+1)<0.12);}}
 let sy=[m[0]^m[1],m[1]^m[2]];
 let co=if arm=="recovery"{match sy{[1,0]=>1,[1,1]=>2,[0,1]=>3,_=>0}}else{0};
 let flip=m[0]^u8::from(co==1);(if flip==1{-v}else{v},m,sy,co)
}
impl Evaluator<Gomoku15> for Eval{
 fn evaluate(&self,s:&Gomoku15)->EvalResult<u16>{
  let (scores,wdl)=crate::mqft_rapfi::query(s,true).expect("pinned MixNet direct evaluation failed");
  let scores=scores.expect("missing policy");let policy=crate::mqft_rapfi::probabilities(&scores).into_iter().map(|(a,p)|(a,p as f32)).collect();
  let clean=crate::mqft_rapfi::value(wdl,s.current_player());
  let mut r=self.records.lock();let id=r.len() as u64;let (delivered,mask,syndrome,corrected)=fault(self.seed,id,&self.arm,clean);
  r.push(EvalRec{id,state_hash:format!("{:016x}",s.tt_hash()),side:s.current_player(),wdl,scores,clean,delivered,mask,syndrome,corrected});
  mqft_trace::set_eval(id);
  EvalResult{policy,value:delivered}
 }
}
fn run(j:Request)->Result<Value,String>{
 if ![32,64,128].contains(&j.budget)||!["clean","single","recovery"].contains(&j.arm.as_str()){return Err("bad arena request".into())}
 let mut s=Gomoku15::freestyle();
 for &a in &j.moves{if s.is_terminal(){return Err("terminal prefix".into())}let m=s.idx_to_move(a).filter(|m|s.legal_moves().contains(m)).ok_or("illegal move")?;s.apply_move_in_place_no_undo(m);}
 if s.is_terminal(){return Err("terminal root".into())}
 let t0=Instant::now();let ev=Arc::new(Eval{seed:j.seed,arm:j.arm.clone(),records:Mutex::new(vec![])});
 let mut cfg=MctsConfig::evaluation(1.4);cfg.tt_enabled=false;cfg.root_forced_win=false;cfg.exact_terminal_value=false;cfg.fpu_reduction=0.;cfg.vl_mode=VlMode::Fixed;cfg.seed=Some(j.seed);cfg.pw=None;
 let mut eng=MctsEngine::new(s.clone(),ev.clone(),cfg);
 let law=LawSpec{kind:"iid".into(),beta_p:vec![10./3.],beta_q:vec![10./3.],proposal:"p".into(),logit_h:0.};
 eng.par_ctrl.mqft_record=true;eng.par_ctrl.mqft_sampler=Some(Mutex::new(Sampler::new_field(law,vec![0.;225],j.seed)));
 let tr=Arc::new(Trace::default());eng.mqft_trace=Some(tr.clone());mqft_trace::clear_eval();
 let mut cuts=vec![];
 for phase in 1..=j.budget/4{
  tr.set_phase(phase);eng.par_ctrl.set_n_threads(4);
  let mut prep=vec![];
  for _ in 0..4{let p=eng.prepare_iteration_async();let path=match &p{PreparedIteration::Immediate{path,..}=>path,PreparedIteration::Pending(p)=>&p.path};assert!(!path.is_empty());prep.push((tr.begin(&s,path),p));}
  let evals:Vec<_>=prep.iter().map(|(_,p)|{mqft_trace::clear_eval();match p{PreparedIteration::Pending(p)=>{let v=eng.evaluator.evaluate(&p.leaf_state);(Some(v),mqft_trace::take_eval())},_ =>(None,None)}}).collect();
  for ((sel,p),(result,id)) in prep.into_iter().zip(evals){
   match p{
    PreparedIteration::Immediate{path,value,reason}=>{if reason!=ImmediateReason::TerminalNode{return Err("unexpected cap".into())}let k=eng.apply_iteration_value_async(path,value);tr.finish(sel,value,None,true,k);},
    PreparedIteration::Pending(p)=>{let value=crate::mcts::expand::expand_with_result(&p.leaf,&p.leaf_state,result.unwrap(),&eng.tt,eng.config.pw.as_ref());let k=eng.apply_iteration_value_async(p.path,value);tr.finish(sel,value,id,false,k);}
   }
  }
  let k=phase*4;
  if [4,8,16,32,64,128].contains(&k){
   let mut n=vec![0u32;225];let mut w=vec![0.;225];let mut s2=vec![0.;225];let mut prior=vec![0.;225];
   for e in eng.root.read_edges(){let a=e.mv as usize;n[a]=e.n.load(Ordering::Acquire);w[a]=atomic_f64_load(&e.w);s2[a]=atomic_f64_load(&e.s2);prior[a]=e.p;}
   assert_eq!(n.iter().sum::<u32>(),k);cuts.push(json!({"k":k,"n":n,"w":w,"s2":s2,"prior":prior}));
  }
 }
 let es=tr.events.lock();let rec=ev.records.lock();
 assert_eq!(es.len(),j.budget as usize);assert_eq!(rec.len() as u64,eng.evaluator_call_count());
 let mut events=vec![];
 for e in es.iter(){let mut z=serde_json::to_value(e).unwrap();let d=z["root_selection"]["law"].as_object_mut().unwrap();assert_eq!(d["p"],d["q"]);assert_eq!(d["p"],d["b"]);for key in ["q","b","log_q","log_b","log_q_path","log_b_path"]{d.remove(key);}events.push(z);}
 let n=cuts.last().unwrap()["n"].as_array().unwrap();let action=(0..225).filter(|&a|n[a].as_u64().unwrap()>0).max_by_key(|&a|(n[a].as_u64().unwrap(),std::cmp::Reverse(a))).unwrap();
 Ok(json!({"schema":"mqft.arena68.search.v1","moves":j.moves,"arm":j.arm,"seed":j.seed,"budget":j.budget,"action":action,"side":s.current_player(),"state_hash":format!("{:016x}",s.tt_hash()),"cuts":cuts,"events":events,"evaluations":&*rec,"elapsed_ns":t0.elapsed().as_nanos() as u64}))
}
pub fn serve()->Result<(),String>{
 let mut out=io::BufWriter::new(io::stdout().lock());
 for line in io::stdin().lock().lines(){let l=line.map_err(|e|e.to_string())?;let result=serde_json::from_str(&l).map_err(|e|e.to_string()).and_then(run);
 let value=match result{Ok(v)=>json!({"ok":true,"result":v}),Err(e)=>json!({"ok":false,"error":e})};
 serde_json::to_writer(&mut out,&value).map_err(|e|e.to_string())?;writeln!(out).map_err(|e|e.to_string())?;out.flush().map_err(|e|e.to_string())?;}
 crate::mqft_rapfi::close();Ok(())
}
#[cfg(test)]mod tests{use super::*;#[test]fn fault_mask(){for seed in 0..100{for id in 0..10{let (v,m,sy,c)=fault(seed,id,"recovery",0.4);assert_eq!(sy,[m[0]^m[1],m[1]^m[2]]);assert_eq!(v,if m.iter().sum::<u8>()>=2{-0.4}else{0.4});if id==0{assert_eq!(c,0);}}}}}
