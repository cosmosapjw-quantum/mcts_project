//! RESTORE60 opt-in canonical MctsEngine + exact R23 BoardNet tensor bridge.
//! BoardNet 2x9x9, 3 padded conv-ReLU, policy 1x1, value 1296->24->1 tanh.
use crate::game::{EvalResult, Evaluator, GameState};
use crate::games::gomoku::Gomoku;
use crate::mcts::{MctsConfig, MctsEngine};
use crate::mcts::node::atomic_f64_load;
use crate::mcts::parallel::VlMode;
use crate::mcts::root::DirichletConfig;
use crate::mqft_trace::{self,Trace};
use parking_lot::Mutex;
use rand::{rngs::StdRng,Rng,SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize,Serialize};
use serde_json::{json,Value};
use std::{fs,io::{self,Write},path::Path,sync::Arc,sync::atomic::Ordering,time::Instant};
const L:usize=9;
const N:usize=L*L;
const NW:usize=36114;
const SHAPES:[usize;12]=[16*2*9,16,16*16*9,16,16*16*9,16,16,1,24*16*N,24,24,1];

#[link(name="openblas")]
extern "C" {
    fn cblas_sgemm(layout:i32,ta:i32,tb:i32,m:i32,n:i32,k:i32,alpha:f32,a:*const f32,lda:i32,b:*const f32,ldb:i32,beta:f32,c:*mut f32,ldc:i32);
    fn openblas_set_num_threads(n:i32);
}
struct Net { w: Vec<Vec<f32>> }
fn gemm(m:usize,n:usize,k:usize,a:&[f32],b:&[f32],c:&mut[f32]) {
    assert_eq!(a.len(),m*k);assert_eq!(b.len(),k*n);assert_eq!(c.len(),m*n);
    // SAFETY: contiguous row-major disjoint input/output slices, checked dimensions,
    // no transposition; CBLAS RowMajor=101, NoTrans=111.
    unsafe {cblas_sgemm(101,111,111,m as i32,n as i32,k as i32,1.,a.as_ptr(),k as i32,b.as_ptr(),n as i32,0.,c.as_mut_ptr(),n as i32)}
}
fn conv(x:&[f32],ci:usize,w:&[f32],b:&[f32])->Vec<f32> {
    assert_eq!(x.len(),ci*N);let k=ci*9;
    let mut col=vec![0.;k*N];
    for ch in 0..ci {for ky in 0..3 {for kx in 0..3 {
        let j=(ch*9+ky*3+kx)*N;
        for y in 0..L {for x0 in 0..L {
            let yy=y as isize+ky as isize-1;let xx=x0 as isize+kx as isize-1;
            if (0..L as isize).contains(&yy)&&(0..L as isize).contains(&xx){
                col[j+y*L+x0]=x[ch*N+yy as usize*L+xx as usize];
            }
        }}
    }}}
    let mut h=vec![0.;16*N];gemm(16,N,k,w,&col,&mut h);
    for i in 0..16 {for z in &mut h[i*N..(i+1)*N]{*z=(*z+b[i]).max(0.);}}
    h
}
impl Net {
    fn load(path:&str)->Result<Self,String>{
        let b=fs::read(path).map_err(|e|e.to_string())?;
        if b.len()!=NW*4{return Err(format!("invalid weight length {}",b.len()))}
        let vals:Vec<f32>=b.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect();
        if vals.iter().any(|x|!x.is_finite()){return Err("nonfinite weights".into())}
        let mut w=vec![];let mut i=0;for n in SHAPES{w.push(vals[i..i+n].to_vec());i+=n;}
        assert_eq!(i,NW);Ok(Self{w})
    }
    fn forward(&self,s:&Gomoku)->(Vec<f32>,f32){
        assert_eq!(s.size,L);assert!(!s.is_terminal());
        let b=s.board_state_record();let me=if s.current_player()==1{1}else{2};
        let mut x=vec![0.;2*N];for i in 0..N{if b[i]==me{x[i]=1.}else if b[i]!=0{x[N+i]=1.}}
        let h=conv(&x,2,&self.w[0],&self.w[1]);let h=conv(&h,16,&self.w[2],&self.w[3]);let h=conv(&h,16,&self.w[4],&self.w[5]);
        let mut p=vec![0.;N];gemm(1,N,16,&self.w[6],&h,&mut p);
        let legal=s.legal_moves();let max=legal.iter().map(|&i|p[i]).fold(f32::NEG_INFINITY,f32::max);
        let mut z=0.;for i in 0..N{if b[i]==0{p[i]=(p[i]-max).exp();z+=p[i]}else{p[i]=0.}}
        for v in &mut p{*v/=z;}
        let mut v=vec![0.;24];gemm(24,1,16*N,&self.w[8],&h,&mut v);
        for (x,b) in v.iter_mut().zip(&self.w[9]){*x=(*x+*b).max(0.);}
        let mut y=[0.];gemm(1,1,24,&self.w[10],&v,&mut y);
        let val=(y[0]+self.w[11][0]).tanh();assert!(val.is_finite()&&p.iter().all(|x|x.is_finite()));
        (p,val)
    }
}
#[derive(Serialize)]
struct EvalRec {id:u64,state_hash:String,side:i8,value:f32}
struct Eval {net:Arc<Net>,records:Mutex<Vec<EvalRec>>}
impl Evaluator<Gomoku> for Eval{
    fn evaluate(&self,s:&Gomoku)->EvalResult<usize>{
        let (p,value)=self.net.forward(s);
        let mut r=self.records.lock();let id=r.len() as u64;
        r.push(EvalRec{id,state_hash:format!("{:016x}",s.tt_hash()),side:s.current_player(),value});
        mqft_trace::set_eval(id);
        EvalResult{policy:s.legal_moves().into_iter().map(|a|(a,p[a])).collect(),value}
    }
}
fn state(moves:&[usize])->Result<Gomoku,String>{
    let mut s=Gomoku::new(L);
    for &a in moves{if s.is_terminal()||!s.legal_moves().contains(&a){return Err("invalid opening".into())}s.apply_move_in_place_no_undo(a);}
    Ok(s)
}
#[derive(Clone,Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
struct Job {id:String,opening:Vec<usize>,seed:u64,budget:u32,training:bool,swap:bool}
fn game(j:&Job,nets:&[Arc<Net>;2])->Result<Value,String>{
    if j.budget<1||j.budget>1024{return Err("invalid budget".into())}
    let t0=Instant::now();let mut s=state(&j.opening)?;if s.is_terminal(){return Err("terminal opening".into())}
    let mut moves=j.opening.clone();let mut rng=StdRng::seed_from_u64(j.seed);let mut rows=vec![];
    while !s.is_terminal(){
        let side=s.current_player();let mi=usize::from(side==-1)^usize::from(j.swap);
        let ev=Arc::new(Eval{net:nets[mi].clone(),records:Mutex::new(vec![])});
        let seed=rng.gen::<u64>();let mut cfg=MctsConfig::evaluation(1.4);
        cfg.tt_enabled=false;cfg.root_forced_win=false;cfg.exact_terminal_value=false;cfg.vl_mode=VlMode::Disabled;cfg.seed=Some(seed);cfg.pw=None;
        if j.training{cfg.dirichlet=Some(DirichletConfig{alpha:0.3,epsilon:0.25});}
        let mut eng=MctsEngine::new(s.clone(),ev.clone(),cfg);
        mqft_trace::clear_eval();let tr=Arc::new(Trace::default());eng.mqft_trace=Some(tr.clone());
        eng.par_ctrl.mqft_record=true;
        for k in 1..=j.budget{tr.set_phase(k);eng.iterate();}
        let mut counts=vec![0u32;N];let mut w=vec![0.;N];let mut s2=vec![0.;N];let mut prior=vec![0.;N];
        for e in eng.root.read_edges(){let a=e.mv;counts[a]=e.n.load(Ordering::Acquire);w[a]=atomic_f64_load(&e.w);s2[a]=atomic_f64_load(&e.s2);prior[a]=e.p;}
        assert_eq!(counts.iter().sum::<u32>(),j.budget);
        assert_eq!(eng.root.n_total.load(Ordering::Acquire),j.budget);
        let events=tr.events.lock();assert_eq!(events.len(),j.budget as usize);
        let mut rcount=vec![0u32;N];let mut rw=vec![0.;N];let mut r2=vec![0.;N];
        for e in events.iter(){let a=e.selection.actions[0];let r=e.root_value as f64;rcount[a]+=1;rw[a]+=r;r2[a]+=r*r;assert_eq!(e.root_value,if e.selection.actions.len()%2==0{e.leaf_value}else{-e.leaf_value});}
        assert_eq!(counts,rcount);for a in 0..N{assert!((w[a]-rw[a]).abs()<1e-9&&(s2[a]-r2[a]).abs()<1e-9);}
        let rec=ev.records.lock();assert_eq!(rec.len() as u64,eng.evaluator_call_count());
        let stochastic=j.training&&rows.len()<12;
        let a=if stochastic{
            let mut q=rng.gen_range(0..j.budget);let mut selected=None;
            for (a,&n) in counts.iter().enumerate(){if q<n{selected=Some(a);break}q-=n;}
            selected.unwrap()
        }else{
            (0..N).filter(|&a|counts[a]>0).max_by_key(|&a|(counts[a],std::cmp::Reverse(a))).unwrap()
        };
        assert!(s.legal_moves().contains(&a));let ply=moves.len();
        rows.push(json!({"ply":ply,"side":side,"model":mi,"search_seed":seed,"state_hash":format!("{:016x}",s.tt_hash()),"board":s.board_state_record(),"counts":counts,"w":w,"s2":s2,"prior":prior,"events":&*events,"evaluations":&*rec,"action":a,"sampled":stochastic}));
        drop(rec);drop(events);drop(eng);
        s.apply_move_in_place_no_undo(a);moves.push(a);
    }
    let winner=if s.outcome()==0.{0}else{if s.outcome()>0.{s.current_player()}else{-s.current_player()}};
    Ok(json!({"schema":"mqft.train9.v1","job":j,"moves":moves,"winner":winner,"terminal":true,"rows":rows,"elapsed_sec":t0.elapsed().as_secs_f64()}))
}
pub fn main_cli()->Result<(),String>{
    unsafe {openblas_set_num_threads(1);}
    let a:Vec<String>=std::env::args().collect();
    match a.get(1).map(String::as_str){
      Some("arena68") if a.len()==2=>crate::mqft_arena68::serve(),
      Some("fault67") if a.len()==6=>controlled61::fault67::run(&a[2],&a[3],&a[4],&a[5]),
      Some("schedule66") if a.len()==6=>controlled61::schedule66::run(&a[2],&a[3],&a[4],&a[5]),
      Some("phase65") if a.len()==6=>controlled61::persistent65::run(&a[2],&a[3],&a[4],&a[5]),
      Some("phase64") if a.len()==6=>controlled61::phase64::run(&a[2],&a[3],&a[4],&a[5]),
      Some("control62") if a.len()==6=>controlled61::covariant62::run(&a[2],&a[3],&a[4],&a[5]),
      Some("control") if a.len()==6=>controlled61::run(&a[2],&a[3],&a[4],&a[5]),
      Some("infer_d4") if a.len()==4=>controlled61::infer(&a[2],&a[3]),
      Some("infer") if a.len()==4=>{
        let n=Net::load(&a[2])?;let moves:Vec<Vec<usize>>=serde_json::from_str(&fs::read_to_string(&a[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        let mut out=vec![];for m in moves{let s=state(&m)?;if s.is_terminal(){return Err("terminal fixture".into())}let (p,v)=n.forward(&s);out.push(json!({"moves":m,"p":p,"v":v,"side":s.current_player()}));}
        println!("{}",serde_json::to_string(&out).unwrap());Ok(())
      }
      Some("play") if a.len()==7=>{
        let n=[Arc::new(Net::load(&a[2])?),Arc::new(Net::load(&a[3])?)];
        let jobs:Vec<Job>=serde_json::from_str(&fs::read_to_string(&a[4]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        let out=Path::new(&a[5]);let workers:usize=a[6].parse().map_err(|_|"worker parse")?;
        if workers==0||workers>24||jobs.is_empty(){return Err("invalid jobs or workers".into())}
        fs::create_dir_all(out).map_err(|e|e.to_string())?;
        let mut ids=std::collections::HashSet::new();
        for j in &jobs{if j.id.is_empty()||!j.id.chars().all(|x|x.is_ascii_alphanumeric()||x=='_'||x=='-')||!ids.insert(&j.id)||out.join(format!("{}.json",j.id)).exists(){return Err("invalid/duplicate/existing job ID".into())}}
        let pool=rayon::ThreadPoolBuilder::new().num_threads(workers).build().map_err(|e|e.to_string())?;
        pool.install(||jobs.par_iter().try_for_each(|j|->Result<(),String>{
          let v=game(j,&n)?;let p=out.join(format!("{}.json",j.id));let tmp=p.with_extension("part");let mut f=fs::OpenOptions::new().write(true).create_new(true).open(&tmp).map_err(|e|e.to_string())?;
          serde_json::to_writer(&mut f,&v).map_err(|e|e.to_string())?;f.flush().map_err(|e|e.to_string())?;f.sync_all().map_err(|e|e.to_string())?;fs::rename(tmp,p).map_err(|e|e.to_string())?;
          eprintln!("GAME {} winner={} plies={}",j.id,v["winner"],v["moves"].as_array().unwrap().len());Ok(())
        }))
      }
      _=>Err("usage: mqft_train infer weights.bin fixtures.json | play black.bin white.bin jobs.json outdir workers".into())
    }
}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn zero_weights_uniform(){
    let w=SHAPES.iter().map(|&n|vec![0.;n]).collect();let n=Net{w};
    let s=state(&[0,1,8]).unwrap();let(p,v)=n.forward(&s);assert_eq!(v,0.);
    for i in 0..N{assert_eq!(p[i],if [0,1,8].contains(&i){0.}else{1./78.});}
 }
 #[test]fn terminal_sign(){let s=state(&[0,9,1,10,2,11,3,12,4]).unwrap();assert!(s.is_terminal());assert_eq!(s.outcome(),-1.);assert_eq!(s.current_player(),-1);}
}

#[path="mqft_train_controlled61.rs"]
mod controlled61;
