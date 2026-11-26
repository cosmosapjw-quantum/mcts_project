//! Controlled finite stochastic root law on canonical MctsEngine, trained BoardNet.
use super::*;
use crate::mcts::{PreparedIteration,ImmediateReason};
use crate::mqft_law::{LawSpec,Sampler};
use std::collections::HashMap;
fn transform(a:usize,g:usize)->usize {
    let(mut y,mut x)=(a/L,a%L);
    if g>=4 {x=L-1-x;}
    for _ in 0..g%4 {(y,x)=(x,L-1-y);}
    y*L+x
}
fn averaged(net:&Net,s:&Gomoku)->(Vec<f32>,f32) {
    let b=s.board_state_record();let side=if s.current_player()==1{1}else{2};
    let mut ps=vec![Vec::<f32>::with_capacity(8);N];let mut vs=Vec::with_capacity(8);
    for g in 0..8{
        let mut bb=vec![0;N];for a in 0..N{bb[transform(a,g)]=b[a];}
        let t=Gomoku::from_board_12(L,5,&bb,side);
        let(p,v)=net.forward(&t);vs.push(v);
        for a in 0..N{ps[a].push(p[transform(a,g)]);}
    }
    // deterministic sorted f64 sum makes the projection insensitive to group order.
    let mean=|x:&mut Vec<f32>|{x.sort_by(f32::total_cmp);(x.iter().map(|&z|z as f64).sum::<f64>()/8.) as f32};
    (ps.iter_mut().map(mean).collect(),mean(&mut vs))
}
#[derive(Clone,Serialize)]
struct FaultRecord67 {id:u64,clean:f32,delivered:f32,mask:[u8;3],syndrome:[u8;2],correction:usize,residual:u8}
struct Fault67 {mode:String,seed:u64,records:Vec<FaultRecord67>}
impl Fault67{
 fn mix(mut x:u64)->u64{x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
 fn u(&self,id:u64,k:u64)->f64{let x=self.seed.wrapping_add(0x9e3779b97f4a7c15u64.wrapping_mul(id*5+k+1));(Self::mix(x)>>11) as f64/9007199254740992.}
 fn apply(&mut self,id:u64,v:f32)->f32{
  let mut mask=[0u8;3];let common=self.mode=="rep_common"||((self.mode=="single_mix"||self.mode=="rep_mix")&&self.u(id,0)<0.5);
  if id>0&&self.mode!="clean"{for k in 0..3{mask[k]=u8::from(self.u(id,if common{1}else{k as u64+1})<0.12);}}
  let syndrome=[mask[0]^mask[1],mask[1]^mask[2]];
  let correction=if self.mode.starts_with("rep"){match syndrome{[1,0]=>1,[1,1]=>2,[0,1]=>3,_=>0}}else{0};
  let residual=mask[0]^u8::from(correction==1);
  let delivered=if residual==1{-v}else{v};
  self.records.push(FaultRecord67{id,clean:v,delivered,mask,syndrome,correction,residual});delivered
 }
}
struct ControlledEval{net:Arc<Net>,d4:bool,records:Mutex<Vec<EvalRec>>,fault67:Mutex<Option<Fault67>>}
impl Evaluator<Gomoku> for ControlledEval{
    fn evaluate(&self,s:&Gomoku)->EvalResult<usize>{
        let(p,mut value)=if self.d4{averaged(&self.net,s)}else{self.net.forward(s)};
        let mut r=self.records.lock();let id=r.len() as u64;
        if let Some(f)=self.fault67.lock().as_mut(){value=f.apply(id,value);}
        r.push(EvalRec{id,state_hash:format!("{:016x}",s.tt_hash()),side:s.current_player(),value});
        mqft_trace::set_eval(id);
        EvalResult{policy:s.legal_moves().into_iter().map(|a|(a,p[a])).collect(),value}
    }
}
#[derive(Clone,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
struct ControlJob{id:String,moves:Vec<usize>,source:usize,seed:u64,budget:u32,width:usize,beta:f64,h:f64,d4:bool,generation:usize,core:usize,orientation:usize,rep:usize,cohort:String}
fn execute(j:&ControlJob,net:Arc<Net>)->Result<Value,String>{execute_field(j,net,None)}
fn execute_field(j:&ControlJob,net:Arc<Net>,field:Option<Vec<f64>>)->Result<Value,String>{execute_instrument(j,net,field,None)}
fn execute_instrument(j:&ControlJob,net:Arc<Net>,field:Option<Vec<f64>>,phase:Option<crate::mqft_law::Phase64>)->Result<Value,String>{execute_persistent(j,net,field,phase,None)}
fn execute_persistent(j:&ControlJob,net:Arc<Net>,field:Option<Vec<f64>>,phase:Option<crate::mqft_law::Phase64>,persistent:Option<crate::mqft_law::Phase65>)->Result<Value,String>{execute_schedule(j,net,field,phase,persistent,None)}
fn execute_schedule(j:&ControlJob,net:Arc<Net>,field:Option<Vec<f64>>,phase:Option<crate::mqft_law::Phase64>,persistent:Option<crate::mqft_law::Phase65>,schedule:Option<Vec<f64>>)->Result<Value,String>{execute_fault67(j,net,field,phase,persistent,schedule,None)}
fn execute_fault67(j:&ControlJob,net:Arc<Net>,field:Option<Vec<f64>>,phase:Option<crate::mqft_law::Phase64>,persistent:Option<crate::mqft_law::Phase65>,schedule:Option<Vec<f64>>,fault:Option<Fault67>)->Result<Value,String>{
    if ![32,64,128,512,1024].contains(&j.budget)||![1,4].contains(&j.width)||j.budget as usize%j.width!=0{return Err("bad budget/width".into())}
    let s=state(&j.moves)?;if s.is_terminal()||!s.legal_moves().contains(&j.source){return Err("bad source or terminal".into())}
    let law=LawSpec{kind:"iid".into(),beta_p:vec![j.beta],beta_q:vec![j.beta],proposal:"p".into(),logit_h:j.h};
    law.validate(j.budget,"wave",Some(j.source))?;
    let ev=Arc::new(ControlledEval{net,d4:j.d4,records:Mutex::new(vec![]),fault67:Mutex::new(fault)});
    let mut cfg=MctsConfig::evaluation(1.4);
    cfg.tt_enabled=false;cfg.root_forced_win=false;cfg.exact_terminal_value=false;cfg.fpu_reduction=0.;cfg.vl_mode=if j.width==1{VlMode::Disabled}else{VlMode::Fixed};cfg.seed=Some(j.seed);cfg.pw=None;
    let mut eng=MctsEngine::new(s.clone(),ev.clone(),cfg);
    eng.par_ctrl.mqft_record=true;eng.par_ctrl.mqft_sampler=Some(Mutex::new(if let Some(hs)=schedule {assert!(field.is_none()&&phase.is_none()&&persistent.is_none()&&hs.len()==j.budget as usize);Sampler::new_schedule66(law,j.source,hs,j.seed)} else if let Some(ph)=persistent {assert!(field.is_none()&&phase.is_none());Sampler::new_phase65(law,ph,j.seed)} else if let Some(ph)=phase {assert!(field.is_none());Sampler::new_phase64(law,ph,j.seed)} else if let Some(f)=field {Sampler::new_field(law,f,j.seed)} else {Sampler::new(law,Some(j.source),j.seed)}));
    let tr=Arc::new(Trace::default());eng.mqft_trace=Some(tr.clone());mqft_trace::clear_eval();
    let mut cuts=vec![];let t0=Instant::now();
    for phase in 1..=j.budget as usize/j.width{
        tr.set_phase(phase as u32);eng.par_ctrl.set_n_threads(j.width as u32);
        let mut prep=vec![];
        for _ in 0..j.width{
            let p=eng.prepare_iteration_async();
            let path=match &p{PreparedIteration::Immediate{path,..}=>path,PreparedIteration::Pending(p)=>&p.path};
            if path.is_empty(){return Err("empty path".into())}
            prep.push((tr.begin(&s,path),p));
        }
        // Deterministic NN evaluations before any commit. Jobs parallelized outside
        // the wave (not claiming concurrent GPU evaluation).
        let evals:Vec<_>=prep.iter().map(|(_,p)|{
            mqft_trace::clear_eval();
            match p{PreparedIteration::Pending(p)=>{let v=eng.evaluator.evaluate(&p.leaf_state);(Some(v),mqft_trace::take_eval())},_ =>(None,None)}
        }).collect();
        for ((sel,p),(result,id)) in prep.into_iter().zip(evals){
            match p{
                PreparedIteration::Immediate{path,value,reason}=>{
                    if reason!=ImmediateReason::TerminalNode{return Err("unexpected cap".into())}
                    let k=eng.apply_iteration_value_async(path,value);tr.finish(sel,value,None,true,k);
                },
                PreparedIteration::Pending(p)=>{
                    let value=crate::mcts::expand::expand_with_result(&p.leaf,&p.leaf_state,result.unwrap(),&eng.tt,eng.config.pw.as_ref());
                    let k=eng.apply_iteration_value_async(p.path,value);tr.finish(sel,value,id,false,k);
                }
            }
        }
        let k=phase*j.width;
        if [4,8,16,32,64,128,256,512,1024].contains(&k){
            let es=tr.events.lock();assert_eq!(es.len(),k);assert_eq!(eng.root.n_total.load(Ordering::Acquire),k as u32);
            let mut prefix=HashMap::<Vec<usize>,(u32,f64,f64)>::new();
            for e in es.iter(){let mut path=vec![];for (d,&a) in e.selection.actions.iter().enumerate(){path.push(a);let r=e.root_value as f64*if d%2==0{1.}else{-1.};let z=prefix.entry(path.clone()).or_insert((0,0.,0.));z.0+=1;z.1+=r;z.2+=r*r;}}
            let mut stack=vec![(eng.root,Vec::<usize>::new())];let mut visited=0;
            while let Some((node,path))=stack.pop(){
                let mut total=0;let mut wsum=0.;
                for e in node.read_edges(){let mut key=path.clone();key.push(e.mv);let q=prefix.remove(&key).unwrap_or((0,0.,0.));let n=e.n.load(Ordering::Acquire);let w=atomic_f64_load(&e.w);let s2=atomic_f64_load(&e.s2);let m2=atomic_f64_load(&e.m2);let mref=if n>0{s2-w*w/n as f64}else{0.};let tol=1e-9*(1.+n as f64);
                    assert!(q.0==n&&(q.1-w).abs()<tol&&(q.2-s2).abs()<tol&&(m2-mref).abs()<tol&&e.virtual_losses.load(Ordering::Acquire)==0&&atomic_f64_load(&e.virtual_value).abs()<tol);
                    total+=n;wsum+=w;if n>0{visited+=1;stack.push((e.child,key));}
                }
                assert_eq!(total,node.n_total.load(Ordering::Acquire));assert!((wsum-atomic_f64_load(&node.w_total)).abs()<1e-9*(1.+total as f64));
            }
            assert!(prefix.is_empty());
            let mut n=vec![0u32;N];let mut w=vec![0.;N];let mut s2=vec![0.;N];let mut p=vec![0.;N];
            for e in eng.root.read_edges(){n[e.mv]=e.n.load(Ordering::Acquire);w[e.mv]=atomic_f64_load(&e.w);s2[e.mv]=atomic_f64_load(&e.s2);p[e.mv]=e.p;}
            cuts.push(json!({"k":k,"n":n,"w":w,"s2":s2,"prior":p,"visited_prefix_edges":visited}));
        }
    }
    let es=tr.events.lock();let rec=ev.records.lock();
    assert_eq!(es.len(),j.budget as usize);assert_eq!(rec.len() as u64,eng.evaluator_call_count());
    // Compact exact trace: p=q=b for this registered law; omit duplicate vectors,
    // redundant probability logs q/b, but retain full scores,p,uniform,actions.
    let mut events=vec![];
    for e in es.iter(){
        let mut v=serde_json::to_value(e).unwrap();let d=v["root_selection"]["law"].as_object_mut().unwrap();
        assert_eq!(d["p"],d["q"]);assert_eq!(d["p"],d["b"]);
        for key in ["q","b","log_q","log_b","log_q_path","log_b_path"]{d.remove(key);}
        events.push(v);
    }
    let mut out=json!({"schema":"mqft.trained.control61.v1","job":j,"board":s.board_state_record(),"side":s.current_player(),"state_hash":format!("{:016x}",s.tt_hash()),"cuts":cuts,"events":events,"evaluations":&*rec,"elapsed_ns":t0.elapsed().as_nanos() as u64});
    if let Some(f)=ev.fault67.lock().as_ref(){out["fault67"]=serde_json::to_value(&f.records).unwrap();}
    Ok(out)
}
pub(super)fn run(weights:&str,plan:&str,out:&str,workers:&str)->Result<(),String>{
    let n=Arc::new(Net::load(weights)?);
    let js:Vec<ControlJob>=serde_json::from_str(&fs::read_to_string(plan).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let out=Path::new(out);let workers:usize=workers.parse().map_err(|_|"worker parse")?;
    if workers==0||workers>24||js.is_empty(){return Err("invalid workers/jobs".into())}
    fs::create_dir_all(out).map_err(|e|e.to_string())?;
    let mut ids=std::collections::HashSet::new();
    for j in &js{if !j.id.chars().all(|c|c.is_ascii_alphanumeric()||c=='_')||j.id.is_empty()||!ids.insert(&j.id)||out.join(format!("{}.json",j.id)).exists(){return Err("bad/duplicate/existing job".into())}}
    let pool=rayon::ThreadPoolBuilder::new().num_threads(workers).build().map_err(|e|e.to_string())?;
    pool.install(||js.par_iter().try_for_each(|j|->Result<(),String>{
        let v=execute(j,n.clone())?;let p=out.join(format!("{}.json",j.id));let t=p.with_extension("part");
        let mut f=fs::OpenOptions::new().write(true).create_new(true).open(&t).map_err(|e|e.to_string())?;
        serde_json::to_writer(&mut f,&v).map_err(|e|e.to_string())?;f.flush().map_err(|e|e.to_string())?;fs::rename(t,p).map_err(|e|e.to_string())?;
        Ok(())
    }))?;
    eprintln!("CONTROL COMPLETE {} jobs",js.len());Ok(())
}
pub(super)fn infer(weights:&str,fixture:&str)->Result<(),String>{
    let n=Net::load(weights)?;let ms:Vec<Vec<usize>>=serde_json::from_str(&fs::read_to_string(fixture).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let mut out=vec![];
    for m in ms{let s=state(&m)?;if s.is_terminal(){return Err("terminal fixture".into())}
        let (p,v)=averaged(&n,&s);out.push(json!({"moves":m,"p":p,"v":v,"side":s.current_player()}));
    }
    println!("{}",serde_json::to_string(&out).unwrap());Ok(())
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn transforms_are_permutations(){for g in 0..8{let mut a=(0..N).map(|a|transform(a,g)).collect::<Vec<_>>();a.sort();assert_eq!(a,(0..N).collect::<Vec<_>>());}}
 #[test]fn d4_zero_weight(){let n=Net{w:SHAPES.iter().map(|&n|vec![0.;n]).collect()};let s=state(&[0,1,8]).unwrap();let(p,v)=averaged(&n,&s);assert_eq!(v,0.);assert_eq!(p[0],0.);assert_eq!(p[2],1./78.);}
}

#[path="mqft_train_covariant62.rs"]
pub(super) mod covariant62;

#[path="mqft_train_phase64.rs"]
pub(super) mod phase64;

#[path="mqft_train_persistent65.rs"]
pub(super) mod persistent65;

#[path="mqft_train_schedule66.rs"]
pub(super) mod schedule66;

#[path="mqft_train_fault67.rs"]
pub(super) mod fault67;
