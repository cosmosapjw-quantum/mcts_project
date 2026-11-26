//! Full legal-support phase instrument experiment, independent game workers.
use super::*;
use crate::mqft_law::Phase64;
use flate2::{write::GzEncoder,Compression};
#[derive(Clone,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
struct Job64{id:String,moves:Vec<usize>,sources:[usize;2],seed:u64,budget:u32,width:usize,beta:f64,generation:usize,core:usize,rep:usize,arm:String}
fn one(j:&Job64,net:Arc<Net>)->Result<Value,String>{
    if j.budget!=128 || ![1,4].contains(&j.width) || !["baseline","dephased","real","plus","minus","rotate0"].contains(&j.arm.as_str()){return Err("invalid phase experiment".into())}
    let s=state(&j.moves)?;let legal=s.legal_moves();
    if s.is_terminal()||j.sources[0]==j.sources[1]||j.sources.iter().any(|a|!legal.contains(a)){return Err("invalid source pair".into())}
    let old=ControlJob{id:j.id.clone(),moves:j.moves.clone(),source:j.sources[0],seed:j.seed,budget:j.budget,width:j.width,beta:j.beta,h:0.,d4:false,generation:j.generation,core:j.core,orientation:0,rep:j.rep,cohort:"phase64".into()};
    let ph=Phase64::new(j.sources,j.arm.clone(),j.width,j.budget as usize);
    let raw=execute_instrument(&old,net,None,Some(ph))?;
    Ok(json!({"schema":"mqft.trained.phase64.v1","input":j,"trace":raw}))
}
pub(in crate::mqft_train)fn run(weights:&str,plan:&str,out:&str,workers:&str)->Result<(),String>{
 let net=Arc::new(Net::load(weights)?);
 let js:Vec<Job64>=serde_json::from_str(&fs::read_to_string(plan).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 let out=Path::new(out);let workers:usize=workers.parse().map_err(|_|"worker parse")?;
 if workers==0||workers>24||js.is_empty(){return Err("bad workers/jobs".into())}
 fs::create_dir_all(out).map_err(|e|e.to_string())?;let mut ids=std::collections::HashSet::new();
 for j in &js{if j.id.is_empty()||!j.id.chars().all(|c|c.is_ascii_alphanumeric()||c=='_')||!ids.insert(&j.id)||out.join(format!("{}.json.gz",j.id)).exists(){return Err("invalid or duplicate job".into())}}
 let pool=rayon::ThreadPoolBuilder::new().num_threads(workers).build().map_err(|e|e.to_string())?;
 pool.install(||js.par_iter().try_for_each(|j|->Result<(),String>{
  let v=one(j,net.clone())?;let p=out.join(format!("{}.json.gz",j.id));let t=p.with_extension("part");
  let f=fs::OpenOptions::new().write(true).create_new(true).open(&t).map_err(|e|e.to_string())?;
  let mut gz=GzEncoder::new(f,Compression::fast());serde_json::to_writer(&mut gz,&v).map_err(|e|e.to_string())?;
  let f=gz.finish().map_err(|e|e.to_string())?;f.sync_all().map_err(|e|e.to_string())?;fs::rename(t,p).map_err(|e|e.to_string())?;Ok(())
 }))?;
 eprintln!("PHASE64 COMPLETE {}",js.len());Ok(())
}
