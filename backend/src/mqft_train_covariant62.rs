//! Exact discrete canonical-frame transport. Original native engine is evaluated
//! in a single shared frame; frame rank is explicit, uniform over ties for the
//! equivariant mixture. All raw coordinates remain canonical, with inverse map.
use super::*;
use flate2::{write::GzEncoder,Compression};
#[derive(Clone,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
struct Job62{
 id:String,moves:Vec<usize>,sources:Vec<usize>,field:Vec<f64>,seed:u64,
 budget:u32,width:usize,beta:f64,d4:bool,generation:usize,core:usize,
 orientation:usize,rep:usize,cohort:String,frame_rank:usize,arm:String
}
fn canonical(j:&Job62)->(Vec<usize>,Vec<usize>,Vec<usize>){
 let mut best:Option<(Vec<usize>,Vec<usize>)>=None;let mut frames=vec![];
 for g in 0..8{
  let key=(j.moves.iter().map(|&a|transform(a,g)).collect::<Vec<_>>(),j.sources.iter().map(|&a|transform(a,g)).collect::<Vec<_>>());
  if best.as_ref().map_or(true,|b|key<*b){best=Some(key);frames=vec![g];}
  else if best.as_ref()==Some(&key){frames.push(g);}
 }
 let(m,s)=best.unwrap();(m,s,frames)
}
fn one(j:&Job62,net:Arc<Net>)->Result<Value,String>{
 if j.sources.is_empty()||j.sources.len()>2||j.sources.len()!=j.field.len()||j.moves.iter().chain(j.sources.iter()).any(|&a|a>=N)||j.field.iter().any(|h|!h.is_finite()||h.abs()>20.){return Err("bad field".into())}
 let st=state(&j.moves)?;let legal=st.legal_moves();let uniq:std::collections::HashSet<_>=j.sources.iter().collect();
 if uniq.len()!=j.sources.len()||j.sources.iter().any(|a|!legal.contains(a)){return Err("bad sources".into())}
 let(m,s,frames)=if j.cohort=="legacy"{(j.moves.clone(),j.sources.clone(),vec![0])}else{canonical(j)};if j.frame_rank>=frames.len(){return Err("bad frame rank".into())}
 let g=frames[j.frame_rank];let mut inv=vec![0;N];for a in 0..N{inv[transform(a,g)]=a;}
 let mut f=vec![0.;N];for (&a,&h) in s.iter().zip(&j.field){f[a]=h;}
 let old=ControlJob{id:j.id.clone(),moves:m,source:s[0],seed:j.seed,budget:j.budget,width:j.width,beta:j.beta,h:0.,d4:j.d4,generation:j.generation,core:j.core,orientation:j.orientation,rep:j.rep,cohort:j.cohort.clone()};
 let raw=execute_field(&old,net,Some(f))?;
 Ok(json!({"schema":"mqft.trained.control62.v1","input":j,"frame":g,"frames":frames,"inverse":inv,"canonical_sources":s,"trace":raw}))
}
pub(in crate::mqft_train) fn run(weights:&str,plan:&str,out:&str,workers:&str)->Result<(),String>{
 let net=Arc::new(Net::load(weights)?);
 let jobs:Vec<Job62>=serde_json::from_str(&fs::read_to_string(plan).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 let out=Path::new(out);let workers:usize=workers.parse().map_err(|_|"worker parse")?;
 if workers==0||workers>24||jobs.is_empty(){return Err("invalid workers/jobs".into())}
 fs::create_dir_all(out).map_err(|e|e.to_string())?;
 let mut ids=std::collections::HashSet::new();
 for j in &jobs{if j.id.is_empty()||!j.id.chars().all(|c|c.is_ascii_alphanumeric()||c=='_')||!ids.insert(&j.id)||out.join(format!("{}.json.gz",j.id)).exists(){return Err("invalid duplicate/existing job".into())}}
 let pool=rayon::ThreadPoolBuilder::new().num_threads(workers).build().map_err(|e|e.to_string())?;
 pool.install(||jobs.par_iter().try_for_each(|j|->Result<(),String>{
  let v=one(j,net.clone())?;let p=out.join(format!("{}.json.gz",j.id));let t=p.with_extension("part");
  let file=fs::OpenOptions::new().write(true).create_new(true).open(&t).map_err(|e|e.to_string())?;
  let mut gz=GzEncoder::new(file,Compression::fast());serde_json::to_writer(&mut gz,&v).map_err(|e|e.to_string())?;let f=gz.finish().map_err(|e|e.to_string())?;f.sync_all().map_err(|e|e.to_string())?;fs::rename(t,p).map_err(|e|e.to_string())?;Ok(())
 }))?;eprintln!("CONTROL62 COMPLETE {}",jobs.len());Ok(())
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn frame_sets(){
  let mut j=Job62{id:"t".into(),moves:vec![40],sources:vec![30],field:vec![0.],seed:1,budget:32,width:4,beta:1.,d4:true,generation:0,core:0,orientation:0,rep:0,cohort:"t".into(),frame_rank:0,arm:"t".into()};
  let(m,s,f)=canonical(&j);assert_eq!(f.len(),2);
  j.moves=vec![transform(40,1)];j.sources=vec![transform(30,1)];let(a,b,g)=canonical(&j);assert_eq!((a,b),(m,s));assert_eq!(g.len(),2);
 }
}
