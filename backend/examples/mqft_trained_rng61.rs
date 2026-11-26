//! Independently reconstruct registered uniform streams, no search code.
use rand::{rngs::StdRng,Rng,SeedableRng};
use serde::Deserialize;
use serde_json::json;
#[derive(Deserialize)]
struct Stream{seed:u64,u:Vec<f64>}
fn main(){
 let p=std::env::args().nth(1).expect("RNG_INPUT61.json path");
 let xs:Vec<Stream>=serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
 let mut n=0;let mut m=0f64;
 for s in &xs{let mut r=StdRng::seed_from_u64(s.seed);
  for &u in &s.u {let e=(r.gen::<f64>()-u).abs();assert!(e<=2e-16);m=m.max(e);n+=1;}}
 println!("{}",json!({"streams":xs.len(),"independent_uniforms":n,"max_error":m}));
}
