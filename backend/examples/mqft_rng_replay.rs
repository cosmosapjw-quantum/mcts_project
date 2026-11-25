//! Read compact trace records, replay root and evaluator PRNG with locked rand.
//! Independent audit executable; no changes to production search.
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{json, Value};
use std::io::{self, BufRead};
fn mix(x: u64) -> u64 {
    let mut x = x.wrapping_add(0x9e3779b97f4a7c15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn hex(x: &Value) -> u64 {
    u64::from_str_radix(x.as_str().unwrap(), 16).unwrap()
}
fn replay(v: Value) -> Value {
    let seed = mix(v["spec_seed"].as_u64().unwrap());
    assert_eq!(seed, hex(&v["replica_seed"]));
    let mut root = StdRng::seed_from_u64(mix(seed ^ 0x52_4f_4f_54));
    let mut nr = 0;
    let mut max_err = 0f64;
    for u in v["root_uniforms"].as_array().unwrap() {
        let q = root.gen::<f64>();
        let e = (q - u.as_f64().unwrap()).abs();
        assert!(e <= 2e-16, "root RNG mismatch at {} {}", nr, e);
        max_err = max_err.max(e);
        nr += 1;
    }
    let mut ne = 0;
    let mut min_margin = 1f64;
    let mut max_log = 0f64;
    for e in v["evaluations"].as_array().unwrap() {
        let s = mix(seed ^ mix(e["id"].as_u64().unwrap()) ^ hex(&e["state_hash"]));
        let scores = e["nnue_scores"].as_array().unwrap();
        let max = scores.iter().map(|x| x[1].as_i64().unwrap()).max().unwrap() as f64;
        let weights: Vec<_> = scores
            .iter()
            .map(|x| ((x[1].as_i64().unwrap() as f64 - max) / 32.).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let p: Vec<_> = weights
            .iter()
            .map(|w| (1. - 1e-6) * w / total + 1e-6 / weights.len() as f64)
            .collect();
        for (j, x) in e["samples"].as_array().unwrap().iter().enumerate() {
            let ss = mix(s ^ j as u64);
            assert_eq!(ss, hex(&x["seed"]));
            let u = StdRng::seed_from_u64(ss).gen::<f64>();
            let mut c = 0.;
            let mut low = 0.;
            let mut i = p.len() - 1;
            for (a, w) in p.iter().enumerate() {
                low = c;
                c += w;
                if u < c {
                    i = a;
                    break;
                }
            }
            assert_eq!(scores[i][0].as_u64(), x["actions"][0].as_u64());
            min_margin = min_margin.min((u - low).abs().min((c - u).abs()));
            let err = (p[i].ln() - x["log_path_prob"].as_f64().unwrap()).abs();
            assert!(err < 1e-10);
            max_log = max_log.max(err);
            ne += 1;
        }
    }
    json!({"root_draws":nr,"sample_draws":ne,"max_root_u_error":max_err,"max_sample_log_error":max_log,"min_sample_cdf_margin":min_margin})
}
fn main() {
    for l in io::stdin().lock().lines() {
        let l = l.unwrap();
        let v: Value = serde_json::from_str(&l).unwrap();
        println!("{}", replay(v));
    }
}
