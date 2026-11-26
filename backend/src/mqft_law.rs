//! Opt-in causal root policy. All other descent, expansion and backup is Quartz.
//! Protocol P/Q differ only in root logit scale; evaluator and downstream kernel shared.
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LawSpec {
    pub kind: String, // iid: frozen completed root; pl: without replacement within wave; reserve: VL-aware.
    pub beta_p: Vec<f64>,
    pub beta_q: Vec<f64>,
    pub proposal: String, // p, q, local, path
    #[serde(default)]
    pub logit_h: f64,
}
impl LawSpec {
    pub fn validate(&self, budget: u32, mode: &str, source: Option<usize>) -> Result<(), String> {
        if !["iid", "pl", "reserve"].contains(&self.kind.as_str())
            || !["p", "q", "local", "path"].contains(&self.proposal.as_str())
        {
            return Err("unknown root law or proposal".into());
        }
        // Shared-tree asynchronous scheduling law not specified. Explicitly reject it.
        if !["serial", "wave", "leaf"].contains(&mode) {
            return Err("root law currently requires serial, leaf or wave".into());
        }
        for v in [&self.beta_p, &self.beta_q] {
            if !(v.len() == 1 || v.len() == budget as usize)
                || v.iter().any(|x| !x.is_finite() || *x < 0. || *x > 20.)
            {
                return Err("beta schedule length 1 or budget, finite in [0,20]".into());
            }
        }
        if !self.logit_h.is_finite()
            || self.logit_h.abs() > 20.
            || (self.logit_h != 0. && source.is_none())
        {
            return Err("logit_h requires legal source action and finite |h|<=20".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct Draw {
    pub t: usize,
    pub kind: String,
    pub proposal: String,
    pub actions: Vec<usize>,
    pub scores: Vec<f32>,
    pub active: Vec<bool>,
    pub beta_p: f64,
    pub beta_q: f64,
    pub logit_h: f64,
    pub source: Option<usize>,
    pub alpha: f64,
    pub u: f64,
    pub picked: usize,
    pub p: Vec<f64>,
    pub q: Vec<f64>,
    pub b: Vec<f64>,
    pub log_p: f64,
    pub log_q: f64,
    pub log_b: f64,
    pub log_p_path: f64,
    pub log_q_path: f64,
    pub log_b_path: f64,
}
pub struct Sampler {
    pub spec: LawSpec,
    source: Option<usize>,
    field: Option<Vec<f64>>,
    rng: StdRng,
    t: usize,
    lp: f64,
    lq: f64,
    lb: f64,
}
fn lse(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}
fn softmax(
    s: &[f32],
    active: &[bool],
    actions: &[usize],
    beta: f64,
    source: Option<usize>,
    h: f64,
    field: Option<&[f64]>,
) -> Vec<f64> {
    let z: Vec<f64> = s
        .iter()
        .zip(actions)
        .zip(active)
        .map(|((&s, &a), &on)| {
            if on {
                beta * s as f64 + if let Some(f)=field {f[a]} else if Some(a) == source { h } else { 0. }
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(m.is_finite(), "nonempty finite support");
    let mut p: Vec<_> = z.iter().map(|x| (x - m).exp()).collect();
    let sum = p.iter().sum::<f64>();
    for x in &mut p {
        *x /= sum;
    }
    p
}
impl Sampler {
    pub fn new(spec: LawSpec, source: Option<usize>, seed: u64) -> Self {
        Self {
            spec,
            source,
            field: None,
            rng: StdRng::seed_from_u64(seed),
            t: 0,
            lp: 0.,
            lq: 0.,
            lb: 0.,
        }
    }
    /// Separate multi-field law; legacy scalar spec/serialization unchanged.
    pub fn new_field(spec: LawSpec, field: Vec<f64>, seed:u64) -> Self {
        assert!(spec.logit_h==0. && !field.is_empty() && field.iter().all(|h|h.is_finite() && h.abs()<=20.));
        let mut out=Self::new(spec,None,seed);out.field=Some(field);out
    }
    pub fn draw(
        &mut self,
        actions: Vec<usize>,
        scores: Vec<f32>,
        active: Vec<bool>,
    ) -> (usize, Draw) {
        let t = self.t;
        let beta_p = self.spec.beta_p[if self.spec.beta_p.len() == 1 { 0 } else { t }];
        let beta_q = self.spec.beta_q[if self.spec.beta_q.len() == 1 { 0 } else { t }];
        let p = softmax(
            &scores,
            &active,
            &actions,
            beta_p,
            self.source,
            self.spec.logit_h,
            self.field.as_deref(),
        );
        let q = softmax(
            &scores,
            &active,
            &actions,
            beta_q,
            self.source,
            self.spec.logit_h,
            self.field.as_deref(),
        );
        let normalizer = lse(self.lp, self.lq);
        let alpha = (self.lp - normalizer).exp();
        let alpha_q = (self.lq - normalizer).exp();
        let b: Vec<f64> = p
            .iter()
            .zip(&q)
            .map(|(&p, &q)| match self.spec.proposal.as_str() {
                "p" => p,
                "q" => q,
                "local" => 0.5 * (p + q),
                _ => alpha * p + alpha_q * q,
            })
            .collect();
        let u = self.rng.gen::<f64>();
        let mut cum = 0.;
        let mut picked = None;
        for (i, &x) in b.iter().enumerate() {
            if x > 0. {
                cum += x;
                if u < cum {
                    picked = Some(i);
                    break;
                }
            }
        }
        let i = picked.unwrap_or_else(|| b.iter().rposition(|&x| x > 0.).unwrap());
        let (lp, lq, lb) = (p[i].ln(), q[i].ln(), b[i].ln());
        self.lp += lp;
        self.lq += lq;
        self.lb += lb;
        self.t += 1;
        (
            i,
            Draw {
                t,
                kind: self.spec.kind.clone(),
                proposal: self.spec.proposal.clone(),
                actions,
                scores,
                active,
                beta_p,
                beta_q,
                logit_h: self.spec.logit_h,
                source: self.source,
                alpha,
                u,
                picked: i,
                p,
                q,
                b,
                log_p: lp,
                log_q: lq,
                log_b: lb,
                log_p_path: self.lp,
                log_q_path: self.lq,
                log_b_path: self.lb,
            },
        )
    }
}
