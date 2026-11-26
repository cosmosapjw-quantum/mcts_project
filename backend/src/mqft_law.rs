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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase64: Option<PhaseRecord64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase65: Option<crate::mqft_law::PhaseRecord65>,
    pub log_q_path: f64,
    pub log_b_path: f64,
}
pub struct Sampler {
    pub spec: LawSpec,
    source: Option<usize>,
    field: Option<Vec<f64>>,
    phase64: Option<Phase64>,
    phase65: Option<Phase65>,
    schedule66: Option<Vec<f64>>,
    rng: StdRng,
    t: usize,
    lp: f64,
    lq: f64,
    lb: f64,
}
/// Auxiliary fresh pair register with causal score-exposure phase schedule.
/// The instrument conserves the pair probability mass and leaves all other
/// legal actions unchanged. It is a software quantum-channel emulator.
#[derive(Clone, Debug, Serialize)]
pub struct PhaseRecord64 {
    pub sources: [usize;2],
    pub wave: usize,
    pub tau: f64,
    pub exposure: f64,
    pub theta: f64,
    pub eta: f64,
    pub pair_mass: f64,
    pub population_a: f64,
    pub x: f64,
    pub y: f64,
    pub output_a: f64,
}
pub struct Phase64 {
    sources: [usize;2],
    mode: String,
    width: usize,
    budget: usize,
    exposure: f64,
    current_exposure: f64,
}
impl Phase64 {
    pub fn new(sources:[usize;2],mode:String,width:usize,budget:usize)->Self{
        assert!(sources[0]!=sources[1] && width>0 && budget>0 && budget%width==0);
        assert!(["baseline","dephased","real","plus","minus","rotate0"].contains(&mode.as_str()));
        Self{sources,mode,width,budget,exposure:0.,current_exposure:0.}
    }
    fn apply(&mut self,t:usize,actions:&[usize],scores:&[f32],base:&[f64])->(Vec<f64>,PhaseRecord64){
        let a=actions.iter().position(|&a|a==self.sources[0]).unwrap();
        let b=actions.iter().position(|&a|a==self.sources[1]).unwrap();
        assert!(base[a]>0. && base[b]>0.);
        if t%self.width==0{
            self.current_exposure=self.exposure;
            self.exposure+=(scores[a] as f64-scores[b] as f64)*self.width as f64/self.budget as f64;
        }
        let wave=t/self.width;let tau=(wave*self.width) as f64/self.budget as f64;
        let eta=if self.mode=="dephased"{0.}else{0.8*(-1.2*tau).exp()};
        let theta=match self.mode.as_str(){
            "plus"=>std::f64::consts::FRAC_PI_2+6.*self.current_exposure,
            "minus"=>-std::f64::consts::FRAC_PI_2+6.*self.current_exposure,
            "rotate0"=>6.*self.current_exposure,
            _=>0.,
        };
        let m=base[a]+base[b];let r=base[a]/m;
        let amp=2.*eta*(r*(1.-r)).sqrt();
        let x=amp*theta.cos();let y=amp*theta.sin();
        let alpha:f64=if self.mode=="baseline"{0.}else{0.35};
        let c=alpha.cos();let s=alpha.sin();
        let z=if self.mode=="baseline"{base[a]}else{m*(c*c*r+s*s*(1.-r)+c*s*y)};
        let mut out=base.to_vec();out[a]=z;out[b]=if self.mode=="baseline"{base[b]}else{m-z};
        assert!(out[a]>0. && out[b]>0. && (out.iter().sum::<f64>()-1.).abs()<1e-12);
        (out,PhaseRecord64{sources:self.sources,wave,tau,exposure:self.current_exposure,theta,eta,pair_mass:m,population_a:r,x,y,output_a:z})
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct PhaseRecord65 {
    pub sources:[usize;2], pub wave:usize, pub tau:f64,
    pub exposure:f64, pub rotation:f64, pub pair_mass:f64, pub population_a:f64,
    pub pre:[f64;3], pub prepared:[f64;3], pub post:[f64;3],
    pub e0:f64, pub e1:f64, pub output_a:f64, pub outcome:usize,
}
/// Persistent auxiliary qubit; actual selection outcomes update it.
pub struct Phase65 {
    sources:[usize;2], mode:String, width:usize, budget:usize,
    state:[f64;3], exposure:f64, previous_gap:f64,
}
impl Phase65 {
    pub fn new(sources:[usize;2],mode:String,width:usize,budget:usize)->Self{
        assert!(sources[0]!=sources[1] && width>0 && budget%width==0);
        assert!(["persist_plus","persist_minus","persist_dephased","reset_plus"].contains(&mode.as_str()));
        let y=if mode=="persist_minus"{-0.8}else if mode=="persist_dephased"{0.0}else{0.8};
        Self{sources,mode,width,budget,state:[0.,y,0.],exposure:0.,previous_gap:0.}
    }
    fn rotate_x(v:[f64;3],angle:f64)->[f64;3]{
        let(c,s)=(angle.cos(),angle.sin());[v[0],c*v[1]-s*v[2],s*v[1]+c*v[2]]
    }
    fn prepare(&mut self,t:usize,actions:&[usize],scores:&[f32],base:&[f64])->(Vec<f64>,PhaseRecord65){
        let a=actions.iter().position(|&a|a==self.sources[0]).unwrap();
        let b=actions.iter().position(|&a|a==self.sources[1]).unwrap();
        let pre=self.state;let dt=self.width as f64/self.budget as f64;
        let mut rotation=0.;
        if t%self.width==0{
            if t>0{
                self.exposure+=self.previous_gap*dt;
                rotation=6.*self.previous_gap*dt;
                let(c,s)=(rotation.cos(),rotation.sin());let eta=(-1.2*dt).exp();
                let [x,y,z]=self.state;self.state=[eta*(c*x-s*y),eta*(s*x+c*y),z];
            }
            self.previous_gap=scores[a] as f64-scores[b] as f64;
        }
        let m=base[a]+base[b];let r=base[a]/m;
        let tau=(t/self.width*self.width) as f64/self.budget as f64;
        if self.mode=="reset_plus"{
            let theta=std::f64::consts::FRAC_PI_2+6.*self.exposure;
            let amp=1.6*(-1.2*tau).exp()*(r*(1.-r)).sqrt();
            self.state=[amp*theta.cos(),amp*theta.sin(),2.*r-1.];
        }
        if self.mode=="persist_dephased"{self.state[0]=0.;self.state[1]=0.;}
        let prepared=self.state;
        let z=Self::rotate_x(prepared,0.7)[2];
        let e1=0.35*r;let e0=e1+0.65;
        let d=0.5*(e0+e1+(e0-e1)*z);
        let mut out=base.to_vec();out[a]=m*d;out[b]=m-out[a];
        assert!(out[a]>0.&&out[b]>0.&&(out.iter().sum::<f64>()-1.).abs()<1e-12);
        let qa=out[a];
        (out,PhaseRecord65{sources:self.sources,wave:t/self.width,tau,exposure:self.exposure,rotation,pair_mass:m,population_a:r,pre,prepared,post:prepared,e0,e1,output_a:qa,outcome:usize::MAX})
    }
    fn finish(&mut self,action:usize,rec:&mut PhaseRecord65){
        rec.outcome=action;
        if self.sources.contains(&action){
            let (e0,e1)=if action==self.sources[0]{(rec.e0,rec.e1)}else{(1.-rec.e0,1.-rec.e1)};
            let [x,y,z]=Self::rotate_x(self.state,0.7);
            let d=0.5*(e0+e1+(e0-e1)*z);
            let g=(e0*e1).sqrt()/d;
            self.state=Self::rotate_x([g*x,g*y,0.5*(e0-e1+(e0+e1)*z)/d],-0.7);
        }
        assert!(self.state.iter().map(|x|x*x).sum::<f64>()<=1.+1e-10);
        rec.post=self.state;
    }
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
            phase64: None,
            phase65: None,
            schedule66: None,
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
    pub fn new_phase64(spec: LawSpec,phase:Phase64,seed:u64)->Self{
        assert!(spec.kind=="iid" && spec.proposal=="p" && spec.logit_h==0. && spec.beta_p==spec.beta_q);
        let mut out=Self::new(spec,None,seed);out.phase64=Some(phase);out
    }
    pub fn new_phase65(spec:LawSpec,phase:Phase65,seed:u64)->Self{
        assert!(spec.kind=="iid" && spec.proposal=="p" && spec.logit_h==0. && spec.beta_p==spec.beta_q);
        let mut out=Self::new(spec,None,seed);out.phase65=Some(phase);out
    }
    /// Opt-in deterministic external source schedule (same-order history).
    pub fn new_schedule66(spec:LawSpec,source:usize,schedule:Vec<f64>,seed:u64)->Self{
        assert!(spec.kind=="iid"&&spec.proposal=="p"&&spec.beta_p==spec.beta_q&&spec.logit_h==0.);
        assert!(!schedule.is_empty()&&schedule.iter().all(|x|x.is_finite()&&x.abs()<=20.));
        let mut out=Self::new(spec,Some(source),seed);out.schedule66=Some(schedule);out
    }
    pub fn draw(
        &mut self,
        actions: Vec<usize>,
        scores: Vec<f32>,
        active: Vec<bool>,
    ) -> (usize, Draw) {
        let t = self.t;
        let h=if let Some(hs)=self.schedule66.as_ref(){hs[t]}else{self.spec.logit_h};
        let beta_p = self.spec.beta_p[if self.spec.beta_p.len() == 1 { 0 } else { t }];
        let beta_q = self.spec.beta_q[if self.spec.beta_q.len() == 1 { 0 } else { t }];
        let mut p = softmax(
            &scores,
            &active,
            &actions,
            beta_p,
            self.source,
            h,
            self.field.as_deref(),
        );
        let mut q = softmax(
            &scores,
            &active,
            &actions,
            beta_q,
            self.source,
            h,
            self.field.as_deref(),
        );
        let phase64=if let Some(ph)=self.phase64.as_mut(){
            assert_eq!(p,q);assert!(active.iter().all(|x|*x));
            let (out,rec)=ph.apply(t,&actions,&scores,&p);
            p=out;q=p.clone();Some(rec)
        }else{None};
        let mut phase65=if let Some(ph)=self.phase65.as_mut(){
            assert_eq!(p,q);assert!(active.iter().all(|x|*x));
            let (out,rec)=ph.prepare(t,&actions,&scores,&p);p=out;q=p.clone();Some(rec)
        }else{None};
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
        if let (Some(ph),Some(rec))=(self.phase65.as_mut(),phase65.as_mut()){ph.finish(actions[i],rec);}
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
                logit_h: h,
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
                phase64,
                phase65,
                log_q_path: self.lq,
                log_b_path: self.lb,
            },
        )
    }
}

#[cfg(test)] mod phase64_tests {
 use super::*;
 #[test] fn pair_mass_and_controls(){
  let a=vec![0,1,2];let scores=vec![0.,0.,0.];let p=vec![0.2,0.3,0.5];
  let run=|mode:&str|Phase64::new([0,1],mode.into(),4,128).apply(0,&a,&scores,&p).0;
  let re=run("real");let de=run("dephased");let plus=run("plus");let minus=run("minus");
  assert_eq!(re,de);assert_eq!(run("baseline"),p);
  assert!((plus[0]+minus[0]-2.*de[0]).abs()<1e-15);
  for q in [re,de,plus,minus]{assert_eq!(q[2],p[2]);assert!((q[0]+q[1]-0.5f64).abs()<1e-15);}
 }
}

#[cfg(test)]mod phase65_tests{
 use super::*;
 #[test]fn persistent_pair_instrument(){
  let mut p=Phase65::new([0,1],"persist_plus".into(),4,128);
  let (q,mut r)=p.prepare(0,&[0,1,2],&[0.,0.,0.],&[0.2,0.3,0.5]);
  assert_eq!(q[2],0.5);assert!((q[0]+q[1]-0.5).abs()<1e-15);
  p.finish(2,&mut r);assert_eq!(r.prepared,r.post);
  let (_,mut r)=p.prepare(1,&[0,1,2],&[0.,0.,0.],&[0.2,0.3,0.5]);p.finish(0,&mut r);assert_ne!(r.prepared,r.post);
  let (_,s)=p.prepare(2,&[0,1,2],&[0.,0.,0.],&[0.2,0.3,0.5]);assert_eq!(s.pre,r.post);
 }
}
