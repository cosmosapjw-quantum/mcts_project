"""Isolated opt-in multi-time phase intervention; canonical source untouched."""
from base61 import *
import os,shutil,subprocess,difflib
P=B/'mqft_restore39_20261125/repo';A=B/'mqft_restore74_20261126/pulse_src';D=S/'phase_src';parent='9168e93ef9669ea6ea8c05d63a20c749b419f466'
assert subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip()==parent
shutil.copytree(A,D);patches={}
def change(f,pairs):
 p=D/'src'/f;old=p.read_text();new=old
 for a,b in pairs:
  assert new.count(a)==1,(f,a,new.count(a));new=new.replace(a,b)
 p.write_text(new);patches[f]=dict(original_sha=sha(A/'src'/f),modified_sha=sha(p),diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+f,tofile='b/'+f)))
modes='"plus81","minus81","eraseplus81","eraseminus81","swap81","base81"'
newimpl=r'''
    fn prepare81(&mut self,t:usize,actions:&[usize],base:&[f64])->(Vec<f64>,PhaseRecord65){
        let ia=actions.iter().position(|&a|a==self.sources[0]).unwrap();
        let ib=actions.iter().position(|&a|a==self.sources[1]).unwrap();
        let pre=self.state;
        let active=(32..44).contains(&t);
        let wave=t/4;
        let mut theta:f64=0.;let mut eta=1.;let erase=t==32&&self.mode.starts_with("erase");
        if active&&t%4==0{
            theta=if t==32{if self.mode=="minus81"||self.mode=="eraseminus81"{-std::f64::consts::FRAC_PI_2}else{std::f64::consts::FRAC_PI_2}}else if t==36{0.37}else{-0.51};
            eta=0.9;
            let(c,s)=(theta.cos(),theta.sin());let[x,y,z]=self.state;
            self.state=if erase{[0.,0.,z]}else{[eta*(c*x-s*y),eta*(s*x+c*y),z]};
        }
        let axis=if !active{[0.,0.,1.]}else if t<36{[0.,1.,0.]}else if (t<40)!=(self.mode=="swap81"){[0.,0.,1.]}else{[1.,0.,0.]};
        let lam=if active&&self.mode!="base81"{0.65}else{0.};
        let m=base[ia]+base[ib];let r=base[ia]/m;let e1=(1.-lam)*r;let e0=e1+lam;
        let prepared=self.state;let z=prepared.iter().zip(axis).map(|(a,b)|a*b).sum::<f64>();
        let mut out=base.to_vec();
        if lam>0.{out[ia]=m*(e0+e1+(e0-e1)*z)/2.;out[ib]=m-out[ia];}
        assert!(out[ia]>0.&&out[ib]>0.&&(out.iter().sum::<f64>()-1.).abs()<1e-12);
        let qa=out[ia];
        (out,PhaseRecord65{sources:self.sources,wave,tau:(wave*4) as f64/128.,exposure:0.,rotation:theta,pair_mass:m,population_a:r,pre,prepared,post:prepared,e0,e1,output_a:qa,outcome:usize::MAX,axis81:Some(axis),lambda81:Some(lam),eta81:Some(eta),erase81:Some(erase)})
    }
    fn finish81(&mut self,action:usize,rec:&mut PhaseRecord65){
        rec.outcome=action;
        if rec.lambda81.unwrap()>0. && self.sources.contains(&action){
            let axis=rec.axis81.unwrap();let(e0,e1)=if action==self.sources[0]{(rec.e0,rec.e1)}else{(1.-rec.e0,1.-rec.e1)};
            let v=self.state;let z=v.iter().zip(axis).map(|(a,b)|a*b).sum::<f64>();
            let d=(e0+e1+(e0-e1)*z)/2.;
            let g=(e0*e1).sqrt()/d;let zp=(e0-e1+(e0+e1)*z)/(2.*d);
            self.state=std::array::from_fn(|i|g*(v[i]-z*axis[i])+zp*axis[i]);
        }
        assert!(self.state.iter().map(|x|x*x).sum::<f64>()<=1.+1e-10);rec.post=self.state;
    }
'''
change('mqft_law.rs',[
 ('pub e0:f64, pub e1:f64, pub output_a:f64, pub outcome:usize,','pub e0:f64, pub e1:f64, pub output_a:f64, pub outcome:usize,\n    #[serde(skip_serializing_if="Option::is_none")] pub axis81:Option<[f64;3]>,\n    #[serde(skip_serializing_if="Option::is_none")] pub lambda81:Option<f64>,\n    #[serde(skip_serializing_if="Option::is_none")] pub eta81:Option<f64>,\n    #[serde(skip_serializing_if="Option::is_none")] pub erase81:Option<bool>,'),
 ('["persist_plus","persist_minus","persist_dephased","reset_plus"].contains','["persist_plus","persist_minus","persist_dephased","reset_plus",'+modes+'].contains'),
 ('Self{sources,mode,width,budget,state:[0.,y,0.],exposure:0.,previous_gap:0.}','let state=if mode.ends_with("81"){assert!(width==4&&budget==128);[0.8,0.,0.]}else{[0.,y,0.]};Self{sources,mode,width,budget,state,exposure:0.,previous_gap:0.}'),
 ('    fn prepare(&mut self,t:usize,actions:&[usize],scores:&[f32],base:&[f64])->(Vec<f64>,PhaseRecord65){','    fn prepare(&mut self,t:usize,actions:&[usize],scores:&[f32],base:&[f64])->(Vec<f64>,PhaseRecord65){\n        if self.mode.ends_with("81"){return self.prepare81(t,actions,base);}'),
 ('e0,e1,output_a:qa,outcome:usize::MAX})','e0,e1,output_a:qa,outcome:usize::MAX,axis81:None,lambda81:None,eta81:None,erase81:None})'),
 ('    fn finish(&mut self,action:usize,rec:&mut PhaseRecord65){','    fn finish(&mut self,action:usize,rec:&mut PhaseRecord65){\n        if self.mode.ends_with("81"){self.finish81(action,rec);return;}'),
 ('\n}\nfn lse(a: f64, b: f64)',newimpl+'\n}\nfn lse(a: f64, b: f64)')
])
change('mqft_train_persistent65.rs',[
 ('"persist_dephased","reference"]','"persist_dephased","reference",'+modes+']')
])
change('mqft_arena68.rs',[
 ('mqft_law::{LawSpec,Sampler}','mqft_law::{LawSpec,Sampler,Phase65}'),
 ('pulse74:Option<f64>}','pulse74:Option<f64>,#[serde(default)] phase81:Option<String>,#[serde(default)] pair81:Option<[usize;2]>}'),
 ('let t0=Instant::now();let ev=','if j.phase81.is_some()!=j.pair81.is_some() || j.phase81.as_ref().is_some_and(|m| !['+modes+'].contains(&m.as_str()) || j.arm!="clean" || j.budget!=128 || j.source74.is_some()) || j.pair81.is_some_and(|p|p[0]==p[1] || p.iter().any(|a| !s.idx_to_move(*a).is_some_and(|m|s.legal_moves().contains(&m)))) {return Err("bad phase81".into())}\n let t0=Instant::now();let ev='),
 ('if let (Some(a),Some(h))=(j.source74,j.pulse74){Sampler::new_schedule66','if let (Some(mode),Some(pair))=(&j.phase81,j.pair81){Sampler::new_phase65(law,Phase65::new(pair,mode.clone(),4,j.budget as usize),j.seed)}else if let (Some(a),Some(h))=(j.source74,j.pulse74){Sampler::new_schedule66'),
 ('"pulse74":j.pulse74,"action":action','"pulse74":j.pulse74,"phase81":j.phase81,"pair81":j.pair81,"action":action')
])
# Add focused Rust regression tests to isolated source. No legacy source change.
p=D/'src/mqft_law.rs'
with p.open('a') as f:f.write(r'''
#[cfg(test)] mod phase81_tests {
 use super::*;
 #[test] fn phase81_controls(){
  let a=vec![0,1,2];let s=vec![0.,0.,0.];let p=vec![0.2,0.3,0.5];
  let mut plus=Phase65::new([0,1],"plus81".into(),4,128);
  let mut minus=Phase65::new([0,1],"minus81".into(),4,128);
  let mut ep=Phase65::new([0,1],"eraseplus81".into(),4,128);
  let mut em=Phase65::new([0,1],"eraseminus81".into(),4,128);
  let mut base=Phase65::new([0,1],"base81".into(),4,128);
  let (x,mut r)=plus.prepare(32,&a,&s,&p);let (y,_)=minus.prepare(32,&a,&s,&p);
  let (u,mut e)=ep.prepare(32,&a,&s,&p);let (v,mut g)=em.prepare(32,&a,&s,&p);
  assert!((x[0]-y[0]-.5*.65*.8*.9).abs()<1e-14);
  assert_eq!(u,v);assert!((x[0]+y[0]-2.*u[0]).abs()<1e-14);
  assert_eq!(base.prepare(32,&a,&s,&p).0,p);
  plus.finish(0,&mut r);ep.finish(0,&mut e);em.finish(0,&mut g);
  assert_eq!(ep.state,em.state);assert!(plus.state!=ep.state);
  assert_eq!(plus.prepare(36,&a,&s,&p).1.axis81,Some([0.,0.,1.]));
  assert_eq!(plus.prepare(40,&a,&s,&p).1.axis81,Some([1.,0.,0.]));
  assert_eq!(plus.prepare(44,&a,&s,&p).0,p);
 }
}
'''.replace('-.5*.65*.8*.9','-0.5*0.65*0.8*0.9'))
# Rust source cannot use leading-dot float literals; test rewritten above.
old=(A/'src/mqft_law.rs').read_text();new=p.read_text();patches['mqft_law.rs']=dict(original_sha=sha(A/'src/mqft_law.rs'),modified_sha=sha(p),diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/mqft_law.rs',tofile='b/mqft_law.rs')))
dump(S/'PHASE_PATCH81.json',dict(parent=parent,base_source_files={str(f.relative_to(A)):sha(f) for f in sorted(A.rglob('*')) if f.is_file()},patches=patches,scope='isolated phase81 module, legacy paths unchanged'))
(S/'phase81.patch').write_text(''.join(x['diff'] for x in patches.values()))
env=dict(os.environ,CARGO_TARGET_DIR=str(P/'backend/target'),CARGO_PROFILE_RELEASE_LTO='false',CARGO_PROFILE_RELEASE_CODEGEN_UNITS='8',CARGO_BUILD_JOBS='8')
cmd=['cargo','build','--offline','--release','--features','mqft-train','--bin','mqft_train','--manifest-path',str(D/'Cargo.toml')]
with (S/'logs/build81.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
shutil.copy2(Path(env['CARGO_TARGET_DIR'])/'release/mqft_train',S/'mqft_phase81')
test=['cargo','test','--offline','--release','--features','mqft-train','--lib','--manifest-path',str(D/'Cargo.toml'),'phase81_controls']
with (S/'logs/test81.log').open('w') as f:subprocess.run(test,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
dump(S/'BUILD81.json',dict(status='PASS_BUILD_AND_FOCUSED_TEST',binary_sha=sha(S/'mqft_phase81'),patch_sha=sha(S/'phase81.patch'),build=cmd,test=test,build_log_sha=sha(S/'logs/build81.log'),test_log_sha=sha(S/'logs/test81.log')))
print('BUILD',sha(S/'mqft_phase81'),flush=True)
