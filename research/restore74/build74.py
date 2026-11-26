from base61 import *
import subprocess,shutil,os,difflib
P=B/'mqft_restore39_20261125/repo';parent='d929a233b26f45669242cf5ee10d8c279203692b'
assert subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip()==parent
D=S/'pulse_src';D.mkdir();shutil.copytree(P/'backend/src',D/'src');shutil.copy2(P/'backend/Cargo.toml',D);shutil.copy2(P/'backend/Cargo.lock',D)
patches={}
def change(f,pairs):
 p=D/'src'/f;old=p.read_text();new=old
 for a,b in pairs:
  assert new.count(a)==1,(f,a,new.count(a));new=new.replace(a,b)
 p.write_text(new);patches[f]=dict(original_sha=sha(P/'backend/src'/f),modified_sha=sha(p),diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/backend/src/'+f,tofile='b/backend/src/'+f)))
change('mqft_train_schedule66.rs',[
 ('["forward","counter"]','["forward","counter","zero74","plus74","minus74"]'),
 ('let hs=(0..128).map(|t|{let h=-0.45+0.9*(t/4) as f64/31.;if j.arm=="forward"{h}else{-h}}).collect();','let hs=(0..128).map(|t|{if j.arm.ends_with("74"){if (32..36).contains(&t){match j.arm.as_str(){"plus74"=>0.35,"minus74"=>-0.35,_=>0.}}else{0.}}else{let h=-0.45+0.9*(t/4) as f64/31.;if j.arm=="forward"{h}else{-h}}}).collect();')
])
change('mqft_arena68.rs',[
 ('budget:u32,arm:String}', 'budget:u32,arm:String,#[serde(default)] source74:Option<usize>,#[serde(default)] pulse74:Option<f64>}'),
 ('if s.is_terminal(){return Err("terminal root".into())}', 'if s.is_terminal(){return Err("terminal root".into())}\n if j.source74.is_some()!=j.pulse74.is_some() || j.source74.is_some_and(|a|j.arm!="clean"||j.budget!=128||!s.idx_to_move(a).is_some_and(|m|s.legal_moves().contains(&m))) || j.pulse74.is_some_and(|h| ![-0.35,0.,0.35].contains(&h)){return Err("bad pulse74".into())}'),
 ('Sampler::new_field(law,vec![0.;225],j.seed)', 'if let (Some(a),Some(h))=(j.source74,j.pulse74){Sampler::new_schedule66(law,a,(0..128).map(|t|if (32..36).contains(&t){h}else{0.}).collect(),j.seed)}else{Sampler::new_field(law,vec![0.;225],j.seed)}'),
 ('"budget":j.budget,"action":action','"budget":j.budget,"source74":j.source74,"pulse74":j.pulse74,"action":action')
])
dump(S/'PULSE_PATCH74.json',dict(parent=parent,patches=patches,scope='isolated experimental source pulse, canonical code untouched'))
(S/'pulse74.patch').write_text(''.join(x['diff'] for x in patches.values()))
env=dict(os.environ,CARGO_TARGET_DIR=str(P/'backend/target'),CARGO_PROFILE_RELEASE_LTO='false',CARGO_PROFILE_RELEASE_CODEGEN_UNITS='8',CARGO_BUILD_JOBS='8')
cmd=['cargo','build','--offline','--release','--features','mqft-train','--bin','mqft_train','--manifest-path',str(D/'Cargo.toml')]
with (S/'logs/build74.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
shutil.copy2(Path(env['CARGO_TARGET_DIR'])/'release/mqft_train',S/'mqft_pulse74')
print('BUILD',sha(S/'mqft_pulse74'),flush=True)
