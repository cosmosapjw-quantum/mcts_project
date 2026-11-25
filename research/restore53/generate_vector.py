"""Opt-in four-sector clock field (three physical tangent directions).
Exact R52 source family, old executables unchanged."""
from pathlib import Path
import json,hashlib,shutil,sys
S=Path(sys.argv[1]);L=int(sys.argv[2]);B=S.parent;F=B/'mqft_restore52_20261126/family'/f'L{L}';out=S/'family'/f'L{L}'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((F/'SOURCE_MANIFEST.json').read_text());assert all(sha(F/f)==v for f,v in m['files'].items())
assert not out.exists()
shutil.copytree(F,out,ignore=shutil.ignore_patterns('__pycache__','*.pyc','SOURCE_MANIFEST.json'))
b=out/'backend';p=out/'python/mqft';old=f'mcts_demo_l{L}_sector52';new=f'mcts_demo_l{L}_vector53'
def edit(f,pairs):
 s=f.read_text()
 for a,z in pairs:
  assert a in s,(f,a);s=s.replace(a,z)
 f.write_text(s)
for f in (b/'Cargo.toml',b/'Cargo.lock',b/'src/main.rs'):edit(f,[(old,new)])
edit(b/'src/mqft_law.rs',[
 ('    pub source_actions: Vec<usize>,\n}', '''    pub source_actions: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sector_schedule: Vec<[f64;4]>,
}'''),
 ('        let has_source=source.is_some()', '''        if !self.sector_schedule.is_empty() &&
            (self.sector_schedule.len()!=budget as usize || source.is_some() ||
             !self.source_actions.is_empty() || self.logit_h!=0. ||
             !self.logit_schedule.is_empty() || self.proposal!="p" ||
             self.sector_schedule.iter().flatten().any(|x|!x.is_finite() || x.abs()>10.)) {
            return Err("vector field requires budget x 4 finite logits |h|<=10, no scalar/group field, proposal p".into());
        }
        let has_source=source.is_some()'''),
 ('    pub alpha: f64,','    #[serde(skip_serializing_if = "Option::is_none")]\n    pub sector_logits: Option<[f64;4]>,\n    pub alpha: f64,'),
 ('    h: f64,\n) -> Vec<f64>', '    h: f64,\n    sector_logits: Option<[f64;4]>,\n) -> Vec<f64>'),
 ('                beta * s as f64 + if Some(a) == source || source_actions.binary_search(&a).is_ok() { h } else { 0. }',f'''                let v = sector_logits.map_or(0., |v| v[2*usize::from(a/{L}>={L//2})+usize::from(a%{L}>={L//2})]-v[3]);
                beta * s as f64 + if Some(a) == source || source_actions.binary_search(&a).is_ok() {{ h }} else {{ v }}'''),
 ('        let beta_p = self.spec.beta_p','        let sector_logits = self.spec.sector_schedule.get(t).copied();\n        let beta_p = self.spec.beta_p'),
 ('            h,\n        );','            h,\n            sector_logits,\n        );'),
 ('                source_actions: self.spec.source_actions.clone(),','                source_actions: self.spec.source_actions.clone(),\n                sector_logits,')
])
edit(b/'src/mqft.rs',[(f'mqft.quartz.trace.v4.L{L}',f'mqft.quartz.trace.v5.L{L}'),
 ('            if max_beta * span + law.logit_schedule.iter().map(|x|x.abs()).fold(law.logit_h.abs(),f64::max) > 640. {','''            let field_span = law.sector_schedule.iter().map(|v|v.iter().copied().fold(f64::NEG_INFINITY,f64::max)-v.iter().copied().fold(f64::INFINITY,f64::min)).fold(0.,f64::max);
            if max_beta * span + law.logit_schedule.iter().map(|x|x.abs()).fold(law.logit_h.abs(),f64::max) + field_span > 640. {''')])
edit(p/'__init__.py',[(f'mqft.quartz.trace.v4.L{L}',f'mqft.quartz.trace.v5.L{L}')])
edit(p/'campaign.py',[(f'mqft.campaign.v4.L{L}',f'mqft.campaign.v5.L{L}'),
 ('"logit_schedule","source_actions"}','"logit_schedule","source_actions","sector_schedule"}'),
 ('            if max(law["beta_p"]+law["beta_q"])','''            vector=law.get("sector_schedule",[])
            if vector:
                if kind!="roots" or law["proposal"]!="p" or h!=0 or hh!=0 or sched or group or pos.get("source_action") is not None or not isinstance(vector,list) or len(vector)!=budget or any(not isinstance(v,list) or len(v)!=4 for v in vector):raise ValueError("invalid vector field")
                vector=[[finite(x,"sector logit",-10,10) for x in v] for v in vector]
                law["sector_schedule"]=vector
            field_span=max((max(v)-min(v) for v in vector),default=0.)
            if max(law["beta_p"]+law["beta_q"])'''),
 ('+max([abs(hh)]+[abs(v) for v in sched])>640','+max([abs(hh)]+[abs(v) for v in sched])+field_span>640')])
edit(p/'path_law.py',[
 ('def softmax(s,beta,active,actions,source,h):','def softmax(s,beta,active,actions,source,h,sector_logits=None):'),
 ('    z=beta*np.asarray(s,float)+h*mask','''    z=beta*np.asarray(s,float)+h*mask
    if sector_logits is not None:
        a=np.asarray(actions);v=np.asarray(sector_logits,float)
        z+=(v-v[3])[2*(a//'''+str(L)+'''>='''+str(L//2)+''').astype(int)+(a%'''+str(L)+'''>='''+str(L//2)+''').astype(int)]'''),
 ('                field=group if group else source','''                field=group if group else source
                field_v=law["sector_schedule"][t] if law.get("sector_schedule") else None
                if a.get("sector_logits")!=field_v:raise ValueError("vector field identity")'''),
 ('active,order,field,h);q=softmax(a["scores"],qq,active,order,field,h)','active,order,field,h,field_v);q=softmax(a["scores"],qq,active,order,field,h,field_v)')])
edit(p/'analysis.py',[(f'mqft.analysis.v4.L{L}',f'mqft.analysis.v5.L{L}')])
files={str(x.relative_to(out)):sha(x) for x in sorted(out.rglob('*')) if x.is_file()}
(out/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(L=L,parent_manifest_sha=sha(F/'SOURCE_MANIFEST.json'),files=files),indent=2)+'\n')
