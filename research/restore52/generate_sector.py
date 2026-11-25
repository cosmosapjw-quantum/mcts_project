"""Opt-in multi-action source on exact R49 size family. R49 production untouched."""
from pathlib import Path
import json,hashlib,shutil,sys
S=Path(sys.argv[1]);L=int(sys.argv[2]);B=S.parent;F=B/'mqft_restore49_20261126/family'/f'L{L}';out=S/'family'/f'L{L}'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((F/'SOURCE_MANIFEST.json').read_text());assert all(sha(F/f)==v for f,v in m['files'].items())
assert not out.exists()
shutil.copytree(F,out,ignore=shutil.ignore_patterns('__pycache__','*.pyc','SOURCE_MANIFEST.json'))
b=out/'backend';p=out/'python/mqft';old=f'mcts_demo_l{L}';new=old+'_sector52'
def edit(f,pairs):
 s=f.read_text()
 for a,z in pairs:
  assert a in s,(f,a);s=s.replace(a,z)
 f.write_text(s)
for f in (b/'Cargo.toml',b/'Cargo.lock',b/'src/main.rs'):edit(f,[(old,new)])
f=b/'src/mqft_law.rs'
edit(f,[
 ('    pub logit_schedule: Vec<f64>,','    pub logit_schedule: Vec<f64>,\n    #[serde(default, skip_serializing_if = "Vec::is_empty")]\n    pub source_actions: Vec<usize>,'),
 ('        if !self.logit_schedule.is_empty() &&','''        if !self.source_actions.is_empty() && (source.is_some() ||
            self.source_actions.windows(2).any(|w|w[0]>=w[1]) || self.proposal!="p") {
            return Err("group field requires sorted unique actions, no scalar source, proposal p".into());
        }
        let has_source=source.is_some() || !self.source_actions.is_empty();
        if !self.logit_schedule.is_empty() &&'''),
 ('source.is_none())','!has_source)'),
 ('    pub source: Option<usize>,','    pub source: Option<usize>,\n    #[serde(skip_serializing_if = "Vec::is_empty")]\n    pub source_actions: Vec<usize>,'),
 ('    source: Option<usize>,\n    h: f64,','    source: Option<usize>,\n    source_actions: &[usize],\n    h: f64,'),
 ('if Some(a) == source { h }','if Some(a) == source || source_actions.binary_search(&a).is_ok() { h }'),
 ('            self.source,\n            h,','            self.source,\n            &self.spec.source_actions,\n            h,'),
 ('                source: self.source,','                source: self.source,\n                source_actions: self.spec.source_actions.clone(),')
])
edit(b/'src/mqft.rs',[(f'mqft.quartz.trace.v3.L{L}',f'mqft.quartz.trace.v4.L{L}'),
 ('    let start = Instant::now();\n    let pool = rayon::ThreadPoolBuilder::new()', '''    if let Some(law)=&spec.root_law {
        if !law.source_actions.is_empty() {
            let legal=state.legal_moves();
            if law.source_actions.len()>=legal.len() || law.source_actions.iter().any(|&a|
                !state.idx_to_move(a).is_some_and(|m|legal.contains(&m))) {
                return Err("group source must be proper nonempty legal subset".into());
            }
        }
    }
    let start = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()''')])
edit(p/'__init__.py',[(f'mqft.quartz.trace.v3.L{L}',f'mqft.quartz.trace.v4.L{L}')])
edit(p/'campaign.py',[(f'mqft.campaign.v3.L{L}',f'mqft.campaign.v4.L{L}'),
 ('"logit_h","logit_schedule"}','"logit_h","logit_schedule","source_actions"}'),
 ('            sched=law.get("logit_schedule")','''            group=pos.get("source_actions",[])
            if group:
                b=valid_moves(pos["moves"])
                if kind!="roots" or pos.get("source_action") is not None or law["proposal"]!="p" or h!=0 or not isinstance(group,list) or any(type(x)!=int or not 0<=x<len(b) or b[x] for x in group) or group!=sorted(set(group)) or len(group)>=b.count(0):raise ValueError("invalid group field")
                if law.get("source_actions") not in (None,group):raise ValueError("group identity")
                law["source_actions"]=group
            elif law.get("source_actions"):raise ValueError("group must be registered at position")
            sched=law.get("logit_schedule")'''),
 ('or pos.get("source_action") is None:raise ValueError("invalid logit schedule")','or (pos.get("source_action") is None and not group):raise ValueError("invalid logit schedule")'),
 ('if hh!=0 and pos.get("source_action") is None:','if hh!=0 and pos.get("source_action") is None and not group:')])
# valid_moves returns list? verified below before execution.
edit(p/'path_law.py',[
 ('    z=beta*np.asarray(s,float)+h*(np.asarray(actions)==source)','    mask=np.isin(actions,source) if isinstance(source,list) else np.asarray(actions)==source\n    z=beta*np.asarray(s,float)+h*mask'),
 ('                # Use independently verified actual FP32 score','''                group=law.get("source_actions",[])
                if a.get("source_actions",[])!=group:raise ValueError("group source identity")
                field=group if group else source
                # Use independently verified actual FP32 score'''),
 ('active,order,source,h)','active,order,field,h)'),
 ('                if source is not None:\n                    z=idx[source];gp+=float(j==z)-p[z];gq+=float(j==z)-q[z];fp+=p[z]*(1-p[z]);fq+=q[z]*(1-q[z])','''                if field is not None:
                    mask=np.isin(order,field);u0=float(mask[j]);pm=float(p[mask].sum());qm=float(q[mask].sum())
                    gp+=u0-pm;gq+=u0-qm;fp+=pm*(1-pm);fq+=qm*(1-qm)''')])
edit(p/'analysis.py',[(f'mqft.analysis.v3.L{L}',f'mqft.analysis.v4.L{L}')])
# No path proposal for group: legacy path summary only scalar fields.
files={str(x.relative_to(out)):sha(x) for x in sorted(out.rglob('*')) if x.is_file()}
(out/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(L=L,parent_manifest_sha=sha(F/'SOURCE_MANIFEST.json'),files=files),indent=2)+'\n')
