from base61 import *
import gzip,zipfile
B0=read(S/'BARRIER88.json');rows=[];err=0.
for c in B0['contexts']:
 p=B/c['source_path'];assert sha(p)==c['source_sha']
 with gzip.open(p,'rt') as f:r=json.load(f)
 t=r['trace'];cut=next(x for x in t['cuts'] if x['k']==32);n=np.array(cut['n'],np.float32);w=np.array(cut['w'],np.float32);prior=np.array(cut['prior'],np.float32)
 q=np.divide(w,n,out=np.zeros_like(w),where=n>0);u=q+np.float32(1.4)*prior*np.sqrt(np.float32(32))/(1+n)
 law=t['events'][32]['root_selection']['law'];i=law['actions'];assert set(i)==set(range(c['L']**2))-set(c['moves']);e=float(abs(u[i]-law['scores']).max());err=max(err,e);assert e<2e-6
 rows.append(dict(family=c['family'],core=c['core'],vertices=i,L=c['L'],moves=c['moves'],Q=q[i].tolist(),visited=(n[i]>0).tolist(),score=law['scores'],component=c['component'],source=c['source'],target=c['target'],path=c['path'],potential=[-v-max(-float(u[c['source']]),-float(u[c['target']])) for v in law['scores']],source_sha=c['source_sha'],source_path=c['source_path']))
dump(S/'LANDSCAPE88.json',dict(contexts=rows,max_score_error=err,scope='Actual signed Q=W/N for visited actions, zero initialization unvisited, PUCT score U, H32 actual cut. G18 projection uses actual board coordinates, no interpolation across occupied sites.'))
# Explicit producer and study provenance; archive hashes already checked by verify88.
src=read(S/'ORIGINAL_SOURCE88.json');home=Path.home();items=[]
with zipfile.ZipFile(home/'Dropbox/omoknuni_quantum-main.zip') as z:
 for f,segments in [('gauge_policy.py',[]),('topological_analysis.py',[(268,358),(398,452)]),('tunneling.py',[(250,271)])]:
  m='omoknuni_quantum-main/python/mcts/quantum/phenomena/'+f;v=z.read(m);ls=v.decode().splitlines()
  items.append(dict(archive='omoknuni_quantum-main.zip',member=m,sha256=hashlib.sha256(v).hexdigest(),segments=[dict(start=a,end=b,text='\n'.join(ls[a-1:b])) for a,b in segments]))
with zipfile.ZipFile(home/'Dropbox/personal_projects/mcts_study.zip') as z:
 for m,segments in [('mcts_study/v5.0/new_quantum_mcts.md',[(44,55),(116,125),(191,204)]),('mcts_study/v6.0/quantum_mcts_foundation.md',[(43,62)])]:
  v=z.read(m);ls=v.decode().splitlines();items.append(dict(archive='mcts_study.zip',member=m,sha256=hashlib.sha256(v).hexdigest(),segments=[dict(start=a,end=b,text='\n'.join(ls[a-1:b])) for a,b in segments]))
prior=read(B/'mqft_restore78_20261127/ORIGINAL_GAUGE_AUDIT78.json');assert items[0]['sha256']==prior['source_member']['sha256']
dump(S/'SOURCE_CORRECTIONS88.json',dict(sources=items,prior_gauge_defects=prior['source_defects'],topology_defects=['Original prev_betti_0 derives from persistent_features not populated until after the loop, so no correct component tracking. Net Betti counts do not identify persistence pairings.','Original routine emits H0 only; distance threshold tied to score threshold has no declared common units.','Local extrema assigned Morse index and stable/unstable without Hessian or flow.'],tunneling_defect='Original WKB proxy uses value barrier duration and temperature rather than a defined spatial kinetic term/incident energy. R75 declared graph reference retained.',study_scope='v5 tree topology alone does not imply diagonal Hessian (categorical Hessian from R78); v6 search topology dynamically constructed. Pruning suppression law not examined here.'))
print('LANDSCAPE',len(rows),err,flush=True)
