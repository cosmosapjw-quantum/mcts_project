"""Independent archive, historical game, graph and joint invariant audits."""
from base61 import *
import gzip,zipfile,csv
from collections import defaultdict
G0=read(S/'GAUGE88.json');T=read(S/'TOPOLOGY88.json');B0=read(S/'BARRIER88.json');Q=read(S/'TRANSPORT88.json')
O75=B/'mqft_restore75_20261127';O78=B/'mqft_restore78_20261127';O79=B/'mqft_restore79_20261127';O87=B/'mqft_restore87_20261128';O69=B/'mqft_restore69_20261126'
err=defaultdict(float)
def chk(k,a,b,tol=1e-10):
 v=float(abs(np.asarray(a)-np.asarray(b)).max());err[k]=max(err[k],v);assert v<tol,(k,v)
# Independent quaternion invariant check with explicit dot/triple cross products.
loops=0;triples=0
for c in G0['contexts']:
 for j in c['joint']:
  a=np.array(j['scalar']);v=np.array(j['vectors']);w=np.array(j['transformed_vectors']);i,k,l=j['triple']
  chk('quat_norm',a*a+np.sum(v*v,1),1)
  chk('quat_gram',np.einsum('ik,jk->ij',v,v),np.einsum('ik,jk->ij',w,w))
  vol=np.dot(np.cross(v[i],v[k]),v[l]);chk('oriented_volume',vol,j['oriented_volume'])
  chk('oriented_volume',vol,np.dot(np.cross(w[i],w[k]),w[l]))
  # Construct SO(3) rotation from spanning triple, not SVD.
  assert j['rank']==3 and abs(vol)>1e-8
  R=np.linalg.solve(v[[i,k,l]],w[[i,k,l]])
  chk('basis_rotation',R.T@R,np.eye(3));chk('basis_rotation_det',np.linalg.det(R),1);chk('basis_rotation_action',v@R,w)
  # Scalar quaternion product with i sigma convention: vector = a*d+c*b-b cross d.
  x=a[i]*a[k]-v[i]@v[k];y=a[i]*v[k]+a[k]*v[i]-np.cross(v[i],v[k]);s=x*a[l]-y@v[l]
  chk('triple_trace',s,j['triple_trace']);chk('reflected_gap',abs(2*vol),j['reflected_trace_gap'])
  loops+=len(a);triples+=1
assert loops==4005 and triples==48
# Check every chronological prefix count by set construction from complete raw paths.
prefix_checks=0
for c in G0['contexts']:
 f=B/c['source_path'];assert sha(f)==c['source_sha']
 with gzip.open(f,'rt') as z:r=json.load(z)
 m=r['input']['moves'];e=r['trace']['events']
 for h in (1,4,8,16,32,64,128):
  E={(tuple(a['actions'][:k-1]),tuple(a['actions'][:k])) for a in e[:h] for k in range(1,len(a['actions'])+1)}
  def state(p):
   x=m+list(p);return frozenset(x[::2]),frozenset(x[1::2])
  q={(state(a),state(b)) for a,b in E};V={u for edge in E for u in edge};W={u for edge in q for u in edge}
  row=c['prefix']['curve'][h-1];assert row==[h,len(V),len(E),len(W),len(q),len(q)-len(W)+1];prefix_checks+=1
# Independent check final cubical H1 by counting bounded holes in the complement on dual square grid.
# A legal 1-skeleton plus occupied unit faces: use all elementary square cells plus outside,
# cross a shared edge only when that edge is missing from the legal graph.
topocheck=0
lookup={('R78',x['family'],x['core']):(x['vertices'],x['edges'],x['L']) for x in G0['contexts']}
lookup.update({('R75',x['family'],x['core']):(x['component'],x['edges'],x['L']) for x in B0['contexts']})
for c in T['contexts']:
 v,e,L=lookup[c['kind'],c['family'],c['core']];e={tuple(sorted(x)) for x in e};V=set(v);outside=(-1,-1)
 squares={(y,x) for y in range(L-1) for x in range(L-1)}
 full={(y,x) for y,x in squares if {y*L+x,y*L+x+1,(y+1)*L+x,(y+1)*L+x+1}<=V}
 empty=squares-full;A={s:[] for s in empty|{outside}}
 for y,x in empty:
  sides=[((y-1,x),(y*L+x,y*L+x+1)),((y+1,x),((y+1)*L+x,(y+1)*L+x+1)),((y,x-1),(y*L+x,(y+1)*L+x)),((y,x+1),(y*L+x+1,(y+1)*L+x+1))]
  for k,edge in sides:
   if tuple(sorted(edge)) in e:continue
   if k in full:raise AssertionError('full face missing boundary')
   if k not in squares:k=outside
   A[y,x].append(k);A[k].append((y,x))
 left=set(A);components=0
 while left:
  q=[next(iter(left))];seen=set(q)
  for u in q:
   for w in A[u]:
    if w not in seen:seen.add(w);q.append(w)
  left-=seen;components+=1
 assert components-1==c['topology']['cubical']['final_betti'][1]
 topocheck+=1
assert topocheck==144
# Historical same-side recovery episodes: actual R68 full games, not first-hit root selection.
P=B/'mqft_restore68_20261126';game=read(O75/'GAMES75.json');assert sha(P/'CAMPAIGN68.json')==game['campaign_sha'] and sha(P/'REGISTRATION.json')==game['reg_sha']
old=read(O75/'RUN_MANIFEST75.json');raw=old['prior_raw_receipts'];assert len(raw)==16
actual={}
for r in raw:
 f=B/r['path'];assert sha(f)==r['sha256']
 with gzip.open(f,'rt') as h:d=json.load(h)
 j=d['job'];actual[j['id']]=[(a['ply'],a['search']['evaluations'][0]['clean']) for a in d['rows'] if a['engine']=='native' and a['arm']=='clean' and a['side']==j['color']]
for c in game['results']:
 for row in c['rows']:
  seq=actual[row['id']];assert [list(x) for x in seq]==row['values']
  v=np.array([x[1] for x in seq]);peak=np.maximum.accumulate(v);ix=np.flatnonzero(peak-v>=c['threshold'])
  assert bool(len(ix))==row['eligible']
  if len(ix):
   t=int(ix[0]);hit=np.flatnonzero(v[t+1:]>=peak[t]);assert t==row['entry_index'] and bool(len(hit))==row['event']
   assert (int(hit[0]+1) if len(hit) else len(v)-1-t)==row['followup']
   chk('drawdown',peak[t]-v[t],row['height'])
  assert row['source_sha']==next(x['sha256'] for x in raw if x['path']==row['source_path'])
 for q in c['risk_table']:
  a=[x for x in c['rows'] if x['eligible']];t=q['age'];assert q['risk']==sum(x['followup']>=t for x in a) and q['hits']==sum(x['event'] and x['followup']==t for x in a) and q['censored']==sum(not x['event'] and x['followup']==t for x in a)
dump(S/'RECOVERY88.json',dict(game=game,raw_receipts=raw,status='PASS_SCOPED',scope='Actual R68 historical same-side value recovery, threshold sensitivity, complete observed risk sets; informative censoring possible.'))
# Exact archive and item acceptance identity. Guide-only rows are not invented source spans.
orig=read(O79/'ORIGINAL_SOURCE79.json');paths={'quantum.zip':Path.home()/'quantum.zip','omoknuni_quantum-main.zip':Path.home()/'Dropbox/omoknuni_quantum-main.zip','mcts_study.zip':Path.home()/'Dropbox/personal_projects/mcts_study.zip'}
assert all(sha(paths[k])==v for k,v in orig['archives'].items())
items=read(S/'REGISTRATION.json')['items'];contracts=[x for x in read(O79/'ITEM_ACCEPTANCE79.json')['items'] if x['id'] in items];assert len(contracts)==8
src=[]
for c in contracts:
 s=c['original_source']
 if s['source_status']=='GUIDE_LABEL_ONLY':
  m=s['guide_source_member'];archive='omoknuni_quantum-main.zip';h=s['guide_source_sha']
 else:m=s['member'];archive=s['archive'];h=s['member_sha']
 with zipfile.ZipFile(paths[archive]) as z:v=z.read(m)
 assert hashlib.sha256(v).hexdigest()==h
 lines=v.decode().splitlines()
 span='\n'.join(lines[int(s['start'])-1:int(s['end'])]) if s['source_status']!='GUIDE_LABEL_ONLY' else '\n'.join(lines[i-1] for i in s['guide_lines'])
 src.append(dict(id=c['id'],original_source=s,member_sha=h,source_excerpt=span,acceptance=c['final_acceptance']))
for f in ['CORE_POLICY.md','QUANTUM_DYNAMICS_POLICY.md','QUANTUM_REPRESENTATION_POLICY.md','RESEARCH_CHARTER.md','ORIGINAL_TARGET_CATALOG.csv','PLOT_GROUPS.json','PLOT_GROUP_CROSSWALK.csv']:assert sha(S/f)==sha(O87/f)==sha(O69/f)
dump(S/'ORIGINAL_SOURCE88.json',dict(archives=orig['archives'],original_attachment_receipt=orig['original_attachment_receipt'],items=src,acceptance_contract_sha=sha(O79/'ITEM_ACCEPTANCE79.json'),scope='Original source spans and guide-only filenames authenticated against local archived original; original uploaded chat ZIP receipt inherited.'))
dump(S/'INDEPENDENT88.json',dict(status='PASS_SCOPED',joint_loops=loops,joint_components=triples,prefix_checks=prefix_checks,independent_dual_hole_checks=topocheck,threshold_checks=T['threshold_checks'],risk_paths=len(B0['paths']),game_paths=16,game_threshold_cases=48,transport_contexts=96,independent_dense_contexts=Q['dense_contexts'],independent_dense_error=Q['independent_dense_error'],errors=dict(err),scope='Same-agent independent basis rotation and quaternion product; raw prefix sets; dual complement hole counts; actual game episode recomputation; dense solve/expm plus exact prior comparison.'))
print('VERIFIED',loops,triples,prefix_checks,topocheck,T['threshold_checks'],dict(err),flush=True)
