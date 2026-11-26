"""R68 historical same-side actual value drawdown, all eligible entrants retained."""
from base61 import *
import gzip
O=B/'mqft_restore68_20261126';A=read(O/'CAMPAIGN68.json');R=read(O/'REGISTRATION.json')
def load(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def episodes(row,threshold):
 j=row['job'];p=O/'data'/(j['id']+'.json.gz');assert sha(p)==row['sha256']
 d=load(p);assert d['job']==j and d['status']==row['status']
 seq=[(r['ply'],r['search']['evaluations'][0]['clean']) for r in d['rows'] if r['engine']=='native' and r['arm']=='clean' and r['side']==j['color']]
 assert len(seq)==row['native'] and len(seq)>0
 assert all(b[0]-a[0]==2 for a,b in zip(seq,seq[1:]))
 vs=np.array([a[1] for a in seq]);assert np.all(abs(vs)<=1+1e-7)
 peak=vs[0];entry=None
 for i,v in enumerate(vs):
  if peak-v>=threshold:entry=i;break
  peak=max(peak,v)
 if entry is None:return dict(id=j['id'],core=j['core'],color=j['color'],values=seq,eligible=False,status=d['status'],source_path=str(p.relative_to(B)),source_sha=row['sha256'])
 hit=np.flatnonzero(vs[entry+1:]>=peak);age=int(hit[0]+1) if len(hit) else len(vs)-1-entry
 return dict(id=j['id'],core=j['core'],color=j['color'],values=seq,eligible=True,entry_index=entry,peak=float(peak),entry_value=float(vs[entry]),height=float(peak-vs[entry]),event=bool(len(hit)),followup=age,horizon=len(vs)-1-entry,status=d['status'],source_path=str(p.relative_to(B)),source_sha=row['sha256'])
def main():
 rows=[x for x in A if x['job']['match']=='ext_clean'];assert len(rows)==16
 out=[]
 for threshold in [.1,.2,.3]:
  r=[episodes(x,threshold) for x in rows];a=[x for x in r if x['eligible']]
  # Risk table is descriptive; informative game termination and variable follow-up.
  tab=[];S0=1.
  for t in range(1,1+max((x['followup'] for x in a),default=0)):
   Y=sum(x['followup']>=t for x in a);d=sum(x['event'] and x['followup']==t for x in a);c=sum((not x['event']) and x['followup']==t for x in a)
   if Y:S0*=1-d/Y
   tab.append(dict(age=t,risk=Y,hits=d,censored=c,descriptive_KM=S0))
  out.append(dict(threshold=threshold,eligible=len(a),events=sum(x['event'] for x in a),censored=sum(not x['event'] for x in a),no_entry=len(r)-len(a),entry_zero_followup=sum(x['followup']==0 for x in a),rows=r,risk_table=tab))
 res=dict(scope='Retrospective R68 ext_clean same-player root evaluator score recovery, all entrants, fixed entry peak, termination/finite exposure censoring potentially informative; no causal or quantum claim; .2 primary, .1/.3 sensitivity',campaign_sha=sha(O/'CAMPAIGN68.json'),reg_sha=sha(O/'REGISTRATION.json'),source_games=16,results=out)
 dump(S/'GAMES75.json',res)
 print([(x['threshold'],x['eligible'],x['events'],x['censored'],x['no_entry']) for x in out],flush=True)
if __name__=='__main__':main()
