"""Independent aggregate of actual R54 paths, exact conditional references,
conditional quantum transfer and reverse-work checks."""
from pathlib import Path
import json,gzip,math,statistics as st
from collections import defaultdict
import numpy as np
from mqft.client import sha256
from mqft.campaign import atomic_json
from mqft.physics54 import lastwave,arm54,exact_count_reference,transfer_channel,crooks_distribution,complex_work_tilt
from mqft.physics52 import tilted_cycle
from mqft.physics50 import noise_decomposition
S=Path(__file__).resolve().parent;B=S.parent;O=B/'mqft_restore53_20261126'
read=lambda p:json.loads(p.read_text())
def main():
 a=[read(S/f'PHYSICS_L{L}.json') for L in (13,21)];native=sum([x['native'] for x in a],[]);games=sum([x['games'] for x in a],[]);groups=sum([x['groups'] for x in a],[]);T=sum([x['temperature'] for x in a],[])
 counts={f"L{x['L']}":x['counts'] for x in a}
 assert len(native)==848 and len(games)==2 and len(groups)==13
 assert sum(v.get('exact_prefix_pairs',0) for c in counts.values() for v in c.values())==768
 assert sum(v.get('exact_prefix_draws',0) for c in counts.values() for v in c.values())==11264
 seeds=defaultdict(list)
 for r in native:seeds[r['seed']].append(r)
 assert len(seeds)==80 and all(len({(r['L'],r['campaign'],r['position'],r['law'],r['B']) for r in v})==1 for v in seeds.values())
 oldpub=read(O/'PUBLICATION_RECEIPT.json');oldresult=read(O/'PHYSICS_RESULTS.json');assert sha256(O/'PHYSICS_RESULTS.json')==oldpub['physics_results_sha256']
 for L in (13,21):assert sha256(O/f'PHYSICS_L{L}.json')==oldresult['size_results_sha'][f'L{L}']
 old=read(O/'PHYSICS_L13.json')['native'];prev=defaultdict(list)
 for r in old:
  if r['campaign']=='g0' and r['law']=='iid' and tuple(r['arm']) in ((-1,0,0,0),(3,-1,0,0),(3,1,0,0)):
   prev[arm54(r)].append(r)
 assert len(prev)==3 and all(len(v)==8 for v in prev.values())
 oldref=lastwave(prev)
 for L in (13,21):assert not {r['seed'] for r in native}&{r['seed'] for r in read(O/f'PHYSICS_L{L}.json')['native']}
 base=defaultdict(list)
 for r in native:
  if r['arm']==[0.,0.,0.]:base[r['L'],r['campaign'],r['position'],r['law']].append(r)
 assert len(base)==13 and sum(len(v) for v in base.values())==80
 exact=[];noise=[];quantum=[];thermal=[];states={};samples={}
 for (L,c,pos,law),rr in sorted(base.items()):
  assert all(r['B']==rr[0]['B'] for r in rr)
  A=np.asarray([r['I'] for r in rr]);P=np.asarray([r['P'] for r in rr])
  noise.append(dict(L=L,campaign=c,position=pos,law=law,**noise_decomposition(A,P)))
  if law=='iid':
   for r in rr:
    q=exact_count_reference(r['P'][-4],np.sum(r['I'][-4:],0))
    exact.append(dict(L=L,campaign=c,position=pos,job=r['job'],B=r['B'],**q))
  if c not in ('new','precision'):continue
  for r in rr:
   n1=int(sum(np.asarray(r['I'])[:4,0]));n2=int(sum(np.asarray(r['I'])[-4:,0]))
   q,J=transfer_channel(n1,n2);quantum.append(dict(L=L,campaign=c,position=pos,law=law,job=r['job'],**q))
   for mode,x in J.items():states[r['job']+'_'+mode]=x
  if c!='new':continue
  p=P.reshape(len(rr),4,4,4).mean((0,2))[[0,2,3]];p/=p.sum(1)[:,None]
  for tau in (4.,16.,64.):
   q,w,v=crooks_distribution(p,tau,4096,546000+2*len(thermal))
   ref=tilted_cycle(p,tau);err=abs(q['forward']['moments'][1]-ref['W']);errR=abs(q['reverse']['moments'][1]-ref['W_reverse'])
   ix=[q['reverse']['k'].index(k) for k in (0,1,4,16,32)];rev=(np.array(q['reverse']['cf_re'])-1j*np.array(q['reverse']['cf_im']))[ix];f=complex_work_tilt(p,tau);e=float(np.max(abs(f-rev)))
   assert max(e,err,errR)<1e-8
   thermal.append(dict(L=L,position=pos,law=law,propensity=p.tolist(),complex_crooks_error=e,mean_work_error=err,mean_reverse_work_error=errR,**q))
   samples[f'{pos}_{law}_{tau:g}_F']=w;samples[f'{pos}_{law}_{tau:g}_R']=v
 assert len(exact)==72 and len(noise)==13 and len(quantum)==48 and len(states)==192 and len(thermal)==12 and len(samples)==24
 np.savez_compressed(S/'CHOI_STATES.npz',**states);np.savez_compressed(S/'WORK_SAMPLES.npz',**samples)
 out=dict(counts=counts,games=games,groups=groups,temperature=T,exact_count=exact,noise=noise,quantum=quantum,thermal=thermal,prior_precision=oldref,prior_precision_source_sha=oldresult['size_results_sha']['L13'],fresh_histories=len(native),unique_seed_groups=len(seeds),max_tilt_error=max(r['max_tilt_error'] for r in native),size_results_sha={f"L{x['L']}":sha256(S/f"PHYSICS_L{x['L']}.json") for x in a},states_sha=sha256(S/'CHOI_STATES.npz'),work_samples_sha=sha256(S/'WORK_SAMPLES.npz'),scope='same actual v5 MixNet; exact IID last-wave conditional occupancy, PL/reserve approximation, new geometry, two h and B scaling, conditional Q/N references')
 atomic_json(S/'PHYSICS_RESULTS.json',out);print('COMBINED',len(native),len(groups),len(exact),len(quantum),len(thermal),flush=True)
if __name__=='__main__':main()
