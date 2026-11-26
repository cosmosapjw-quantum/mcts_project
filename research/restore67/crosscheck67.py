from base61 import *
import gzip
def one(arg):
 tag,js=arg
 traces=[]
 for j in js:
  with gzip.open(S/'data'/tag/(j['id']+'.json.gz'),'rt') as f:q=json.load(f)['trace']
  q.pop('fault67');q.pop('elapsed_ns');q.pop('job')
  for e in q['events']:e.pop('worker',None)
  traces.append(q)
 assert traces[0]==traces[1]==traces[2]
 return len(traces[0]['events'])
def main():
 jobs=[]
 for g in (0,3):
  tag=f'fault{g}';js=read(S/'plans'/f'{tag}.json');d=collections.defaultdict(dict)
  for j in js:
   if j['arm'] in ('single_iid','single_mix','rep_common'):d[(j['core'],j['rep'])][j['arm']]=j
  for v in d.values():jobs.append((tag,[v[k] for k in ('single_iid','single_mix','rep_common')]))
 with cf.ProcessPoolExecutor(max_workers=12) as p:out=list(p.map(one,jobs,chunksize=8))
 assert len(out)==512 and sum(out)==65536
 dump(S/'COMMON_CONTROL67.json',dict(scope='same seed and evaluator ordinal: single_iid/single_mix/rep_common deliver identical first-bit flips; full native traces exactly equal excluding job, fault metadata, time, worker',triples=len(out),backups_per_arm=sum(out),semantic_equal=True))
 print('COMMON CONTROL',len(out),sum(out))
if __name__=='__main__':main()
