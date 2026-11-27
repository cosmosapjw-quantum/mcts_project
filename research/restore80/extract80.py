"""Authentic R79 empirical reference and R80 matched/novel complete path measures."""
from base61 import *
import gzip
from audit_native74 import one as audit9
from audit_rapfi74 import audit_search
O79=B/'mqft_restore79_20261127';FAMS=['g0','g3','Rapfi'];TAGS=['schedule0','schedule3','rapfi'];CUT=[4,8,16,32,64,128]
def one(args):
 cohort,tag,j,old=args;root=O79 if cohort==0 else S;f=root/'data'/tag/(j['id']+'.json.gz')
 h=sha(f)
 if old:assert h==old['sha256']
 with gzip.open(f,'rt') as z:d=json.load(z)
 assert d['input']==j;tr=d['trace'];g=TAGS.index(tag);L=15 if g==2 else 9;es=tr['events'];cuts={x['k']:x for x in tr['cuts']}
 a=audit_search(tr,j['moves'],'clean',j['seed'],128,j['sources'][0],0.) if cohort and g==2 else audit9((tag,j)) if cohort else old
 aa=np.array([e['actions'][0] for e in es]);z=np.array([e['root_value'] for e in es]);law=[e['root_selection']['law'] for e in es];lp=np.array([q['log_p'] for q in law]);pp=np.array([q['log_p_path'] for q in law]);u=np.array([q['u'] for q in law]);src=j['sources'];x=np.isin(aa,src);b=np.asarray(cuts[128]['n']);legal=np.asarray(law[0]['actions']);assert len(aa)==128 and abs(np.cumsum(lp)-pp).max()<1e-9
 # exact finite root path action under ideal random categorical sampling, deterministic downstream.
 Sact=-float(lp.sum());assert Sact>=0 and np.all(b[legal]>=0)
 unseen=legal[b[legal]==0];q=np.asarray(law[-1]['p']);pos={v:i for i,v in enumerate(law[-1]['actions'])}
 mass=float(sum(q[pos[i]] for i in unseen));assert mass>=0
 # count of observed suboptimal root score choices and unvisited legal support at endpoint.
 nongreedy=sum(int(e['root_selection']['law']['scores'][e['root_selection']['law']['picked']] < max(e['root_selection']['law']['scores'])-1e-8) for e in es)
 # Root-action entropy and negative-log occupancy represent endpoint, not history probability.
 X=np.array([x.mean(),z.mean()]);Y=np.array([x.sum(),np.sum(z>0)],int)
 # Strict source pair before all runs, keep signed mark continuous and positive-mark lattice.
 assert np.all(abs(z)<=1) and np.array_equal(b,np.bincount(aa,minlength=L*L))
 receipt=dict(cohort=cohort,family=g,core=j['core'],rep=j['rep'],seed=j['seed'],path=str(f.relative_to(B)),sha256=h,bytes=f.stat().st_size,max_score=a['max_score'],max_probability=a['max_probability'],max_log=a['max_log'])
 return receipt,X,Y,Sact,mass,len(unseen),nongreedy,u,aa,z,lp
def main():
 R=read(S/'REGISTRATION.json');R0=read(O79/'REGISTRATION.json');assert R['weights']==R0['weights'] and R['external']==R0['external'] and sha(S/'mqft_pulse74')==R['pulse_binary_sha']==R0['pulse_binary_sha']
 for a,b in zip(R['cores'][:32],R0['cores']):assert a['moves']==b['moves'] and a['sources']==b['sources'] and not set(a['seeds'])&set(b['seeds']) and not set(a['rapfi_seeds'])&set(b['rapfi_seeds'])
 old={x['path']:x for x in read(O79/'AUDIT79.json')['raw_receipts'] if x['cohort']==1};args=[]
 for cohort,root,r in ((0,O79,R0),(1,S,R)):
  for tag in TAGS:
   p=root/'plans'/f'{tag}.json';assert sha(p)==r['plans'][tag]['sha256'];js=read(p)
   for j in js:
    f=root/'data'/tag/(j['id']+'.json.gz');args.append((cohort,tag,j,old[str(f.relative_to(B))] if cohort==0 else None))
 assert len(args)==1920
 with cf.ProcessPoolExecutor(max_workers=12) as pool:out=list(pool.map(one,args,chunksize=8))
 rec=[r[0] for r in out];M=np.array([[r['cohort'],r['family'],r['core'],r['rep']] for r in rec]);assert len(set(map(tuple,M)))==1920
 X,Y,act,mass,unseen,ng,u,actions,marks,logp=map(np.stack,zip(*(r[1:] for r in out)))
 np.savez_compressed(S/'MARKED80.npz',meta=M,X=X,Y=Y,action=act,unseen_mass=mass,unseen=unseen,nongreedy=ng,uniforms=u,root_actions=actions,root_marks=marks,root_logp=logp)
 for g in range(3):
  ix=(M[:,0]==1)&(M[:,1]==g);print(FAMS[g],int(ix.sum()),'unseen',int((mass[ix]>0).sum()),'mass',float(mass[ix].mean()),'action range',act[ix].min(),act[ix].max(),'nongreedy',float(ng[ix].mean()),flush=True)
 r=dict(paths=1152,prior_paths=768,backups=147456,prior_backups=98304,raw_bytes=sum(x['bytes'] for x in rec if x['cohort']==1),prior_raw_bytes=sum(x['bytes'] for x in rec if x['cohort']==0),max_score=max(x['max_score'] for x in rec),max_probability=max(x['max_probability'] for x in rec),max_log=max(x['max_log'] for x in rec),raw_receipts=rec,array_sha=sha(S/'MARKED80.npz'),scope='All fresh path laws, CDF, backup and marks checked; R79 byte-authenticated and direct reduction. Recorded uniform vs seed separately checked if available.')
 dump(S/'AUDIT80.json',r);print('AUDIT',{k:v for k,v in r.items() if k!='raw_receipts'},flush=True)
if __name__=='__main__':main()
