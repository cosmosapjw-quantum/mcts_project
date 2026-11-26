"""Apply explicit ideal syndrome recovery to actual previously registered persistent register states.
This is a software counterfactual channel on frozen states, not new native searches."""
from qec67 import *
import gzip
R=B/'mqft_restore65_20261126'
def td(a,b):return float(np.linalg.norm(a-b,ord='nuc')/2)
def pure_overlap(a,b):return float(np.trace(a@b).real)
def main():
 A=[I]+[kron([X if k==j else I for k in range(3)]) for j in range(3)]
 A[0]=np.eye(8);err=0.
 for a in A:
  for b in A:
   q=dag(V)@dag(a)@b@V;err=max(err,float(abs(q-np.trace(q)*I/2).max()))
 assert err==0.
 p=.12;c=np.sqrt(1-p);s=np.sqrt(p)
 channels=[]
 for mu in (0.,.5,1.):
  K=[]
  for mask in masks:
   w=(1-mu)*p**sum(mask)*(1-p)**(3-sum(mask))+mu*(1-p if sum(mask)==0 else p if sum(mask)==3 else 0.)
   K.append(np.sqrt(w)*kron([X if i else I for i in mask]))
  channels.append((f'bit_mu{mu:g}',K,[np.sqrt(1-p)*I,np.sqrt(p)*X]))
 channels += [('coherent',[kron([c*I-1j*s*X]*3)],[c*I-1j*s*X]),('phase_single',[kron([Z,I,I])],[Z])]
 aud={x['job']['id']:x for x in read(R/'AUDIT65.json')['jobs']}
 out=[]
 for g in (0,3):
  js=[j for j in read(R/f'plans/phase{g}.json') if j['rep']==0 and j['arm'] in ('persist_plus','persist_minus')]
  assert len(js)==96
  for j in js:
   f=R/'data'/f'phase{g}'/(j['id']+'.json.gz');assert sha(f)==aud[j['id']]['sha256']
   with gzip.open(f,'rt') as z:q=json.load(z)['trace']
   for t in (0,7,15,31):
    x,y,z=q['events'][4*t]['root_selection']['law']['phase65']['prepared'];rho=(I+x*X+y*Y+z*Z)/2
    assert np.linalg.eigvalsh(rho).min()>-1e-12
    for name,K,K0 in channels:
     physical=channel(K,V@rho@dag(V));decoded=sum(dag(V)@C0@P0@physical@P0@dag(C0)@V for C0,P0 in zip(C,P))
     direct=channel(recover(K),rho);er=float(abs(decoded-direct).max());assert er<1e-12
     raw=channel(K0,rho)
     out.append(dict(generation=g,core=j['core'],arm=j['arm'],wave=t,pilot_id=j['id'],pilot_sha=aud[j['id']]['sha256'],channel=name,uncoded_trace_distance=td(raw,rho),recovered_trace_distance=td(decoded,rho),uncoded_Y_error=float(abs(np.trace(Y@(raw-rho)))),recovered_Y_error=float(abs(np.trace(Y@(decoded-rho)))),max_error=er))
 summary=[]
 for g in (0,3):
  for name,_,_ in channels:
   q=[x for x in out if x['generation']==g and x['channel']==name];assert len(q)==384
   summary.append(dict(generation=g,channel=name,n=len(q),**{key:float(np.mean([x[key] for x in q])) for key in ('uncoded_trace_distance','recovered_trace_distance','uncoded_Y_error','recovered_Y_error')}))
 dump(S/'STATE_RECOVERY67.json',dict(scope='768 prior R65 persistent conditional states, 3840 software channel applications; not fresh native state preparation or physical recovery',KL_max_error=err,max_matrix_error=max(x['max_error'] for x in out),summary=summary,rows=out))
 print('STATE_RECOVERY',len(out),summary,flush=True)
if __name__=='__main__':main()
