"""Conditional spatial-sector quantum lift with explicit phase generator."""
from pathlib import Path
import json,numpy as np
from mqft.client import sha256
from mqft.campaign import atomic_json
from mqft.conditional import density,von_neumann
from mqft.framework import qfi
S=Path(__file__).resolve().parent;r=json.loads((S/'PHYSICS_RESULTS.json').read_text());out=[];states={}
G=np.diag(np.arange(4));u=np.ones(4)/2
for x in r['conditional_records']:
 p=np.array(x['p']);F=np.full((4,4),.65**2);np.fill_diagonal(F,1.)
 rho=density(np.sqrt(p[:,None]*p[None,:])*F);diag=np.diag(p);a=[]
 for theta in (0.,.3,.6):
  U=np.diag(np.exp(-1j*theta*np.arange(4)));R=density(U@rho@U.conj().T);dr=-1j*(G@R-R@G)
  assert np.max(abs(np.diag(R)-p))<1e-10
  val=qfi(R,dr);assert val>=-1e-10
  a.append(dict(theta=theta,QFI=val,diag_QFI=qfi(diag,np.zeros((4,4))),probe=float((u@R@u).real),dephased_probe=.25,S=von_neumann(R),imag_norm=float(np.linalg.norm(R.imag))))
  states[f"{x['position']}_{x['law']}_{theta}"]=R
 out.append(dict(L=x['L'],position=x['position'],law=x['law'],values=a))
assert len(out)==30 and len(states)==90
np.savez_compressed(S/'SPATIAL_QUANTUM_STATES.npz',**states)
atomic_json(S/'PHASE_RESULTS.json',dict(source_sha=sha256(S/'PHYSICS_RESULTS.json'),states_sha=sha256(S/'SPATIAL_QUANTUM_STATES.npz'),data=out,scope='declared coherent phase rotation of empirical spatial-sector record state; same diagonal, no native phase identification'))
print('phase',len(out),len(states))
