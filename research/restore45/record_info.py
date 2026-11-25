import json,numpy as np
from pathlib import Path
from mqft.physics45 import cq_record_information
from mqft.campaign import atomic_json
s=Path(__file__).resolve().parent;r=json.loads((s/"PHYSICS_RESULTS.json").read_text())
out=[]
for pos in sorted(r["references"]):
 for law in ("iid","pl","reserve"):
  a=np.array([x["target_ind"] for x in r["native"] if x["position"]==pos and x["law"]==law and x["h"]==0])
  assert a.shape==(8,16)
  y=(a[:,12:].sum(1)>0).astype(int)
  for window,sl in (("fragment1",slice(0,4)),("fragment2",slice(4,8)),("fragment3",slice(8,12)),("union",slice(0,12))):
   for kind in ("iid","static"):
    out.append(dict(position=pos,law=law,window=window,noise=kind,**cq_record_information(a[:,sl],y,kind=kind)))
assert len(out)==96
atomic_json(s/"RECORD_INFORMATION.json",dict(data=out,scope="same actual native h0 paths drive declared qubit records; last-wave binary label disjoint; empirical eight-sample mixture, not independent measured quantum records"))
print("record cases",len(out),"nonconstant",sum(x["pointer_H"]>0 for x in out),"max",max(x["Holevo"] for x in out))
