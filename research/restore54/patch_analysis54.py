from pathlib import Path
import shutil,hashlib,json
S=Path(__file__).resolve().parent;O=S.parent/'mqft_restore53_20261126'
for L in (13,21):
 src=O/'family'/f'L{L}'/'python';dst=S/'python'/f'L{L}'
 assert not dst.exists();shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 p=dst/'mqft/analysis.py';s=p.read_text()
 a='        elif actual[k]!=v:raise ValueError(f"spec mismatch {k}")'
 b='''        elif k=="root_law" and v is not None and v.get("sector_schedule"):
            a0=dict(actual[k]);v0=dict(v)
            x=np.asarray(a0.pop("sector_schedule"),float);y=np.asarray(v0.pop("sector_schedule"),float)
            if a0!=v0 or x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("vector law identity")
            # Zero is exact; nonzero values may differ only by <= 2 ULP
            # from JSON serialization. Engine's actual probabilities replayed below.
            if np.any((x==0)!=(y==0)) or np.any(abs(x-y)>2*abs(np.spacing(y))):
                raise ValueError("vector schedule exceeds 2 ULP")
        elif actual[k]!=v:raise ValueError(f"spec mismatch {k}")'''
 assert s.count(a)==1;p.write_text(s.replace(a,b))
 h=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
 (dst/'PATCH_MANIFEST.json').write_text(json.dumps(dict(parent=h(src/'mqft/analysis.py'),modified=h(p),files={str(f.relative_to(dst)):h(f) for f in sorted(dst.rglob('*.py'))}),indent=2)+'\n')
