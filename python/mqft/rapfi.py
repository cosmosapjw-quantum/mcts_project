"""Quartz COG01B's pinned external Rapfi protocol, independently packaged.
Rapfi is an external GPL-3.0 program, not a minimax oracle or MCTS trace source.
"""
import os
import re
import selectors
import subprocess
import time
import tempfile
from pathlib import Path
from .client import sha256

ARCHIVE_SHA="1a3e24024062a153ac079060ee9589a37c6bdd1ecc54fed3908793c519594e05"
RUNTIME={
    "config.toml":"32b9180c91247541d399ceef62906d5ffeb48e34eb5f261612987633820d5d3c",
    "pbrain-rapfi-linux-clang-avx2":"4217c3202cdd8dbb58524110d9064299e314a695e58ddbd055d9a4dd99de201c",
    "mix9svqfreestyle_bsmix.bin.lz4":"6bc0d1b0ff8e1d857f7f412923cd458f38f1a435087c91ae71fec3676c23ef62",
}
WEIGHT="mix9svqfreestyle_bsmix.bin.lz4"
EXE="pbrain-rapfi-linux-clang-avx2"

def verify(directory):
    d=Path(directory).resolve(strict=True)
    for name,h in RUNTIME.items():
        if sha256(d/name)!=h:raise ValueError(f"Rapfi pinned runtime mismatch: {name}")
    if not os.access(d/EXE,os.X_OK):raise ValueError("Rapfi binary not executable")
    return dict(release="250615",rules="freestyle15",files=RUNTIME.copy(),license="GPL-3.0",
                source="https://github.com/dhbloo/rapfi",release_url="https://github.com/dotb1ue/rapfi/releases/tag/250615",
                score_semantics="external strong move proxy, not exact")

def install(archive,directory):
    archive=Path(archive).resolve(strict=True);d=Path(directory).resolve()
    if sha256(archive)!=ARCHIVE_SHA:raise ValueError("official archive hash mismatch")
    if d.exists():raise FileExistsError(d)
    d.parent.mkdir(parents=True,exist_ok=True)
    # Archive bytes are checked before extraction. Do not redistribute the binary.
    with tempfile.TemporaryDirectory(dir=d.parent,prefix="rapfi-intake-") as t:
        out=Path(t)/"release";out.mkdir()
        subprocess.run(["7z","x","-y","-o"+str(out),str(archive)],check=True,stdout=subprocess.DEVNULL)
        candidates=list(out.rglob(EXE))
        if len(candidates)!=1:raise ValueError("expected exactly one pinned executable")
        src=candidates[0].parent
        (src/EXE).chmod((src/EXE).stat().st_mode|0o111)
        verify(src)
        src.rename(d)
    return verify(d)

def protocol(moves,timeout_ms,threads):
    if len(set(moves))!=len(moves) or any(not 0<=a<225 for a in moves):
        raise ValueError("invalid opening/history")
    # BOARD numbers are relative to engine to-move, not always absolute black/white.
    side=len(moves)%2
    rows=["START 15","INFO rule 0",f"INFO timeout_turn {timeout_ms}",f"INFO thread_num {threads}"]
    if not moves:rows.append("BEGIN")
    else:
        rows.append("BOARD")
        rows.extend(f"{a%15},{a//15},{1 if i%2==side else 2}" for i,a in enumerate(moves))
        rows.append("DONE")
    return "\n".join(rows)+"\n"

def parse(output,moves):
    lines=output.splitlines()
    if not any(x.strip()=="OK" for x in lines):raise ValueError("Rapfi start not acknowledged")
    # Same known successful message as the original Quartz experiment; reject error lines.
    low=output.lower()
    if f"nnue: load weight from {WEIGHT}" not in low or any(re.search(r"(failed|error).*(nnue|weight)|(nnue|weight).*(failed|error)",x,re.I) for x in lines):
        raise ValueError("freestyle NNUE load not confirmed")
    matches=re.findall(r"(?m)^\s*(\d{1,2}),(\d{1,2})\s*$",output)
    if not matches:raise ValueError("no move line")
    x,y=map(int,matches[-1]);a=y*15+x
    if not (0<=x<15 and 0<=y<15) or a in moves:raise ValueError("illegal Rapfi move")
    return a

class Rapfi:
    def __init__(self,directory,timeout_ms=500,threads=1,wall_timeout=12):
        self.dir=Path(directory).resolve(strict=True);self.identity=verify(self.dir)
        self.ms=int(timeout_ms);self.threads=int(threads);self.timeout=float(wall_timeout)
        if not 1<=self.threads<=64 or self.ms<=0 or self.timeout<=self.ms/1000:raise ValueError("invalid external limits")
    def best(self,moves):
        t=time.monotonic()
        p=subprocess.Popen([str(self.dir/EXE)],cwd=self.dir,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
        sel=selectors.DefaultSelector();sel.register(p.stdout,selectors.EVENT_READ,"stdout");sel.register(p.stderr,selectors.EVENT_READ,"stderr")
        buf={"stdout":bytearray(),"stderr":bytearray()};a=None
        try:
            p.stdin.write(protocol(moves,self.ms,self.threads).encode());p.stdin.flush()
            while time.monotonic()-t<self.timeout:
                for key,_ in sel.select(timeout=min(.25,max(0,self.timeout-(time.monotonic()-t)))):
                    data=os.read(key.fileobj.fileno(),16384)
                    if not data:sel.unregister(key.fileobj);continue
                    buf[key.data].extend(data)
                    if len(buf[key.data])>4<<20:raise RuntimeError("external log limit")
                # Parse only completed stdout lines.
                if b"\n" in buf["stdout"]:
                    complete=bytes(buf["stdout"]).rsplit(b"\n",1)[0].decode(errors="replace")
                    try:a=parse(complete,moves)
                    except ValueError:pass
                if a is not None:break
                if p.poll() is not None:break
            if a is None:raise TimeoutError("no verified external move: "+bytes(buf["stdout"][-1200:]).decode(errors="replace"))
            p.stdin.write(b"END\n");p.stdin.flush()
            p.wait(timeout=3)
            if p.returncode!=0:raise RuntimeError(f"Rapfi exit {p.returncode}")
            return dict(action=a,status="external_proxy",timeout_ms=self.ms,threads=self.threads,
                        wall_seconds=time.monotonic()-t,stdout=bytes(buf["stdout"]).decode(errors="replace"),
                        stderr=bytes(buf["stderr"]).decode(errors="replace"),identity=self.identity)
        finally:
            sel.close()
            if p.poll() is None:p.kill();p.wait()
            for f in (p.stdin,p.stdout,p.stderr):f.close()
