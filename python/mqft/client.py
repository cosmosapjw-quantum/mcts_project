"""Persistent JSONL bridge to the native Rust backend; no fallback engine."""
import hashlib
import math
import json
import queue
import subprocess
import threading
from pathlib import Path
from . import SCHEMA

def sha256(path):
    with open(path, "rb") as f:
        h=hashlib.sha256()
        for block in iter(lambda:f.read(1<<20), b""):
            h.update(block)
    return h.hexdigest()

def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False)

def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()

class Backend:
    def __init__(self, executable, timeout=180):
        self.exe=Path(executable).resolve(strict=True)
        self.timeout=float(timeout)
        if not math.isfinite(self.timeout) or self.timeout<=0:raise ValueError("invalid backend timeout")
        self._lock=threading.RLock();self._closed=threading.Event()
        self.sha=sha256(self.exe)
        self.proc=subprocess.Popen([str(self.exe)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,text=True,encoding="utf-8",bufsize=1)
        self.q=queue.Queue(maxsize=2)
        self.stderr=[]
        def put(item):
            while not self._closed.is_set():
                try:self.q.put(item,timeout=.1);return
                except queue.Full:pass
        def read():
            try:
                for line in self.proc.stdout:
                    if self._closed.is_set():break
                    put(line)
            except (OSError,ValueError):pass
            finally:put(None)
        def errors():
            for line in self.proc.stderr:
                self.stderr.append(line[-1000:])
                if len(self.stderr)>40:del self.stderr[:-40]
        self._reader=threading.Thread(target=read,daemon=True);self._err_reader=threading.Thread(target=errors,daemon=True)
        self._reader.start();self._err_reader.start()
        try:
            self.identity=self.call({"op":"describe"})
            if self.identity["schema"]!=SCHEMA or self.identity["board_size"]!=15 or self.identity["rules"]!="freestyle":
                raise RuntimeError("unexpected backend schema or rules")
        except BaseException:
            self.close()
            raise
    def call(self,req):
        with self._lock:return self._call(req)
    def _call(self,req):
        if self._closed.is_set() or self.proc.poll() is not None:
            raise RuntimeError(f"backend exited {self.proc.returncode}: {self.stderr}")
        self.proc.stdin.write(canonical(req)+"\n");self.proc.stdin.flush()
        try: line=self.q.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError("native search timeout") from None
        if line is None:raise RuntimeError(f"backend EOF: {self.stderr}")
        ans=json.loads(line)
        if not ans.get("ok"):raise ValueError(ans.get("error","native backend rejected request"))
        return ans["result"]
    def state(self,moves):return self.call({"op":"state","moves":list(moves)})
    def search(self,moves,spec):return self.call({"op":"search","moves":list(moves),"spec":spec})
    def close(self):
        with self._lock:
            if self._closed.is_set():return
            self._closed.set()
            self._close()
    def _close(self):
        if self.proc.poll() is None:
            try:self.proc.stdin.close()
            except OSError:pass
            try:self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
        self._reader.join(timeout=2);self._err_reader.join(timeout=2)
        for f in (self.proc.stdout,self.proc.stderr):
            if f: f.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
