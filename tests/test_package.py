"""Small mathematical and in-memory software fixtures, never research data."""
import copy
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
from mqft.campaign import make_plan,load_plan,atomic_json,execute,valid_moves,win,choose
from mqft.client import Backend
from mqft.analysis import check_trace
from mqft.observables import fit_beta,frozen_action,tangent_response,root_observables,delta,empirical_mi,graph_and_coarse
from mqft.rapfi import protocol,parse,WEIGHT

class Math(unittest.TestCase):
    def test_beta_boundaries(self):
        self.assertEqual(fit_beta([1,0],[0,1])["status"],"zero")
        self.assertEqual(fit_beta([0,1],[0,1])["status"],"infinite")
        self.assertEqual(fit_beta([.5,.5],[1,1])["status"],"flat")
        p=np.exp([0.,2.]);p/=p.sum()
        self.assertAlmostEqual(fit_beta(p,[0,1])["beta"],2.,places=9)
    def test_kkt_and_response(self):
        q=np.array([.12,.05,.09]);p=np.array([.3,.4,.3]);x=np.array([.3,.4,.3]);field=np.array([1.,0.,0.])
        f=frozen_action(q,p,12,2,x);self.assertLess(f["kkt_residual"],1e-8)
        z,_=tangent_response(q,p,12,2,x,field)
        h=1e-5;a=frozen_action(q+h*field,p,12,2,x);b=frozen_action(q-h*field,p,12,2,x)
        np.testing.assert_allclose(z,(np.array(a["optimum"])-b["optimum"])/(2*h),rtol=2e-7,atol=2e-8)
        self.assertAlmostEqual(z.sum(),0.)
    def test_moments_balance(self):
        a=root_observables([dict(action=0,n=2,w=0.,s2=2.,prior=.5),dict(action=1,n=1,w=1.,s2=1.,prior=.5)],3,1.4)
        b=root_observables([dict(action=0,n=3,w=1.,s2=3.,prior=.5),dict(action=1,n=1,w=1.,s2=1.,prior=.5)],4,1.4)
        self.assertAlmostEqual(a["variance"],a["within"]+a["between"])
        d=delta(a,b);self.assertAlmostEqual(d["dE"],d["field"]+d["redistribution"])
        self.assertAlmostEqual(empirical_mi([0,0,1,1],[1,1,0,0])["I"],np.log(2))
    def test_board_protocol(self):
        self.assertTrue(win(valid_moves([0,15,1,16,2,17,3,18,4]),1))
        with self.assertRaises(ValueError):valid_moves([0,15,1,16,2,17,3,18,4,22])
        with self.assertRaises(ValueError):valid_moves([0,0])
        self.assertIn("7,7,2",protocol([112],500,1))
        self.assertEqual(parse(f"OK\nMESSAGE mix9svq nnue: load weight from {WEIGHT}\n7,8\n",[112]),127)
        with self.assertRaises(ValueError):parse("OK\n7,8\n",[112])
    def test_plan_dry_and_gate(self):
        cfg=json.loads(Path("configs/pilot_games.json").read_text());cfg["replicates"]=1
        p=make_plan(cfg)
        self.assertEqual(len(p["jobs"]),120)
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/"plan.json";atomic_json(f,p);self.assertEqual(p,load_plan(f))
            dest=Path(t)/"no_data"
            with self.assertRaises(PermissionError):execute(p,dest,"missing")
            self.assertFalse(dest.exists())
    def test_graph_partition(self):
        e=[{"actions":[1,2],"root_value":1.},{"actions":[1,2],"root_value":0.},{"actions":[3,4],"root_value":-1.},{"actions":[3,4],"root_value":1.}]
        z=graph_and_coarse(e)
        self.assertEqual(z["eligible_nodes"],4);self.assertEqual(z["sampled_pairs"],6)
        for r in z["coarse"]:self.assertAlmostEqual(r["within"]+r["between"],r["total_variance"])

@unittest.skipUnless(os.environ.get("MQFT_TEST_BACKEND"),"set MQFT_TEST_BACKEND")
class Bridge(unittest.TestCase):
    def test_all_schedules_contract(self):
        # A tiny fixed board fixture, no game loop, no persistent trace/plots.
        moves=[0,15,1,16,2,17,3,18]
        with Backend(os.environ["MQFT_TEST_BACKEND"],20) as b:
            for mode in ("serial","root","tree","leaf","wave"):
                with self.subTest(mode=mode):
                    spec=dict(mode=mode,workers=1 if mode=="serial" else 2,budget=5,cuts=[1,2,5],seed=42,c_puct=1.4,rollout_depth=2,samples_per_eval=2 if mode=="leaf" else 1,prior="local",source_action=None,source_h=0.)
                    tr=b.search(moves,spec);q=check_trace(tr,moves,spec)
                    self.assertEqual(q["global_rows"][-1]["k"],5)
                    self.assertEqual(sum(q["global_rows"][-1]["counts"]),5)
                    # Corrupt a native moment, analysis must reject.
                    bad=copy.deepcopy(tr);bad["replicas"][0]["cuts"][-1]["edges"][0]["s2"]+=1
                    with self.assertRaises(ValueError):check_trace(bad,moves,spec)
    def test_root_source(self):
        with Backend(os.environ["MQFT_TEST_BACKEND"],20) as b:
            spec=dict(mode="serial",workers=1,budget=2,cuts=[2],seed=42,c_puct=1.4,rollout_depth=1,samples_per_eval=1,prior="uniform",source_action=113,source_h=.2)
            tr=b.search([112],spec);q=check_trace(tr,[112],spec)
            z=q["local"][0];i=z["actions"].index(113);j=z["actions"].index(111)
            self.assertAlmostEqual(z["prior"][i]/z["prior"][j],np.exp(.2),places=6)

class Conditional(unittest.TestCase):
    def test_complex_density_and_reset(self):
        from mqft.conditional import density,information,finite_reset
        psi=np.array([1,0,0,1j])/np.sqrt(2);r=np.outer(psi,psi.conj())
        z=information(r,[2,2],[1]);self.assertAlmostEqual(z["QMI"],2*np.log(2));self.assertAlmostEqual(z["Holevo"],np.log(2))
        self.assertAlmostEqual(z["basis_MI"],np.log(2));self.assertGreater(z["imag_norm"],0)
        t=finite_reset(r,2,[0.,1.],1.2,np.pi/2);self.assertAlmostEqual(t["Sigma"],t["I_SM_B"]+t["D_B_Gibbs"])
        with self.assertRaises(ValueError):density([[1.1,0],[0,-.1]])

@unittest.skipUnless(os.environ.get("MQFT_TEST_BACKEND"),"set MQFT_TEST_BACKEND")
class ReceiptFixture(unittest.TestCase):
    def test_read_and_render_temporary_fixture(self):
        # No campaign/generate execution. One 2-backup bridge unit fixture, temporary only.
        import gzip
        from mqft.client import sha256,digest
        from mqft.analysis import analyze
        from mqft.plots import render
        cfg=dict(kind="roots",seed=1,replicates=1,budgets=[2],c_puct=[1.4],modes=[dict(mode="serial",workers=1)],positions=[dict(id="unit",moves=[112],source_action=None)],sources=[0.],rollout_depth=1,jobs=1,cpu_limit=1,max_output_bytes=1048576)
        p=make_plan(cfg);j=p["jobs"][0]
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);folder=t/"fixture";folder.mkdir();out=t/"analysis"
            with Backend(os.environ["MQFT_TEST_BACKEND"],20) as be:
                tr=be.search(j["opening"],j["spec"]);h=be.sha;identity=be.identity
            atomic_json(folder/"run_lock.json",dict(plan_hash=p["plan_hash"],backend_sha=h,external=None))
            f=folder/(j["id"]+".jsonl.gz")
            with gzip.open(f,"wt") as g:
                for row in (dict(type="header",job=j,plan_hash=p["plan_hash"],backend_sha=h,backend=identity,external=None),dict(type="root",moves=j["opening"],trace=tr),dict(type="end",status="root_complete",moves=j["opening"],backups=2,searched_records=1,terminal=False,winner=None)):
                    g.write(json.dumps(row)+"\n")
            atomic_json(folder/(j["id"]+".receipt.json"),dict(job_id=j["id"],job_hash=digest(j),plan_hash=p["plan_hash"],backend_sha=h,status="root_complete",sha256=sha256(f),file=f.name,bytes=f.stat().st_size,backups=2,records=1))
            a=analyze(p,folder,out);self.assertEqual(a["completed"],1)
            v=render(out,t/"plots");self.assertTrue(any(x["status"].startswith("observed") for x in v["panels"]))
            self.assertEqual(v["source_manifest_sha256"],sha256(out/"manifest.json"))

class Lifecycle(unittest.TestCase):
    def test_reject_fractional_config(self):
        c=json.loads(Path("configs/commissioning_games.json").read_text())
        for field,value in (("replicates",1.2),("max_plies",True),("search_timeout",float("nan"))):
            b=copy.deepcopy(c);b[field]=value
            with self.assertRaises(ValueError):make_plan(b)
    def test_staged_crash_recovery(self):
        from mqft.campaign import recover_staged
        from mqft.client import sha256,digest
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);j=dict(id="unit");f=p/"unit.jsonl.gz.part";f.write_bytes(b"closed unit shard")
            r=dict(job_id="unit",job_hash=digest(j),plan_hash="plan",backend_sha="exe",sha256=sha256(f),bytes=f.stat().st_size,file="unit.jsonl.gz")
            atomic_json(p/"unit.receipt.json.staged",r)
            self.assertEqual(recover_staged(j,p,"exe","plan"),r)
            self.assertTrue((p/"unit.jsonl.gz").exists());self.assertFalse(f.exists())
            self.assertTrue((p/"unit.receipt.json").exists())
    @unittest.skipUnless(os.environ.get("MQFT_TEST_BACKEND"),"set MQFT_TEST_BACKEND")
    def test_real_root_runner_and_resume(self):
        # Tiny temporary software integration fixture, no game or study dataset.
        from mqft.campaign import execute
        from mqft.analysis import analyze
        cfg=dict(kind="roots",seed=1,replicates=1,budgets=[2],c_puct=[1.4],modes=[dict(mode="serial",workers=1)],positions=[dict(id="unit",moves=[112],source_action=None)],sources=[0.],rollout_depth=1,jobs=1,cpu_limit=1,max_output_bytes=1048576)
        p=make_plan(cfg)
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);out=t/"data"
            a=execute(p,out,os.environ["MQFT_TEST_BACKEND"],authorized=True)
            self.assertEqual(a["failures"],[]);self.assertEqual(a["completed"][0]["backups"],2)
            b=execute(p,out,os.environ["MQFT_TEST_BACKEND"],authorized=True)
            self.assertEqual(a,b)
            self.assertEqual(analyze(p,out,t/"analysis")["completed"],1)
