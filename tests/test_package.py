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
                for row in (dict(type="header",schema=p["schema"],job=j,plan_hash=p["plan_hash"],backend_sha=h,backend=identity,external=None),dict(type="root",moves=j["opening"],trace=tr),dict(type="end",status="root_complete",moves=j["opening"],backups=2,searched_records=1,terminal=False,winner=None)):
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

@unittest.skipUnless(os.environ.get("MQFT_TEST_BACKEND"),"set MQFT_TEST_BACKEND")
class RootLaw(unittest.TestCase):
    def test_native_stochastic_laws(self):
        from mqft.path_law import replay
        with Backend(os.environ["MQFT_TEST_BACKEND"],30) as b:
            for kind in ("iid","pl","reserve"):
                for prop in ("p","q","local","path"):
                    with self.subTest(kind=kind,proposal=prop):
                        law=dict(kind=kind,beta_p=[.8],beta_q=[3.2],proposal=prop,logit_h=.2)
                        spec=dict(mode="wave",workers=4,budget=9,cuts=[1,5,9],seed=42,c_puct=1.4,rollout_depth=3,samples_per_eval=1,prior="local",source_action=111,source_h=0.,root_law=law,record_rollouts=True)
                        tr=b.search([112,113],spec);z=check_trace(tr,[112,113],spec)
                        a=z["path_laws"][0];self.assertEqual(a["root_operations"],18)
                        if prop=="path":self.assertAlmostEqual(a["weight_p"]+a["weight_q"],2.)
                        if kind=="pl":
                            from collections import defaultdict
                            phases=defaultdict(list)
                            for e in tr["replicas"][0]["events"]:phases[e["phase"]].append(e["actions"][0])
                            self.assertTrue(all(len(x)==len(set(x)) for x in phases.values()))
                        bad=copy.deepcopy(tr);bad["replicas"][0]["events"][0]["root_selection"]["law"]["p"][0]+=.01
                        with self.assertRaises(ValueError):check_trace(bad,[112,113],spec)
    def test_illegal_stochastic_mode(self):
        with Backend(os.environ["MQFT_TEST_BACKEND"],20) as b:
            spec=dict(mode="tree",workers=4,budget=4,cuts=[4],seed=1,c_puct=1.4,rollout_depth=1,samples_per_eval=1,prior="local",source_action=None,source_h=0.,root_law=dict(kind="iid",beta_p=[1.],beta_q=[2.],proposal="path"))
            with self.assertRaises(ValueError):b.search([112],spec)

class Framework(unittest.TestCase):
    def test_complex_gram_instrument_and_qfi(self):
        from mqft.framework import gram_density,gram_from_environment,purify,instrument,coarse_density,qfi,phase_probe,negativity,gibbs,unitary
        p=[.5,.5];r=gram_density(p,[0,np.pi/2],np.ones((2,2)))
        Y=np.array([[0,-1j],[1j,0]])
        self.assertAlmostEqual(phase_probe(r,Y),1.)
        X,meta=purify(r);self.assertEqual(meta["numerical_rank"],1);np.testing.assert_allclose(X@X.conj().T,r,atol=1e-12)
        f=gram_from_environment([np.eye(2),Y],np.eye(2)/2);np.testing.assert_allclose(f,np.eye(2))
        branches=instrument(r,[[np.diag([1,0])],[np.diag([0,1])]])
        self.assertAlmostEqual(sum(x[0] for x in branches),1.)
        with self.assertRaises(ValueError):instrument(r,[[np.eye(2)/2]])
        np.testing.assert_allclose(coarse_density(r,[0,0],[0,1]),[[1.]])
        self.assertAlmostEqual(qfi(np.diag([.5,.5]),np.diag([1.,-1.])),4.)
        v=np.array([1,0,0,1j])/np.sqrt(2);self.assertAlmostEqual(negativity(np.outer(v,v.conj()),2,2),.5)
        rho,therm=gibbs(np.diag([0.,1.]),1.);self.assertGreater(therm["C_over_kB"],0)
        U=unitary(Y,.4,1.);np.testing.assert_allclose(U.conj().T@U,np.eye(2),atol=1e-12)
        with self.assertRaises(ValueError):gram_density(p,[0,0],[[1,2],[2,1]])
    def test_exact_wave_fisher_projection(self):
        from mqft.framework import exact_wave,pushforward,graph_reduction
        from mqft.path_law import categorical_characteristic,path_weight_statistics
        for mode in ("iid","pl","reserve"):
            x=exact_wave([1,2,0],[.1,-.2,0],[.2,.3,.5],1.4,1.,2,mode,source=1)
            d=pushforward(x["p"],x["counts"],x["score"])
            np.testing.assert_allclose(d["fine_Fisher"],d["coarse_Fisher"]+d["missing_Fisher"],atol=1e-10)
            z,mu,cov=categorical_characteristic(x["p"],np.array(x["counts"]),[0,0,0])
            self.assertAlmostEqual(z.real,1.);self.assertAlmostEqual(mu.sum(),2.);np.testing.assert_allclose(cov@np.ones(3),0,atol=1e-10)
        d=graph_reduction([[0,1,0],[1,0,2],[0,2,0]],[1,2,3],[0,0,1],.5,.3)
        np.testing.assert_allclose(d["isometry"].T@d["isometry"],np.eye(2))
        self.assertGreaterEqual(np.linalg.eigvalsh(d["memory_defect"]).min(),-1e-9)
        z=path_weight_statistics([dict(proposal="path",weight_p=1.5,weight_q=.5,mean_mark=.5,score_p=.2,score_q=-.2,conditional_fisher_p=.2,conditional_fisher_q=.2)])
        self.assertAlmostEqual(z["p"]["mark_IS"],.75)

@unittest.skipUnless(os.environ.get("MQFT_TEST_BACKEND"),"set MQFT_TEST_BACKEND")
class FullRolloutChronology(unittest.TestCase):
    def test_full_rollouts(self):
        with Backend(os.environ["MQFT_TEST_BACKEND"],30) as b:
            for mode,w,s in (("serial",1,1),("root",4,1),("tree",8,1),("leaf",4,4),("wave",4,1)):
                spec=dict(mode=mode,workers=w,budget=37,cuts=[1,9,37],seed=88,c_puct=1.4,rollout_depth=225,samples_per_eval=s,prior="local",source_action=None,source_h=0.,record_rollouts=True)
                z=check_trace(b.search([112,113,97,127],spec),[112,113,97,127],spec)
                self.assertEqual(z["global_rows"][-1]["k"],37)

class ClosureFixture(unittest.TestCase):
    def test_causal_group_ablation(self):
        from mqft.closure import evaluate
        rows=[]
        for g in ("a","b","c"):
            for j in range(18):
                rows.append(dict(kind="roots",position_id=g,label=j%3,state=[float(j%2),0.],memory=[float(j%3)],geometry=[.1]))
        a=evaluate(rows,[("roots","a"),("roots","b")],[("roots","c")])
        self.assertEqual(a["test_events"],18)
        self.assertLess(a["models"]["history"]["nll"],a["models"]["state"]["nll"])
        with self.assertRaises(ValueError):evaluate(rows,[("roots","a")],[("roots","a")])

class PreflightMath(unittest.TestCase):
    def test_paired_block_sizing(self):
        from mqft.preflight import paired_blocks
        self.assertGreater(paired_blocks(1,.1),paired_blocks(1,.2))
        with self.assertRaises(ValueError):paired_blocks(0,.1)

@unittest.skipUnless(os.environ.get("MQFT_TEST_BACKEND"),"set MQFT_TEST_BACKEND")
class LawAnalysisFixture(unittest.TestCase):
    def test_stochastic_receipts_and_summary(self):
        from mqft.campaign import execute
        from mqft.analysis import analyze
        from mqft.plots import render
        cfg=dict(kind="roots",seed=3,replicates=2,budgets=[4],c_puct=[1.4],
                 modes=[dict(mode="wave",workers=2,root_law=dict(kind="reserve",beta_p=[.5],beta_q=[2.],proposal="path",logit_h=.1))],
                 positions=[dict(id="unit",moves=[112,113],source_action=111)],sources=[0.],
                 rollout_depth=3,record_rollouts=True,jobs=1,cpu_limit=2,max_output_bytes=2097152)
        p=make_plan(cfg)
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);r=execute(p,t/"data",os.environ["MQFT_TEST_BACKEND"],authorized=True)
            self.assertEqual(r["failures"],[]);analyze(p,t/"data",t/"analysis")
            rr=[json.loads(x) for x in (t/"analysis/law_summary.jsonl").read_text().splitlines()]
            self.assertEqual(len(rr),1);self.assertEqual(rr[0]["n_paths"],2)
            self.assertAlmostEqual(rr[0]["targets"]["p"]["weight_mean"]+rr[0]["targets"]["q"]["weight_mean"],2.)
            self.assertEqual(len((t/"analysis/history_features.jsonl").read_text().splitlines()),8)
            z=render(t/"analysis",t/"plots");self.assertTrue(any(x["name"]=="N_A_path_law" and x["status"]=="observed" for x in z["panels"]))

@unittest.skipUnless(os.environ.get("MQFT_TEST_BACKEND") and os.environ.get("MQFT_TEST_RAPFI"),"actual pinned MixNet runtime required")
class SharedMixNet(unittest.TestCase):
    def test_pinned_real_mixnet_all_modes(self):
        from mqft.client import Backend
        from mqft.analysis import check_trace
        # Fixed-board software fixtures only; no research data.
        for moves in ([112,113],[112,113,97]):
            results={}
            for mode,workers in (("serial",1),("root",4),("tree",4),("leaf",4),("wave",4)):
                spec=dict(mode=mode,workers=workers,budget=4,cuts=[1,4],seed=16,c_puct=1.4,rollout_depth=1,samples_per_eval=4,prior="local",source_action=None,source_h=0.,record_rollouts=True,evaluator="rapfi_nnue_lookahead_v1")
                with Backend(os.environ["MQFT_TEST_BACKEND"],30,os.environ["MQFT_TEST_RAPFI"],workers) as b:
                    tr=b.search(moves,spec)
                    self.assertTrue(all(r.get("nnue_scores") for x in tr["replicas"] for r in x["evaluations"]))
                    z=check_trace(tr,moves,spec)["global_rows"][-1]
                    self.assertEqual(z["k"],4)
                    results[mode]=tr
                    bad=copy.deepcopy(tr)
                    bad["replicas"][0]["evaluations"][0]["samples"][0]["log_path_prob"]+=.1
                    with self.assertRaises(ValueError):check_trace(bad,moves,spec)
            def signature(tr):
                return [[(x["actions"],x["root_value"]) for x in r["events"]] for r in tr["replicas"]]
            self.assertEqual(signature(results["serial"]),signature(results["leaf"]))
    def test_common_evaluator_plan(self):
        cfg=json.loads(Path("configs/mixnet_games.json").read_text())
        cfg["opponents"]=["rapfi_black"]
        with self.assertRaises(ValueError):make_plan(cfg)

@unittest.skipUnless(os.environ.get("MQFT_TEST_BACKEND") and os.environ.get("MQFT_TEST_RAPFI"),"actual pinned MixNet runtime required")
class CommonArena(unittest.TestCase):
    def test_color_assignment_and_receipt(self):
        from mqft.campaign import execute,ply_spec
        from mqft.analysis import analyze
        cfg=dict(kind="games",seed=42000,replicates=1,budgets=[4],c_puct=[1.4],
          modes=[dict(mode="wave",workers=2,samples_per_eval=4)],openings=[[112,113]],
          opponents=["serial_black","serial_white"],rollout_depth=1,prior="local",evaluator="rapfi_nnue_lookahead_v1",
          record_rollouts=True,move_temperature=0.,max_plies=4,jobs=1,cpu_limit=4,max_output_bytes=10485760)
        pl=make_plan(cfg);self.assertEqual(len(pl["jobs"]),2)
        self.assertEqual(len({x["spec"]["seed"] for x in pl["jobs"]}),1)
        for j in pl["jobs"]:
            self.assertEqual(ply_spec(j,2,1)["mode"],"serial" if j["opponent"]=="serial_black" else "wave")
            self.assertEqual(ply_spec(j,3,-1)["mode"],"serial" if j["opponent"]=="serial_white" else "wave")
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);out=p/"data"
            r=execute(pl,out,os.environ["MQFT_TEST_BACKEND"],os.environ["MQFT_TEST_RAPFI"],authorized=True)
            self.assertEqual(r["failures"],[]);self.assertEqual(len(r["completed"]),2)
            m=analyze(pl,out,p/"analysis");self.assertEqual(m["statuses"],{"truncated":2})
            z=[json.loads(l) for l in (p/"analysis/global.jsonl").read_text().splitlines() if json.loads(l)["k"]==4]
            self.assertEqual(len(z),4)
            self.assertEqual({x["mode"] for x in z},{"serial","wave"})
            self.assertTrue(all(x["evaluator"]=="rapfi_nnue_lookahead_v1" for x in z))
    def test_finite_source_response(self):
        from mqft.campaign import execute
        from mqft.analysis import analyze
        cfg=dict(kind="roots",seed=42003,replicates=1,budgets=[4],c_puct=[1.4],
          modes=[dict(mode="serial",workers=1,samples_per_eval=4)],positions=[dict(id="probe",moves=[112,113],source_action=111)],
          sources=[-.2,0.,.2],rollout_depth=1,prior="local",evaluator="rapfi_nnue_lookahead_v1",
          record_rollouts=True,paired_schedule_seed=True,jobs=1,cpu_limit=2,max_output_bytes=10485760)
        pl=make_plan(cfg)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);r=execute(pl,p/"data",os.environ["MQFT_TEST_BACKEND"],os.environ["MQFT_TEST_RAPFI"],authorized=True)
            self.assertEqual(r["failures"],[]);analyze(pl,p/"data",p/"analysis")
            rr=[json.loads(x) for x in (p/"analysis/source_response.jsonl").read_text().splitlines()]
            self.assertEqual(len(rr),1);self.assertTrue(np.isfinite(rr[0]["adaptive_kkt_diagnostic"]))

class ProspectiveResponse(unittest.TestCase):
    def test_frozen_features_and_bounded_model(self):
        from mqft.prospective import field_features,predict,fit,evaluate
        b=dict(prior=[.3,.4,.3],action_q=[.02,.01,-.02],pi=[.25,.5,.25],actions=[0,1,2],source_action=1,k=32,c_puct=1.4,mode="serial",budget=32,replicate=0,job="zero")
        q,pi,i,X,w,opt=field_features(b,.2)
        self.assertAlmostEqual(float(X[:,1].sum()),0.,places=10)
        self.assertGreaterEqual(X[i,1],0.)
        z=predict(b,.2,{"q_coef":[0.,0.],"gain":1.})
        self.assertAlmostEqual(z["frozen"],z["prospective"],places=10)
        self.assertAlmostEqual(z["frozen"],z["gain"],places=10)
        a=dict(b,pi=[.25,.46875,.28125],action_q=[.02,-.01,-.02],job="minus")
        c=dict(b,pi=[.25,.53125,.21875],action_q=[.02,.03,-.02],job="plus")
        case=[(("r","s","serial",32,0,1),.2,a,b,c)]
        model=fit(case);out=evaluate(case,model)
        self.assertEqual(model["serial"]["n_train"],1)
        self.assertAlmostEqual(out[0]["observed"],.15625)
        self.assertTrue(np.isfinite(out[0]["prospective"]))
        self.assertEqual(out[0]["baseline_job"],"zero")

class GroupedPhysics44(unittest.TestCase):
    def test_exact_collective_and_coarse_fisher(self):
        from mqft.physics44 import exact_wave_full_state,projected_information
        e=[dict(action=i,n=2,w=.1*(i-2),prior=.2) for i in range(5)]
        for law in ("iid","pl","reserve"):
            z=exact_wave_full_state(e,list(range(5)),4,law,0)
            self.assertLess(max(z["errors"].values()),1e-10)
            f=projected_information(z)
            self.assertAlmostEqual(f["ordered"]["relative"],1.,places=10)
            self.assertLess(f["constant"]["relative"],1e-10)
            if law=="iid":np.testing.assert_allclose(z["R"],z["C"],atol=1e-10)
        self.assertEqual(len(exact_wave_full_state(e,list(range(5)),4,"pl",0)["words"]),120)
    def test_conditional_quantum_and_thermal(self):
        from mqft.physics44 import exact_wave_full_state,quantum_memory,record_fragments,thermal_reference,saddlepoint
        e=[dict(action=i,n=2,w=.1*(i-2),prior=.2) for i in range(5)]
        z=exact_wave_full_state(e,list(range(5)),2,"reserve",0)
        q,states=quantum_memory(z,alphas=(0.,.5))
        self.assertAlmostEqual(q[0]["trace_distance"],0.,places=10)
        self.assertGreater(q[1]["trace_distance"],0.)
        self.assertLess(q[1]["diag_error"],1e-10)
        d=record_fragments(z);self.assertAlmostEqual(d[-1]["QMI"],2*d[-1]["S_system"],places=10)
        t=thermal_reference(z);self.assertAlmostEqual(t["JE"],t["JE_expected"],places=10)
        self.assertAlmostEqual(t["sigma"],t["KL"],places=10)
        self.assertAlmostEqual(t["delta_U"],t["work"]+t["heat"],places=10)
        f=saddlepoint(exact_wave_full_state(e,list(range(5)),4,"iid",0),replicas=(8,64))
        self.assertLess(f[-1]["one_loop_error"],f[-1]["leading_error"])
    def test_projector_and_euler_obstruction(self):
        rho=np.diag([1.,0.]).astype(complex);H=np.array([[0.,1.],[1.,0.]])
        a=rho-1j*.1*(H@rho-rho@H)
        self.assertAlmostEqual(np.linalg.det(a).real,-.01)
        self.assertLess(np.linalg.eigvalsh(a).min(),0.)

class MultivariatePhysics45(unittest.TestCase):
    def test_lattice_coordinate_covolume_and_dp(self):
        from mqft.physics44 import exact_wave_full_state
        from mqft.physics45 import lattice_action
        e=[dict(action=i,n=2,w=.1*(i-2),prior=.2) for i in range(5)]
        z=exact_wave_full_state(e,list(range(5)),4,"pl",0)
        for coarse in (False,True):
            a=lattice_action(z,(8,),coarse)[0]
            self.assertLess(a["dp_residual"],1e-9)
            self.assertLess(a["geometry_residual"],1e-9)
            self.assertAlmostEqual(a["lattice_covolume"],np.sqrt(a["d"]))
    def test_nonthermal_ratio_and_echo(self):
        from mqft.physics45 import protocol_entropy,probe_from_words,cq_record_information
        e=[dict(action=i,n=2,w=.1*(i-2),prior=.2) for i in range(5)]
        a=protocol_entropy(e,list(range(5)),"iid",0)
        self.assertAlmostEqual(a["KL"],0.,places=10)
        for law in ("pl","reserve"):
            a=protocol_entropy(e,list(range(5)),law,0)
            self.assertAlmostEqual(a["IFT"],1.,places=10)
            self.assertGreaterEqual(a["KL"],a["coarse_KL"]-1e-10)
        z=probe_from_words([[1,1],[1,1]],[.5,.5],1,.5,True)
        self.assertAlmostEqual(z["static"]["coherence"],1.,places=10)
        self.assertGreater(z["trace_distance"],0.)
        a=cq_record_information([[0,0],[1,1],[0,0],[1,1]],[0,1,0,1])
        self.assertGreater(a["Holevo"],0.)
        self.assertLessEqual(a["X_readout_MI"],a["Holevo"]+1e-10)
    def test_centered_score_is_sample_covariance(self):
        from mqft.physics45 import centered_estimates
        a=np.array([.2,.4,.3,.6]);g=np.array([.3,-.1,.2,-.2]);d=g*.5
        z=centered_estimates(a,g,d)
        self.assertAlmostEqual(z["centered"].mean(),np.cov(a,g,ddof=1)[0,1])
        self.assertAlmostEqual(z["centered_fd"].mean(),np.cov(a,d,ddof=1)[0,1])

class PrecisionPhysics46(unittest.TestCase):
    def test_lr_predictable_bernoulli(self):
        from mqft.physics46 import log_ratio,score_evidence
        p=.3;h=.6
        l0,k0=log_ratio([p],[0],h);l1,k1=log_ratio([p],[1],h)
        self.assertAlmostEqual((1-p)*np.exp(l0)+p*np.exp(l1),1.)
        self.assertAlmostEqual(-((1-p)*l0+p*l1),k0)
        self.assertAlmostEqual(k0,k1)
        a=score_evidence([dict(conditional_p=[p],target_ind=[1],G=1-p,F=p*(1-p))])
        self.assertTrue(np.isfinite(a["e_mixture"]))
    def test_discrete_projected_memory_recurrence(self):
        from mqft.physics46 import fit_count_memory
        train=[dict(full_wave_counts=list(x)) for x in [(0,0,1),(1,0,0),(0,1,1),(1,1,0)]*4]
        test=[dict(full_wave_counts=r["full_wave_counts"],position="test") for r in train]
        z=fit_count_memory(train,test,"pl")
        self.assertLess(z["mz_residual"],1e-10)
        self.assertGreater(z["kernel_norms"][0],0.)
    def test_causal_break_same_preparation(self):
        from mqft.physics46 import causal_break
        z=causal_break([1,0,0,0,1,0,0,0])
        self.assertGreater(z["static_MI"],0.)
        self.assertLess(z["iid_MI"],1e-10)
        self.assertAlmostEqual(z["correlation"],np.sin(1.)**2)
        self.assertLess(z["residual"],1e-10)
        z=causal_break([0,0,0,0,1,0,0,0])
        self.assertLess(z["static_MI"],1e-10)

class ScalePhysics47(unittest.TestCase):
    def test_joint_dual_and_clock_projection(self):
        from mqft.physics47 import joint_action,nonstationary_projection
        a=[]
        for i,(x,y) in enumerate([(0.,-.5),(.5,-.3),(.3,.7),(.8,.1)]):
            a.append(dict(occupancy=x,mean_mark=y,G=float(i)-1.5,F=.5,
                conditional_p=[.2,.3],target_ind=[i%2,i//2]))
        # G must be the derivative of the empirical likelihood.
        for x in a:x["G"]=sum(x["target_ind"])-sum(x["conditional_p"])
        z=joint_action(a,a)
        self.assertEqual(z["status"],"full_rank")
        self.assertLess(z["duality_error"],2e-6)
        L=np.array([[.7,.1,.2],[.2,.7,.3],[.1,.2,.5]])
        E=np.array([[.5,0],[.5,0],[0,1.]])
        R=np.array([[1,1,0],[0,0,1]])
        z=nonstationary_projection([L,L.T/np.sum(L.T,axis=0),L],E,R)
        self.assertLess(z["residual"],1e-10)
    def test_matched_block_memory_and_phase(self):
        from mqft.physics47 import process_probe
        z,_=process_probe(1,0)
        self.assertLess(z["choi_static_reset"],1e-10)
        z,_=process_probe(1,1)
        self.assertGreater(z["choi_static_reset"],0.)
        self.assertEqual(z["channels"]["dephased"]["phase_QFI"],0.)
        self.assertLess(max(x["tp_error"] for x in z["channels"].values()),1e-10)
    def test_partial_reset_thermal_cycle(self):
        from mqft.physics47 import thermal_cycle
        for eta in (.25,.5,1.):
            x=thermal_cycle([.2,.7,.4],eta)
            self.assertAlmostEqual(x["JE"],1.,places=10)
            self.assertAlmostEqual(x["work"],x["entropy_production"]+x["final_lag_KL"],places=10)
        self.assertLess(abs(x["delta_U"]),1e-10)
        self.assertLess(abs(x["final_lag_KL"]),1e-10)

class PrecisionPhysics48(unittest.TestCase):
    def test_causal_kernel_and_paired_jackknife(self):
        from mqft.physics48 import causal_kernel,response_precision
        rng=np.random.default_rng(48);a=[]
        for i in range(16):
            I=rng.binomial(1,.3,8).tolist();M=rng.uniform(-1,1,8).tolist()
            a.append(dict(seed=i,k=8,target_ind=I,marks=M,conditional_p=[.3]*8,G=sum(I)-2.4,F=8*.3*.7,occupancy=sum(I)/8,mean_mark=float(np.mean(M))))
        z=causal_kernel(a)
        self.assertAlmostEqual(z["estimates"][0][0],z["estimates"][1][0]+z["estimates"][4][0])
        arms={h:[dict(x,occupancy=x["occupancy"]+.01*h) for x in a] for h in (-.2,-.1,0.,.1,.2)}
        p,r=response_precision(arms)
        self.assertAlmostEqual(p[0]["values"]["direct"][0],.01)
        self.assertAlmostEqual(r["values"][0][0],.01)
        self.assertTrue(np.isfinite(p[0]["se"]["LR_minus_direct"]).all())
    def test_tomography_repair_and_duration(self):
        from mqft.physics48 import shot_channel,duration_cycle
        from mqft.physics47 import process_probe
        z,_=process_probe(1,1)
        for name in ("static","reset_static","dephased"):
            J,R,m,a=shot_channel(z["channels"][name]["ptm"],128,np.random.default_rng(48))
            self.assertGreaterEqual(np.linalg.eigvalsh(R).min(),-1e-10)
        for T in (1.,16.,64.):
            x=duration_cycle([.2,.7,.4],T,8)
            self.assertAlmostEqual(x["JE"],1.,places=10)
            self.assertAlmostEqual(x["work"],x["sigma"]+x["lag"],places=10)
        self.assertLess(duration_cycle([.4,.4,.4],1.,8)["work"],1e-12)

    def test_prefix_likelihood_finite(self):
        from mqft.physics48 import prefix_finite
        rng=np.random.default_rng(49);a=[]
        for i in range(8):
            I=rng.binomial(1,.4,8).tolist()
            a.append(dict(seed=i,k=8,target_ind=I,marks=[.2]*8,conditional_p=[.4]*8))
        r=prefix_finite({h:a for h in (-.2,0.,.2)})[0]
        self.assertEqual(r["n"],8)
        self.assertTrue(np.isfinite(r["se"]["prefix"]).all())
        self.assertAlmostEqual(r["values"]["prefix"][1],0.)
        self.assertAlmostEqual(r["values"]["direct"][0],0.)
