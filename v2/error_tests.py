"""Contracts for signed errors, ordered quantiles, and fresh task calibration."""
from statistics import NormalDist
from pathlib import Path
import copy
import hashlib
import json
import tempfile
import unittest
import numpy as np
import torch
from error_model import (DEFAULT_ALPHAS,ErrorConfig,SignedErrorQuantiles,
                         interval_from_pair,pinball_loss,uncertainty_features)
from calibrate import conformal_expansions,predict_intervals
from train_error import task_seed,read_cache,error_bindings,CACHE_FORMAT
from posttrain_error import PosttrainedError,balanced_weights
from error_binding import observed_table_sha,assert_bound_prediction
from error_model import SUMMARY_FEATURES


def tiny_cache(directory):
    from dgp import sample_task,registry,SOURCE_COMMIT
    seed=42
    t=sample_task("he_song_xu",64,task_seed(seed,"train",0,"observations"),
                  mechanism_seed=task_seed(seed,"train",0,"mechanism"))
    provenance=dict(t["metadata"],n=64,fold_seed=task_seed(seed,"train",0,"folds"))
    groups=sorted(set(r["source_group"] for r in registry() if r["default_split"]=="train"))
    names=["context_0","context_1"]+list(SUMMARY_FEATURES)
    metadata=dict(format=CACHE_FORMAT,split="train",cases=1,completed_cases=1,failures=0,
                  factual_checkpoint_sha256="a"*64,eta_checkpoint_sha256="b"*64,
                  relative_scale_floor=.001,probability_floor=.01,outer_folds=2,
                  feature_names=names,source_sha256=error_bindings(),registered_source_groups=groups,
                  sample_sizes=[64],truth_target="superpopulation_NIE_1",provenance=[provenance],
                  source_group_counts={provenance["source_group"]:1},
                  arguments=dict(seed=seed,split="train",cases=1,relative_scale_floor=.001,probability_floor=.01,n_values=[64]))
    arrays=dict(features=np.zeros((1,len(names)),np.float32),response=np.asarray([2.],np.float64),
                estimate=np.asarray([t["truth"]+1.],np.float64),scale=np.asarray([.5],np.float64),truth=np.asarray([t["truth"]],np.float64))
    path=Path(directory)/"fixture.npz"
    def write():
        with path.open("wb") as handle:
            np.savez_compressed(handle,**arrays)
        metadata["cache_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".json").write_text(json.dumps(metadata),encoding="utf-8")
    write()
    return path,arrays,metadata,write


def bound_fixture():
    from dgp import registry
    table=dict(x=np.zeros((4,1),np.float32),m=np.zeros((4,1),np.float32),
               a=np.asarray([0,1,0,1],np.float32),y=np.asarray([-2,-1,1,2],np.float32))
    contract=dict(factual_checkpoint_sha256="a"*64,eta_checkpoint_sha256="b"*64,
                  relative_scale_floor=.001,probability_floor=.01,outer_folds=2,
                  feature_names=["context_0","context_1"]+list(SUMMARY_FEATURES),source_sha256=error_bindings(),
                  registered_source_groups=sorted(set(r["source_group"] for r in registry() if r["default_split"]=="train")),
                  sample_sizes=[64],truth_target="superpopulation_NIE_1")
    result=dict(n=4,estimate=.0025,standard_error=.001,scores=[.001,.002,.003,.004],
                pooled_context=[.2,.3],probability_floor=.01,outer_folds=2,
                diagnostics=[dict(query_rows=4,pi_clipped=0.,g_clipped=.25,ratio_max=3.)],
                nuisance_binding={k:contract[k] for k in ("factual_checkpoint_sha256","eta_checkpoint_sha256")},
                point_source_sha256={k:contract["source_sha256"][k] for k in ("model.py","evaluate.py")},
                observed_table_sha256=observed_table_sha(table))
    return table,result,contract


class ErrorContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)

    def test_initial_quantiles_are_normal_and_nested(self):
        model=SignedErrorQuantiles(ErrorConfig(feature_count=9,width=16))
        pairs=model(torch.randn(6,9)).detach().numpy()
        for i,alpha in enumerate(DEFAULT_ALPHAS):
            z=NormalDist().inv_cdf(1-alpha/2)
            np.testing.assert_allclose(pairs[:,i,:],np.tile([-z,z],(6,1)),atol=1e-6)
        self.assertTrue(np.all(np.diff(pairs[:,:,0],axis=1)<0))
        self.assertTrue(np.all(np.diff(pairs[:,:,1],axis=1)>0))

    def test_optimization_preserves_ordered_quantiles(self):
        model=SignedErrorQuantiles(ErrorConfig(feature_count=9,width=16))
        optimizer=torch.optim.Adam(model.parameters(),lr=.05)
        x=torch.randn(100,9)
        e=3*x[:,0]+.2*torch.randn(100)
        for _ in range(8):
            optimizer.zero_grad()
            loss=pinball_loss(model(x),e)
            loss.backward()
            optimizer.step()
        pairs=model(x).detach().numpy()
        self.assertTrue(np.isfinite(pairs).all())
        self.assertTrue(np.all(pairs[:,:,0]<pairs[:,:,1]))
        self.assertTrue(np.all(np.diff(pairs[:,:,0],axis=1)<0))
        self.assertTrue(np.all(np.diff(pairs[:,:,1],axis=1)>0))

    def test_signed_error_interval_orientation(self):
        # theta_hat=10, s=2 and e in [-1,3] imply theta in [4,12].
        self.assertEqual(interval_from_pair(10,2,[-1,3]),[4.,12.])
        self.assertEqual(interval_from_pair(10,2,[-1,3],.5),[3.,13.])

    def test_observed_features_ignore_truth_and_share_scale_floor(self):
        table=dict(y=np.asarray([-2.,-1.,1.,2.]),a=np.asarray([0.,1.,0.,1.]),truth=2.)
        result=dict(scores=[.001,.002,.003,.004],estimate=.0025,standard_error=0.,
                    diagnostics=[dict(query_rows=4,pi_clipped=0.,g_clipped=.25,ratio_max=3.)],
                    pooled_context=[.2,.3])
        first,scale=uncertainty_features(result,table,.01)
        table["truth"]=1e30
        second,other_scale=uncertainty_features(result,table,.01)
        np.testing.assert_array_equal(first,second)
        self.assertEqual(scale,other_scale)
        self.assertAlmostEqual(scale,.01*np.std(table["y"],ddof=1))
        e=(result["estimate"]-1.)/scale
        endpoints=interval_from_pair(result["estimate"],scale,[e-.1,e+.1])
        self.assertLess(endpoints[0],1.)
        self.assertGreater(endpoints[1],1.)

    def test_finite_sample_calibration_rank(self):
        response=np.linspace(-4,4,1000)
        pairs=np.tile(np.asarray([[-1,1],[-1.5,1.5],[-2,2]]),(1000,1,1))
        result=conformal_expansions(pairs,response,DEFAULT_ALPHAS)
        self.assertEqual(result["ranks"],[901,951,991])
        self.assertTrue(np.all(np.diff(result["expansions"])>=0))
        for i,alpha in enumerate(DEFAULT_ALPHAS):
            c=result["expansions"][i]
            covered=(response>=pairs[:,i,0]-c)&(response<=pairs[:,i,1]+c)
            self.assertGreaterEqual(covered.mean(),1-alpha)

    def test_small_calibration_is_rejected_instead_of_inventing_finite_tail(self):
        with self.assertRaises(ValueError):
            conformal_expansions(np.tile([[[-1.,1.]]],(20,1,1)),np.zeros(20),[.01])

    def test_stage_seed_namespaces_are_disjoint(self):
        values={split:{task_seed(42,split,i,"mechanism") for i in range(10)}
                for split in ("train","dev","calibration","post_train","post_dev")}
        self.assertTrue(all(min(v)>2**160 for v in values.values()))
        self.assertFalse(values["train"]&values["dev"])
        self.assertFalse(values["train"]&values["calibration"])
        self.assertFalse(values["dev"]&values["calibration"])

    def test_balanced_weights_equalize_mechanisms_and_source_groups(self):
        provenance=[dict(source_group="one",mechanism_id="a")]*3+[
            dict(source_group="one",mechanism_id="b"),dict(source_group="two",mechanism_id="c")]
        weights=balanced_weights(provenance)
        self.assertAlmostEqual(float(weights.sum()),1.)
        self.assertAlmostEqual(float(weights[:3].sum()),.25)
        self.assertAlmostEqual(float(weights[3]),.25)
        self.assertAlmostEqual(float(weights[4]),.5)

    def test_confidence_monotonicity_and_posttraining_nesting(self):
        base=SignedErrorQuantiles(ErrorConfig(feature_count=9,width=16))
        model=PosttrainedError(base,confidence_width=8,correction_width=8)
        features=torch.randn(20,9)
        _,_,_,_,scales,probabilities=model.confidence_geometry(features)
        self.assertTrue(torch.all(torch.diff(probabilities,dim=-1)>=0))
        self.assertTrue(torch.all(scales>0))
        optimizer=torch.optim.Adam(model.correction.parameters(),lr=.02)
        response=torch.randn(20)
        for _ in range(5):
            optimizer.zero_grad()
            loss=pinball_loss(model(features),response)
            loss.backward()
            optimizer.step()
        pairs=model(features).detach().numpy()
        self.assertTrue(np.all(pairs[:,:,0]<pairs[:,:,1]))
        self.assertTrue(np.all(np.diff(pairs[:,:,0],axis=1)<=0))
        self.assertTrue(np.all(np.diff(pairs[:,:,1],axis=1)>=0))

    def test_cache_rejects_inconsistent_response_even_with_matching_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path,arrays,metadata,write=tiny_cache(directory)
            read_cache(path,"train")
            arrays["response"][0]+=1
            write()
            with self.assertRaisesRegex(ValueError,"response"):
                read_cache(path,"train")

    def test_cache_rejects_missing_provenance_and_ood_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path,arrays,metadata,write=tiny_cache(directory)
            original=copy.deepcopy(metadata["provenance"])
            metadata["provenance"]=[]
            write()
            with self.assertRaisesRegex(ValueError,"provenance"):
                read_cache(path,"train")
            metadata["provenance"]=original
            metadata["provenance"][0]["source_id"]="eric"
            write()
            with self.assertRaisesRegex(ValueError,"OOD"):
                read_cache(path,"train")

    def test_bound_prediction_rejects_point_model_floor_folds_and_table_mismatch(self):
        table,result,contract=bound_fixture()
        assert_bound_prediction(result,table,contract)
        changes=(lambda r:r["nuisance_binding"].update(factual_checkpoint_sha256="c"*64),
                 lambda r:r.update(probability_floor=.02),lambda r:r.update(outer_folds=3),
                 lambda r:r["point_source_sha256"].update(**{"evaluate.py":"d"*64}),
                 lambda r:r.update(observed_table_sha256="e"*64),lambda r:r.pop("nuisance_binding"))
        for change in changes:
            with self.subTest(change=change):
                altered=copy.deepcopy(result)
                change(altered)
                with self.assertRaises(ValueError):
                    assert_bound_prediction(altered,table,contract)

    def test_interval_prediction_enforces_bound_model_and_calibration_identity(self):
        table,result,contract=bound_fixture()
        model=SignedErrorQuantiles(ErrorConfig(feature_count=len(contract["feature_names"]),width=16))
        model._deployment_contract=contract
        model._checkpoint_sha256="f"*64
        calibration=dict(contract,alphas=list(model.config.alphas),expansions=[0.,0.,0.],error_checkpoint_sha256="f"*64)
        predict_intervals(result,table,model,calibration)
        calibration["error_checkpoint_sha256"]="a"*64
        with self.assertRaisesRegex(ValueError,"uncertainty checkpoint"):
            predict_intervals(result,table,model,calibration)


if __name__=="__main__":
    unittest.main(verbosity=2)
