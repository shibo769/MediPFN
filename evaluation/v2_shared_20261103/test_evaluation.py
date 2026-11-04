"""CPU-only scientific audit contracts; no trained model or OOD data needed."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from evaluate_shard import EvaluationBindingError,finite_prediction,source_guard
from summarize import weighted_uncertainty,weighted_quantile,metrics_and_pairs
from audit_summary import independent_standard_errors


def cases():
    return [dict(case_id=str(i),analysis_weight=.25,source_group="family",source_id="preset",stratum="id",truth=0.,
                 n=200,cluster_id="one" if i<2 else "two") for i in range(4)]


class EvaluationContracts(unittest.TestCase):
    def test_population_and_conditional_table_uncertainty_differ(self):
        result=weighted_uncertainty([0,1,2,3],cases())
        self.assertEqual(result["mean"],1.5)
        self.assertAlmostEqual(result["cluster_se"],1.)
        self.assertAlmostEqual(result["conditional_table_mc_se"],np.sqrt(.125))

    def test_identical_populations_are_not_independent_mechanisms(self):
        sample=cases()
        for c in sample:c["cluster_id"]="same_population"
        result=weighted_uncertainty([0,1,2,3],sample)
        self.assertEqual(result["config_clusters"],1)
        self.assertIsNone(result["cluster_se"])
        self.assertAlmostEqual(result["conditional_table_mc_se"],np.sqrt(5/12))

    def test_cluster_finite_sample_correction_is_per_allocation_stratum(self):
        first=cases();second=copy.deepcopy(first)
        for c in first:c["analysis_weight"]=.125
        for c in second:
            c["analysis_weight"]=.125;c["source_group"]="second";c["source_id"]="second"
            c["case_id"]="second_"+c["case_id"];c["cluster_id"]="second_"+c["cluster_id"]
        result=weighted_uncertainty([0,1,2,3,0,1,2,3],first+second)
        self.assertAlmostEqual(result["cluster_se"],np.sqrt(.5))
        cluster,mc=independent_standard_errors(first+second,[0,1,2,3,0,1,2,3])
        self.assertAlmostEqual(cluster,np.sqrt(.5))
        self.assertAlmostEqual(mc,result["conditional_table_mc_se"])

    def test_failed_intervals_remain_in_coverage_denominator(self):
        sample=cases();rows={}
        for i,c in enumerate(sample):
            for route in ("custom_v2","frozen_v3","adapted_v3"):
                success=not(route=="custom_v2" and i==3)
                cp=success and (route!="frozen_v3" or i<2)
                length={"custom_v2":2.,"frozen_v3":1.,"adapted_v3":4.}[route] if success else None
                rows[(c["case_id"],route)]=dict(status="success" if success else "failure",seconds=1.,error=0. if success else None,
                    intervals={str(level):([-length/2,length/2] if cp else [1.,1.+length]) for level in (90,95,99)} if success else None,
                    covered={str(level):bool(cp) for level in (90,95,99)},length={str(level):length for level in (90,95,99)})
        metrics,paired=metrics_and_pairs(sample,rows)
        main=next(r for r in metrics if r["scope"]=="stratum" and r["route"]=="custom_v2" and r["nominal"]==.95)
        self.assertEqual(main["coverage"],.75)
        self.assertEqual(main["tables"],4);self.assertEqual(main["length_success_denominator"],3)
        self.assertEqual(main["mean_length"],2.)
        difference=next(r for r in paired if r["scope"]=="stratum" and r["left"]=="custom_v2" and r["right"]=="frozen_v3" and r["nominal"]==.95)
        self.assertEqual(difference["coverage_difference"],.25)
        self.assertEqual(difference["length_difference_joint_success"],1.)
        self.assertEqual(difference["jointly_successful"],3)

    def test_weighted_runtime_quantiles_respect_registered_mixture(self):
        self.assertEqual(weighted_quantile([1,10,100],[.8,.1,.1],.5),1.)
        self.assertEqual(weighted_quantile([1,10,100],[.8,.1,.1],.95),100.)

    def test_actual_native_fold_mismatch_is_binding_failure(self):
        prediction=dict(fold_assignment=[0,1],estimate=0.,standard_error=.1,interval_scale=.1,
                         intervals={"90":[-1,1],"95":[-2,2],"99":[-3,3]})
        finite_prediction(prediction,np.asarray([0,1],np.int8),2)
        with self.assertRaises(EvaluationBindingError):
            finite_prediction(prediction,np.asarray([1,0],np.int8),2)

    def test_interval_nonfinite_and_non_nested_are_rejected(self):
        prediction=dict(fold_assignment=[0,1],estimate=0.,standard_error=.1,interval_scale=.1,
                         intervals={"90":[-1,1],"95":[-2,2],"99":[-3,3]})
        invalid=copy.deepcopy(prediction);invalid["intervals"]["95"]=[-.5,.5]
        with self.assertRaises(ValueError):finite_prediction(invalid,np.asarray([0,1],np.int8),2)
        invalid=copy.deepcopy(prediction);invalid["estimate"]=float("nan")
        with self.assertRaises(FloatingPointError):finite_prediction(invalid,np.asarray([0,1],np.int8),2)

    def test_frozen_source_byte_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);program=root/"program.py";program.write_text("pass\n")
            manifest=root/"release_manifest.json"
            manifest.write_text(json.dumps(dict(files={"program.py":hashlib.sha256(program.read_bytes()).hexdigest()})))
            expected=hashlib.sha256(manifest.read_bytes()).hexdigest()
            source_guard(manifest,expected)
            program.write_text("changed\n")
            with self.assertRaises(ValueError):source_guard(manifest,expected)

    def test_model_hash_registry_matches_route_source(self):
        root=Path(__file__).parent;tree=ast.parse((root/"routes.py").read_text())
        value=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="ASSET_HASHES" for t in n.targets))
        self.assertEqual(ast.literal_eval(value),json.loads((root/"frozen_model_hashes.json").read_text()))


if __name__=="__main__":unittest.main(verbosity=2)
