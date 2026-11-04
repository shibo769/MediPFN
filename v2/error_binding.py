"""Strict construction contracts and an actual-checkpoint-bound NIE estimator.

This wrapper leaves the frozen factual/evaluation source unchanged. It records
the hashes of the checkpoint bytes actually loaded and the point-estimator
source files actually imported, and binds every result to its observed table.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from error_model import SUMMARY_FEATURES

FEATURE_CONTRACT_KEYS=("factual_checkpoint_sha256","eta_checkpoint_sha256",
                       "relative_scale_floor","probability_floor","outer_folds",
                       "feature_names","source_sha256","registered_source_groups",
                       "sample_sizes","truth_target")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_contract(metadata):
    missing=[k for k in FEATURE_CONTRACT_KEYS if k not in metadata]
    if missing:
        raise ValueError("Missing feature-construction contract fields: "+",".join(missing))
    return {k:metadata[k] for k in FEATURE_CONTRACT_KEYS}


def assert_same_feature_contract(left,right):
    first,second=feature_contract(left),feature_contract(right)
    for key in FEATURE_CONTRACT_KEYS:
        if first[key]!=second[key]:
            raise ValueError("Feature-construction contract differs: "+key)


def validate_feature_metadata(metadata):
    contract=feature_contract(metadata)
    for key in ("factual_checkpoint_sha256","eta_checkpoint_sha256"):
        if not isinstance(contract[key],str) or re.fullmatch(r"[0-9a-f]{64}",contract[key]) is None:
            raise ValueError("Invalid checkpoint SHA: "+key)
    if (not np.isfinite(contract["probability_floor"]) or not np.isfinite(contract["relative_scale_floor"])
            or not 0<float(contract["probability_floor"])<.5 or not float(contract["relative_scale_floor"])>0):
        raise ValueError("Invalid probability/uncertainty scale floor")
    if contract["outer_folds"]!=2:
        raise ValueError("This registered feature pipeline uses two outer folds")
    sizes=contract["sample_sizes"]
    if not isinstance(sizes,list) or not sizes or len(set(sizes))!=len(sizes) or any(type(n) is not int or n<16 for n in sizes):
        raise ValueError("Registered sample sizes must be distinct integers >=16")
    names=contract["feature_names"]
    context_count=len(names)-len(SUMMARY_FEATURES)
    if context_count<1 or names!=["context_"+str(i) for i in range(context_count)]+list(SUMMARY_FEATURES):
        raise ValueError("Feature order does not match the observed uncertainty interface")
    hashes=contract["source_sha256"]
    required=("model.py","evaluate.py","error_model.py","error_binding.py","train_error.py","dgp/generators.py")
    if not isinstance(hashes,dict) or any(k not in hashes for k in required):
        raise ValueError("Missing feature-source SHA bindings")
    if any(not isinstance(v,str) or re.fullmatch(r"[0-9a-f]{64}",v) is None for v in hashes.values()):
        raise ValueError("Invalid feature-source SHA")
    if contract["truth_target"]!="superpopulation_NIE_1":
        raise ValueError("Feature cache has a different estimand")
    return contract


def observed_table_sha(table):
    digest=hashlib.sha256()
    for key in ("x","a","m","y"):
        value=np.ascontiguousarray(np.asarray(table[key],dtype=np.float32))
        digest.update(key.encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class BoundNIEEstimator:
    def __init__(self,factual_checkpoint,eta_checkpoint,device="cpu",probability_floor=.01,outer_folds=2):
        from evaluate import load_nuisances
        import evaluate
        import model
        self.factual,self.eta=load_nuisances(factual_checkpoint,eta_checkpoint,device)
        self.factual.eval().requires_grad_(False)
        self.eta.eval().requires_grad_(False)
        self.device=device
        self.probability_floor=probability_floor
        self.outer_folds=outer_folds
        self.binding=dict(factual_checkpoint_sha256=file_sha(factual_checkpoint),
                          eta_checkpoint_sha256=file_sha(eta_checkpoint))
        self.point_source_sha256={"model.py":file_sha(model.__file__),"evaluate.py":file_sha(evaluate.__file__)}

    def estimate(self,table,seed=123,return_scores=True):
        from evaluate import estimate_table
        result=estimate_table(self.factual,table,self.eta,self.device,seed=seed,folds=self.outer_folds,
                              probability_floor=self.probability_floor,return_scores=return_scores)
        result["nuisance_binding"]=dict(self.binding)
        result["point_source_sha256"]=dict(self.point_source_sha256)
        result["observed_table_sha256"]=observed_table_sha(table)
        return result


def assert_bound_prediction(result,table,contract):
    """Reject mismatched actual checkpoint/source/settings/table witnesses."""
    validate_feature_metadata(contract)
    binding=result.get("nuisance_binding")
    if not isinstance(binding,dict):
        raise ValueError("Prediction needs a result from BoundNIEEstimator")
    for key in ("factual_checkpoint_sha256","eta_checkpoint_sha256"):
        if binding.get(key)!=contract[key]:
            raise ValueError("Point-estimator checkpoint differs from uncertainty model: "+key)
    if result.get("probability_floor")!=contract["probability_floor"] or result.get("outer_folds")!=contract["outer_folds"]:
        raise ValueError("Point-estimator floor/fold settings differ from uncertainty model")
    for key in ("model.py","evaluate.py"):
        if result.get("point_source_sha256",{}).get(key)!=contract["source_sha256"][key]:
            raise ValueError("Point-estimator source differs from uncertainty training: "+key)
    if result.get("observed_table_sha256")!=observed_table_sha(table):
        raise ValueError("Point estimate belongs to a different observed table")
    expected=["context_"+str(i) for i in range(len(result.get("pooled_context",[])))]+list(SUMMARY_FEATURES)
    if expected!=contract["feature_names"]:
        raise ValueError("Point-estimator uncertainty feature order differs")
    if result.get("n")!=len(np.asarray(table["a"]).reshape(-1)):
        raise ValueError("Point estimate and observed table row counts differ")
