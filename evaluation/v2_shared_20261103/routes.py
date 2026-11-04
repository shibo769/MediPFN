"""Read-only, resident inference wrappers around three frozen v2 routes.

No fitting/selection/optimizer is introduced. Native point estimators receive
the same observed table and outer-fold seed. The V3 wrapper reproduces its
frozen head/calibration transformation while exposing the native seed argument.
"""
from __future__ import annotations
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROUTES=("custom_v2","frozen_v3","adapted_v3")
EXPECTED_RELEASES={
 "primary_code":{"release_sha256":"475a2abf480e9e36bfebeac6d584dc6a5d5139377340fac2ea0c221e41daedf6",
                 "manifest_sha256":"c356f36617860a0fa6fbd80a12c34ca0dfcf8dd5effe9a5df39291dd2b008e68"},
 "optional_code":{"release_sha256":"67409e5e616eea452020131b7beef22de3c6833c1c441e2fa7b30df569d43408",
                  "manifest_sha256":"794b4fbd6d6b38a7716bc4c20364e743b992cd8f2d2a991b62b386d7661a2975"}}
ASSET_HASHES={
 "factual":"722172242bb4f80957b02a9f2227d557d29379db20906a63a5e5e2aa01a2d66b",
 "eta":"82fc00689e50526f130263beebecd09afefe4ab4cbe1fccf7239dd599d659cd7",
 "custom_head":"2a8e8c3ac56ceb34e718acc29838ec4b0a1785fd62c92b1dced910e856bcc3c3",
 "custom_calibration":"dc857f520a4af6405d1e26922d28faae677ee294fb232b97f92127b5fda22427",
 "frozen_head":"df406ce3d43bc097860473ba83817cb19f3e48e23d6dfabece0bfcd545e40ba3",
 "frozen_calibration":"e013c2d849375d9ad5eeaee7fd1ca03234fa7b34871678e4faa962e8917df3cc",
 "adapter":"dbc4c0fbc7c099e665ea65e8b26c9594a721711bef7825aed7cb5f56cf79f201",
 "adapted_head":"ca7c0bf657fe6c927799daa4e74c05de3efe790a8419c11665f559c50f87a7e3",
 "adapted_calibration":"dcef6f5e668b0756d78f9a7d3a4acfc1ce22bc994711e69536f5991537b66ad3",
 "classifier":"d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988",
 "regressor":"311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301"}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8388608),b""):
            h.update(chunk)
    return h.hexdigest()


def paths(root):
    root=Path(root)
    return dict(primary_code=root/"releases/error_475a2abf480e9e36",
                optional_code=root/"optional_releases/67409e5e616eea45",
                factual=root/"runs/factual_seed0/best.pt",eta=root/"runs/nested_seed0/best.pt",
                custom_head=root/"runs/error_pipeline_seed0/post_fit/best.pt",
                custom_calibration=root/"runs/error_pipeline_seed0/final_calibration.json",
                frozen_head=root/"runs/frozen_v3_error_seed0/best.pt",
                frozen_calibration=root/"runs/frozen_v3_error_seed0/calibration.json",
                adapter=root/"runs/adapter_v3_seed0/best.pt",
                adapted_head=root/"runs/adapter_v3_error_seed0/best.pt",
                adapted_calibration=root/"runs/adapter_v3_error_seed0/calibration.json",
                classifier=root/"tabular_backbones/v3_runtime/weights/tabpfn-v3-classifier-v3_default.ckpt",
                regressor=root/"tabular_backbones/v3_runtime/weights/tabpfn-v3-regressor-v3_default.ckpt")


def verify_assets(root):
    p=paths(root)
    actual={k:sha256(p[k]) for k in ASSET_HASHES}
    if actual!=ASSET_HASHES:
        raise ValueError("A frozen checkpoint/calibration/upstream model changed")
    releases={}
    for key in ("primary_code","optional_code"):
        manifest_path=p[key]/"release_manifest.json"
        if sha256(manifest_path)!=EXPECTED_RELEASES[key]["manifest_sha256"]:
            raise ValueError("Unrecognized native source manifest: "+key)
        manifest=json.loads(manifest_path.read_text())
        if manifest["release_sha256"]!=EXPECTED_RELEASES[key]["release_sha256"]:
            raise ValueError("Unrecognized native source release identity: "+key)
        for relative,expected in manifest["files"].items():
            if sha256(p[key]/relative)!=expected:
                raise ValueError("Frozen source changed: "+str(p[key]/relative))
        releases[key]=dict(release_sha256=manifest["release_sha256"],files=len(manifest["files"]))
    return dict(assets=actual,releases=releases)


def configure_imports(root):
    p=paths(root)
    for location in (str(p["optional_code"]),str(p["primary_code"])):
        if location not in sys.path:
            sys.path.insert(0,location)
    return p


def validate_observed(table):
    allowed={"x","a","m","y","x_types","mediator_types","outcome_type"}
    if set(table)-allowed:
        raise ValueError("Inference accepts observed arrays/type schema only")
    x,m=np.asarray(table["x"]),np.asarray(table["m"])
    a,y=np.asarray(table["a"]).reshape(-1),np.asarray(table["y"]).reshape(-1)
    if x.ndim!=2 or m.ndim!=2 or len(m)!=len(a) or len(x)!=len(a) or len(y)!=len(a):
        raise ValueError("Observed table dimensions differ")
    if not all(np.isfinite(v).all() for v in (x,m,a,y)) or not np.isin(a,[0,1]).all():
        raise ValueError("Invalid observed table values")


class CustomRoute:
    def __init__(self,root,device="cuda"):
        p=configure_imports(root)
        from error_binding import BoundNIEEstimator
        from calibrate import load_calibrated_error
        self.estimator=BoundNIEEstimator(p["factual"],p["eta"],device,probability_floor=.01,outer_folds=2)
        self.head,_,self.calibration=load_calibrated_error(p["custom_head"],p["custom_calibration"],device)
        self.route="custom_v2"

    @torch.inference_mode()
    def predict(self,table,fold_seed):
        validate_observed(table)
        from calibrate import predict_intervals
        point=self.estimator.estimate(table,seed=int(fold_seed),return_scores=True)
        predicted=predict_intervals(point,table,self.head,self.calibration)
        return dict(estimate=point["estimate"],standard_error=point["standard_error"],
                    interval_scale=predicted["scale"],fold_assignment=point["fold_assignment"],
                    intervals={k:v["endpoints"] for k,v in predicted["intervals"].items()},
                    diagnostics=point["diagnostics"])


class V3Route:
    def __init__(self,root,adapted=False,device="cuda"):
        p=configure_imports(root)
        from tabular_backbones.frozen import BackboneConfig,make_shared_frozen_v3_factory
        from tabular_backbones.error_head import load_head
        from tabular_backbones.train_uncertainty import canonical_hash
        self.root=Path(root);self.adapted=adapted;self.route="adapted_v3" if adapted else "frozen_v3"
        self.head_path=p["adapted_head" if adapted else "frozen_head"]
        self.calibration_path=p["adapted_calibration" if adapted else "frozen_calibration"]
        self.adapter_path=p["adapter"] if adapted else None
        self.calibration=json.loads(self.calibration_path.read_text())
        if self.calibration["status"]!="CALIBRATED_ID_MIXTURE" or self.calibration["head_checkpoint_sha256"]!=sha256(self.head_path):
            raise ValueError("Wrong frozen V3 uncertainty checkpoint/calibration")
        self.binding=self.calibration["backbone_binding"]
        self.config=replace(BackboneConfig(**self.binding["base_config"]),device=device)
        for key,expected in (("model_version","v3"),("package_version","9.2.0"),("n_estimators",1),("n_preprocessing_jobs",1),("probability_floor",.01)):
            if getattr(self.config,key)!=expected:
                raise ValueError("Different registered V3 configuration: "+key)
        for key in ("classifier","regressor"):
            if getattr(self.config,key+"_sha256")!=ASSET_HASHES[key]:
                raise ValueError("Different official V3 weights")
        if self.binding["feature_program_sha256"]!=sha256(p["optional_code"]/"tabular_backbones/train_uncertainty.py"):
            raise ValueError("Different frozen V3 uncertainty feature program")
        if self.binding["frozen_program_sha256"]!=sha256(p["optional_code"]/"tabular_backbones/frozen.py"):
            raise ValueError("Different frozen V3 point estimator")
        self.head,doc=load_head(self.head_path,self.binding,map_location=device)
        self.head=self.head.to(device).eval().requires_grad_(False)
        self.schema=doc["feature_schema"]
        if canonical_hash(self.schema)!=self.calibration["feature_schema_sha256"]:
            raise ValueError("Different frozen V3 feature schema")
        if adapted:
            from tabular_backbones.adapter import load_adapter,adapted_factory
            if self.binding["adapter_sha256"]!=sha256(self.adapter_path):
                raise ValueError("Different frozen adapter checkpoint")
            self.adapter,_=load_adapter(self.adapter_path,self.config)
            self.factory=adapted_factory(self.config,self.adapter)
        else:
            if self.binding["adapter_sha256"] is not None:
                raise ValueError("Frozen route must not contain an adapter")
            self.factory=make_shared_frozen_v3_factory(self.config)

    def predict(self,table,fold_seed):
        validate_observed(table)
        from tabular_backbones.frozen import estimate_table
        from tabular_backbones.train_uncertainty import feature_vector,quantile_indices
        point=estimate_table(table,self.config,seed=int(fold_seed),factory=self.factory,
                             return_scores=True,collect_embeddings=True)
        features=(feature_vector(point,table)-np.asarray(self.schema["center"]))/np.asarray(self.schema["scale"])
        with torch.no_grad():
            tensor=torch.as_tensor(features[None,:],dtype=torch.float32,device=self.config.device)
            quantiles=self.head(tensor)[0].cpu().numpy()
        scale=max(float(point["standard_error"]),float(self.calibration["standard_error_floor"]))
        intervals={}
        for level in (.90,.95,.99):
            low,high=quantile_indices(level)
            correction=self.calibration["corrections"][str(round(level*100))]["correction"]
            intervals[str(round(level*100))]=[float(point["estimate"]-scale*(quantiles[high]+correction)),
                                            float(point["estimate"]-scale*(quantiles[low]-correction))]
        return dict(estimate=point["estimate"],standard_error=point["standard_error"],interval_scale=scale,
                    intervals=intervals,fold_assignment=point["fold_assignment"],diagnostics=point["diagnostics"])


def load_route(name,root,device="cuda"):
    if name=="custom_v2":
        return CustomRoute(root,device)
    if name in ("frozen_v3","adapted_v3"):
        return V3Route(root,adapted=name=="adapted_v3",device=device)
    raise ValueError("Unknown frozen route")


def release_route(route):
    del route
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
