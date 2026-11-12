"""Bound observed-only point estimators; the two existing backbones never train."""
from __future__ import annotations
from dataclasses import asdict
import hashlib,json,sys
from pathlib import Path
import numpy as np
from common import ROUTES,sha256,observed_hash,digest,assert_contract

NATIVE_RELEASE='67409e5e616eea452020131b7beef22de3c6833c1c441e2fa7b30df569d43408'
NATIVE_MANIFEST='794b4fbd6d6b38a7716bc4c20364e743b992cd8f2d2a991b62b386d7661a2975'
CLASSIFIER_SHA='d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988'
REGRESSOR_SHA='311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301'
ADAPTER_SHA='dbc4c0fbc7c099e665ea65e8b26c9594a721711bef7825aed7cb5f56cf79f201'

class BoundV3Point:
    def __init__(self,models_root,route,device='cuda',standard_error_floor=1e-3):
        if route not in ROUTES or standard_error_floor<=0:raise ValueError('Unknown route or invalid interval scale')
        self.root=Path(models_root);self.native=self.root/'optional_releases/67409e5e616eea45'
        manifest_path=self.native/'release_manifest.json'
        if sha256(manifest_path)!=NATIVE_MANIFEST:raise ValueError('Unrecognized native source manifest')
        manifest=json.loads(manifest_path.read_text())
        if manifest['release_sha256']!=NATIVE_RELEASE:raise ValueError('Unrecognized native source release')
        for relative,expected in manifest['files'].items():
            if sha256(self.native/relative)!=expected:raise ValueError('Native point source changed')
        if str(self.native) not in sys.path:sys.path.insert(0,str(self.native))
        from tabular_backbones.frozen import BackboneConfig,make_shared_frozen_v3_factory
        import tabular_backbones.frozen as frozen
        if Path(frozen.__file__).resolve()!=self.native/'tabular_backbones/frozen.py':raise ValueError('Unexpected imported native estimator')
        weights=self.root/'tabular_backbones/v3_runtime/weights'
        classifier=weights/'tabpfn-v3-classifier-v3_default.ckpt';regressor=weights/'tabpfn-v3-regressor-v3_default.ckpt'
        if sha256(classifier)!=CLASSIFIER_SHA or sha256(regressor)!=REGRESSOR_SHA:raise ValueError('Official weights changed')
        self.config=BackboneConfig(model_version='v3',package_version='9.2.0',classifier_path=str(classifier),
          classifier_sha256=CLASSIFIER_SHA,regressor_path=str(regressor),regressor_sha256=REGRESSOR_SHA,
          device=device,n_estimators=1,n_preprocessing_jobs=1,probability_floor=.01)
        adapter_sha=None
        self.adapter=None
        if route=='adapted_v3':
            from tabular_backbones.adapter import load_adapter,adapted_factory
            path=self.root/'runs/adapter_v3_seed0/best.pt'
            if sha256(path)!=ADAPTER_SHA:raise ValueError('Existing selected adapter changed')
            self.adapter,_=load_adapter(path,self.config);self.adapter.requires_grad_(False)
            self.factory=adapted_factory(self.config,self.adapter);adapter_sha=ADAPTER_SHA
        else:self.factory=make_shared_frozen_v3_factory(self.config)
        cfg=asdict(self.config);cfg.pop('device')
        # Absolute runtime paths are excluded; actual model bytes are bound.
        cfg.pop('classifier_path');cfg.pop('regressor_path')
        self.contract=dict(route=route,native_release_sha256=NATIVE_RELEASE,native_manifest_sha256=NATIVE_MANIFEST,
          classifier_sha256=CLASSIFIER_SHA,regressor_sha256=REGRESSOR_SHA,adapter_sha256=adapter_sha,
          point_config=cfg,outer_folds=2,feature_dimension=1038,standard_error_floor=float(standard_error_floor),
          feature_program_sha256=sha256(__file__),common_program_sha256=sha256(Path(__file__).with_name('common.py')),
          native_point_program_sha256=sha256(self.native/'tabular_backbones/frozen.py'),
          native_feature_program_sha256=sha256(self.native/'tabular_backbones/train_uncertainty.py'),
          target='NIE1=E[Y(1,M(1))-Y(1,M(0))]',scale_definition='max(score standard error, absolute registered floor)',
          input_policy='observed arrays and type metadata only; no source IDs/true effects/nuisance truth')
        self.route=route;self.device=device

    def extract(self,observed,fold_seed,expected_folds=None):
        before=observed_hash(observed)
        from tabular_backbones.frozen import estimate_table
        from tabular_backbones.train_uncertainty import feature_vector
        result=estimate_table(observed,self.config,seed=int(fold_seed),factory=self.factory,
          return_scores=True,collect_embeddings=True)
        if observed_hash(observed)!=before:raise ValueError('Frozen point estimator mutated the observed table')
        folds=np.asarray(result['fold_assignment'],np.int8)
        if expected_folds is not None and not np.array_equal(folds,np.asarray(expected_folds,np.int8)):
            raise ValueError('Native outer-fold assignment differs from the shared plan')
        features=feature_vector(result,observed)
        if features.shape!=(1038,) or not np.isfinite(features).all():raise ValueError('Unexpected V3 uncertainty features')
        estimate=float(result['estimate']);se=float(result['standard_error'])
        if not np.isfinite(estimate) or not np.isfinite(se) or se<0:raise FloatingPointError('Nonfinite point estimator')
        return dict(features=features,estimate=estimate,standard_error=se,
          scale=max(se,self.contract['standard_error_floor']),contract=self.contract,
          observed_sha256=before,fold_seed=int(fold_seed),
          fold_assignment_sha256=hashlib.sha256(folds.tobytes()).hexdigest(),
          point_result=result)
