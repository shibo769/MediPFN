"""Frozen deployment of offline cross-task TabPFN weights; no test-time SGD."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from .frozen import official_factory,sha256
from .train_backbone import FORMAT,make_training_dataset


def make_finetuned_factory(config,checkpoint,expected_checkpoint_sha256):
    import torch
    from tabpfn import TabPFNClassifier,TabPFNRegressor
    from tabpfn.finetune_utils import clone_model_for_evaluation
    if sha256(checkpoint)!=expected_checkpoint_sha256:raise ValueError('Changed fine-tuned checkpoint')
    c=torch.load(checkpoint,map_location=config.device,weights_only=False)
    if c.get('format')!=FORMAT:raise ValueError('Wrong fine-tuned format')
    base=c['base_config']
    for key in ('model_version','package_version','classifier_sha256','regressor_sha256'):
        if getattr(config,key)!=base[key]:raise ValueError('Fine-tune base differs: '+key)

    @lru_cache(2)
    def loaded(classification):
        est=official_factory(config,classification,119)
        r=np.random.default_rng(18);x=r.normal(size=(32,2))
        y=np.tile([0,1],16) if classification else r.normal(size=32)
        make_training_dataset(est,x,y,16)
        key='classifier_state' if classification else 'regressor_state'
        est.models_[0].load_state_dict(c[key],strict=True)
        est.models_[0].eval()
        for p in est.models_[0].parameters():p.requires_grad_(False)
        return est

    def factory(actual_config,classification,seed):
        if actual_config!=config:raise ValueError('Factory called with another backbone configuration')
        cls=TabPFNClassifier if classification else TabPFNRegressor
        args=dict(device=config.device,n_estimators=config.n_estimators,
                  n_preprocessing_jobs=config.n_preprocessing_jobs,random_state=int(seed),
                  ignore_pretraining_limits=False)
        # Official ModelSpecs clone retains updated weights even though normal
        # sklearn fit() initializes a new table-specific inference engine.
        return clone_model_for_evaluation(loaded(classification),args,cls)
    return factory
