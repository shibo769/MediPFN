"""Regression gate for NumPy/Tensor mismatches in differentiable embedding paths."""
import unittest
import numpy as np
import torch
from .adapter import differentiable_query_embeddings


class Executor:
    def __init__(self):self.seen=None
    def iter_outputs(self,query,autocast,task_type,only_return_standard_out):
        if not isinstance(query,torch.Tensor):raise TypeError('Executor requires Tensor for differentiable preprocessing')
        self.seen=dict(dtype=str(query.dtype),device=str(query.device),requires_grad=query.requires_grad,
                       task_type=task_type,standard_only=only_return_standard_out)
        yield {'test_embeddings':torch.ones(len(query),1,5,device=query.device,dtype=query.dtype)},None


class Estimator:
    def __init__(self):self.executor_=Executor();self.use_autocast_=False


class EmbeddingContract(unittest.TestCase):
    def test_tensor_device_dtype_retained_and_output_detached(self):
        est=Estimator();q=torch.randn(7,3,dtype=torch.float32,requires_grad=True)
        out=differentiable_query_embeddings(est,q)
        self.assertEqual(out.shape,(1,7,5));self.assertEqual(out.dtype,np.float32)
        self.assertEqual(est.executor_.seen,dict(dtype='torch.float32',device='cpu',requires_grad=False,
                         task_type='regression',standard_only=False))

    def test_numpy_is_rejected_before_preprocessing(self):
        with self.assertRaises(TypeError):differentiable_query_embeddings(Estimator(),np.zeros((7,3)))


if __name__=='__main__':unittest.main(verbosity=2)
