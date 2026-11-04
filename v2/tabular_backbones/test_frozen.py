"""Offline orchestration/leakage contracts; mocks are explicitly not model results."""
import unittest
import numpy as np
from .frozen import BackboneConfig,estimate_table


class MeanModel:
    def __init__(self,classification,log):self.classification=classification;self.log=log
    def fit(self,x,y):
        self.mean=float(np.mean(y));self.classes_=np.array([0,1]);self.log.append((x.copy(),y.copy()));return self
    def predict(self,q):return np.full(len(q),self.mean)
    def predict_proba(self,q):return np.tile([1-self.mean,self.mean],(len(q),1))


class Contracts(unittest.TestCase):
    def setUp(self):
        self.config=BackboneConfig('v2','mock','unused','0'*64,'unused','0'*64)
        r=np.random.default_rng(7);n=128
        self.table=dict(x=r.normal(size=(n,3)),a=np.tile([0.,1.],n//2),
                        m=r.normal(size=(n,2)),y=np.arange(n,dtype=float)+r.normal(size=n))

    def test_no_truth_argument_and_nested_fits(self):
        log=[]
        factory=lambda c,classification,seed:MeanModel(classification,log)
        out=estimate_table(self.table,self.config,seed=18,factory=factory,return_scores=True)
        # 7 fits per outer fold: pi,g,mu,inner-mu twice,eta0,eta1.
        self.assertEqual(len(log),14)
        self.assertEqual(len(out['scores']),len(self.table['a']))
        fold=np.asarray(out['fold_assignment'])
        for k in (0,1):
            query=np.flatnonzero(fold==k)
            support=set(np.flatnonzero(fold!=k))
            # Every full outcome-fit label comes exclusively from outer support.
            labels=log[k*7+2][1]
            self.assertEqual(set(labels),set(self.table['y'][list(support)]))
            self.assertFalse(set(labels)&set(self.table['y'][query]))
        self.assertTrue(np.isfinite(out['estimate']))

    def test_truth_cannot_affect_output(self):
        factory=lambda c,classification,seed:MeanModel(classification,[])
        a=estimate_table(dict(self.table,truth=-1000),self.config,factory=factory)
        b=estimate_table(dict(self.table,truth=1000),self.config,factory=factory)
        self.assertEqual(a['estimate'],b['estimate'])
        self.assertEqual(a['standard_error'],b['standard_error'])

    def test_no_baseline_and_vector_m(self):
        t=dict(self.table,x=np.empty((128,0)),m=np.ones((128,100)))
        out=estimate_table(t,self.config,factory=lambda c,cl,s:MeanModel(cl,[]))
        self.assertTrue(np.isfinite(out['estimate']))

    def test_single_arm_rejected(self):
        t=dict(self.table,a=np.ones(128))
        with self.assertRaises(ValueError):estimate_table(t,self.config)


if __name__=='__main__':unittest.main(verbosity=2)
