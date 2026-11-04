"""NumPy-only tests of the actual calibration functions, without requiring torch."""
import ast
from pathlib import Path
import unittest
import numpy as np

# Load precisely the product's pure numerical functions; no duplicated algorithm.
tree=ast.parse(Path(__file__).with_name('train_uncertainty.py').read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('calibrate','quantile_indices')]
namespace={'np':np};exec(compile(ast.Module(body=functions,type_ignores=[]),'<actual calibration functions>','exec'),namespace)
calibrate=namespace['calibrate'];quantile_indices=namespace['quantile_indices']


class CalibrationMath(unittest.TestCase):
    def test_correct_finite_sample_order_statistic(self):
        errors=np.linspace(-3,3,512);q=np.zeros((512,6));c=calibrate(errors,q)
        for level in (.9,.95,.99):
            row=c[str(round(level*100))]
            self.assertEqual(row['rank'],int(np.ceil(513*level)))
            self.assertGreaterEqual(np.mean(abs(errors)<=row['correction']),row['rank']/512)

    def test_nested_endpoint_corrections_and_quantile_order(self):
        errors=np.linspace(-1,1,512);q=np.tile([-10,-9,0,0,9,10.],(512,1));c=calibrate(errors,q)
        intervals=[]
        for level in (.9,.95,.99):
            i,j=quantile_indices(level);delta=c[str(round(level*100))]['correction']
            intervals.append((-q[0,j]-delta,-q[0,i]+delta))
        self.assertLessEqual(intervals[2][0],intervals[1][0]);self.assertLessEqual(intervals[1][0],intervals[0][0])
        self.assertGreaterEqual(intervals[2][1],intervals[1][1]);self.assertGreaterEqual(intervals[1][1],intervals[0][1])

    def test_asymmetric_error_sign(self):
        errors=np.full(512,2.);q=np.tile([-3,-2,-1,1,2,3.],(512,1));c=calibrate(errors,q)
        # Positive estimator error requires the interval to extend below estimate.
        i,j=quantile_indices(.9);d=c['90']['correction'];estimate=10.;scale=.5;truth=estimate-2*scale
        lo=estimate-scale*(q[0,j]+d);hi=estimate-scale*(q[0,i]-d)
        self.assertLessEqual(lo,truth);self.assertGreaterEqual(hi,truth)

    def test_insufficient_calibration_tasks_rejected(self):
        with self.assertRaises(ValueError):calibrate(np.zeros(12),np.zeros((12,6)))


if __name__=='__main__':unittest.main(verbosity=2)
