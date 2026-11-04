"""Small observed-NPZ interface contracts; no models, truth or training loaded."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from predict import load_observed_table


class ObservedTableContract(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'observed.npz'
        self.arrays=dict(x=np.empty((20,0),np.float32),a=np.array([0,1]*10),
                         m=np.arange(20,dtype=np.float32)[:,None],y=np.arange(20,dtype=np.float32),
                         x_types=np.asarray([],dtype='U'),mediator_types=np.asarray(['continuous']))

    def write(self,**changes):
        np.savez(self.path,**dict(self.arrays,**changes));return self.path

    def test_no_covariates_and_extra_pickled_truth(self):
        observed=load_observed_table(self.write(truth=np.asarray([object()],dtype=object)))
        self.assertEqual(observed['x'].shape,(20,0))
        self.assertEqual(observed['x_types'],[])
        self.assertNotIn('truth',observed)

    def test_feature_metadata_must_match_columns(self):
        with self.assertRaisesRegex(ValueError,'mediator_types'):
            load_observed_table(self.write(mediator_types=np.asarray(['continuous','binary'])))

    def test_small_treatment_arm_rejected(self):
        with self.assertRaisesRegex(ValueError,'at least eight'):
            load_observed_table(self.write(a=np.asarray([0]*13+[1]*7)))

    def test_nonregistered_treatment_contrast_rejected(self):
        with self.assertRaisesRegex(ValueError,'treatment values'):
            load_observed_table(self.write(a=np.asarray([0,2]*10)))


if __name__=='__main__':unittest.main()
