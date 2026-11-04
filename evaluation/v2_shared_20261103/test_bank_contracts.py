"""Frozen input/anti-leakage contracts using the four model-free ID fixtures."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
import bank

HERE=Path(__file__).resolve().parent


class BankContracts(unittest.TestCase):
    def setUp(self):
        self.root=HERE/'id_validation_fixtures';self.manifest=bank.load_manifest(self.root)

    def test_inference_loader_excludes_truth_source_and_configuration(self):
        for case in self.manifest['cases']:
            observed=bank.load_observed_case(self.root,case)
            self.assertEqual(set(observed),set(bank.OBSERVED_KEYS))
            self.assertTrue(all(observed[k].dtype==np.float32 for k in ('x','a','m','y')))
            self.assertNotIn('truth',observed);self.assertNotIn('metadata',observed)

    def test_changed_archive_bytes_rejected(self):
        case=self.manifest['cases'][0]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);p=root/case['path'];p.parent.mkdir();shutil.copyfile(self.root/case['path'],p)
            raw=bytearray(p.read_bytes());raw[-1]^=1;p.write_bytes(raw)
            with self.assertRaises(ValueError):bank.load_case(root,case)

    def test_wrong_offline_truth_rejected(self):
        case=dict(self.manifest['cases'][0]);case['truth']+=1.
        with self.assertRaises(ValueError):bank.load_case(self.root,case)

    def test_path_escape_rejected(self):
        case=dict(self.manifest['cases'][0]);case['path']='../bank.py'
        with self.assertRaises(ValueError):bank.load_case(self.root,case)

    def test_frozen_manifest_change_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'manifest.json').write_bytes((self.root/'manifest.json').read_bytes()+b' ')
            (root/'manifest.sha256').write_bytes((self.root/'manifest.sha256').read_bytes())
            with self.assertRaises(ValueError):bank.load_manifest(root)

    def test_actual_formal_bank_replicates_and_weights(self):
        m=bank.load_manifest(HERE/'frozen_bank_final')
        self.assertEqual(len(m['cases']),1152);self.assertEqual(m['unique_structural_populations'],576)
        blocks={}
        for case in m['cases']:blocks.setdefault(case['mechanism_block_id'],[]).append(case)
        for cases in blocks.values():
            self.assertEqual(len(cases),2);a,b=cases
            self.assertEqual(a['mechanism_seed'],b['mechanism_seed']);self.assertEqual(a['cluster_id'],b['cluster_id'])
            self.assertNotEqual(a['observation_seed'],b['observation_seed']);self.assertNotEqual(a['observed_sha256'],b['observed_sha256'])
        for stratum in ('id','source_ood','composition_ood'):
            for n in (200,600,1000):
                self.assertAlmostEqual(sum(c['analysis_weight_within_n'] for c in m['cases'] if c['stratum']==stratum and c['n']==n),1.)


if __name__=='__main__':unittest.main(verbosity=2)
