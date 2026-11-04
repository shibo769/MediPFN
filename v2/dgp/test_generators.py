"""Scientific generator gates. Run: python -m dgp.test_generators."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import unittest

import numpy as np

from . import generators as g


class GeneratorTests(unittest.TestCase):
    def test_source_hashes(self):
        manifest=json.loads((Path(__file__).parent/'source_manifest.json').read_text())
        self.assertEqual(len(manifest['sources']),15)
        for src in manifest['sources']:
            p=g._VENDOR/(src['name']+'.py')
            # Vendored files preserve the source's exact LF/CRLF bytes.
            b=p.read_bytes()
            digest=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
            self.assertEqual(digest,src['sha'],p.name)

    def test_schema_and_reproducibility(self):
        for s in g.available_sources('all'):
            for ns in (17,200):
                a=g.sample_task(s,ns,379,mechanism_seed=48)
                b=g.sample_task(s,ns,379,mechanism_seed=48)
                for key in ('x','a','m','y'):
                    self.assertTrue(np.array_equal(a[key],b[key]),(s,key))
                    self.assertTrue(np.isfinite(a[key]).all())
                self.assertEqual(a['x'].shape[0],ns)
                self.assertEqual(a['m'].shape[0],ns)
                self.assertEqual(a['m'].shape[1],len(a['mediator_types']))
                self.assertEqual(a['x'].shape[1],len(a['x_types']))
                self.assertTrue(np.isfinite(a['truth']))
                self.assertTrue(set(np.unique(a['a'])) <= {0,1})
                c=g.sample_task(s,ns,380,mechanism_seed=48)
                self.assertEqual(a['mechanism_id'],c['mechanism_id'])
                self.assertEqual(a['truth'],c['truth'])
                self.assertFalse(np.array_equal(a['y'],c['y']))

    def test_ninety_native_smoke_tables(self):
        # Fifteen records (including duplicate aliases), two n, three table seeds.
        count=0
        for s in g._ALIASES:
            for n in (40,120):
                for seed in (11,12,13):
                    t=g.sample_task(s,n,seed,randomize=False)
                    self.assertTrue(np.isfinite(t['truth']))
                    count+=1
        self.assertEqual(count,90)

    def test_no_source_leakage(self):
        train=set(g.available_sources('train'))
        self.assertFalse(train&set(g.available_sources('ood_source')))
        self.assertFalse(train&set(g.available_sources('ood_composition')))
        self.assertEqual(len(set(g._FAMILIES[k] for k in train)),8)
        self.assertEqual(set(g.available_sources('dev')),train)
        self.assertEqual(set(g.available_sources('calibration')),train)
        self.assertEqual(g._FAMILIES['jones_lnl'],g._FAMILIES['jones_correlated'])
        self.assertEqual(g._FAMILIES['xia_binary'],g._FAMILIES['xia_continuous'])

    def test_full_data_duplicates(self):
        a=g.sample_task('shan',300,9,mechanism_seed=18)
        b=g.sample_task('shan_internal',300,9,mechanism_seed=18)
        for k in ('x','a','m','y'):
            self.assertTrue(np.array_equal(a[k],b[k]))
        self.assertEqual(a['truth'],b['truth'])
        self.assertEqual(a['x'].shape[1],3)
        # Fully observed data do not contain masks, shadows, event D or induced C.
        self.assertEqual(set(a.keys()),set(b.keys()))

    def test_native_analytic_truths(self):
        expected={'diaz_hejazi':-.11950701713951144,'he_song_xu':.04,
                  'jones_lnl':2.4,'jones_lnn':2.4,'jones_nnn':2.4,
                  'jones_correlated':2.,'miles':-.51,'shan':1.,
                  'eric':.2366199012683949,'xia_continuous':8.,'zuo':2.}
        for s,v in expected.items(): self.assertAlmostEqual(g.truth(s),v,places=8,msg=s)

    def test_null_paths(self):
        for s in g.available_sources('all'):
            for path in ('a_to_m','m_to_y'):
                c=g.sample_mechanism(s,96)
                g._zero_path(s,c,path)
                self.assertAlmostEqual(g.truth(s,c),0.,places=12,msg=(s,path))
            c=g.sample_mechanism(s,96)
            g._zero_path(s,c,'a_to_m');g._zero_path(s,c,'m_to_y')
            self.assertAlmostEqual(g.truth(s,c),0.,places=12,msg=(s,'both'))
        c=g.sample_mechanism('jones_lnl',randomize=False)
        c['alpha0']=[1.,1.]+[0.]*8;c['beta0']=[1.,-1.]+[0.]*8
        self.assertAlmostEqual(g.truth('jones_lnl',c),0.,places=12)

    def test_adaptations_remain_removed_under_override(self):
        for s,keys in [('cheng',['mediator_event_coefficient','outcome_event_coefficient']),
                       ('diaz_intermediate',['m_z','m_az','y_z','y_az']),
                       ('miles',['mediator_confounder','mediator_exposure_confounder','outcome_confounder']),
                       ('xia_binary',['y_c','y_ac']),('xia_continuous',['y_c','y_ac'])]:
            c=g.sample_mechanism(s,randomize=False)
            for k in keys: c[k]=[.4,.5,.6] if isinstance(c[k],list) else .8
            t=g.sample_task(s,20,1,config=c)
            for k in keys:self.assertTrue(np.all(np.asarray(t['metadata']['config'][k])==0),(s,k))

    def test_monte_carlo_conditional_truth(self):
        for s in ('cheng','diaz_hejazi','diaz_intermediate','eric','xia_binary','shan'):
            for seed in (0,2):
                c=g.sample_mechanism(s,seed)
                v=g.conditional_effect_sample(s,180000,384+seed,c)
                se=v.std(ddof=1)/np.sqrt(len(v))
                self.assertLess(abs(v.mean()-g.truth(s,c)),6*se+2e-7,(s,seed))

    def test_vectorized_observed_distribution_matches_frozen_source(self):
        """Compare independently sampled margins with the unedited frozen generator.

        Adapted confounder coefficients are passed as zero, then D/C discarded.
        Full-data missingness adaptations are validated separately by their SEM.
        """
        cases={
            'cheng':(['X1','X2','X3','X4'],'Z',['M'],'Y'),
            'diaz_hejazi':(['W1','W2','W3'],'A',['Z1','Z2','Z3'],'Y'),
            'diaz_intermediate':(['W1','W2','W3'],'A',['M'],'Y'),
            'he_song_xu':(['X1','X2'],'A',['M'],'Y'),
            'jones_lnl':(['X1','X2','X3'],'D',['M'+str(i) for i in range(1,11)],'Y'),
            'miles':(['C0'],'A',['M'],'Y'),
            'eric':(['X1','X2','X3'],'A',['M'],'Y'),
            'xia_binary':(['X'],'A',['M'],'Y'),
            'xia_continuous':(['X'],'A',['M'],'Y'),
        }
        n=18000
        for s,(xs,ak,ms,yk) in cases.items():
            path=g._VENDOR/(g._ALIASES[s]+'.py')
            spec=importlib.util.spec_from_file_location('vendor_'+s,path)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            c=g.sample_mechanism(s,randomize=False)
            v=mod.generate(c,n,874)
            native=np.column_stack([v[k] for k in xs+[ak]+ms+[yk]])
            t=g.sample_task(s,n,875,randomize=False)
            ours=np.column_stack([t['x'],t['a'],t['m'],t['y']])
            se=np.sqrt(native.var(0)/n+ours.var(0)/n)
            self.assertTrue(np.all(abs(native.mean(0)-ours.mean(0))<6*se+1e-5),s)
            # Large variance discrepancies would reveal accidental structural changes.
            ratio=ours.var(0)/np.maximum(native.var(0),1e-12)
            self.assertTrue(np.all((ratio>.75)&(ratio<1.33)),(s,ratio))

    def test_correlation_factorization(self):
        c=g.sample_mechanism('jones_correlated',randomize=False);c['p']=10
        t=g.sample_task('jones_correlated',100000,8,config=c)
        e=t['m'].astype(float)
        e[:,np.array(c['a_to_m_indices'])-1]-=t['a'][:,None]*c['a_to_m_coefficient']
        cov=np.cov(e.T)
        self.assertLess(abs(cov[0,1]-.5),.02)
        self.assertLess(abs(cov[2,3]-.5),.02)
        self.assertLess(abs(cov[0,3]-.25),.02)

    def test_parameter_diversity(self):
        for s in g.available_sources('all'):
            ids={g.sample_task(s,12,95,mechanism_seed=k)['mechanism_id'] for k in range(20)}
            self.assertGreater(len(ids),15,s)


if __name__=='__main__':
    started=time.monotonic()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(GeneratorTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'passed':result.wasSuccessful(),'tests':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),
            'seconds':time.monotonic()-started,'source_commit':g.SOURCE_COMMIT,
            'train_sources':g.available_sources('train'),
            'ood_source':g.available_sources('ood_source'),
            'ood_composition':g.available_sources('ood_composition')}
    (Path(__file__).parent/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
