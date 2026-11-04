"""Scientific/model contracts; run in the same torch environment as training."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from model import MediationPFNV2, ModelConfig, load_checkpoint, nested_contexts, save_checkpoint, subset
from train import collate, factual_loss, make_batch
from evaluate import bayes_ratio, estimate_table, nie_score
from train_nested import nested_loss


def fake_table(n=64,p=3,q=2,seed=1):
    r=np.random.default_rng(seed)
    a=np.tile([0,1],n//2).astype(np.float32)
    r.shuffle(a)
    x=r.normal(size=(n,p)).astype(np.float32)
    m=(r.normal(size=(n,q))+.3*a[:,None]).astype(np.float32)
    y=(m.sum(1)+.4*a+r.normal(size=n)).astype(np.float32)
    return dict(x=x,a=a,m=m,y=y,truth=.3*q,mediator_types=["continuous"]*q,x_types=["continuous"]*p,
                metadata=dict(source_id="fictional",source_group="fictional",mechanism_id="fictional"))


class ModelContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        torch.set_num_threads(2)
        self.config=ModelConfig(width=32,heads=4,depth=1,inducing_tokens=4,max_x=8,max_m=128)
        self.model=MediationPFNV2(self.config)

    def test_variable_dimensions_finite_gradient(self):
        support=collate([fake_table(64,0,100),fake_table(64,3,1,2)],self.config)
        batch=dict(support=subset(support,slice(None,32)),query=subset(support,slice(32,None)))
        loss,_=factual_loss(self.model,batch,"cpu",eta_weight=0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
        self.assertIsNotNone(self.model.m_map[0].weight.grad)

    def test_pi_has_no_mediator_query_path(self):
        self.model.eval()
        table=collate([fake_table()],self.config)
        context=self.model.encode(table)
        first=self.model.predict(context,table["x"],table["m"],"pi")
        second=self.model.predict(context,table["x"],table["m"]+100,"pi")
        torch.testing.assert_close(first,second,rtol=0,atol=0)

    def test_nested_teacher_stays_frozen(self):
        teacher=MediationPFNV2(self.config).eval().requires_grad_(False)
        before={k:v.clone() for k,v in teacher.state_dict().items()}
        packed=collate([fake_table(64)],self.config)
        batch=dict(support=subset(packed,slice(None,32)),query=subset(packed,slice(32,None)))
        loss=nested_loss(self.model,teacher,batch,"cpu")
        loss.backward()
        for key,value in teacher.state_dict().items():
            torch.testing.assert_close(before[key],value,rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(any(p.grad is not None for p in self.model.parameters()))

    def test_checkpoint_roundtrip(self):
        self.model.eval()
        packed=collate([fake_table()],self.config)
        with torch.inference_mode():
            expected=self.model.predict(self.model.encode(packed),packed["x"],packed["m"],"g")
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"model.pt"
            save_checkpoint(path,self.model,stage="fictional")
            restored,_=load_checkpoint(path)
            restored.eval()
            with torch.inference_mode():
                actual=restored.predict(restored.encode(packed),packed["x"],packed["m"],"g")
            torch.testing.assert_close(expected,actual)

    def test_crossfit_output_has_full_denominator(self):
        result=estimate_table(self.model,fake_table(64),self.model,return_scores=True)
        self.assertEqual(len(result["scores"]),64)
        self.assertEqual(sum(d["query_rows"] for d in result["diagnostics"]),64)
        self.assertEqual(set(result["fold_assignment"]),{0,1})
        self.assertTrue(np.isfinite(result["estimate"]))

    def test_oracle_bayes_score_targets_nie(self):
        r=np.random.default_rng(9)
        n=200000
        alpha,beta,direct=.4,1.7,.3
        a=r.binomial(1,.5,n)
        m=alpha*a+r.normal(size=n)
        y=direct*a+beta*m+r.normal(size=n)
        pi=np.full(n,.5)
        g=1/(1+np.exp(-(alpha*m-alpha**2/2)))
        ratio=np.exp(-alpha*m+alpha**2/2)
        np.testing.assert_allclose(bayes_ratio(pi,g),ratio,rtol=1e-12)
        score=nie_score(a,y,pi,g,direct+beta*m,np.full(n,direct),np.full(n,direct+alpha*beta))
        self.assertLess(abs(score.mean()-alpha*beta),5*score.std(ddof=1)/np.sqrt(n))

    def test_generator_truth_not_needed_for_factual_batch(self):
        batch=make_batch(44,1,2,self.config,[64],16,workers=1)
        self.assertEqual(batch["query"]["a"].shape,(2,16))
        self.assertTrue(all(p["mechanism_seed"]%2==0 for p in batch["provenance"]))
        dev=make_batch(44,1,2,self.config,[64],16,dev=True,workers=1)
        self.assertTrue(all(p["mechanism_seed"]%2==1 for p in dev["provenance"]))
        self.assertFalse(set(p["mechanism_id"] for p in batch["provenance"])&set(p["mechanism_id"] for p in dev["provenance"]))


if __name__=="__main__":
    unittest.main(verbosity=2)
