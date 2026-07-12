import random
import numpy as np
import torch
from sdhtg.losses.class_balanced import effective_number_weights, class_balanced_focal_loss
from sdhtg.losses.contrastive import supervised_info_nce
from sdhtg.training.curriculum import Curriculum
from sdhtg.training.reproducibility import seed_everything


def test_minority_class_receives_larger_weight():
    weights=effective_number_weights(torch.tensor([1000,10]))
    assert weights[1]>weights[0]


def test_focal_loss_has_gradient():
    logits=torch.tensor([0.1,-0.2],requires_grad=True); labels=torch.tensor([1.,0.])
    loss=class_balanced_focal_loss(logits,labels,torch.tensor([100,5])); loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_contrastive_identical_pairs_are_better():
    torch.manual_seed(1); z=torch.randn(8,16); shuffled=z.roll(1,0)
    assert supervised_info_nce(z,z)<supervised_info_nce(z,shuffled)


def test_curriculum_is_monotonic():
    cfg={"warmup_epochs":2,"boundary_initial_temperature":1.,"boundary_final_temperature":.1,
         "film_initial_strength":0.,"film_final_strength":1.,"boundary_loss_warmup_epochs":5}
    states=[Curriculum(cfg).at(e,10) for e in range(10)]
    assert all(a.boundary_temperature>=b.boundary_temperature for a,b in zip(states,states[1:]))
    assert all(a.film_strength<=b.film_strength for a,b in zip(states,states[1:]))


def test_seed_reproducibility():
    seed_everything(42); first=(random.random(),np.random.rand(),torch.rand(2))
    seed_everything(42); second=(random.random(),np.random.rand(),torch.rand(2))
    assert first[0]==second[0] and first[1]==second[1] and torch.equal(first[2],second[2])
