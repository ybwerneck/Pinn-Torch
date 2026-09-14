"""Regression tests for externally configured optimizers (CPU only)."""

import unittest

import torch

from fisiocomPinn.Loss import LOSS
from fisiocomPinn.Trainer import Trainer


class TrainerOptimizerTests(unittest.TestCase):
    def make_trainer(self, adaptive=False, external=True):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        trainer = Trainer(
            1, model, adaptive=adaptive,
            optimizer=optimizer if external else None,
        )
        loss = LOSS(device="cpu", criterium="MSE", name="data")
        loss.add_data(torch.ones(1, 1), torch.ones(1, 1))
        trainer.add_loss(loss)
        return trainer, model, optimizer

    def test_external_sgd_update_and_state_are_preserved(self):
        trainer, model, optimizer = self.make_trainer()
        trained_model, history = trainer.train()
        self.assertIs(trained_model, model)
        self.assertIs(trainer.optimizer, optimizer)
        self.assertEqual(history["data"], [1.0])
        self.assertAlmostEqual(model.weight.item(), 0.2, places=6)
        trainer.train()
        self.assertAlmostEqual(model.weight.item(), 0.54, places=6)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.1)

    def test_adaptive_parameters_update_without_duplicate_groups(self):
        trainer, model, optimizer = self.make_trainer(adaptive=True)
        trainer.train()
        weights = trainer.adaptive_weights
        trainer.train()
        self.assertIs(trainer.adaptive_weights, weights)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertNotEqual(weights.log_vars.item(), 0.0)
        self.assertAlmostEqual(model.weight.item(), 0.54, places=6)

    def test_default_adam_in_both_modes(self):
        for adaptive in (False, True):
            with self.subTest(adaptive=adaptive):
                trainer, model, _ = self.make_trainer(adaptive, external=False)
                trainer.train()
                self.assertIsInstance(trainer.optimizer, torch.optim.Adam)
                self.assertGreater(model.weight.item(), 0)

    def test_rejects_invalid_optimizer(self):
        model = torch.nn.Linear(1, 1)
        with self.assertRaises(TypeError):
            Trainer(1, model, optimizer=torch.optim.SGD)
        with self.assertRaisesRegex(ValueError, "closure"):
            Trainer(1, model, optimizer=torch.optim.LBFGS(model.parameters()))
        other_model = torch.nn.Linear(1, 1)
        with self.assertRaisesRegex(ValueError, "model parameters"):
            Trainer(1, model, optimizer=torch.optim.SGD(other_model.parameters()))

    def test_rejects_changed_adaptive_loss_count(self):
        trainer, _, _ = self.make_trainer(adaptive=True)
        trainer.train()
        trainer.add_loss(LOSS(device="cpu", name="extra"))
        with self.assertRaisesRegex(ValueError, "number of adaptive losses"):
            trainer.train()


if __name__ == "__main__":
    unittest.main()
