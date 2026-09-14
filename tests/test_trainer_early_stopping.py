"""Deterministic CPU checks for early stopping in both training loops."""

import unittest

import torch

from fisiocomPinn.Loss import LOSS
from fisiocomPinn.Trainer import Trainer


class TrainerEarlyStoppingTests(unittest.TestCase):
    def make_trainer(self, adaptive=False, patience=2, tolerance=0.0, lr=0.0):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        trainer = Trainer(
            10, model, adaptive=adaptive, optimizer=optimizer,
            patience=patience, tolerance=tolerance,
        )
        loss = LOSS(device="cpu", criterium="MSE", name="data")
        loss.add_data(torch.ones(1, 1), torch.zeros(1, 1))
        trainer.add_loss(loss, weigth=3.0)
        return trainer

    def test_plateau_stops_at_exact_patience_and_resets_on_reuse(self):
        for adaptive in (False, True):
            trainer = self.make_trainer(adaptive)
            for _ in range(2):
                _, history = trainer.train()
                self.assertTrue(trainer.stopped_early)
                self.assertEqual(trainer.n_epochs_run, 3)
                self.assertEqual(history["data"], [4.0] * 3)
                self.assertEqual(trainer.monitor_history, [4.0 if adaptive else 12.0] * 3)

    def test_disabled_stopping_completes_budget(self):
        for adaptive in (False, True):
            trainer = self.make_trainer(adaptive, patience=None)
            trainer.train()
            self.assertEqual(trainer.n_epochs_run, 10)
            self.assertFalse(trainer.stopped_early)

    def test_improving_loss_completes_budget(self):
        for adaptive in (False, True):
            trainer = self.make_trainer(adaptive, lr=0.01)
            trainer.train()
            self.assertEqual(trainer.n_epochs_run, 10)
            self.assertEqual(trainer.patience_count, 0)
            self.assertLess(trainer.monitor_history[-1], trainer.monitor_history[0])

    def test_tolerance_accumulates_small_improvements(self):
        trainer = self.make_trainer(tolerance=0.5)
        trainer._reset_early_stopping()
        for value in (4.0, 3.75, 3.375, 3.0):
            self.assertFalse(trainer._check_early_stopping(value))
        self.assertTrue(trainer._check_early_stopping(2.875))
        self.assertEqual(trainer.best_loss, 3.375)

    def test_adaptive_weight_changes_do_not_hide_plateau(self):
        trainer = self.make_trainer(adaptive=True)
        trainer.train()
        trainer.optimizer.param_groups[1]["lr"] = 0.1
        trainer.train()
        self.assertNotEqual(trainer.adaptive_weights.log_vars.item(), 0.0)
        self.assertEqual(trainer.n_epochs_run, 3)
        self.assertEqual(trainer.monitor_history, [4.0] * 3)

    def test_nonfinite_loss_fails_before_update(self):
        for adaptive in (False, True):
            for value in (float("nan"), float("inf")):
                trainer = self.make_trainer(adaptive, lr=0.1)
                trainer.losses[0].target.fill_(value)
                with self.assertRaises(FloatingPointError):
                    trainer.train()
                self.assertEqual(trainer.model.weight.item(), 2.0)
                self.assertEqual(trainer.n_epochs_run, 0)

    def test_invalid_configuration(self):
        for patience in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                self.make_trainer(patience=patience)
        for tolerance in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                self.make_trainer(tolerance=tolerance)

    def test_physics_derivative_remains_differentiable(self):
        # For u(x)=a*x and u'(x)=0, the residual MSE is a**2.
        for adaptive in (False, True):
            trainer = self.make_trainer(adaptive, lr=0.01)

            def residual(batch, model):
                x = batch.detach().clone().requires_grad_(True)
                u = model(x)
                return torch.autograd.grad(
                    u, x, grad_outputs=torch.ones_like(u), create_graph=True,
                )[0]

            trainer.losses[0].setEvalFunction(residual)
            _, history = trainer.train()
            self.assertEqual(history["data"][0], 4.0)
            self.assertLess(history["data"][-1], 4.0)
            self.assertLess(trainer.model.weight.item(), 2.0)
            self.assertEqual(trainer.n_epochs_run, 10)


if __name__ == "__main__":
    unittest.main()
