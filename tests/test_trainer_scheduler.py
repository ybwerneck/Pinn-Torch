"""CPU regression checks for optional PyTorch learning-rate schedulers."""

import math
import unittest

import torch

from fisiocomPinn.Loss import LOSS
from fisiocomPinn.Trainer import Trainer


class TrainerSchedulerTests(unittest.TestCase):
    def make_trainer(self, factory, adaptive=False, n_epochs=2, **kwargs):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = factory(optimizer)
        trainer = Trainer(
            n_epochs, model, optimizer=optimizer, scheduler=scheduler,
            adaptive=adaptive, **kwargs,
        )
        loss = LOSS(device="cpu", criterium="MSE", name="data")
        loss.add_data(torch.ones(1, 1), torch.ones(1, 1))
        trainer.add_loss(loss, weigth=2.0)
        return trainer, optimizer, scheduler

    def test_step_lr_runs_after_update_and_persists(self):
        trainer, optimizer, scheduler = self.make_trainer(
            lambda opt: torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.5),
        )
        trainer.train()
        # Fixed objective 2*(w-1)**2: w=0 -> 0.4 -> 0.52.
        self.assertAlmostEqual(trainer.model.weight.item(), 0.52, places=6)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.025)
        trainer.train()
        self.assertIs(trainer.scheduler, scheduler)
        self.assertEqual(scheduler.last_epoch, 4)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.00625)

    def test_logarithmic_lambda_in_both_modes(self):
        for adaptive in (False, True):
            trainer, optimizer, scheduler = self.make_trainer(
                lambda opt: torch.optim.lr_scheduler.LambdaLR(
                    opt, lambda k: 1 / (1 + math.log1p(k)),
                ), adaptive=adaptive,
            )
            trainer.train()
            trainer.train()
            self.assertAlmostEqual(
                optimizer.param_groups[0]["lr"], 0.1 / (1 + math.log1p(4)),
            )
            self.assertEqual(len(optimizer.param_groups), 1)
            if adaptive:
                params = optimizer.param_groups[0]["params"]
                self.assertEqual(len(params), len({id(p) for p in params}))
                self.assertTrue(any(p is trainer.adaptive_weights.log_vars for p in params))
                self.assertNotEqual(trainer.adaptive_weights.log_vars.item(), 0.0)

    def test_plateau_receives_monitor_metric_in_both_modes(self):
        for adaptive in (False, True):
            trainer, optimizer, scheduler = self.make_trainer(
                lambda opt: torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="min", patience=0, factor=0.5,
                ), adaptive=adaptive, n_epochs=3,
            )
            # Zero inputs keep the model's loss at four, while adaptive weights
            # can still decrease the adaptive objective.
            trainer.losses[0].data_in.zero_()
            trainer.losses[0].target.fill_(2.0)
            trainer.train()
            self.assertEqual(scheduler.best, 4.0 if adaptive else 8.0)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.025)

    def test_scheduler_stops_with_early_stopping(self):
        for adaptive in (False, True):
            trainer, _, scheduler = self.make_trainer(
                lambda opt: torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.5),
                adaptive=adaptive, n_epochs=20, patience=2,
            )
            trainer.losses[0].data_in.zero_()
            trainer.train()
            self.assertTrue(trainer.stopped_early)
            self.assertEqual(scheduler.last_epoch, 3)

    def test_invalid_loss_does_not_advance_scheduler(self):
        trainer, _, scheduler = self.make_trainer(
            lambda opt: torch.optim.lr_scheduler.StepLR(opt, 1),
        )
        initial_epoch = scheduler.last_epoch
        trainer.losses[0].target.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            trainer.train()
        self.assertEqual(scheduler.last_epoch, initial_epoch)

    def test_rejects_invalid_or_mismatched_scheduler(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        with self.assertRaises(TypeError):
            Trainer(1, model, optimizer=optimizer, scheduler=object())
        with self.assertRaisesRegex(ValueError, "supplied optimizer"):
            Trainer(1, model, scheduler=scheduler)
        other_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with self.assertRaisesRegex(ValueError, "supplied optimizer"):
            Trainer(1, model, optimizer=other_optimizer, scheduler=scheduler)


if __name__ == "__main__":
    unittest.main()
