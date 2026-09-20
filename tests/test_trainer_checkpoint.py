"""Deterministic CPU checks for checkpointing and resumption."""

import os
import tempfile
import unittest

import torch

from fisiocomPinn.Loss import LOSS
from fisiocomPinn.Trainer import Trainer


def make_trainer(
    n_epochs,
    ckpt_path=None,
    ckpt_freq=None,
    adaptive=False,
    lr=0.1,
    patience=None,
    scheduler_factory=None,
):
    """Objective (w-1)**2 from w=0 with plain SGD: a closed-form trajectory."""
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = scheduler_factory(optimizer) if scheduler_factory else None
    trainer = Trainer(
        n_epochs,
        model,
        adaptive=adaptive,
        optimizer=optimizer,
        scheduler=scheduler,
        patience=patience,
        ckpt_path=ckpt_path,
        ckpt_freq=ckpt_freq,
    )
    loss = LOSS(device="cpu", criterium="MSE", name="data")
    loss.add_data(torch.ones(1, 1), torch.ones(1, 1))
    trainer.add_loss(loss)
    return trainer, model, optimizer, scheduler


class TrainerCheckpointTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt = os.path.join(self._tmp.name, "nested", "ckpt.pt")

    def tearDown(self):
        self._tmp.cleanup()

    # -- disabled by default ------------------------------------------------

    def test_disabled_writes_nothing(self):
        trainer, _, _, _ = make_trainer(2)
        trainer.train()
        self.assertFalse(os.path.exists(self.ckpt))
        self.assertEqual(os.listdir(self._tmp.name), [])

    def test_rejects_invalid_ckpt_freq(self):
        for freq in (0, -1, 1.5, True):
            with self.subTest(freq=freq):
                with self.assertRaisesRegex(ValueError, "ckpt_freq"):
                    make_trainer(2, ckpt_path=self.ckpt, ckpt_freq=freq)

    # -- the core contract --------------------------------------------------

    def test_final_save_is_unconditional_and_atomic(self):
        # No ckpt_freq at all: the end-of-train save must still happen.
        trainer, _, _, _ = make_trainer(2, ckpt_path=self.ckpt)
        trainer.train()
        self.assertTrue(os.path.exists(self.ckpt))
        self.assertFalse(os.path.exists(self.ckpt + ".tmp"))
        state = torch.load(self.ckpt, weights_only=False)
        self.assertEqual(state["it"], 2)
        self.assertEqual(state["n_epochs"], 2)

    def test_resume_matches_uninterrupted_run(self):
        # Two epochs, stop, then resume to four -- must land exactly where a
        # single uninterrupted four-epoch run lands.
        first, _, _, _ = make_trainer(2, ckpt_path=self.ckpt)
        first.train()
        resumed, resumed_model, _, _ = make_trainer(4, ckpt_path=self.ckpt)
        resumed.train()

        straight, straight_model, _, _ = make_trainer(4)
        straight.train()

        self.assertAlmostEqual(
            resumed_model.weight.item(), straight_model.weight.item(), places=6
        )

    def test_resume_of_finished_run_is_a_noop(self):
        trainer, _, _, _ = make_trainer(2, ckpt_path=self.ckpt)
        _, history = trainer.train()
        final_weight = trainer.model.weight.item()

        again, again_model, _, _ = make_trainer(2, ckpt_path=self.ckpt)
        _, again_history = again.train()
        self.assertAlmostEqual(again_model.weight.item(), final_weight, places=6)
        self.assertEqual(len(again_history["data"]), len(history["data"]))

    def test_periodic_save_tracks_progress(self):
        trainer, _, _, _ = make_trainer(5, ckpt_path=self.ckpt, ckpt_freq=2)
        trainer.train()
        self.assertEqual(torch.load(self.ckpt, weights_only=False)["it"], 5)

    # -- issue 1: scheduler state must survive ------------------------------

    def test_scheduler_state_survives_resume(self):
        factory = lambda opt: torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.5)

        first, _, _, _ = make_trainer(2, ckpt_path=self.ckpt, scheduler_factory=factory)
        first.train()

        resumed, r_model, r_opt, r_sched = make_trainer(
            4, ckpt_path=self.ckpt, scheduler_factory=factory
        )
        resumed.train()

        straight, s_model, s_opt, s_sched = make_trainer(4, scheduler_factory=factory)
        straight.train()

        # Without restoring scheduler state the LR schedule would restart and
        # both last_epoch and the resulting weight would diverge.
        self.assertEqual(r_sched.last_epoch, s_sched.last_epoch)
        self.assertAlmostEqual(
            r_opt.param_groups[0]["lr"], s_opt.param_groups[0]["lr"], places=9
        )
        self.assertAlmostEqual(
            r_model.weight.item(), s_model.weight.item(), places=6
        )

    # -- issue 2: early-stopping state must survive -------------------------

    def test_early_stopping_budget_survives_resume(self):
        # lr=0 -> a permanent plateau. Iteration 0 only establishes best_loss,
        # so two iterations leave patience_count at 1; with patience=2 the
        # resumed run must trip on its very first iteration.
        first, _, _, _ = make_trainer(2, ckpt_path=self.ckpt, lr=0.0, patience=2)
        first.train()
        self.assertEqual(first.patience_count, 1)
        self.assertFalse(first.stopped_early)

        resumed, _, _, _ = make_trainer(
            10, ckpt_path=self.ckpt, lr=0.0, patience=2
        )
        _, history = resumed.train()

        self.assertTrue(resumed.stopped_early)
        # Three recorded evaluations in total: two restored, one new. A reset
        # budget would instead run three fresh iterations and record five.
        self.assertEqual(len(history["data"]), 3)
        self.assertEqual(resumed.n_epochs_run, 3)

    # -- issue 3: incompatible param-group layout must fail loudly ----------

    def test_rejects_mismatched_param_groups(self):
        # Adaptive + external optimizer, no scheduler -> adaptive weights get
        # their own group (2 groups).
        saver, _, saver_opt, _ = make_trainer(
            1, ckpt_path=self.ckpt, adaptive=True
        )
        saver.train()
        self.assertEqual(len(saver_opt.param_groups), 2)
        self.assertEqual(
            torch.load(self.ckpt, weights_only=False)["param_groups"], 2
        )

        # Same config but with a scheduler -> adaptive weights join group 0
        # (1 group), so the checkpoint is not loadable here.
        factory = lambda opt: torch.optim.lr_scheduler.StepLR(opt, 1)
        loader, _, loader_opt, _ = make_trainer(
            1, ckpt_path=self.ckpt, adaptive=True, scheduler_factory=factory
        )
        # torch would also fail here, but with an opaque message. Pin the
        # trainer's own diagnostic, which names the scheduler/adaptive cause.
        with self.assertRaisesRegex(
            ValueError, "cannot be resumed under a different configuration"
        ):
            loader.train()
        self.assertEqual(len(loader_opt.param_groups), 1)

    # -- adaptive weights round-trip ----------------------------------------

    def test_adaptive_weights_survive_resume(self):
        first, _, _, _ = make_trainer(2, ckpt_path=self.ckpt, adaptive=True)
        first.train()
        saved = first.adaptive_weights.log_vars.item()
        self.assertNotEqual(saved, 0.0)

        resumed, _, _, _ = make_trainer(4, ckpt_path=self.ckpt, adaptive=True)
        resumed.train()

        straight, _, _, _ = make_trainer(4, adaptive=True)
        straight.train()
        self.assertAlmostEqual(
            resumed.adaptive_weights.log_vars.item(),
            straight.adaptive_weights.log_vars.item(),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
