"""Deterministic CPU checks for the validator hook in both training loops."""

import unittest

import torch

from fisiocomPinn.Loss import LOSS
from fisiocomPinn.Trainer import Trainer


class RecordingValidator:
    """Minimal object satisfying the validator interface: .val(model)."""

    def __init__(self):
        self.seen = []

    def val(self, model):
        self.seen.append(model)


class TrainerValidatorTests(unittest.TestCase):
    def make_trainer(self, adaptive=False, n_epochs=10):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2.0)
        # lr=0 keeps the objective flat; patience=None keeps the full budget,
        # so the iteration count is exact and the hook timing is unambiguous.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        trainer = Trainer(
            n_epochs, model, adaptive=adaptive, optimizer=optimizer,
            patience=None,
        )
        loss = LOSS(device="cpu", criterium="MSE", name="data")
        loss.add_data(torch.ones(1, 1), torch.zeros(1, 1))
        trainer.add_loss(loss)
        return trainer, model

    def test_runs_on_frequency_in_both_modes(self):
        for adaptive in (False, True):
            with self.subTest(adaptive=adaptive):
                trainer, _ = self.make_trainer(adaptive)
                validator = RecordingValidator()
                trainer.add_validator(validator, freq=3)
                trainer.train()
                # iterations 0..9, fired on 0, 3, 6, 9
                self.assertEqual(len(validator.seen), 4)

    def test_receives_the_trained_model(self):
        trainer, model = self.make_trainer()
        validator = RecordingValidator()
        trainer.add_validator(validator, freq=10)
        trainer.train()
        self.assertEqual(len(validator.seen), 1)
        self.assertIs(validator.seen[0], model)

    def test_multiple_validators_keep_independent_frequencies(self):
        trainer, _ = self.make_trainer()
        every_two, every_five = RecordingValidator(), RecordingValidator()
        trainer.add_validator(every_two, freq=2)
        trainer.add_validator(every_five, freq=5)
        trainer.train()
        self.assertEqual(len(every_two.seen), 5)   # 0,2,4,6,8
        self.assertEqual(len(every_five.seen), 2)  # 0,5

    def test_absent_validators_is_a_noop(self):
        for adaptive in (False, True):
            with self.subTest(adaptive=adaptive):
                trainer, _ = self.make_trainer(adaptive)
                self.assertFalse(hasattr(trainer, "validators"))
                trainer.train()  # must not raise

    def test_registration_persists_across_train_calls(self):
        trainer, _ = self.make_trainer()
        validator = RecordingValidator()
        trainer.add_validator(validator, freq=10)
        trainer.train()
        trainer.train()
        self.assertEqual(len(validator.seen), 2)


if __name__ == "__main__":
    unittest.main()
