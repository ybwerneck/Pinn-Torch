from fisiocomPinn.Loss import *
from math import isfinite
from numbers import Integral, Real
import time
import os
from inspect import signature, Parameter
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


class AdaptiveLossWeights(nn.Module):
    def __init__(self, n_terms: int):
        super().__init__()
        # começa tudo com peso igual
        self.log_vars = nn.Parameter(torch.zeros(n_terms, dtype=torch.float32))

    def forward(self, losses):
        """
        losses: lista ou tupla de tensores escalares
        """
        total_loss = 0.0
        weighted_losses = []

        for i, loss_i in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted = precision * loss_i + self.log_vars[i]
            total_loss = total_loss + weighted
            weighted_losses.append(weighted)

        return total_loss, weighted_losses


class Trainer:
    def __init__(
        self,
        n_epochs,
        model,
        device="cpu",
        batch_size=1000,
        adaptive=True,
        patience=300,
        tolerance=1e-3,
        print_steps=5000,
        lr=1e-3,
        betas=(0.9, 0.9999),
        optimizer=None,
        scheduler=None,
        ckpt_path=None,
        ckpt_freq=None,
    ):
        """Create a trainer with an optional torch.optim.Optimizer instance.

        Move the model to its target device before constructing an external
        optimizer. Its settings and state are preserved; lr and betas only
        configure the default Adam. Optimizers requiring a closure are not
        supported by these training loops.

        An optional PyTorch scheduler must reference the supplied optimizer.
        It advances after each optimizer step. ReduceLROnPlateau receives the
        early-stopping metric. With a scheduler, adaptive parameters join the
        first optimizer group and share its settings and learning-rate schedule.

        Early stopping minimizes the fixed weighted loss, or the raw loss sum
        in adaptive mode. A decrease strictly greater than tolerance resets
        patience. Set patience=None to disable it. The final iterate is returned;
        best weights are not restored.
        """
        if patience is not None and (
            isinstance(patience, bool)
            or not isinstance(patience, Integral)
            or patience < 1
        ):
            raise ValueError("patience must be a positive integer or None")
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, Real)
            or not isfinite(tolerance)
            or tolerance < 0
        ):
            raise ValueError("tolerance must be finite and non-negative")
        if optimizer is not None:
            if not isinstance(optimizer, optim.Optimizer):
                raise TypeError("optimizer must be a torch.optim.Optimizer instance")
            closure = signature(optimizer.step).parameters.get("closure")
            if closure is not None and closure.default is Parameter.empty:
                raise ValueError("Optimizers requiring a closure are not supported")

        if scheduler is not None:
            if not isinstance(scheduler, (LRScheduler, ReduceLROnPlateau)):
                raise TypeError("scheduler must be a PyTorch LR scheduler instance")
            if optimizer is None or scheduler.optimizer is not optimizer:
                raise ValueError("scheduler must reference the supplied optimizer")

        self.scheduler = scheduler
        self.model = model.to(device)
        if optimizer is not None:
            optimizer_params = {
                id(param)
                for group in optimizer.param_groups
                for param in group["params"]
            }
            if any(
                param.requires_grad and id(param) not in optimizer_params
                for param in self.model.parameters()
            ):
                raise ValueError(
                    "optimizer must include all trainable model parameters; "
                    "move the model to device before creating the optimizer"
                )
        self.optimizer = optimizer
        self._external_optimizer = optimizer is not None
        self.adaptive_weights = None
        self.device = device
        self.tolerance = tolerance
        self.patience = patience
        self.print_steps = print_steps
        self.losses = []
        self.lossesW = []
        self.adaptive = adaptive
        self.lr = lr
        self.betas = betas
        self.n_it = n_epochs

        # Checkpointing. ckpt_path=None (the default) is a strict no-op: no
        # checkpoint I/O happens and callers that do not opt in observe no
        # behavioural change whatsoever.
        if ckpt_freq is not None and (
            isinstance(ckpt_freq, bool)
            or not isinstance(ckpt_freq, Integral)
            or ckpt_freq < 1
        ):
            raise ValueError("ckpt_freq must be a positive integer or None")
        self.ckpt_path = ckpt_path
        self.ckpt_freq = ckpt_freq

        return

    def _reset_early_stopping(self):
        """Reset monitoring state for each train call."""
        self.best_loss = float("inf")
        self.patience_count = 0
        self.stopped_early = False
        self.n_epochs_run = 0
        self.monitor_history = []

    def _step_scheduler(self, metric):
        """Advance an optional scheduler once per completed optimizer step."""
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(metric)
        elif self.scheduler is not None:
            self.scheduler.step()

    def _check_early_stopping(self, value):
        """Record the pre-update metric after a completed optimizer step."""
        self.n_epochs_run += 1
        self.monitor_history.append(value)
        if value < self.best_loss - self.tolerance:
            self.best_loss = value
            self.patience_count = 0
        else:
            self.patience_count += 1
        self.stopped_early = (
            self.patience is not None and self.patience_count >= self.patience
        )
        if self.stopped_early:
            print(
                f"Early stopping after {self.n_epochs_run} iterations: "
                f"no improvement greater than {self.tolerance} "
                f"for {self.patience_count} evaluations."
            )
        return self.stopped_early

    def save_checkpoint(self, it, loss_dict):
        """Atomically persist training state to ckpt_path; no-op if unset.

        `it` is the NEXT iteration to run on resume. The optimizer parameter
        group count is recorded so a checkpoint cannot be silently restored
        into an incompatible configuration.
        """
        if self.ckpt_path is None:
            return
        state = {
            "it": it,
            "n_epochs": self.n_it,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "param_groups": len(self.optimizer.param_groups),
            "loss_dict": loss_dict,
            "early_stopping": {
                "best_loss": self.best_loss,
                "patience_count": self.patience_count,
                "n_epochs_run": self.n_epochs_run,
                "monitor_history": list(self.monitor_history),
                "stopped_early": self.stopped_early,
            },
        }
        if self.adaptive_weights is not None:
            state["adaptive_weights"] = self.adaptive_weights.state_dict()
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        os.makedirs(os.path.dirname(self.ckpt_path) or ".", exist_ok=True)
        tmp = self.ckpt_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.ckpt_path)  # atomic: never a half-written file

    def load_checkpoint(self):
        """Restore from ckpt_path if present; returns (start_it, loss_dict|None).

        Must run AFTER the optimizer, adaptive weights and any scheduler reach
        their final configuration, since their states are restored in place.
        """
        if self.ckpt_path is None or not os.path.exists(self.ckpt_path):
            return 0, None
        state = torch.load(
            self.ckpt_path, map_location=self.device, weights_only=False
        )

        saved_groups = state.get("param_groups")
        current_groups = len(self.optimizer.param_groups)
        if saved_groups is not None and saved_groups != current_groups:
            raise ValueError(
                "Checkpoint has %d optimizer parameter group(s) but this trainer "
                "has %d. Adaptive weights occupy a new group without a scheduler "
                "and group 0 with one, so a checkpoint cannot be resumed under a "
                "different configuration." % (saved_groups, current_groups)
            )

        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.adaptive_weights is not None and "adaptive_weights" in state:
            self.adaptive_weights.load_state_dict(state["adaptive_weights"])
        if self.scheduler is not None and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])

        es = state.get("early_stopping")
        if es is not None:
            # Resume is not reuse. A fresh train() resets this state, but a
            # resumed run must continue the same patience budget, or a requeued
            # cell would train longer than one that never stopped.
            self.best_loss = es["best_loss"]
            self.patience_count = es["patience_count"]
            self.n_epochs_run = es["n_epochs_run"]
            self.monitor_history = list(es["monitor_history"])
            self.stopped_early = es["stopped_early"]

        start_it = int(state["it"])
        print(
            "[checkpoint] resumed %s at iteration %d/%s"
            % (self.ckpt_path, start_it, state.get("n_epochs", self.n_it))
        )
        return start_it, state.get("loss_dict")

    def shuffle_data(self, *arrays):
        indices = np.random.permutation(arrays[0].shape[0])

        return tuple(array[indices] for array in arrays)

    def add_loss(self, loss_obj, weigth=1):
        self.losses.append(loss_obj)
        if not self.adaptive:
            self.lossesW.append(weigth)

    def default_loop(self, loss_dict, start_it=0):
        self._resume_it = start_it
        for it in range(start_it, self.n_it):
            start_time = time.time()  # Start timing the iteration

            self.model.zero_grad()
            self.optimizer.zero_grad()
            total_loss = 0
            losses = []

            for i, (weighth, loss_obj) in enumerate(zip(self.lossesW, self.losses)):

                loss = loss_obj.forward(self.model)

                total_loss += loss * weighth

                losses.append(loss * weighth)

                loss_dict[loss_obj.name].append(loss.item())

            if not torch.isfinite(total_loss).all().item():
                raise FloatingPointError("Non-finite training loss before optimizer step")

            # Backward pass
            total_loss.backward()

            # Update weights
            self.optimizer.step()
            self._step_scheduler(total_loss.item())

            iteration_time = time.time() - start_time  # Calculate iteration duration

            if it % self.print_steps == 0:
                print(
                    "Iteration {}: total loss {:.8f}, losses: {}, time: {:.4f}s".format(
                        it,
                        total_loss.item(),
                        [los.item() for los in losses],
                        iteration_time,
                    )
                )

            self._resume_it = it + 1
            if self.ckpt_freq and it > start_it and it % self.ckpt_freq == 0:
                self.save_checkpoint(it + 1, loss_dict)

            if self._check_early_stopping(total_loss.item()):
                break

        return loss_dict

    def adaptive_loop(self, loss_dict, adaptive_weights, start_it=0):

        self._resume_it = start_it
        for it in range(start_it, self.n_it):
            start_time = time.time()  # Start timing the iteration

            self.model.zero_grad()
            self.optimizer.zero_grad()
            total_loss = 0

            losses = []

            for i, loss_obj in enumerate(self.losses):

                loss = loss_obj.forward(self.model)

                losses.append(loss)

                loss_dict[loss_obj.name].append(loss.item())

            # Adaptive weighting
            total_loss, weighted_losses = adaptive_weights(losses)

            monitor_value = sum(loss.item() for loss in losses)
            if not isfinite(monitor_value) or not torch.isfinite(total_loss).all().item():
                raise FloatingPointError("Non-finite training loss before optimizer step")

            # Backward pass
            total_loss.backward()

            # Update weights
            self.optimizer.step()
            self._step_scheduler(monitor_value)

            iteration_time = time.time() - start_time  # Calculate iteration duration

            if it % self.print_steps == 0:
                log_vars = adaptive_weights.log_vars.detach().cpu().numpy()

                print(
                    "Iteration {}: total loss {:.4f}, losses: {}, weights: {}, time: {:.4f}s".format(
                        it,
                        total_loss.item(),
                        [ten.item() for ten in losses],
                        log_vars,
                        iteration_time,
                    )
                )

            self._resume_it = it + 1
            if self.ckpt_freq and it > start_it and it % self.ckpt_freq == 0:
                self.save_checkpoint(it + 1, loss_dict)

            if self._check_early_stopping(monitor_value):
                break

        return loss_dict

    def train(
        self,
    ):

        if self.losses == []:
            print("No loss function added")
            return

        loss_dict = {}

        for loss in self.losses:
            loss_dict[loss.name] = []

        self._reset_early_stopping()

        if self.adaptive:

            if self._external_optimizer:
                if self.adaptive_weights is None:
                    self.adaptive_weights = AdaptiveLossWeights(len(self.losses)).to(
                        self.device
                    )
                    if self.scheduler is None:
                        self.optimizer.add_param_group(
                            {"params": list(self.adaptive_weights.parameters())}
                        )
                    else:
                        # Schedulers capture per-group settings at construction.
                        # Keep that group layout intact, including on reuse.
                        self.optimizer.param_groups[0]["params"].extend(
                            self.adaptive_weights.parameters()
                        )
                elif len(self.adaptive_weights.log_vars) != len(self.losses):
                    raise ValueError(
                        "Cannot change the number of adaptive losses after training "
                        "with an external optimizer; create a new trainer and optimizer"
                    )
                adaptive_weights = self.adaptive_weights
            else:
                adaptive_weights = AdaptiveLossWeights(n_terms=len(self.losses)).to(
                    self.device
                )
                self.optimizer = optim.Adam(
                    list(self.model.parameters()) + list(adaptive_weights.parameters()),
                    lr=self.lr,
                    betas=self.betas,
                )
                self.adaptive_weights = adaptive_weights

            start_it, resumed = self.load_checkpoint()
            if resumed is not None:
                loss_dict = resumed

            loss_dict = self.adaptive_loop(
                loss_dict, adaptive_weights, start_it=start_it
            )

        else:
            if not self._external_optimizer:
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.lr,
                    betas=self.betas,
                )

            start_it, resumed = self.load_checkpoint()
            if resumed is not None:
                loss_dict = resumed

            loss_dict = self.default_loop(loss_dict, start_it=start_it)

        # Final save is unconditional (independent of ckpt_freq) so the trained
        # model is always on disk when train() returns -- both as the resume
        # target and as the artifact for post-hoc analysis.
        self.save_checkpoint(getattr(self, "_resume_it", self.n_it), loss_dict)

        return self.model, loss_dict
