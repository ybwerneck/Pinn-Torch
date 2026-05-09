from fisiocomPinn.Loss import *
from math import ceil
import time


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
    ):

        self.model = model.to(device)
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

        return

    def shuffle_data(self, *arrays):
        indices = np.random.permutation(arrays[0].shape[0])

        return tuple(array[indices] for array in arrays)

    def add_loss(self, loss_obj, weigth=1):
        self.losses.append(loss_obj)
        if not self.adaptive:
            self.lossesW.append(weigth)

    def default_loop(self, loss_dict):
        for it in range(self.n_it):
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

            # Backward pass
            total_loss.backward()

            # Update weights
            self.optimizer.step()

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

        return loss_dict

    def adaptive_loop(self, loss_dict, adaptive_weights):

        for it in range(self.n_it):
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

            # Backward pass
            total_loss.backward()

            # Update weights
            self.optimizer.step()

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

        patience_count = 0

        if self.adaptive:

            adaptive_weights = AdaptiveLossWeights(n_terms=len(self.losses)).to(
                self.device
            )

            self.optimizer = optim.Adam(
                list(self.model.parameters()) + list(adaptive_weights.parameters()),
                lr=self.lr,
                betas=self.betas,
            )

            loss_dict = self.adaptive_loop(loss_dict, adaptive_weights)

        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                betas=self.betas,
            )

            loss_dict = self.default_loop(loss_dict)

        return self.model, loss_dict
