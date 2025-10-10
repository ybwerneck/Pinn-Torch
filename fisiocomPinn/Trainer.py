from fisiocomPinn.Loss import *
from math import ceil
import time


class Trainer:

    def shuffle_data(self, *arrays):
        indices = np.random.permutation(arrays[0].shape[0])

        return tuple(array[indices] for array in arrays)

    def train_test_split(
        self,
        *arrays,
        test_size=0.5,
        shuffle=True,
    ):
        with torch.no_grad():
            if shuffle:
                arrays = self.shuffle_data(*arrays)

            # Determine train-test split index
            total_samples = arrays[0].shape[0]
            if 0 < test_size < 1:
                split_idx = total_samples - int(total_samples * test_size)
            elif isinstance(test_size, int) and 1 <= test_size < total_samples:
                split_idx = total_samples - test_size
            else:
                raise ValueError(
                    "Invalid test_size: must be a float (0 < x < 1) or int < len(data)"
                )

            # Perform the split for each array
            train_set = tuple(array[:split_idx].to(self.device) for array in arrays)
            test_set = tuple(array[split_idx:].to(self.device) for array in arrays)

            # Flatten and return
            return (*train_set, *test_set)

    def add_loss(self, loss_obj, weigth=1):
        self.losses.append(loss_obj)
        self.lossesW.append(weigth)

    def __init__(
        self,
        n_epochs,
        model,
        device="cpu",
        batch_size=1000,
        target=[],
        data=[],
        patience=300,
        tolerance=1e-3,
        val_steps=None,
        print_steps=5000,
        validation=None,
        optimizer=None,
    ):

        self.model = model.to(device)
        self.device = device
        self.validation = validation
        self.tolerance = tolerance
        self.patience = patience
        self.optimizer = optimizer
        self.print_steps = print_steps
        self.losses = []
        self.lossesW = []

        if len(data) != 0 and len(target) != 0:

            self.n_batchs = int(ceil(len(data) / batch_size))

            self.val_steps = val_steps if val_steps else self.n_batchs

            self.n_it = int(n_epochs * self.n_batchs)

            if self.validation:
                (
                    self.data_train,
                    self.target_train,
                    self.data_test,
                    self.target_test,
                ) = self.train_test_split(
                    data,
                    target,
                    test_size=self.validation,
                )

            else:

                self.data_train = data
                self.data_test = None
                self.target_train = target
                self.target_test = None

            data_loss = LOSS(
                device,
                name="Data Loss",
                batch_size=batch_size,
            )

            data_loss.add_data(
                self.data_train,
                self.target_train,
            )

            self.add_loss(data_loss)

            return

        self.n_batchs = 1
        self.n_it = n_epochs
        self.val_steps = n_epochs

        return

    def train(
        self,
    ):

        if self.losses == []:
            print("No loss function added")
            return

        loss_dict = {}

        for loss in self.losses:
            loss_dict[loss.name] = []

        loss_dict["val"] = []

        patience_count = 0
        val_min = torch.tensor([1000])
        val_loss = torch.tensor([1000])

        for it in range(self.n_it):
            start_time = time.time()  # Start timing the iteration

            self.model.zero_grad()
            self.optimizer.zero_grad()
            total_loss = 0
            losses = []

            for weighth, loss_obj in zip(self.lossesW, self.losses):

                loss = loss_obj.forward(self.model)

                total_loss += loss * weighth

                losses.append((loss * weighth).item())

                if it % self.n_batchs == 0:
                    loss_dict[loss_obj.name].append(loss.item())

            # Backward pass
            total_loss.backward()

            # Update weights
            self.optimizer.step()

            iteration_time = time.time() - start_time  # Calculate iteration duration

            if it / self.n_batchs % self.print_steps == 0:
                print(
                    "Iteration {}: total loss {:.4f}, losses: {}, time: {:.4f}s".format(
                        it // self.n_batchs,
                        total_loss.item(),
                        losses,
                        iteration_time,
                    )
                )


            # Computing validation loss

            if (it + 1) % (self.val_steps + 1) // self.n_batchs == 0:
                with torch.no_grad():

                    val_loss = torch.mean(
                        torch.sum(
                            ((self.target_test - self.model(self.data_test))) ** 2,
                            dim=1,
                        )
                        ** 0.5
                    )

                if it % self.n_batchs == 0:
                    loss_dict["val"].append(val_loss.item())

                if self.tolerance and self.validation:

                    if val_loss.item() < val_min.item() * (1 - self.tolerance):

                        val_min = val_loss

                        patience_count = 0

                        best_state = self.model.state_dict()

                        it_break = it

                    else:
                        patience_count += 1

                    if patience_count >= self.patience:

                        print(
                            "Iteration {}: total loss {:.4f}, losses: {}, time: {:.4f}s".format(
                                it // self.n_batchs,
                                total_loss.item(),
                                losses,
                                iteration_time,
                            )
                        )

                        self.model.load_state_dict(best_state)

                        print("Early break on iteration: ", it_break)

                        break

        return self.model, loss_dict
