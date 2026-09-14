# 🧠 FisiocomPINN Library

A lightweight **Physics-Informed Neural Network (PINN)** framework designed for flexible training, loss composition, and dataset handling — built with **PyTorch**.

> Developed by **Yan Werneck** and **Thiago Esterci** under the [FisiocomPINN framework](https://github.com/ybwerneck/Pinn-Torch).

---

## 📦 Installation

```bash
git clone https://github.com/ybwerneck/Pinn-Torch.git
cd Pinn-Torch
pip install -e .
```

### Dependencies

Installed automatically via `setup.py`:

* `torch>=2.0.0`
* `numpy>=1.24`
* `matplotlib>=3.7`
* `h5py>=3.8`
* `chaospy>=4.3`

---

## 🧩 Library Overview

```
fisiocomPinn/
├── dependencies.py      # Utility imports and folder management
├── Utils.py             # Dataset generation, loading, and validation helpers
├── Loss.py              # Generic and custom loss functions (MSE, RMSE, LP, etc.)
├── Trainer.py           # Training with configurable optimizers and loss weights
├── Net.py               # Fully connected networks
├── Loss_PINN.py         # Physics and initial-condition losses
└── Validator.py         # Validation and result export
```

---

## ⚙️ Core Modules

### 1. `dependencies.py`

Utility imports and helper functions.

---

### 2. `Utils.py`

Contains dataset utilities, batch generation, and FitzHugh–Nagumo (FHN) examples.

#### `default_batch_generator(size, ranges, device)`

Generates uniform random batches within given variable ranges.

```python
batch = default_batch_generator(1000, [(0, 1), (-1, 1)], "cuda")
```

#### `cp_batch_generator(size, ranges)`

Uses **Chaospy** to generate samples from uniform or normal distributions.

```python
batch = cp_batch_generator(500, [(0, 1), (0.5, 0.5)])
```

#### `LoadDataSet(folder, data_in, data_out, device, dtype)`

Loads `.npy` datasets from disk.

```python
X, Y = LoadDataSet("data/", ["T.npy"], ["SOLs.npy"], device="cuda")
```

#### `FHN_LOSS_fromODE(...)` / `FHN_VAL_fromODE(...)`

Generate training losses and validation datasets from **ODE systems**, using SciPy’s `solve_ivp`.

---

### 3. `Loss.py`

The `LOSS` class (defined in `Loss.py`) is the **core abstraction for handling losses** in the FisiocomPINN framework.

It allows:

* flexible batching from stored datasets or generators,
* custom evaluation functions (e.g. PDE residuals),
* multiple loss combination (via `Trainer.add_loss`),
* and easy integration with neural network training.

---

#### 🔧 Class Definition

```python
from fisiocomPinn.Loss import LOSS
```

```python
class LOSS(torch.nn.Module):
    def __init__(
        self,
        device=torch.device("cuda"),
        criterium="RMSE",
        name="Loss",
        batch_size=10000,
    )
```

| Parameter      | Type           | Default  | Description                                         |
| -------------- | -------------- | -------- | --------------------------------------------------- |
| **device**     | `torch.device` | `'cuda'` | Device where tensors and computations are performed |
| **criterium**  | `str`          | `"RMSE"` | The loss metric name (see below)                    |
| **name**       | `str`          | `"Loss"` | Name used to identify this loss in logs             |
| **batch_size** | `int`          | `10000`  | Batch size used when sampling from data             |

---

#### ⚗️ Supported Criteria (`loss_map`)

The following built-in loss types are defined internally:

| Key                      | Description                      | Formula                                           |             |   |
| ------------------------ | -------------------------------- | ------------------------------------------------- | ----------- | - |
| `"MAE"`                  | Mean Absolute Error              | (\frac{1}{N}\sum                                  | y - \hat{y} | ) |
| `"MSE"`                  | Mean Squared Error               | (\frac{1}{N}\sum (y - \hat{y})^2)                 |             |   |
| `"RMSE"`                 | Root Mean Squared Error          | (\sqrt{\frac{1}{N}\sum (y - \hat{y})^2})          |             |   |
| `"KLDivergenceLoss"`     | KL Divergence                    | (D_{KL}(p | q))                                   |             |   |
| `"CosineSimilarityLoss"` | Cosine distance (1 - similarity) | (1 - \cos(y, \hat{y}))                            |             |   |
| `"LPthLoss"`             | General L<sub>p</sub> norm       | (|y - \hat{y}|_p)                                 |             |   |
| `"L2"`                   | Normalized L2 distance           | (\frac{|y - \hat{y}|_2}{|y|_2 + \varepsilon})     |             |   |
| `"L2_squared"`           | Squared normalized L2            | (\frac{|y - \hat{y}|_2^2}{|y|_2^2 + \varepsilon}) |             |   |

You can pass any of these keys to the constructor as the `criterium` argument.

---

#### 🧰 Methods

##### `add_data(data_in, target)`

Registers in-memory data for supervised learning.

```python
loss = LOSS(criterium="MSE")
loss.add_data(X_train, Y_train)
```

---

##### `getBatch()`

Returns the next `(inputs, targets)` batch for training, using `batch_size`.

If the end of the dataset is reached, it wraps around automatically.

```python
batch, tgt = loss.getBatch()
```

---

##### `setBatchGenerator(batch_generator, *args)`

Links a **custom batch generation function** to dynamically produce training points (for PINNs, this usually generates collocation points).

```python
def my_batch_gen(batch_size, device, range_):
    x = torch.linspace(*range_, batch_size, device=device).view(-1, 1)
    y = x**2
    return x, y

loss.setBatchGenerator(my_batch_gen, (0, 1))
```

---

##### `setEvalFunction(eval_func, *args)`

Defines how model outputs should be computed — useful for **physics-informed losses** or **operator residuals**.

```python
def pde_residual(batch, model, mu):
    x = batch.requires_grad_(True)
    y = model(x)
    dy_dx = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    return dy_dx - mu * y  # residual

loss.setEvalFunction(pde_residual, 0.1)
```

Extra evaluation arguments are positional. For a homogeneous residual equation,
configure the batch generator to return zero targets with the residual shape.

---

##### `forward(model, *loss_args)`

Computes the current loss value.

The logic is:

1. Get batch data (from dataset or generator),
2. Compute prediction (via model or custom `eval_func`),
3. Apply criterium.

```python
value = loss(model)
print("Current loss:", value.item())
```

If `batchGen` and `eval_func` are not defined, the loss will use internal `data_in` and `target`.

---

#### 🧩 Integration Example

```python
from fisiocomPinn.Loss import LOSS
from fisiocomPinn.Trainer import Trainer
import torch

# Example model
model = torch.nn.Sequential(
    torch.nn.Linear(1, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 1)
)

# Prepare data
x = torch.linspace(0, 1, 100).view(-1, 1)
y = torch.sin(2 * torch.pi * x)

# Create loss object
data_loss = LOSS(device="cpu", criterium="RMSE", name="Data Loss", batch_size=32)
data_loss.add_data(x, y)

# Create trainer
trainer = Trainer(
    n_epochs=500,
    model=model,
    batch_size=32,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
)

trainer.add_loss(data_loss)
trainer.train()
```

---

#### ⚙️ Extending `LOSS`

You can easily define **custom loss functions** by extending `loss_map` in `Loss.py`:

```python
loss_map["L1Smooth"] = lambda tgt, pred: torch.mean(torch.sqrt((tgt - pred)**2 + 1e-6))
```

Then:

```python
loss = LOSS(criterium="L1Smooth")
```

---

#### 🔄 Combined Loss Example

For fixed weights, create the trainer with `adaptive=False` and register each
loss using `Trainer.add_loss(loss_obj, weigth=1)`. The keyword is spelled
`weigth` in the current API:

```python
trainer.add_loss(data_loss, weigth=1.0)
trainer.add_loss(physics_loss, weigth=0.5)
```

The total loss during training is computed as:

$$
\mathcal{L}_{total} = \sum_i w_i \mathcal{L}_i
$$

---

### 4. `Trainer.py`

Manages training with fixed or adaptive loss weights. By default, the trainer
creates Adam using `lr` and `betas`. Pass an optimizer instance to use its own
hyperparameters, parameter groups and existing state instead.

#### Constructor

```python
Trainer(
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
)
```

| Parameter | Behavior |
| --------- | -------- |
| `n_epochs` | Maximum number of optimizer steps per `train()` call |
| `model` | PyTorch module to train |
| `device` | Target device; move the model there before constructing an external optimizer |
| `optimizer` | Optimizer instance, not its class; `None` creates Adam |
| `scheduler` | Optional PyTorch LR scheduler instance bound to the supplied `optimizer`; `None` leaves the learning rate unchanged by Trainer |
| `lr`, `betas` | Settings for the default Adam only |
| `adaptive` | Learn loss weights when `True`; use `add_loss` weights when `False` |
| `print_steps` | Logging interval in iterations; use a positive integer |
| `batch_size` | Currently unused by Trainer; configure batching on each `LOSS` |
| `patience` | Consecutive evaluations without sufficient improvement before stopping; positive integer, or `None` to disable (default: 300) |
| `tolerance` | Minimum absolute decrease required to reset patience; finite and non-negative (default: 1e-3) |

#### Complete example: external SGD

This CPU example fits `y = 2x` using a supervised loss and fixed weighting.

```python
import torch
from fisiocomPinn.Loss import LOSS
from fisiocomPinn.Trainer import Trainer

torch.manual_seed(0)
device = "cpu"
model = torch.nn.Linear(1, 1).to(device)
x = torch.linspace(-1, 1, 32, device=device).view(-1, 1)
y = 2 * x

data_loss = LOSS(device=device, criterium="MSE", name="data", batch_size=32)
data_loss.add_data(x, y)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
trainer = Trainer(
    n_epochs=100,
    model=model,
    device=device,
    optimizer=optimizer,
    adaptive=False,
    print_steps=50,
)
trainer.add_loss(data_loss, weigth=1.0)
trained_model, loss_history = trainer.train()
print(loss_history["data"][-1])
```

To choose a different optimizer for a new trainer, construct it with the model
parameters and its own settings:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
trainer = Trainer(n_epochs=100, model=model, device=device, optimizer=optimizer)
trainer.add_loss(data_loss)
trained_model, loss_history = trainer.train()
```

To retain the original Adam behavior, omit `optimizer`:

```python
trainer = Trainer(n_epochs=100, model=model, device=device, lr=1e-3)
trainer.add_loss(data_loss)
trained_model, loss_history = trainer.train()
```

#### Optimizer lifecycle and adaptive weights

`optimizer` must be a `torch.optim.Optimizer` instance containing all trainable
model parameters. For example, SGD, AdamW and RMSprop can be passed this way.
Trainer's `lr` and `betas` are ignored when an external optimizer is provided.
Optimizers requiring a closure, such as LBFGS, are currently rejected explicitly.

With `adaptive=True` (the default) and no scheduler, the trainer adds a parameter group for the
learnable loss weights on the first `train()` call. That group inherits the
optimizer defaults. Repeated calls reuse the external optimizer and adaptive
weights without adding duplicate groups. Register all losses before the first
call: changing their number afterward requires a new trainer and optimizer.
Keep the loss order and meaning unchanged when continuing an adaptive run.
Create a separate optimizer for each independent trainer.

With adaptive weighting, the objective is
`sum(exp(-s_i) * loss_i + s_i)`, where `s_i` are learned log variables.
The fixed `weigth` argument is ignored in this mode. The additional parameter
group inherits optimizer defaults, including weight decay when configured.

When `optimizer=None`, each `train()` call creates a fresh Adam and, in adaptive
mode, fresh loss weights. Model parameters retain their current values.
With an external optimizer, its state (such as momentum) persists across calls.

#### Optional learning-rate scheduler

Pass an optimizer and a scheduler constructed with that same optimizer.
Omitting `scheduler` (or setting it to `None`) preserves optimizer-only behavior.
The scheduler advances once after each optimizer update, including the final
update that triggers early stopping. Its state persists across `train()` calls.
Schedule durations therefore count optimizer steps, not full dataset passes.

For a logarithmic learning-rate decay, using the `model` and `data_loss` above:

```python
import math

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: 1.0 / (1.0 + math.log1p(step)),
)
trainer = Trainer(
    n_epochs=10000, model=model, optimizer=optimizer,
    scheduler=scheduler, patience=500, tolerance=1e-6,
)
trainer.add_loss(data_loss)
trained_model, loss_history = trainer.train()
```

This schedules the learning rate, not the loss value. `StepLR`, `LambdaLR`, and
other PyTorch schedulers with a parameterless `step()` use the same interface.
`ReduceLROnPlateau` instead receives the pre-update training metric: the fixed
weighted sum in fixed mode, or the raw loss sum in adaptive mode. Use `mode="min"`
to reduce the learning rate when loss stops improving. No validation set is
evaluated automatically. Its patience and thresholds are independent of early
stopping; allow enough training iterations after reductions to assess their effect.

When a scheduler is supplied with `adaptive=True`, the learned loss parameters
join the **first existing optimizer group** on the first `train()` call. They
share all its settings, including learning rate, weight decay, and momentum.
This preserves the group layout already captured by the scheduler. Repeated
calls do not duplicate these parameters. Without a scheduler, the separate-group
behavior described above remains unchanged. Register losses before training and
keep the optimizer's parameter-group layout unchanged during the run.

Passing a scheduler without an explicit optimizer, or one bound to a different
optimizer, raises `ValueError`. Pass an instance, not a scheduler class.

#### Results and current limitations

`train()` returns `(model, loss_history)`. Each history key is a registered loss
name and contains raw, unweighted loss values measured before each update.
Use unique names for losses. History starts afresh on each `train()` call.
If no loss has been registered, the method prints a message and returns `None`.
The trainer does not automatically split datasets or run `Validator`.

| Method | Description |
| ------ | ----------- |
| `add_loss(loss_obj, weigth=1)` | Register a loss; fixed weights apply when `adaptive=False` |
| `train()` | Return the trained model and per-loss history |

#### Early stopping

Both loops now use `patience` and `tolerance`. Existing runs may therefore stop
earlier than before; use `patience=None` to retain the full iteration budget.

```python
trainer = Trainer(
    n_epochs=10000, model=model, adaptive=True,
    patience=500, tolerance=1e-6,
)
trainer.add_loss(data_loss)
trained_model, loss_history = trainer.train()
print(trainer.stopped_early, trainer.n_epochs_run)
```

The monitored metric is the fixed weighted sum for `adaptive=False`, and the
sum of raw losses for `adaptive=True`. This keeps learned weight changes out of
the stopping criterion. The latter sum is scale-dependent: normalize loss terms
appropriately when combining physical quantities with different units or scales.
A value strictly below `best_loss - tolerance` becomes the new reference and
resets the counter; smaller improvements accumulate relative to that reference.
Thus `best_loss` stores the last significant improvement, not necessarily the
smallest observed value. `tolerance` is an absolute decrease, not a target loss.

The first finite evaluation establishes the reference. With `patience=2`, a
constant loss stops after three optimizer steps. Metrics are measured before
each update, and the stopping decision follows that update. The returned model
and optimizer retain their last state; best weights are not restored.
`monitor_history`, `best_loss`, `patience_count`, `stopped_early`, and
`n_epochs_run` describe the current call and reset at the next `train()` call.
Non-finite objectives raise `FloatingPointError` before the optimizer update.

This criterion monitors training batches, which may be resampled or noisy; it
does not evaluate an independent validation set. A plateau does not establish
physical accuracy. Evaluate solution errors, residuals, and initial/boundary
conditions separately on independent points.

Run optimizer regression tests in an environment with the package dependencies:

```bash
python -B -m unittest discover -s tests -v
```

---

## 🪪 License

This library is distributed under the **GNU General Public License v3.0**.
See [`LICENSE.md`](LICENSE.md) for full details.
