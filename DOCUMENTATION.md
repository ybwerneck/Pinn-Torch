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
├── Trainer.py           # Training loop with early stopping and validation
└── LICENSE.md           # GNU GPLv3 License
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
    x = torch.linspace(0, 1, batch_size).view(-1, 1).to(device)
    y = x**2
    return x, y

loss.setBatchGenerator(my_batch_gen, (0, 1))
```

---

##### `setEvalFunction(eval_func, *args)`

Defines how model outputs should be computed — useful for **physics-informed losses** or **operator residuals**.

```python
def pde_residual(batch, model, mu):
    x = batch
    y = model(x)
    dy_dx = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    return dy_dx - mu * y  # residual

loss.setEvalFunction(pde_residual, mu=0.1)
```

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
data_loss = LOSS(device="cuda", criterium="RMSE", name="Data Loss", batch_size=32)
data_loss.add_data(x, y)

# Create trainer
trainer = Trainer(
    n_epochs=500,
    model=model,
    batch_size=32,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    data=x,
    target=y,
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

Use the `Trainer.add_loss(loss_obj, weight)` method to combine multiple loss terms:

```python
trainer.add_loss(data_loss, weight=1.0)
trainer.add_loss(physics_loss, weight=0.5)
```

The total loss during training is computed as:

[
\mathcal{L}_{total} = \sum_i w_i , \mathcal{L}_i
]

---

### 4. `Trainer.py`

Manages the **training loop**, batching, validation, and early stopping.

#### `Trainer` Class

```python
from fisiocomPinn.Trainer import Trainer

trainer = Trainer(
    n_epochs=5000,
    model=my_model,
    device="cuda",
    batch_size=256,
    data=X_train,
    target=Y_train,
    optimizer=torch.optim.Adam(my_model.parameters(), lr=1e-3),
    validation=0.2,
)
```

##### Key Methods

| Method                         | Description                                                   |
| ------------------------------ | ------------------------------------------------------------- |
| `add_loss(loss_obj, weight=1)` | Add a custom loss term                                        |
| `train_test_split()`           | Split data into training/testing sets                         |
| `train()`                      | Run training loop with validation and patience-based stopping |

##### Early stopping parameters

* **`patience`**: number of iterations without improvement before stopping
* **`tolerance`**: minimum relative improvement threshold

##### Example

```python
# Add physics-informed loss
trainer.add_loss(physics_loss, weight=0.5)

# Train
trained_model, loss_history = trainer.train()
```

---

## 📊 Visualization Utilities

Functions like `default_file_val_plot()` can automatically generate result plots from a validation object by calling:

```python
default_file_val_plot(validation_object, dump=True)
```

---

## 🧠 Example Workflow

```python
from fisiocomPinn.Utils import FHN_LOSS_fromODE
from fisiocomPinn.Trainer import Trainer
from fisiocomPinn.Loss import LOSS
import torch

# Define ODE and neural network
def fhn_ode(t, y):
    v, w = y
    dvdt = v - v**3/3 - w
    dwdt = 0.08 * (v + 0.7 - 0.8 * w)
    return [dvdt, dwdt]

model = torch.nn.Sequential(
    torch.nn.Linear(1, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 2)
)

# Generate loss from ODE data
loss_obj = FHN_LOSS_fromODE(fhn_ode, (0, 20), [1.0, 0.0])

# Trainer setup
trainer = Trainer(
    n_epochs=10000,
    model=model,
    batch_size=256,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)
)

trainer.add_loss(loss_obj)
trainer.train()
```

---

## 🪪 License

This library is distributed under the **GNU General Public License v3.0**.
See [`LICENSE.md`](LICENSE.md) for full details.


