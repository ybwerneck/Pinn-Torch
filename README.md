# 🧠 FisiocomPINN — Physics-Informed Neural Networks in PyTorch

**FisiocomPINN** is a modular Python library built on **PyTorch** for the development, training, and validation of **Physics-Informed Neural Networks (PINNs)**.
It is designed for **scientific computing** and **computational modeling**, integrating physical laws (PDEs/ODEs) into neural network training.

This framework was developed as part of a **PhD research project** within the *Fisiocomp Group* at the Federal University of Juiz de Fora (UFJF), Brazil.

---

## 📚 Main Features

* 🧩 **General-purpose PINN architecture** for solving PDEs and ODEs
* ⚙️ **Modular losses**: MSE, MAE, RMSE, LP-norms, KL divergence, cosine similarity, etc.
* 🧠 **Training engine** with early stopping, validation, and adaptive learning rate
* 🔢 **Utility functions** for dataset generation, stochastic sampling, and plotting
* 💻 **GPU/CPU compatible** using native PyTorch tensors

---

## 📁 Project Structure

```
Pinn-Torch/
│
├── examples/
│   └── Example_PINN_EDO.ipynb        # Example: PINN modeling the innate immune system response
│
├── fisiocomPinn/
│   ├── dependencies.py               # Core dependencies and shared utilities
│   ├── Utils.py                      # Data loaders, ODE solvers, and validation helpers
│   ├── Loss.py                       # Custom loss functions for PINN training
│   ├── Trainer.py                    # Training loop, early stopping, and optimizer management
│
├── setup.py                          # Installation and packaging script
├── LICENSE.md                        # GNU GPL v3 license
└── README.md                         # Project documentation (this file)
```

---

## ⚙️ Installation

### 🧩 Option 1 — From Source

```bash
git clone https://github.com/ybwerneck/Pinn-Torch.git
cd Pinn-Torch
pip install .
```

### 🧪 Option 2 — Development Mode

```bash
git clone https://github.com/ybwerneck/Pinn-Torch.git
cd Pinn-Torch
pip install -e .
```

This installs the package in editable mode, so code changes take effect immediately.

---

## 🧮 Dependencies

FisiocomPINN automatically installs its core requirements:

| Library      | Version ≥ | Purpose                  |
| ------------ | --------- | ------------------------ |
| `torch`      | 2.0.0     | Neural network engine    |
| `numpy`      | 1.24      | Numerical computations   |
| `matplotlib` | 3.7       | Visualization            |
| `h5py`       | 3.8       | HDF5 I/O support         |
| `chaospy`    | 4.3       | Stochastic sampling / UQ |

You can also install them manually via:

```bash
pip install torch numpy matplotlib h5py chaospy
```

For a detailed scientific example (including biological ODEs), see
👉 [`examples/Example_PINN_EDO.ipynb`](./examples/Example_PINN_EDO.ipynb)

---

## 🧬 Citation

If you use **FisiocomPINN** in your research, please cite it as:

> Werneck, Y., & Esterci, T. (2025). *FisiocomPINN: A Physics-Informed Neural Network framework for scientific computing in PyTorch*. GitHub repository.
> Available at: [https://github.com/ybwerneck/Pinn-Torch](https://github.com/ybwerneck/Pinn-Torch)

### BibTeX

```bibtex
@misc{fisiocompinn2025,
  author       = {Yan Werneck and Thiago Esterci},
  title        = {FisiocomPINN: A Physics-Informed Neural Network framework for scientific computing in PyTorch},
  year         = {2025},
  howpublished = {\url{https://github.com/ybwerneck/Pinn-Torch}},
  note         = {Version 0.1.0. Part of the FisiocomPINN PhD research project, Federal University of Juiz de Fora (UFJF).}
}
```

---

## 📖 License

Distributed under the **GNU General Public License v3.0 (GPLv3)**.
See [`LICENSE.md`](./LICENSE.md) or [the full text](https://www.gnu.org/licenses/gpl-3.0.txt) for details.

---

## 👩‍🔬 Authors & Acknowledgments

**Developed by:**

* **Yan Werneck** — Lead Developer (UFJF / Fisiocomp Research Group)

* **Thiago Esterci** — Co-developer (UFJF / Fisiocomp Research Group)

**Supervised Research:**
Federal University of Juiz de Fora (UFJF) — *PhD in Computational Modeling (PPGMC)*
Fisiocom Group — *Computational Physiology and High-Perfomance Computing Laboratory*


