# 🤝 Contributing to FisiocomPINN

Thank you for your interest in contributing to **FisiocomPINN** — a Python framework for **Physics-Informed Neural Networks (PINNs)** developed within the *Fisiocomp Group (UFJF)*.

Your contributions help improve the reproducibility, scalability, and scientific value of this PhD research project.
Whether you are fixing a bug, adding a feature, or writing documentation, your help is welcome and appreciated!

---

## 📋 Table of Contents

1. [Code of Conduct](#-code-of-conduct)
2. [How to Contribute](#-how-to-contribute)
3. [Project Setup](#-project-setup)
4. [Development Guidelines](#-development-guidelines)
5. [Testing](#-testing)
6. [Commit and Pull Request Rules](#-commit-and-pull-request-rules)
7. [Citation and Acknowledgment](#-citation-and-acknowledgment)

---

## 🧭 Code of Conduct

This project adheres to the principles of **collaboration, academic integrity, and open science**.
We expect all contributors to:

* Be respectful and professional in all communications.
* Properly attribute scientific ideas and code.
* Avoid plagiarism and data misuse.
* Maintain reproducibility in experiments and results.

By participating in this project, you agree to uphold these principles.

---

## 🛠️ How to Contribute

### 1. Report Bugs or Issues

* Open a **GitHub Issue** with:

  * a clear and concise title,
  * a description of the problem,
  * steps to reproduce,
  * and (if applicable) code snippets or screenshots.

### 2. Suggest Enhancements

* Use the “Feature request” label in Issues.
* Explain *why* the feature is useful and, if possible, link relevant literature or prior work.

### 3. Submit Code Contributions

* Fork the repository:

  ```bash
  git clone https://github.com/ybwerneck/Pinn-Torch.git
  cd Pinn-Torch
  git checkout -b feature/your-feature-name
  ```
* Implement your changes following the [Development Guidelines](#-development-guidelines).
* Commit and push your branch:

  ```bash
  git add .
  git commit -m "Add feature: short description"
  git push origin feature/your-feature-name
  ```
* Open a **Pull Request (PR)** on GitHub, describing your contribution and linking related Issues.

---

## ⚙️ Project Setup

To set up a development environment:

```bash
git clone https://github.com/ybwerneck/Pinn-Torch.git
cd Pinn-Torch
pip install -e .[dev]
```

If you prefer a clean installation without developer extras:

```bash
pip install -e .
```

---

## 💡 Development Guidelines

Please follow these guidelines to maintain code quality and readability:

### Code Style

* Follow **PEP8** conventions (`black` or `autopep8` formatters are recommended).
* Use **descriptive variable names** and **type hints** where appropriate.
* Comment equations, derivations, and assumptions — this is crucial for reproducibility.

### Documentation

* Each public function or class should include a **docstring**:

  ```python
  def train_model(model, data, epochs):
      """
      Train a PINN model with the given dataset.
      
      Args:
          model (torch.nn.Module): Neural network to train.
          data (torch.Tensor): Input training data.
          epochs (int): Number of training epochs.
      Returns:
          dict: Training loss history.
      """
  ```
* If you add a new example notebook, place it in:

  ```
  examples/
  ```

  and document it at the top with the problem, equations, and boundary conditions used.

### Folder Structure

```
fisiocomPinn/
│
├── Loss.py          # Loss definitions
├── Trainer.py       # Training logic
├── Utils.py         # Data utilities
├── dependencies.py  # Common imports and helper tools
└── ...
```
 
* Keep example notebooks (`examples/*.ipynb`) clean and runnable without external dependencies beyond those listed in `setup.py`.

---

## Testing

Run the test suite locally after installing the package dependencies:

```bash
python -B -m unittest discover -s tests -v
```

The [Tests workflow](.github/workflows/tests.yml) runs this suite on Ubuntu with
Python 3.11 and CPU-only PyTorch whenever a pull request targeting `main` is
opened, reopened, or updated with new commits. It installs dependencies from
`setup.py`; no development extra is required. New commits cancel any older run
still in progress for the same pull request.

To block merging when tests fail, configure a branch protection rule or ruleset
for `main` requiring the `Unit tests (CPU, Python 3.11)` status check. The workflow
alone reports results and does not configure branch protection.

---

## 🧾 Commit and Pull Request Rules

Follow these conventions for clarity and traceability:

### Commit Messages

Use **short, descriptive** messages, prefixed with a category:

| Prefix      | Purpose                    |
| ----------- | -------------------------- |
| `feat:`     | new feature                |
| `fix:`      | bug fix                    |
| `docs:`     | documentation              |
| `refactor:` | non-functional code change |
| `test:`     | add or modify tests        |
| `chore:`    | maintenance tasks          |

Example:

```
feat: add adaptive learning rate scheduler to Trainer class
```

### Pull Requests

* Reference related Issues in your PR description.
* Include **summary of the change**, **impact**, and **testing evidence**.
* Ensure code passes all tests and lint checks before submission.

