# OverfitLab

A small-scale deep learning experiment explicitly designed to demonstrate the diagnosis of overfitting and the practical
application of correction techniques.

OverfitLab proves fundamentals in Deep Learning diagnostic loops: observing a failure mode (train/validation
divergence), reasoning about its cause, implementing targeted fixes (regularization/dropout), and validating the
improved generalization.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/overfitlab.git
cd overfitlab

# Install package and dev dependencies
pip install -e .[dev]

# Run the complete experiment (Train baseline -> Train corrected -> Plot results)
python -m overfitlab.experiment
```

## 📚 Documentation Index

- [Diagnostic Report & Results](docs/report/README.md): Detailed breakdown of the overfitting cause and applied fixes.
- [Code Quality Guidelines](docs/code_quality.md): Instructions on linting, formatting, and commit hooks.
- [Usage Guide](docs/usage/README.md): Instructions on running experiments and utilizing the data.
- [Testing Guide](docs/testing/README.md): Information regarding unit tests and CI integration.
- [Architecture](docs/architecture/README.md): Details about the neural network layers and training loops.
