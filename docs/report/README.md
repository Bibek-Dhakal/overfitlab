# Diagnostic Report: Overfitting in Deep Learning

This document covers the required reasoning trail demonstrating a diagnostic loop in Deep Learning: observing an issue,
hypothesizing its cause, and applying a documented fix.

## 1. The Experiment Context

We simulate a classic failure mode using a highly non-linear, small synthetic dataset (`sklearn.datasets.make_moons`
with substantial noise, N=300).

## 2. Baseline Run (The Failure Mode)

- **Architecture**: A deep Multi-Layer Perceptron (MLP) featuring 3 hidden layers of 256 units each.
- **Result**: The model quickly memorizes the training data.
- **Diagnosis**:
    * The divergence between Train Loss and Validation Loss is severe.
    * Train Loss rapidly drops toward 0, while Validation Loss rebounds and continually increases.
    * *Hypothesis:* The model is massively over-parameterized relative to the dataset size. It possesses enough capacity
      to "memorize" the gaussian noise present in the training set instead of learning the underlying distribution,
      destroying its ability to generalize.

## 3. Applied Corrections

To combat this, we implement two specific, named techniques in the `RegularizedMLP`:

1. **Dropout (p=0.5)**: Inserted between hidden layers. This randomly zeroes out a fraction of activations during
   training, preventing complex co-adaptations of neurons and forcing the network to distribute representations
   robustly.
2. **Weight Decay (L2 Regularization, 1e-3)**: Added to the Adam optimizer. This adds a penalty proportional to the sum
   of squared weights to the loss function, heavily discouraging abnormally large weight values that characteristically
   appear when memorizing noisy outliers.

## 4. Corrected Run (The Fix)

- **Architecture**: The identical hidden layer sizes, but intertwined with Dropout and optimized with Weight Decay.
- **Result**:
    * Train Loss remains higher (it can no longer perfectly memorize the noise).
    * Validation Loss closely tracks Train Loss, plateauing smoothly rather than spiking upward.
    * Generalization is restored.

*You can visualize this exactly by running `python -m overfitlab.experiment` and viewing `experiment_results.png`.*