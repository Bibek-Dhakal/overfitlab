import matplotlib.pyplot as plt

from overfitlab.data import get_dataloaders
from overfitlab.model import BaselineMLP, RegularizedMLP
from overfitlab.train import train_model


def plot_results(
    baseline_history: dict, regularized_history: dict, filename: str = "experiment_results.png"
):
    """
    Plots the comparison between the overfitted model and the regularized model.
    """
    epochs = range(1, len(baseline_history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Baseline
    ax1.plot(epochs, baseline_history["train_loss"], label="Train Loss", color="blue")
    ax1.plot(epochs, baseline_history["val_loss"], label="Val Loss", color="red", linestyle="--")
    ax1.set_title("Baseline (Overfitted)")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Binary Cross Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Regularized
    ax2.plot(epochs, regularized_history["train_loss"], label="Train Loss", color="blue")
    ax2.plot(epochs, regularized_history["val_loss"], label="Val Loss", color="red", linestyle="--")
    ax2.set_title("Corrected (Dropout + L2 Weight Decay)")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Binary Cross Entropy Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\nPlot saved to {filename}")


def run_experiment():
    print("--- Starting OverfitLab Experiment ---")

    print("\n1. Generating Dataset...")
    train_loader, val_loader = get_dataloaders(n_samples=300, noise=0.3, batch_size=64)

    print("\n2. Training Baseline Model (Expect Overfitting)...")
    baseline = BaselineMLP()
    # Train for 800 epochs, no weight decay
    baseline_hist = train_model(
        baseline, train_loader, val_loader, epochs=800, lr=2e-3, weight_decay=0.0
    )

    print("\n3. Training Regularized Model (Expect Generalization)...")
    regularized = RegularizedMLP(dropout_rate=0.5)
    # Train for 800 epochs, WITH weight decay
    regularized_hist = train_model(
        regularized, train_loader, val_loader, epochs=800, lr=2e-3, weight_decay=1e-3
    )

    print("\n4. Generating Comparison Plot...")
    plot_results(baseline_hist, regularized_hist)
    print("\nExperiment Complete.")


if __name__ == "__main__":
    run_experiment()
