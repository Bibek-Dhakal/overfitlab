import torch
from overfitlab.model import BaselineMLP, RegularizedMLP
from overfitlab.train import train_model
from torch.utils.data import DataLoader, TensorDataset


def test_model_forward():
    baseline = BaselineMLP(input_dim=2, hidden_dim=16)
    regularized = RegularizedMLP(input_dim=2, hidden_dim=16)

    dummy_input = torch.randn(5, 2)

    out_base = baseline(dummy_input)
    out_reg = regularized(dummy_input)

    assert out_base.shape == (5, 1)
    assert out_reg.shape == (5, 1)


def test_train_loop():
    # Setup dummy data
    X = torch.randn(20, 2)
    y = torch.randint(0, 2, (20, 1)).float()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10)

    model = BaselineMLP(input_dim=2, hidden_dim=4)

    # Run only 2 epochs to verify it doesn't crash
    history = train_model(model, loader, loader, epochs=2, lr=0.1)

    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) == 2
