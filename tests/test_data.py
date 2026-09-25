import torch
from overfitlab.data import get_dataloaders


def test_get_dataloaders():
    train_loader, val_loader = get_dataloaders(n_samples=100, batch_size=10, random_state=42)

    assert len(train_loader.dataset) == 67  # 67% of 100
    assert len(val_loader.dataset) == 33  # 33% of 100

    # Check shape of batch
    for X, y in train_loader:
        assert X.shape == (10, 2)
        assert y.shape == (10, 1)
        assert isinstance(X, torch.FloatTensor)
        break
