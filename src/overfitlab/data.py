import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def get_dataloaders(
    n_samples: int = 300, noise: float = 0.25, batch_size: int = 32, random_state: int = 42
) -> tuple[DataLoader, DataLoader]:
    """
    Generates a deterministic dataset prone to overfitting due to high noise.
    Splits into train and validation sets, returning PyTorch DataLoaders.
    """
    # 1. Generate non-linear dummy data with noise
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    # 2. Deterministic split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33, random_state=random_state
    )

    # 3. Convert to Float tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)

    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).view(-1, 1)

    # 4. Create datasets and loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
