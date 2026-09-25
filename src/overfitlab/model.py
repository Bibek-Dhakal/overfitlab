import torch.nn as nn


class BaselineMLP(nn.Module):
    """
    A heavily over-parameterized Multi-Layer Perceptron.
    Designed to easily memorize small datasets.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


class RegularizedMLP(nn.Module):
    """
    The corrected architecture. Matches capacity of BaselineMLP but introduces
    Dropout layers to prevent severe co-adaptation of neurons (overfitting).
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 256, dropout_rate: float = 0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)
