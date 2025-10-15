import torch
import torch.nn as nn


class MLPStateEncoder(nn.Module):
    """Encoder mặc định (có thể thay bằng GNN sau).
    Nhận state vector (đã flatten + concat prefs) -> latent.
    """
    def __init__(self, in_dim: int, hid_dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, out_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
