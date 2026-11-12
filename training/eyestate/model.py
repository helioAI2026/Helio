import torch
from torch import nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class EyeNet(nn.Module):
    """Tiny open/closed eye classifier.

    Input  (N, 1, 64, 64) grayscale in [0, 1]
    Output (N, 1) logit of "closed"
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(_block(1, 16), _block(16, 32), _block(32, 64), _block(64, 96))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(96, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=(2, 3))  # global average pool (exports to ReduceMean)
        return self.head(x)
