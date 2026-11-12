import torch

from eyestate.model import EyeNet
from eyestate.train import train


def test_eyenet_output_shape_and_size():
    m = EyeNet()
    assert m(torch.zeros(2, 1, 64, 64)).shape == (2, 1)
    assert sum(p.numel() for p in m.parameters()) < 250_000


def test_train_smoke(fake_mrl, tmp_path):
    ckpt = train(fake_mrl, tmp_path / "ckpt", epochs=2, batch_size=32, device="cpu")
    assert ckpt.exists()
    data = torch.load(ckpt, weights_only=False)
    assert "state_dict" in data
    assert data["meta"]["test_subjects"]
    assert "split_seed" in data["meta"]
