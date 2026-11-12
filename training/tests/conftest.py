import cv2
import numpy as np
import pytest


@pytest.fixture
def fake_mrl(tmp_path):
    """Árvore no formato MRL: 10 sujeitos, olhos fechados escuros e abertos claros."""
    root = tmp_path / "mrl"
    rng = np.random.default_rng(0)
    for s in range(1, 11):
        d = root / f"s{s:04d}"
        d.mkdir(parents=True)
        for i in range(20):
            state = i % 2  # 0 = fechado, 1 = aberto (campo E do MRL)
            base = 200 if state == 1 else 40
            img = np.clip(base + rng.normal(0, 10, (40, 48)), 0, 255).astype(np.uint8)
            name = f"s{s:04d}_{i:05d}_0_0_{state}_0_0_01.png"
            cv2.imwrite(str(d / name), img)
    (root / "README.txt").write_text("not an image")
    return root
