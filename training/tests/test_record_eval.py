import numpy as np

from eyestate.datasets import LiveClipAdapter
from eyestate.record_eval import save_crop


def test_save_crop_numbering(tmp_path):
    crop = np.full((64, 64), 0.5, np.float32)
    a = save_crop(crop, tmp_path, "closed")
    b = save_crop(crop, tmp_path, "closed")
    assert (a.name, b.name) == ("00000.png", "00001.png")
    samples = LiveClipAdapter(tmp_path).samples()
    assert len(samples) == 2 and all(s.label == 1 for s in samples)
