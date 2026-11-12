from pathlib import Path

import cv2
import numpy as np
import torch

from eyestate.datasets import EyeDataset, MRLAdapter, Sample, subject_split


def test_mrl_parses_label_and_subject(tmp_path):
    d = tmp_path / "s0003"
    d.mkdir()
    img = np.zeros((10, 10), np.uint8)
    cv2.imwrite(str(d / "s0003_00012_0_1_0_0_1_01.png"), img)
    cv2.imwrite(str(d / "s0003_00013_0_1_1_0_1_01.png"), img)
    by_name = {s.path.name: s for s in MRLAdapter(tmp_path).samples()}
    closed = by_name["s0003_00012_0_1_0_0_1_01.png"]
    opened = by_name["s0003_00013_0_1_1_0_1_01.png"]
    assert closed.label == 1 and closed.subject == "s0003"
    assert opened.label == 0


def test_mrl_ignores_non_png(fake_mrl):
    samples = MRLAdapter(fake_mrl).samples()
    assert len(samples) == 200
    assert all(s.path.suffix == ".png" for s in samples)


def test_subject_split_no_leakage(fake_mrl):
    samples = MRLAdapter(fake_mrl).samples()
    tr, va, te = subject_split(samples)
    subj = [{s.subject for s in part} for part in (tr, va, te)]
    assert all(subj)
    assert not (subj[0] & subj[1]) and not (subj[0] & subj[2]) and not (subj[1] & subj[2])
    assert subj[0] | subj[1] | subj[2] == {s.subject for s in samples}


def test_subject_split_deterministic(fake_mrl):
    samples = MRLAdapter(fake_mrl).samples()
    assert subject_split(samples, seed=3) == subject_split(samples, seed=3)


def test_eye_dataset_shape(fake_mrl):
    samples = MRLAdapter(fake_mrl).samples()
    for train in (True, False):
        x, y = EyeDataset(samples[:4], train=train)[0]
        assert x.shape == (1, 64, 64)
        assert x.dtype == torch.float32
        assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
        assert y.dtype == torch.float32
