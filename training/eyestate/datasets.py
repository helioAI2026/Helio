"""Datasets of eye crops for the open/closed classifier.

Every source is a *DatasetAdapter* that lists `Sample(path, label, subject)`;
label 1 = closed, 0 = open. Splits are made by subject so the same person
never appears in train and test.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

INPUT_SIZE = 64


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int  # 1 = closed, 0 = open
    subject: str


class DatasetAdapter(Protocol):
    def samples(self) -> list[Sample]: ...


class MRLAdapter:
    """MRL Eye Dataset: sXXXX_IIIII_gender_glasses_EYESTATE_reflections_lighting_sensor.png
    EYESTATE: 0 = closed, 1 = open."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def samples(self) -> list[Sample]:
        out = []
        for p in sorted(self.root.rglob("*.png")):
            parts = p.stem.split("_")
            if len(parts) != 8 or not parts[0].startswith("s"):
                continue
            out.append(Sample(p, 1 - int(parts[4]), parts[0]))
        return out


class LiveClipAdapter:
    """Webcam clip recorded with record_eval: root/{open,closed}/NNNNN.png."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def samples(self) -> list[Sample]:
        out = []
        for label_name, label in (("open", 0), ("closed", 1)):
            for p in sorted((self.root / label_name).glob("*.png")):
                out.append(Sample(p, label, "live"))
        return out


def subject_split(
    samples: list[Sample], val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    subjects = sorted({s.subject for s in samples})
    random.Random(seed).shuffle(subjects)
    n = len(subjects)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    test_s = set(subjects[:n_test])
    val_s = set(subjects[n_test:n_test + n_val])
    train = [s for s in samples if s.subject not in test_s | val_s]
    val = [s for s in samples if s.subject in val_s]
    test = [s for s in samples if s.subject in test_s]
    return train, val, test


def load_gray(path: Path, size: int = INPUT_SIZE) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = img.shape[0]
    # contrast / brightness ±30%
    img = (img - 0.5) * rng.uniform(0.7, 1.3) + 0.5 + rng.uniform(-0.3, 0.3)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    angle = rng.uniform(-10, 10)
    scale = rng.uniform(0.9, 1.1)
    m = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
    img = cv2.warpAffine(img, m, (size, size), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        img = img[:, ::-1]
    return np.clip(img, 0.0, 1.0)


class EyeDataset(Dataset):
    def __init__(self, samples: list[Sample], train: bool, seed: int = 0):
        self.samples = samples
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = load_gray(s.path)
        if self.train:
            img = augment(img, self.rng)
        x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None]
        return x, torch.tensor(float(s.label), dtype=torch.float32)
