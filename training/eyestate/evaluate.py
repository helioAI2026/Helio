"""Avalia o modelo ONNX nos sujeitos de teste do MRL e, opcionalmente, no clipe ao vivo.

    uv run python -m eyestate.evaluate --onnx ../model/eye_state.onnx \
        --checkpoint checkpoints/best.pt [--live data/live]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

from eyestate.datasets import LiveClipAdapter, MRLAdapter, Sample, load_gray


def predict(session: ort.InferenceSession, samples: list[Sample], batch: int = 512) -> np.ndarray:
    probs = []
    for i in range(0, len(samples), batch):
        x = np.stack([load_gray(s.path) for s in samples[i:i + batch]])[:, None]
        logit = session.run(None, {"input": x.astype(np.float32)})[0][:, 0]
        probs.append(1.0 / (1.0 + np.exp(-logit)))
    return np.concatenate(probs) if probs else np.zeros(0)


def evaluate(onnx_path: Path, samples: list[Sample], threshold: float = 0.5) -> dict:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    y = np.array([s.label for s in samples])
    p = predict(session, samples)
    pred = (p > threshold).astype(int)
    auc = float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")
    return {
        "n": len(samples),
        "accuracy": float((pred == y).mean()) if len(y) else 0.0,
        "roc_auc": auc,
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),  # linhas: real aberto/fechado
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, default=Path("../model/eye_state.onnx"))
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    ap.add_argument("--data", type=Path, default=Path("data/mrl"))
    ap.add_argument("--live", type=Path, default=None)
    a = ap.parse_args()

    test_subjects = set(torch.load(a.checkpoint, weights_only=False)["meta"]["test_subjects"])
    test = [s for s in MRLAdapter(a.data).samples() if s.subject in test_subjects]
    print("MRL (sujeitos de teste):", json.dumps(evaluate(a.onnx, test)))
    if a.live:
        print("Ao vivo (webcam):", json.dumps(evaluate(a.onnx, LiveClipAdapter(a.live).samples())))


if __name__ == "__main__":
    main()
