from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.eye_state.classifier import EyeStateClassifier
from src.eye_state.geometry import Box, Face, crop_eye, eye_boxes
from src.eye_state.yunet import YuNet


class ModelMissingError(Exception):
    pass


@dataclass
class EyeStateResult:
    face: Face | None
    boxes: tuple[Box | None, Box | None]  # (right, left)
    p_right: float | None
    p_left: float | None
    p_closed: float | None

    @property
    def face_detected(self) -> bool:
        return self.p_closed is not None


def _require(path: str, hint: str) -> None:
    if not Path(path).exists():
        raise ModelMissingError(f"Modelo não encontrado: {path}. Rode {hint}")


class EyeStatePipeline:
    """Camera frame (RGB float [0,1]) -> face -> eye crops -> p_closed."""

    def __init__(self, yunet_path: str, classifier_path: str):
        _require(yunet_path, "uv run python scripts/download_models.py")
        _require(classifier_path, "cd training && uv run python -m eyestate.train && uv run python -m eyestate.export "
                                  "(veja o README), ou use DETECTOR = \"mediapipe\" em config.py")
        self.detector = YuNet(yunet_path)
        self.classifier = EyeStateClassifier(classifier_path)

    def process(self, frame: np.ndarray) -> EyeStateResult:
        rgb = (frame * 255).astype(np.uint8) if frame.dtype != np.uint8 else frame
        face = self.detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if face is None:
            return EyeStateResult(None, (None, None), None, None, None)

        boxes = eye_boxes(face, rgb.shape)
        valid = [i for i, b in enumerate(boxes) if b is not None]
        if not valid:
            return EyeStateResult(face, boxes, None, None, None)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        probs = self.classifier.predict(np.stack([crop_eye(gray, boxes[i]) for i in valid]))
        p: list[float | None] = [None, None]
        for i, pr in zip(valid, probs):
            p[i] = float(pr)
        return EyeStateResult(face, boxes, p[0], p[1], float(np.mean(probs)))
