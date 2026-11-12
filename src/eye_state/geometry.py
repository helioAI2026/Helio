from dataclasses import dataclass

import cv2
import numpy as np

import config

Box = tuple[int, int, int, int]  # (x0, y0, x1, y1), x1/y1 exclusive


@dataclass
class Face:
    box: np.ndarray        # (x, y, w, h)
    right_eye: np.ndarray  # (x, y) pixels — the subject's right eye
    left_eye: np.ndarray
    score: float


def eye_boxes(
    face: Face,
    frame_shape: tuple[int, int],
    scale: float = config.EYE_CROP_SCALE,
    min_dist: float = config.MIN_EYE_DISTANCE_PX,
) -> tuple[Box | None, Box | None]:
    """Square crop around each eye, side = scale * inter-eye distance.

    A box that leaves the frame is None; both are None when the face is too small.
    """
    h, w = frame_shape[:2]
    dist = float(np.linalg.norm(face.left_eye - face.right_eye))
    if dist < min_dist:
        return None, None
    half = scale * dist / 2.0

    def box(center: np.ndarray) -> Box | None:
        x0, y0 = int(round(center[0] - half)), int(round(center[1] - half))
        x1, y1 = int(round(center[0] + half)), int(round(center[1] + half))
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None
        return x0, y0, x1, y1

    return box(face.right_eye), box(face.left_eye)


def crop_eye(gray: np.ndarray, box: Box, size: int = 64) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = cv2.resize(gray[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
    return crop.astype(np.float32) / 255.0
