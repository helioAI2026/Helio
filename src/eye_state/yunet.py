import cv2
import numpy as np

from src.eye_state.geometry import Face


class YuNet:
    """OpenCV YuNet face detector. Each detection row:
    x, y, w, h, right_eye(x, y), left_eye(x, y), nose(x, y), right_mouth(x, y), left_mouth(x, y), score
    """

    def __init__(self, model_path: str, score_threshold: float = 0.7):
        self._det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold, 0.3, 5000)
        self._size: tuple[int, int] | None = None

    def detect(self, bgr: np.ndarray) -> Face | None:
        h, w = bgr.shape[:2]
        if self._size != (w, h):
            self._det.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        f = faces[int(np.argmax(faces[:, -1]))]
        return Face(box=f[0:4].copy(), right_eye=f[4:6].copy(), left_eye=f[6:8].copy(), score=float(f[-1]))
