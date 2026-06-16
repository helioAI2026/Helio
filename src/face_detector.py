import cv2
import mediapipe as mp
from mediapipe.tasks import python
import numpy as np


class FaceDetector:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self, model_path):
        opts = python.vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            num_faces=1,
        )
        self.detector = python.vision.FaceLandmarker.create_from_options(opts)

    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        if not result.face_landmarks:
            return None, None

        lm = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]], dtype=np.float32)

        if lm[33][0] >= lm[362][0]:
            return None, None

        return lm[self.LEFT_EYE], lm[self.RIGHT_EYE]

    def _ear(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.0