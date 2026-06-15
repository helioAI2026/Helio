import mediapipe as mp
from mediapipe.tasks import python
import numpy as np


class FaceDetector:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    MOUTH_LOWER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    MOUTH_MAR = [61, 291, 39, 13, 269, 375, 14, 405]

    def __init__(self, model_path):
        opts = python.vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            num_faces=1,
        )
        self.detector = python.vision.FaceLandmarker.create_from_options(opts)

    def detect(self, frame):
        h, w = frame.shape[:2]
        img = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))

        if not result.face_landmarks:
            return None, None, None, None

        lm = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]], dtype=np.float32)
        
        return lm[self.LEFT_EYE], lm[self.RIGHT_EYE], [lm[i] for i in self.MOUTH_MAR], {
            'upper': lm[self.MOUTH_UPPER],
            'lower': lm[self.MOUTH_LOWER],
        }

    def _ear(self, eye):
        # Eye Aspect Ratio (EAR) = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
        # Mede a proporção altura/largura do olho (0.0 fechado → ~0.4 aberto)
        A = np.linalg.norm(eye[1] - eye[5])  # Distância vertical superior
        B = np.linalg.norm(eye[2] - eye[4])  # Distância vertical inferior
        C = np.linalg.norm(eye[0] - eye[3])  # Distância horizontal (largura)
        return (A + B) / (2.0 * C) if C > 0 else 0.0