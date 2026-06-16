import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


class FaceDetector:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    MOUTH_LOWER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    MOUTH_MAR = [61, 291, 39, 13, 269, 375, 14, 405]

    _INPUT_SIZE = 192

    def __init__(self, model_path):
        landmark_path = model_path.replace("face_landmarker.task", "face_landmark.tflite")
        detection_path = model_path.replace("face_landmarker.task", "face_detection.tflite")

        self._landmark = tflite.Interpreter(model_path=landmark_path, num_threads=4)
        self._landmark.allocate_tensors()
        self._landmark_in = self._landmark.get_input_details()[0]
        self._landmark_out = self._landmark.get_output_details()[0]

        self._detector = tflite.Interpreter(model_path=detection_path, num_threads=4)
        self._detector.allocate_tensors()
        self._detector_in = self._detector.get_input_details()[0]
        self._detector_out = self._detector.get_output_details()

    def _detect_face(self, frame):
        h, w = frame.shape[:2]
        size = 128
        blob = cv2.resize(frame, (size, size)).astype(np.float32) / 127.5 - 1.0
        self._detector.set_tensor(self._detector_in['index'], blob[np.newaxis])
        self._detector.invoke()
        scores = self._detector.get_tensor(self._detector_out[0]['index'])[0]
        boxes = self._detector.get_tensor(self._detector_out[1]['index'])[0]

        best = int(np.argmax(scores))
        if scores[best] < 0.5:
            return None

        y1, x1, y2, x2 = boxes[best]
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
        pad = int(max(x2 - x1, y2 - y1) * 0.2)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        return x1, y1, x2, y2

    def detect(self, frame):
        h, w = frame.shape[:2]
        img = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

        bbox = self._detect_face(img)
        if bbox is None:
            return None, None, None, None

        x1, y1, x2, y2 = bbox
        face = img[y1:y2, x1:x2]
        face_h, face_w = face.shape[:2]

        resized = cv2.resize(face, (self._INPUT_SIZE, self._INPUT_SIZE)).astype(np.float32) / 127.5 - 1.0
        self._landmark.set_tensor(self._landmark_in['index'], resized[np.newaxis])
        self._landmark.invoke()
        raw = self._landmark.get_tensor(self._landmark_out['index'])[0]

        n = len(raw) // 3
        lm = raw[:n * 3].reshape(n, 3)
        lm[:, 0] = lm[:, 0] / self._INPUT_SIZE * face_w + x1
        lm[:, 1] = lm[:, 1] / self._INPUT_SIZE * face_h + y1
        lm = lm[:, :2].astype(np.float32)

        return lm[self.LEFT_EYE], lm[self.RIGHT_EYE], [lm[i] for i in self.MOUTH_MAR], {
            'upper': lm[self.MOUTH_UPPER],
            'lower': lm[self.MOUTH_LOWER],
        }

    def _ear(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.0
