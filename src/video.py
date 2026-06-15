import cv2
import numpy as np

from src.alert import Alert


class Camera:
    def __init__(self, cam_id, width, height, fps=30):
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Câmera {cam_id} não disponível")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def release(self):
        self.cap.release()


class Display:
    def __init__(self, title, width, height):
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, width, height)
        self.title = title
        self.hud_h = Alert.HUD_HEIGHT

    def render(self, frame, left_eye, right_eye, drowsy, mouth_draw=None, yawning=False):
        img = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        eye_color = (60, 60, 220) if drowsy else (60, 200, 60)
        for eye in (left_eye, right_eye):
            if eye is not None:
                pts = eye.astype(np.int32)
                if pts[:, 1].min() > self.hud_h:
                    cv2.polylines(img, [pts], True, eye_color, 1, cv2.LINE_AA)
                    for p in pts:
                        cv2.circle(img, tuple(p), 2, eye_color, -1, cv2.LINE_AA)

        if mouth_draw is not None:
            mouth_color = (0, 200, 220) if yawning else (200, 200, 60)
            upper = mouth_draw['upper'].astype(np.int32)
            lower = mouth_draw['lower'].astype(np.int32)
            if upper[:, 1].min() > self.hud_h:
                cv2.polylines(img, [upper], False, mouth_color, 1, cv2.LINE_AA)
                cv2.polylines(img, [lower], False, mouth_color, 1, cv2.LINE_AA)
                for pts in (upper, lower):
                    for p in pts:
                        cv2.circle(img, tuple(p), 2, mouth_color, -1, cv2.LINE_AA)

        return img

    def show(self, img):
        cv2.imshow(self.title, img)
        return cv2.waitKey(1) & 0xFF != ord('q')

    def close(self):
        cv2.destroyAllWindows()