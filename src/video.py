import cv2, numpy as np

class Camera:
    def __init__(self, cam_id, width, height):
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Câmera {cam_id} não disponível")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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

    def render(self, frame, left_eye, right_eye, drowsy):
        img   = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        color = (60, 60, 220) if drowsy else (60, 200, 60)
        for eye in (left_eye, right_eye):
            if eye is not None:
                pts = eye.astype(np.int32)
                cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)
                for p in pts:
                    cv2.circle(img, tuple(p), 2, color, -1, cv2.LINE_AA)
        return img

    def show(self, img):
        cv2.imshow(self.title, img)
        return cv2.waitKey(1) & 0xFF != ord('q')

    def close(self):
        cv2.destroyAllWindows()