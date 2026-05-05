import config
from src.face_detector import FaceDetector
from src.drowsiness    import Drowsiness
from src.perclos       import PERCLOS
from src.yawn_detector import YawnDetector
from src.alert         import Alert
from src.video         import Camera, Display


class App:
    def __init__(self):
        self.detector = FaceDetector(config.FACE_MODEL)

        self.drowsiness = Drowsiness(
            config.EAR_THRESHOLD,
            config.CONSEC_FRAMES,
            config.ALERT_COOLDOWN,
        )

        self.perclos = PERCLOS(
            ear_threshold  = config.EAR_THRESHOLD,
            fps            = config.CAMERA_FPS,
            window_seconds = config.PERCLOS_WINDOW,
            alert_ratio    = config.PERCLOS_ALERT_RATIO,
        )

        self.yawn = YawnDetector(
            mar_threshold = config.MAR_THRESHOLD,
            consec_frames = config.YAWN_CONSEC_FRAMES,
            cooldown      = config.YAWN_COOLDOWN,
            history       = config.YAWN_HISTORY,
        )

        self.alert   = Alert()
        self.camera  = Camera(config.CAMERA_ID, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        self.display = Display(config.WINDOW_TITLE, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

    def run(self):
        while self.camera.cap.isOpened():
            frame = self.camera.read()
            if frame is None:
                break

            left, right, mouth_mar, mouth_draw = self.detector.detect(frame)
            face_detected = left is not None  

            drowsy, _, ear                       = self.drowsiness.update(left, right)
            perclos_ratio, perclos_alert         = self.perclos.update(ear, face_detected)
            mar, yawning, yawn_alert, yawn_count = self.yawn.update(mouth_mar)

            img = self.display.render(
                frame,
                left_eye   = left,
                right_eye  = right,
                drowsy     = drowsy or perclos_alert,
                mouth_draw = mouth_draw,
                yawning    = yawning,
            )

            self.alert.draw(
                img,
                drowsy        = drowsy,
                ear           = ear,
                perclos       = perclos_ratio,
                perclos_alert = perclos_alert,
                yawning       = yawning,
                yawn_alert    = yawn_alert,
                mar           = mar,
                yawn_count    = yawn_count,
            )

            if not self.display.show(img):
                break

        self.camera.release()
        self.display.close()