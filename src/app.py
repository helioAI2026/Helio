import config
from src.face_detector import FaceDetector
from src.drowsiness    import Drowsiness
from src.alert         import Alert
from src.video         import Camera, Display

class App:
    def __init__(self):
        self.detector  = FaceDetector(config.FACE_MODEL)

        self.drowsiness = Drowsiness(
            config.EAR_THRESHOLD, 
            config.CONSEC_FRAMES, 
            config.ALERT_COOLDOWN
        )

        self.alert     = Alert()

        self.camera    = Camera(
            config.CAMERA_ID, 
            config.CAMERA_WIDTH, 
            config.CAMERA_HEIGHT
        )
        
        self.display   = Display(
            config.WINDOW_TITLE, 
            config.WINDOW_WIDTH, 
            config.WINDOW_HEIGHT
        )

    def run(self):
        while self.camera.cap.isOpened():
            frame = self.camera.read()
            if frame is None:
                break

            left, right = self.detector.detect(frame)
            drowsy, should_alert, ear = self.drowsiness.update(left, right)

            img = self.display.render(frame, left, right, drowsy)
            self.alert.draw(img, drowsy, ear)

            if not self.display.show(img):
                break

        self.camera.release()
        self.display.close()