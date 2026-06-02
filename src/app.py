import config
from helpers.aws import MQTT
from helpers.consts import DEVICE_ID, DEVICE_SECRET
from helpers.ride import Ride
from models.event import Event
from src.face_detector import FaceDetector
from src.sleepiness_detector import SleepinessDetector
from src.event_logger import EventLogger
from src.alert import Alert
from src.video import Camera, Display


class App:
    def __init__(self):
        self.ride = Ride(
            device_id=DEVICE_ID,
            device_secret=DEVICE_SECRET.encode('utf-8'),
        )

        ride_id = self.ride.generate_ride_id()

        self.mqtt = MQTT(
            region='us-east-1',
            topic='test'
        )

        test_event = Event(
            device_id=DEVICE_ID,
            **ride_id,
            data=[
                {"test": "123"}
            ]
        )

        self.mqtt.send_payload(test_event)

        self.detector = FaceDetector(config.FACE_MODEL)
        self.sleepiness = SleepinessDetector(ear_threshold=config.EAR_THRESHOLD, fps=config.CAMERA_FPS)
        self.logger = EventLogger()
        self.alert = Alert()
        self.camera = Camera(config.CAMERA_ID, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        self.display = Display(config.WINDOW_TITLE, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self._last_alert_frame = 0
        self._alert_cooldown = int(config.CAMERA_FPS * config.ALERT_COOLDOWN_SECONDS)
        self._alert_start_frame = None
        self._alert_overlay_duration = config.ALERT_OVERLAY_DURATION_SECONDS

    def run(self):
        frame_count = 0
        was_alert = False
        while self.camera.cap.isOpened():
            frame = self.camera.read()
            if frame is None:
                break

            left, right, _, mouth_draw = self.detector.detect(frame)
            face_detected = left is not None
            ear = (self.detector._ear(left) + self.detector._ear(right)) / 2.0 if face_detected else 0.0

            result = self.sleepiness.compute(ear=ear, face_detected=face_detected)

            if result['alert']:
                if self._alert_start_frame is None:
                    self._alert_start_frame = frame_count
                if not was_alert and frame_count - self._last_alert_frame >= self._alert_cooldown:
                    self.logger.log_alert(result)
                    self._last_alert_frame = frame_count
                was_alert = True
            else:
                self._alert_start_frame = None
                was_alert = False

            alert_duration = (frame_count - self._alert_start_frame) / config.CAMERA_FPS if self._alert_start_frame is not None else 0.0
            img = self.display.render(frame, left_eye=left, right_eye=right, drowsy=result['alert'], mouth_draw=mouth_draw, yawning=False)
            self.alert.draw(img, result, alert_duration=alert_duration, overlay_threshold=self._alert_overlay_duration)

            if not self.display.show(img):
                break

            frame_count += 1

        self.camera.release()
        self.display.close()
        print(f"Logs: {self.logger.get_session_file()}")


if __name__ == "__main__":
    app = App()
    app.run()